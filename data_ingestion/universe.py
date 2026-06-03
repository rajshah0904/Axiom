"""
data_ingestion/universe.py — Universe management and contract filter.

Decides which contracts from the Kalshi API are admitted to the signal
computation universe. Called once per rebalance cycle from build_universe().

Three public entry points:

  is_in_universe(contract, book_depth_usd, now)   — pure filter, no I/O
  build_universe(client, db)                       — daily rebalance entry point
  persist_event_contract(db, contract, snap)       — thin bridge to ingest.py
"""

import logging
from datetime import datetime, timezone
from typing import Dict, FrozenSet, List, Tuple

from kalshi_client import (
    ContractObject,
    EventData,
    KalshiClient,
    PriceSnapshot,
)
from storage.db import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sport configuration constants — verified against Kalshi sandbox May 2026
# ---------------------------------------------------------------------------

SPORT_SERIES_TICKERS: Dict[str, str] = {
    "NBA":   "KXNBAGAME",
    "NFL":   "KXNFLGAME",
    "MLB":   "KXMLBGAME",
    "NHL":   "KXNHLGAME",
    "NCAAB": "KXNCAAMBGAME",
    "NCAAF": "KXNCAAFGAME",
    "UFC":   "KXUFCFIGHT",
}

# All sports actively polled.
ACTIVE_SPORTS: FrozenSet[str] = frozenset(SPORT_SERIES_TICKERS.keys())

# Universe filter thresholds — all named, none hardcoded elsewhere.
MIN_BOOK_DEPTH_USD: float = 10_000.0          # USD notional of YES bids
# CLAUDE.md universe filter requires "minimum 50 trades in past 7 days". The 7-day
# trade count needs price history and is enforced by the storage layer. volume_24h
# (24-hour contract count from the market object) is used here as a fast proxy to
# exclude completely illiquid contracts before the order book call. Not equivalent
# to the 7-day check — the full check runs separately after ingestion.
MIN_VOLUME_24H_CONTRACTS: float = 8.0          # proxy: 24h contract count, NOT 7-day trades
MIN_DAYS_TO_RESOLUTION: int = 1               # contracts resolving < 1d are excluded
MAX_DAYS_TO_RESOLUTION: int = 30              # too-distant contracts are excluded


# ---------------------------------------------------------------------------
# Universe filter — pure function, no I/O
# ---------------------------------------------------------------------------

def is_in_universe(
    contract: ContractObject,
    book_depth_usd: float,
    now: datetime,
) -> Tuple[bool, str]:
    """Apply all seven universe filters in order.

    Returns (True, "") when all pass. Returns (False, reason) on the first
    failure. ``reason`` is a short snake_case string for structured logging.

    Filters applied in order:
      1. already_resolved         — contract.is_resolved is True
      2. sport_not_active         — sport not in ACTIVE_SPORTS
      3. resolution_too_soon      — resolves within MIN_DAYS_TO_RESOLUTION days
      4. resolution_too_far       — resolves beyond MAX_DAYS_TO_RESOLUTION days
      5. price_out_of_range       — yes_price outside [0.01, 0.99]
      6. insufficient_book_depth  — book_depth_usd < MIN_BOOK_DEPTH_USD
      7. insufficient_volume      — volume_24h < MIN_VOLUME_24H_CONTRACTS

    Args:
        contract: Parsed ContractObject from get_events().
        book_depth_usd: USD notional of YES bids from the live order book.
        now: Current UTC datetime (injected for testability; never call
            datetime.now() inside this function).

    Returns:
        Tuple of (admitted: bool, rejection_reason: str).
    """
    if contract.is_resolved:
        return False, "already_resolved"
    if contract.sport not in ACTIVE_SPORTS:
        return False, "sport_not_active"
    days_to_res = (contract.resolution_date - now).days
    if days_to_res < MIN_DAYS_TO_RESOLUTION:
        return False, "resolution_too_soon"
    if days_to_res > MAX_DAYS_TO_RESOLUTION:
        return False, "resolution_too_far"
    if not (0.01 <= contract.yes_price <= 0.99):
        return False, "price_out_of_range"
    if book_depth_usd < MIN_BOOK_DEPTH_USD:
        return False, "insufficient_book_depth"
    if contract.volume_24h < MIN_VOLUME_24H_CONTRACTS:
        return False, "insufficient_volume"
    return True, ""


# ---------------------------------------------------------------------------
# Persistence bridge — thin wrapper around ingest.persist_event
# ---------------------------------------------------------------------------

def persist_event_contract(
    db: Database,
    contract: ContractObject,
    snap: PriceSnapshot,
) -> None:
    """Persist one admitted contract and its pre-fetched order-book snapshot.

    Wraps the single contract in a minimal EventData and delegates to
    persist_event(), passing the pre-fetched snapshot so no redundant
    get_order_book() call is made.

    Args:
        db: Storage layer with migrate_schema already applied.
        contract: ContractObject that has passed all universe filters.
        snap: Pre-fetched PriceSnapshot from the build_universe() loop.

    Raises:
        Any exception from insert_contract() or insert_price_snapshot().
    """
    from data_ingestion.ingest import persist_event  # avoid circular at module level
    event = EventData(
        event_ticker=contract.event_ticker,
        series_ticker=contract.series_ticker,
        title="",
        sub_title="",
        mutually_exclusive=False,
        markets=[contract],
    )
    persist_event(db, None, event, snap=snap)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_sport(
    sport: str,
    series_ticker: str,
    client: KalshiClient,
    db: Database,
    now: datetime,
) -> Tuple[List[ContractObject], int, int]:
    """Fetch, filter, and persist all contracts for one sport.

    Args:
        sport: Human-readable sport name, e.g. ``"NBA"``.
        series_ticker: Kalshi series ticker, e.g. ``"KXNBAGAME"``.
        client: Authenticated Kalshi API client.
        db: Storage layer.
        now: Rebalance wall-clock time (UTC); shared across all sports in one cycle.

    Returns:
        Tuple of (admitted_contracts, events_fetched_count, rejected_count).

    Raises:
        Any exception from get_events() — caller (build_universe) catches it
        per-sport so one sport failing does not abort others.
    """
    resp = client.get_events(series_ticker=series_ticker)
    events: List[EventData] = resp.payload
    admitted: List[ContractObject] = []
    rejected: int = 0

    for event in events:
        for contract in event.markets:
            try:
                book_resp = client.get_order_book(contract.contract_id)
                snap: PriceSnapshot = book_resp.payload
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Order book fetch failed — contract skipped",
                    extra={
                        "contract_id": contract.contract_id,
                        "sport": sport,
                        "error_type": type(exc).__name__,
                        "error_detail": str(exc),
                    },
                )
                rejected += 1
                continue

            ok, reason = is_in_universe(contract, snap.book_depth_usd, now)
            if not ok:
                logger.debug(
                    "Contract rejected from universe",
                    extra={
                        "contract_id": contract.contract_id,
                        "sport": sport,
                        "reason": reason,
                    },
                )
                rejected += 1
                continue

            persist_event_contract(db, contract, snap)
            admitted.append(contract)

    return admitted, len(events), rejected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_universe(
    client: KalshiClient,
    db: Database,
) -> List[ContractObject]:
    """Daily universe entry point — call once per rebalance cycle.

    For each sport in SPORT_SERIES_TICKERS:
      1. Fetches events via client.get_events(series_ticker=...).
      2. For each contract in each event, fetches the live order book.
      3. Applies is_in_universe() — rejected contracts are logged and dropped.
      4. Admitted contracts are persisted via persist_event_contract().

    One sport raising an exception does not abort the others. One contract
    failing get_order_book() or is_in_universe() does not abort its sport.

    Args:
        client: Authenticated Kalshi API client.
        db: Storage layer with migrate_schema already applied.

    Returns:
        List of ContractObject instances that passed all universe filters
        and were successfully persisted.
    """
    now = datetime.now(timezone.utc)
    universe: List[ContractObject] = []

    for sport, series_ticker in SPORT_SERIES_TICKERS.items():
        try:
            admitted, events_fetched, rejected = _process_sport(
                sport, series_ticker, client, db, now
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Sport fetch failed — skipping",
                extra={
                    "sport": sport,
                    "series_ticker": series_ticker,
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                },
            )
            continue

        logger.info(
            "Sport processed",
            extra={
                "sport": sport,
                "events_fetched": events_fetched,
                "contracts_admitted": len(admitted),
                "contracts_rejected": rejected,
            },
        )
        universe.extend(admitted)

    return universe
