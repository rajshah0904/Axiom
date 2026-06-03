"""
data_ingestion/backfill.py — Historical contract and price backfill.

Fetches historical contracts from the Kalshi historical API and settled/closed
events from the live API, then writes full price history to the database.
Unlike the live ingestion path, backfill:

  - Applies no universe filters (no liquidity check, no date-range gate).
  - Stores all resolved contracts regardless of sport or price range.
  - Uses daily candlestick snapshots (total_liquidity=0.0, no order book).
  - Routes price fetching: always tries historical candlesticks API first;
    falls back to live candlesticks on 404 (handles far-future resolution_date
    placeholders that Kalshi sets on many finalized contracts).
  - Respects a configurable inter-contract sleep to avoid rate limits.

Three public entry points:

  backfill_sport(client, db, series_ticker, sport, start_ts)  — one sport
  backfill_all(client, db, start_date, sleep_between_contracts)  — all sports
  run_backfill()                                                 — CLI entry point
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from kalshi_client import ClientError, ContractObject, KalshiClient, PriceSnapshot
from storage.db import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_SLEEP_BETWEEN_CONTRACTS: float = 0.5     # seconds; override via AXIOM_BACKFILL_SLEEP
_DEFAULT_START_DATE: str = "2025-01-01"           # ISO date; Kalshi sports launched early 2025
_BACKFILL_STATUSES: List[str] = ["settled", "closed"]   # Kalshi event status values
_CUTOFF_FALLBACK_DAYS: int = 90                   # fallback when market_settled_ts absent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_all_contracts(
    client: KalshiClient,
    series_ticker: str,
    cutoff_ts: datetime,
) -> List[ContractObject]:
    """Fetch all historical and live-resolved contracts for one series, deduplicated.

    Three sources are merged in priority order:
      1. Historical API (get_historical_markets) — authoritative for old data.
      2. Live API settled events (get_events status="settled").
      3. Live API closed events (get_events status="closed").

    Deduplication by contract_id — first seen wins (historical result takes priority).
    Logs one INFO entry with per-source counts and total after dedup.

    Args:
        client: Authenticated Kalshi API client.
        series_ticker: Kalshi series ticker, e.g. ``"KXNBAGAME"``.
        cutoff_ts: Historical API cutoff; used by caller for routing only.

    Returns:
        Deduplicated flat list of ContractObject instances.

    Raises:
        Any exception from the API calls — caller is responsible for catching.
    """
    seen: Dict[str, ContractObject] = {}

    hist_resp = client.get_historical_markets(series_ticker)
    from_historical_api = 0
    for contract in hist_resp.payload:
        if contract.contract_id not in seen:
            seen[contract.contract_id] = contract
            from_historical_api += 1

    settled_resp = client.get_events(series_ticker=series_ticker, status="settled")
    from_live_settled = 0
    for event in settled_resp.payload:
        for contract in event.markets:
            if contract.contract_id not in seen:
                seen[contract.contract_id] = contract
                from_live_settled += 1

    closed_resp = client.get_events(series_ticker=series_ticker, status="closed")
    from_live_closed = 0
    for event in closed_resp.payload:
        for contract in event.markets:
            if contract.contract_id not in seen:
                seen[contract.contract_id] = contract
                from_live_closed += 1

    logger.info(
        "Contracts fetched for series",
        extra={
            "series_ticker": series_ticker,
            "from_historical_api": from_historical_api,
            "from_live_settled": from_live_settled,
            "from_live_closed": from_live_closed,
            "total_after_dedup": len(seen),
        },
    )
    return list(seen.values())


def _fetch_candlesticks_with_fallback(
    client: KalshiClient,
    contract: ContractObject,
    start_ts: datetime,
) -> List[PriceSnapshot]:
    """Try historical candlesticks first, fall back to live on 404.

    Args:
        client: Authenticated Kalshi API client.
        contract: ContractObject to fetch history for.
        start_ts: UTC start timestamp for candle range.

    Returns:
        List of PriceSnapshot objects, possibly empty if no candles exist.

    Raises:
        APIError: if both historical and live endpoints fail with non-404 errors.
        ClientError: if historical returns 404 AND live also returns 404.
    """
    try:
        resp = client.get_historical_candlesticks(
            contract.contract_id, contract.series_ticker, start_ts
        )
        return resp.payload
    except ClientError as exc:
        if exc.status_code != 404:
            raise
        logger.debug(
            "Historical candlesticks 404 — falling back to live endpoint",
            extra={
                "contract_id": contract.contract_id,
                "series_ticker": contract.series_ticker,
            },
        )

    resp = client.get_price_history(
        contract_id=contract.contract_id,
        series_ticker=contract.series_ticker,
        start_timestamp=start_ts,
    )
    return resp.payload


def _store_price_history(
    client: KalshiClient,
    db: Database,
    contract: ContractObject,
    start_ts: datetime,
    cutoff_ts: datetime,
) -> int:
    """Fetch candlestick history for one contract and write to DB.

    Routing strategy: try historical endpoint first (covers all finalized
    contracts regardless of resolution_date placeholder values). On 404,
    fall back to the live candlesticks endpoint (covers very recent contracts
    not yet moved to the historical tier).

    Both paths write identically: total_liquidity=0.0, daily_volume=snap.volume_fp,
    bid_depth=None, ask_depth=None.

    Args:
        client: Authenticated Kalshi API client.
        db: Storage layer.
        contract: ContractObject whose history to fetch.
        start_ts: UTC datetime; only candles at or after this time are returned.
        cutoff_ts: Unused — retained for API compatibility with callers.
            Routing is now endpoint-driven (try historical, fallback live)
            rather than date-driven.

    Returns:
        Number of candle rows written to contract_prices.

    Raises:
        APIError: if both endpoints fail.
        Any exception from DB writes.
    """
    snaps = _fetch_candlesticks_with_fallback(client, contract, start_ts)
    for snap in snaps:
        db.insert_price_snapshot(
            contract_id=contract.contract_id,
            timestamp=snap.timestamp.isoformat(),
            yes_price=snap.yes_price,
            no_price=snap.no_price,
            total_liquidity=0.0,
            daily_volume=snap.volume_fp,
            bid_depth=None,
            ask_depth=None,
        )
    return len(snaps)


def _backfill_one_contract(
    client: KalshiClient,
    db: Database,
    contract: ContractObject,
    start_ts: datetime,
    cutoff_ts: datetime,
    sleep_s: float,
) -> int:
    """Persist one historical contract row and its full price history.

    Steps:
      1. ``db.insert_contract()`` — idempotent via INSERT OR IGNORE.
      2. ``_write_ticker_columns()`` — writes series_ticker and event_ticker.
      3. ``_store_price_history()`` — fetches candles (routed by cutoff_ts).
      4. If resolved with known outcome, calls ``db.mark_contract_resolved()``.
      5. Sleeps ``sleep_s`` seconds to avoid rate limiting.

    Args:
        client: Authenticated Kalshi API client.
        db: Storage layer.
        contract: ContractObject to persist.
        start_ts: UTC datetime; price history start.
        cutoff_ts: Boundary between historical and live API routing.
        sleep_s: Seconds to sleep after this contract is fully processed.

    Returns:
        Number of candle rows written for this contract.

    Raises:
        Any exception from DB writes or API calls — caller catches per-contract.
    """
    from data_ingestion.ingest import _write_ticker_columns  # avoid circular at module level

    db.insert_contract(
        contract_id=contract.contract_id,
        sport=contract.sport,
        home_team=contract.home_team,
        away_team=contract.away_team,
        game_date=contract.game_date.isoformat(),
        resolution_date=contract.resolution_date.isoformat(),
        ingestion_timestamp=contract.ingestion_timestamp.isoformat(),
        open_yes_price=contract.open_yes_price,
        resolution_outcome=contract.resolution_outcome,
        is_resolved=1 if contract.is_resolved else 0,
    )
    _write_ticker_columns(db, contract.contract_id, contract.series_ticker, contract.event_ticker)
    # Note: if _store_price_history() raises, the contracts row is already
    # written and will remain without price history. This is intentional —
    # contract metadata is valid even when Kalshi has no candlestick data.
    # The signal layer must handle contracts with empty contract_prices.
    candles_written = _store_price_history(client, db, contract, start_ts, cutoff_ts)

    if contract.is_resolved and contract.resolution_outcome is not None:
        db.mark_contract_resolved(contract.contract_id, contract.resolution_outcome)

    time.sleep(sleep_s)
    return candles_written


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def backfill_sport(
    client: KalshiClient,
    db: Database,
    series_ticker: str,
    sport: str,
    start_ts: datetime,
    sleep_between_contracts: float = _DEFAULT_SLEEP_BETWEEN_CONTRACTS,
) -> int:
    """Backfill all historical contracts for one sport series.

    Calls get_historical_cutoff() once to determine the historical/live routing
    boundary, then merges contracts from all three sources (historical API,
    settled events, closed events) and processes each one.

    Applies no universe or liquidity filters — all resolved contracts are
    persisted for signal validation purposes.

    Args:
        client: Authenticated Kalshi API client.
        db: Storage layer with migrate_schema already applied.
        series_ticker: Kalshi series ticker, e.g. ``"KXNBAGAME"``.
        sport: Human-readable sport name, e.g. ``"NBA"``.
        start_ts: UTC datetime; price history will begin from this point.
        sleep_between_contracts: Seconds to sleep after each contract.

    Returns:
        Total number of contracts successfully backfilled.

    Raises:
        Any exception from get_historical_cutoff() or _fetch_all_contracts() —
        caller (backfill_all) catches per-sport.
    """
    cutoff_resp = client.get_historical_cutoff()
    if "market_settled_ts" in cutoff_resp.payload:
        cutoff_ts = cutoff_resp.payload["market_settled_ts"]
    else:
        cutoff_ts = datetime.now(timezone.utc) - timedelta(days=_CUTOFF_FALLBACK_DAYS)
        logger.warning(
            "market_settled_ts absent from historical cutoff — using fallback",
            extra={
                "series_ticker": series_ticker,
                "fallback_days": _CUTOFF_FALLBACK_DAYS,
                "cutoff_ts": cutoff_ts.isoformat(),
            },
        )

    contracts = _fetch_all_contracts(client, series_ticker, cutoff_ts)

    contracts_backfilled = 0
    contracts_failed = 0
    for contract in contracts:
        try:
            candles = _backfill_one_contract(
                client, db, contract, start_ts, cutoff_ts, sleep_between_contracts
            )
            logger.debug(
                "Contract backfilled",
                extra={
                    "contract_id": contract.contract_id,
                    "candles_written": candles,
                    "is_resolved": contract.is_resolved,
                },
            )
            contracts_backfilled += 1
        except Exception as exc:  # noqa: BLE001
            contracts_failed += 1
            logger.error(
                "Backfill failed for contract — skipping",
                extra={
                    "contract_id": contract.contract_id,
                    "series_ticker": series_ticker,
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                },
            )

    logger.info(
        "Sport backfill complete",
        extra={
            "sport": sport,
            "series_ticker": series_ticker,
            "total_contracts": len(contracts),
            "contracts_backfilled": contracts_backfilled,
            "contracts_failed": contracts_failed,
        },
    )
    return contracts_backfilled


def backfill_all(
    client: KalshiClient,
    db: Database,
    start_date: str = _DEFAULT_START_DATE,
    sleep_between_contracts: float = _DEFAULT_SLEEP_BETWEEN_CONTRACTS,
) -> None:
    """Backfill historical contracts for all supported sports.

    Iterates every sport in ``SPORT_SERIES_TICKERS``.  One sport raising an
    exception does not abort the others — per-sport try/except throughout.

    Args:
        client: Authenticated Kalshi API client.
        db: Storage layer with migrate_schema already applied.
        start_date: ISO 8601 date string (e.g. ``"2025-01-01"``). Price history
            begins from midnight UTC of this date.
        sleep_between_contracts: Seconds to sleep after each contract.
    """
    from data_ingestion.universe import SPORT_SERIES_TICKERS  # avoid circular at module level

    start_ts = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    for sport, series_ticker in SPORT_SERIES_TICKERS.items():
        try:
            backfill_sport(client, db, series_ticker, sport, start_ts, sleep_between_contracts)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Sport backfill failed — skipping",
                extra={
                    "sport": sport,
                    "series_ticker": series_ticker,
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                },
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_backfill() -> None:
    """CLI entry point for historical backfill.

    Environment variables:
        AXIOM_BACKFILL_SLEEP: float seconds to sleep between contracts.
            Overrides the default of ``_DEFAULT_SLEEP_BETWEEN_CONTRACTS``.

    Command-line arguments (positional, both optional):
        --start-date <YYYY-MM-DD>  Backfill start date. Default: ``_DEFAULT_START_DATE``.
        --sport <NAME>             Single sport to backfill (e.g. ``NBA``). If omitted,
                                   all sports in SPORT_SERIES_TICKERS are backfilled.

    Exits with code 1 on configuration failure.
    """
    import dotenv
    dotenv.load_dotenv()

    db_path = os.environ.get("AXIOM_DB_PATH", "")
    if not db_path:
        print(
            "Configuration error: AXIOM_DB_PATH environment variable is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    sleep_s = _DEFAULT_SLEEP_BETWEEN_CONTRACTS
    raw_sleep = os.environ.get("AXIOM_BACKFILL_SLEEP", "")
    if raw_sleep:
        try:
            sleep_s = float(raw_sleep)
        except ValueError:
            print(
                f"Warning: AXIOM_BACKFILL_SLEEP={raw_sleep!r} is not a valid float. "
                f"Using default {_DEFAULT_SLEEP_BETWEEN_CONTRACTS}s.",
                file=sys.stderr,
            )

    start_date = _DEFAULT_START_DATE
    sport_filter: Optional[str] = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--start-date" and i + 1 < len(args):
            start_date = args[i + 1]
            i += 2
        elif args[i] == "--sport" and i + 1 < len(args):
            sport_filter = args[i + 1].upper()
            i += 2
        else:
            i += 1

    from kalshi_client import KalshiClient, KalshiConfig
    from storage.db import Database
    from data_ingestion.ingest import migrate_schema
    from data_ingestion.universe import SPORT_SERIES_TICKERS

    kcfg = KalshiConfig.from_env()
    client = KalshiClient(kcfg)
    db = Database(db_path)
    migrate_schema(db)

    start_ts = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    if sport_filter is not None:
        if sport_filter not in SPORT_SERIES_TICKERS:
            print(
                f"Unknown sport {sport_filter!r}. "
                f"Valid options: {sorted(SPORT_SERIES_TICKERS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        series_ticker = SPORT_SERIES_TICKERS[sport_filter]
        backfill_sport(client, db, series_ticker, sport_filter, start_ts, sleep_s)
    else:
        backfill_all(client, db, start_date=start_date, sleep_between_contracts=sleep_s)
