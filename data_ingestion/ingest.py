"""
data_ingestion/ingest.py — Kalshi-to-SQLite bridge layer.

Connects kalshi_client.py to storage/db.py. Pure functions only; no classes,
no frameworks.  Three public entry points:

  migrate_schema(db)                     — run once at startup
  persist_event(db, client, event)       — called per EventData from get_events()
  persist_price_snapshots(db, client, [ids]) — called every 15 minutes

Mapping rules (non-negotiable):
  total_liquidity  ← snap.book_depth_usd   (USD notional of YES bids)
  daily_volume     ← snap.daily_volume     (24-hour contract count, NOT USD)
  Candlestick PriceSnapshot objects (book_depth_usd=0.0, empty bids/asks)
  must never be passed to insert_price_snapshot() for liquidity purposes.
"""

import logging
import sqlite3
from typing import List, Optional

from kalshi_client import ContractObject, EventData, KalshiClient, PriceSnapshot
from storage.db import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

# Columns added to contracts after the initial schema was shipped.
# Tuples of (column_name, sqlite_type).  Append new migrations here.
_MIGRATION_COLUMNS: List[tuple] = [
    ("series_ticker", "TEXT"),   # needed for get_price_history()
    ("event_ticker", "TEXT"),    # needed to group mutually-exclusive markets
]


def migrate_schema(db: Database) -> None:
    """Apply additive column migrations to the contracts table.

    Uses ALTER TABLE for each missing column.  If the column already exists,
    SQLite raises OperationalError("duplicate column name: …") which is
    silently caught — making this call idempotent and safe to run at startup
    on both fresh and existing databases.

    Args:
        db: Initialised :class:`~storage.db.Database` with schema applied.

    Raises:
        sqlite3.OperationalError: On any database error other than a
            duplicate-column conflict.
    """
    for col_name, col_type in _MIGRATION_COLUMNS:
        try:
            with db._conn:
                db._conn.execute(
                    f"ALTER TABLE contracts ADD COLUMN {col_name} {col_type}"
                )
            logger.info(
                "Schema migration applied",
                extra={"column": col_name, "table": "contracts"},
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column" in str(exc).lower():
                logger.debug(
                    "Migration already applied — column exists",
                    extra={"column": col_name, "table": "contracts"},
                )
            else:
                raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_ticker_columns(
    db: Database,
    contract_id: str,
    series_ticker: str,
    event_ticker: str,
) -> None:
    """Persist series_ticker and event_ticker on an existing contracts row.

    Uses COALESCE so the first ingestion wins (point-in-time discipline):
    if the column is already populated, the existing value is kept.

    Args:
        db: Storage layer.
        contract_id: Primary key of the row to update.
        series_ticker: Parent series, e.g. ``"KXNBAGAME"``.
        event_ticker: Parent event, e.g. ``"KXNBAGAME-26MAY28OKCSAS"``.
    """
    with db._conn:
        db._conn.execute(
            """
            UPDATE contracts
            SET series_ticker = COALESCE(series_ticker, ?),
                event_ticker  = COALESCE(event_ticker, ?)
            WHERE contract_id = ?
            """,
            (series_ticker, event_ticker, contract_id),
        )


def _store_price_snapshot(
    db: Database,
    contract_id: str,
    snap: PriceSnapshot,
) -> None:
    """Write a pre-fetched order-book snapshot to contract_prices. No API call.

    Mapping rules:
      ``total_liquidity`` ← ``snap.book_depth_usd``  (USD notional of YES bids)
      ``daily_volume``    ← ``snap.daily_volume``     (24-hour contract count)

    Args:
        db: Storage layer.
        contract_id: Primary key of the corresponding contracts row.
        snap: Already-fetched PriceSnapshot (from get_order_book or pre-fetched).

    Raises:
        Any database exception from insert_price_snapshot().
    """
    bid_depth = [{"price": b.price, "size": b.size} for b in snap.bids]
    ask_depth = [{"price": a.price, "size": a.size} for a in snap.asks]
    db.insert_price_snapshot(
        contract_id=contract_id,
        timestamp=snap.timestamp.isoformat(),
        yes_price=snap.yes_price,
        no_price=snap.no_price,
        total_liquidity=snap.book_depth_usd,          # USD notional — NEVER volume_fp
        daily_volume=snap.daily_volume,               # 24h contract count
        bid_depth=bid_depth if bid_depth else None,
        ask_depth=ask_depth if ask_depth else None,
    )


def _fetch_and_store_snapshot(
    db: Database,
    client: KalshiClient,
    contract_id: str,
) -> None:
    """Fetch a live order-book snapshot and write it to contract_prices.

    Delegates the DB write to _store_price_snapshot. Logs one structured entry
    with latency and endpoint on every call.

    Args:
        db: Storage layer.
        client: Authenticated Kalshi API client.
        contract_id: Kalshi contract ticker.

    Raises:
        Any exception from the API call or DB write — caller is responsible
        for catching and deciding whether to skip or abort.
    """
    resp = client.get_order_book(contract_id)
    snap = resp.payload
    _store_price_snapshot(db, contract_id, snap)
    logger.debug(
        "Price snapshot stored",
        extra={
            "contract_id": contract_id,
            "success": True,
            "latency_ms": resp.latency_ms,
            "endpoint": resp.endpoint,
            "book_depth_usd": snap.book_depth_usd,
        },
    )


def _persist_one_contract(
    db: Database,
    client: Optional[KalshiClient],
    c: ContractObject,
    snap: Optional[PriceSnapshot] = None,
) -> None:
    """Persist one contract row and (if liquid) its order-book snapshot.

    Writes the base contract row via ``insert_contract``, then writes
    ``series_ticker`` and ``event_ticker`` via the migrated columns.
    Skips the order-book call for illiquid contracts (``volume_24h`` below
    the ``is_liquid()`` threshold) and logs at DEBUG.

    When ``snap`` is provided it is written directly, avoiding a redundant
    ``get_order_book()`` call. ``client`` may be ``None`` only when ``snap``
    is always pre-fetched for every liquid contract.

    Args:
        db: Storage layer.
        client: Authenticated Kalshi API client. May be None when snap is given.
        c: Parsed :class:`~kalshi_client.ContractObject` from ``get_events()``.
        snap: Pre-fetched PriceSnapshot. When not None, skips the API call.

    Raises:
        Any exception from DB writes or the API call — caller (persist_event)
        catches these per-contract so one failure does not abort the event.
    """
    db.insert_contract(
        contract_id=c.contract_id,
        sport=c.sport,
        home_team=c.home_team,
        away_team=c.away_team,
        game_date=c.game_date.isoformat(),
        resolution_date=c.resolution_date.isoformat(),
        ingestion_timestamp=c.ingestion_timestamp.isoformat(),
        open_yes_price=c.open_yes_price,
        resolution_outcome=c.resolution_outcome,
        is_resolved=1 if c.is_resolved else 0,
    )
    _write_ticker_columns(db, c.contract_id, c.series_ticker, c.event_ticker)

    if not KalshiClient.is_liquid(c):
        logger.debug(
            "Contract below liquidity threshold — order book skipped",
            extra={"contract_id": c.contract_id, "volume_24h": c.volume_24h},
        )
        return

    if snap is not None:
        _store_price_snapshot(db, c.contract_id, snap)
    else:
        _fetch_and_store_snapshot(db, client, c.contract_id)


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------

def persist_event(
    db: Database,
    client: Optional[KalshiClient],
    event: EventData,
    snap: Optional[PriceSnapshot] = None,
) -> None:
    """Persist all contracts nested inside one Kalshi event.

    For each :class:`~kalshi_client.ContractObject` in ``event.markets``:

    1. Calls ``db.insert_contract()`` with all fields; converts every
       :class:`~datetime.datetime` to ``isoformat()`` before writing.
    2. Writes ``series_ticker`` and ``event_ticker`` to the migrated columns.
    3. If ``KalshiClient.is_liquid(c)`` is ``True``:
       - When ``snap`` is provided, writes it directly (no API call).
       - Otherwise calls ``get_order_book()`` and stores the snapshot.
    4. If illiquid, skips any order-book call entirely (logs DEBUG).

    One bad contract does not abort the event — exceptions are caught per
    contract and logged at ERROR.

    Args:
        db: Storage layer with ``migrate_schema`` already applied.
        client: Authenticated Kalshi API client. May be ``None`` only when
            ``snap`` is always pre-fetched for every liquid contract.
        event: Parsed :class:`~kalshi_client.EventData` from ``get_events()``.
        snap: Pre-fetched PriceSnapshot. When provided, skips ``get_order_book()``.
    """
    logger.info(
        "Ingesting event",
        extra={
            "event_ticker": event.event_ticker,
            "series_ticker": event.series_ticker,
            "market_count": len(event.markets),
        },
    )
    for c in event.markets:
        try:
            _persist_one_contract(db, client, c, snap=snap)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to persist contract — skipping",
                extra={
                    "contract_id": c.contract_id,
                    "event_ticker": event.event_ticker,
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                },
            )


def persist_price_snapshots(
    db: Database,
    client: KalshiClient,
    contract_ids: List[str],
) -> None:
    """Refresh live order-book price snapshots for a batch of contracts.

    Called by the 15-minute polling loop.  For each ``contract_id``:

    * Calls ``get_order_book()`` and stores the snapshot with
      ``total_liquidity = snap.book_depth_usd``.
    * Logs one structured entry per call (success or failure, with latency).
    * One failure does not abort the remaining contracts in the batch.

    Args:
        db: Storage layer.
        client: Authenticated Kalshi API client.
        contract_ids: Kalshi contract tickers to refresh.
    """
    logger.info(
        "Price snapshot batch started",
        extra={"batch_size": len(contract_ids)},
    )
    for contract_id in contract_ids:
        try:
            _fetch_and_store_snapshot(db, client, contract_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Price snapshot failed — skipping",
                extra={
                    "contract_id": contract_id,
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                    "success": False,
                },
            )
