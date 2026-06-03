"""
Tests for data_ingestion/ingest.py — Kalshi-to-SQLite bridge layer.

All tests use a real SQLite database (temp file via tmp_path) and a mocked
KalshiClient. No live API calls, no network, no shared state between tests.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_client import (
    APIResponse,
    BidAskLevel,
    ContractObject,
    EventData,
    KalshiClient,
    PriceSnapshot,
)
from storage.db import Database
from data_ingestion.ingest import (
    migrate_schema,
    persist_event,
    persist_price_snapshots,
)


# ---------------------------------------------------------------------------
# Shared timestamps
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
_FUTURE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> Database:
    """Fresh database with schema initialised and migration applied."""
    db = Database(str(tmp_path / "test.db"))
    db.initialise_schema()
    migrate_schema(db)
    return db


def _make_contract(
    contract_id: str = "C1",
    series_ticker: str = "KXNBAGAME",
    event_ticker: str = "KXNBAGAME-EVENT1",
    volume_24h: float = 20.0,
    is_resolved: bool = False,
) -> ContractObject:
    """Minimal ContractObject suitable for ingestion tests."""
    return ContractObject(
        contract_id=contract_id,
        series_ticker=series_ticker,
        event_ticker=event_ticker,
        sport="NBA",
        home_team="TeamA",
        away_team="TeamB",
        game_date=_FUTURE,
        resolution_date=_FUTURE,
        yes_price=0.55,
        no_price=0.45,
        volume_fp=1000.0,
        volume_24h=volume_24h,
        resolution_criteria_text="Resolves YES if TeamA wins.",
        ingestion_timestamp=_NOW,
        open_yes_price=0.50,
        is_resolved=is_resolved,
        resolution_outcome=None,
    )


def _make_snapshot(
    contract_id: str = "C1",
    book_depth_usd: float = 15_000.0,
    volume_fp: float = 999_999.0,   # intentionally different from book_depth_usd
    daily_volume: float = 50.0,
) -> PriceSnapshot:
    """PriceSnapshot with distinct book_depth_usd and volume_fp for mapping tests."""
    return PriceSnapshot(
        contract_id=contract_id,
        timestamp=_NOW,
        yes_price=0.55,
        no_price=0.45,
        volume_fp=volume_fp,
        daily_volume=daily_volume,
        bids=[BidAskLevel(price=0.55, size=100.0)],
        asks=[BidAskLevel(price=0.57, size=80.0)],
        book_depth_usd=book_depth_usd,
    )


def _make_api_response(
    payload,
    latency_ms: float = 42.0,
    endpoint: str = "/trade-api/v2/markets/TEST/orderbook",
) -> APIResponse:
    """Wrap a payload in a successful APIResponse."""
    return APIResponse(
        payload=payload,
        http_status_code=200,
        latency_ms=latency_ms,
        endpoint=endpoint,
        success=True,
    )


def _make_event(
    contracts: List[ContractObject],
    event_ticker: str = "KXNBAGAME-EVENT1",
    series_ticker: str = "KXNBAGAME",
) -> EventData:
    """EventData wrapping a list of contracts."""
    return EventData(
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title="Test Game",
        sub_title="TeamA at TeamB",
        mutually_exclusive=True,
        markets=contracts,
    )


def _make_mock_client(
    snapshot: Optional[PriceSnapshot] = None,
) -> MagicMock:
    """Mock KalshiClient whose get_order_book() returns the given snapshot."""
    if snapshot is None:
        snapshot = _make_snapshot()
    client = MagicMock(spec=KalshiClient)
    client.get_order_book.return_value = _make_api_response(snapshot)
    return client


# ---------------------------------------------------------------------------
# TestMigrateSchema
# ---------------------------------------------------------------------------

class TestMigrateSchema:
    """migrate_schema() adds series_ticker and event_ticker columns."""

    def test_adds_required_columns(self, tmp_path) -> None:
        """Both new columns appear in contracts table after migration."""
        db = Database(str(tmp_path / "test.db"))
        db.initialise_schema()
        migrate_schema(db)

        cols = {
            row[1]
            for row in db._conn.execute("PRAGMA table_info(contracts)").fetchall()
        }
        assert "series_ticker" in cols
        assert "event_ticker" in cols

    def test_idempotent_safe_to_call_twice(self, tmp_path) -> None:
        """Calling migrate_schema() a second time does not raise."""
        db = Database(str(tmp_path / "test.db"))
        db.initialise_schema()
        migrate_schema(db)
        migrate_schema(db)   # must not raise OperationalError


# ---------------------------------------------------------------------------
# TestPersistEvent
# ---------------------------------------------------------------------------

class TestPersistEvent:
    """persist_event() persists contracts and order-book snapshots."""

    def test_happy_path_two_liquid_contracts(self, tmp_path) -> None:
        """Two liquid contracts → 2 insert_contract rows, 2 order-book calls,
        2 price snapshot rows."""
        db = _make_db(tmp_path)
        client = _make_mock_client()
        c1 = _make_contract("C1", volume_24h=20.0)
        c2 = _make_contract("C2", volume_24h=20.0)

        persist_event(db, client, _make_event([c1, c2]))

        assert client.get_order_book.call_count == 2
        contracts = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        snapshots = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices"
        ).fetchone()[0]
        assert contracts == 2
        assert snapshots == 2

    def test_ticker_columns_stored_on_contract_row(self, tmp_path) -> None:
        """series_ticker and event_ticker are written to the migrated columns."""
        db = _make_db(tmp_path)
        client = _make_mock_client()
        c = _make_contract(
            "C1",
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-EVENT1",
        )

        persist_event(db, client, _make_event([c]))

        row = db._conn.execute(
            "SELECT series_ticker, event_ticker FROM contracts WHERE contract_id = 'C1'"
        ).fetchone()
        assert row["series_ticker"] == "KXNBAGAME"
        assert row["event_ticker"] == "KXNBAGAME-EVENT1"

    def test_illiquid_contract_skips_order_book_call(self, tmp_path) -> None:
        """Contract with volume_24h=0.0 skips get_order_book() entirely."""
        db = _make_db(tmp_path)
        client = _make_mock_client()
        c = _make_contract("C1", volume_24h=0.0)   # below is_liquid() threshold

        persist_event(db, client, _make_event([c]))

        client.get_order_book.assert_not_called()
        # Contract row still inserted
        count = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        assert count == 1
        # No price snapshot
        snaps = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices"
        ).fetchone()[0]
        assert snaps == 0

    def test_one_bad_contract_does_not_abort_others(self, tmp_path) -> None:
        """DataValidationError on one contract's order book does not prevent
        the other contract from being fully persisted."""
        from kalshi_client import DataValidationError

        db = _make_db(tmp_path)
        client = MagicMock(spec=KalshiClient)
        # C1 order book raises; C2 succeeds
        client.get_order_book.side_effect = [
            DataValidationError("bad price", raw_response="{}", contract_id="C1"),
            _make_api_response(_make_snapshot("C2")),
        ]

        c1 = _make_contract("C1", volume_24h=20.0)
        c2 = _make_contract("C2", volume_24h=20.0)
        persist_event(db, client, _make_event([c1, c2]))

        # Both contract rows must exist
        contracts = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        assert contracts == 2

        # C2 gets its price snapshot; C1 does not
        snap_c2 = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C2'"
        ).fetchone()[0]
        snap_c1 = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()[0]
        assert snap_c2 == 1
        assert snap_c1 == 0

    def test_total_liquidity_uses_book_depth_usd_not_volume_fp(
        self, tmp_path
    ) -> None:
        """total_liquidity stored in contract_prices must equal snap.book_depth_usd,
        not snap.volume_fp (which is a contract count, not a USD amount)."""
        db = _make_db(tmp_path)
        snap = _make_snapshot(
            contract_id="C1",
            book_depth_usd=15_000.0,
            volume_fp=999_999.0,   # deliberately different — should NOT appear in DB
        )
        client = _make_mock_client(snapshot=snap)
        c = _make_contract("C1", volume_24h=20.0)

        persist_event(db, client, _make_event([c]))

        row = db._conn.execute(
            "SELECT total_liquidity FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()
        assert row is not None
        assert row["total_liquidity"] == pytest.approx(15_000.0)

    def test_daily_volume_stored_from_snap_daily_volume(self, tmp_path) -> None:
        """daily_volume in contract_prices matches snap.daily_volume (contract count)."""
        db = _make_db(tmp_path)
        snap = _make_snapshot(contract_id="C1", daily_volume=42.0)
        client = _make_mock_client(snapshot=snap)
        c = _make_contract("C1", volume_24h=20.0)

        persist_event(db, client, _make_event([c]))

        row = db._conn.execute(
            "SELECT daily_volume FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()
        assert row is not None
        assert row["daily_volume"] == pytest.approx(42.0)

    def test_resolved_contract_stores_is_resolved_one(self, tmp_path) -> None:
        """is_resolved=True on ContractObject becomes is_resolved=1 in the DB."""
        db = _make_db(tmp_path)
        client = _make_mock_client()
        c = _make_contract("C1", volume_24h=20.0, is_resolved=True)
        # Make it truly resolved so it passes date validation
        c.resolution_outcome = 1

        persist_event(db, client, _make_event([c]))

        row = db._conn.execute(
            "SELECT is_resolved FROM contracts WHERE contract_id = 'C1'"
        ).fetchone()
        assert row["is_resolved"] == 1

    def test_empty_event_does_not_raise(self, tmp_path) -> None:
        """An event with zero markets is handled gracefully."""
        db = _make_db(tmp_path)
        client = _make_mock_client()
        persist_event(db, client, _make_event([]))
        client.get_order_book.assert_not_called()


# ---------------------------------------------------------------------------
# TestPersistPriceSnapshots
# ---------------------------------------------------------------------------

class TestPersistPriceSnapshots:
    """persist_price_snapshots() refreshes order-book data for a batch."""

    def _pre_insert(self, db: Database, contract_id: str) -> None:
        """Insert a bare contract row so FK constraint is satisfied."""
        db.insert_contract(
            contract_id=contract_id,
            sport="NBA",
            home_team="A",
            away_team="B",
            game_date=_FUTURE.isoformat(),
            resolution_date=_FUTURE.isoformat(),
            ingestion_timestamp=_NOW.isoformat(),
        )

    def test_happy_path_two_contracts(self, tmp_path) -> None:
        """Two contract_ids → two get_order_book calls and two price snapshots."""
        db = _make_db(tmp_path)
        self._pre_insert(db, "C1")
        self._pre_insert(db, "C2")

        client = MagicMock(spec=KalshiClient)
        client.get_order_book.side_effect = [
            _make_api_response(_make_snapshot("C1")),
            _make_api_response(_make_snapshot("C2")),
        ]

        persist_price_snapshots(db, client, ["C1", "C2"])

        assert client.get_order_book.call_count == 2
        count = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices"
        ).fetchone()[0]
        assert count == 2

    def test_one_failure_does_not_abort_others(self, tmp_path) -> None:
        """Exception on C1 does not prevent C2's snapshot from being stored."""
        db = _make_db(tmp_path)
        self._pre_insert(db, "C1")
        self._pre_insert(db, "C2")

        client = MagicMock(spec=KalshiClient)
        client.get_order_book.side_effect = [
            RuntimeError("network timeout"),
            _make_api_response(_make_snapshot("C2")),
        ]

        persist_price_snapshots(db, client, ["C1", "C2"])

        # Both order book calls were attempted
        assert client.get_order_book.call_count == 2
        # C2 snapshot stored; C1 has none
        c2_snaps = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C2'"
        ).fetchone()[0]
        c1_snaps = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()[0]
        assert c2_snaps == 1
        assert c1_snaps == 0

    def test_empty_list_makes_no_api_calls(self, tmp_path) -> None:
        """An empty contract_ids list makes no API calls and does not raise."""
        db = _make_db(tmp_path)
        client = _make_mock_client()

        persist_price_snapshots(db, client, [])

        client.get_order_book.assert_not_called()

    def test_total_liquidity_is_book_depth_usd(self, tmp_path) -> None:
        """total_liquidity in contract_prices equals snap.book_depth_usd."""
        db = _make_db(tmp_path)
        self._pre_insert(db, "C1")

        snap = _make_snapshot("C1", book_depth_usd=12_500.0, volume_fp=888_888.0)
        client = MagicMock(spec=KalshiClient)
        client.get_order_book.return_value = _make_api_response(snap)

        persist_price_snapshots(db, client, ["C1"])

        row = db._conn.execute(
            "SELECT total_liquidity FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()
        assert row is not None
        assert row["total_liquidity"] == pytest.approx(12_500.0)
