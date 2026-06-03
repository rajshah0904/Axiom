"""
Tests for data_ingestion/universe.py — universe filter and build_universe().

All tests use a real SQLite database (temp file via tmp_path) and a fully
mocked KalshiClient. No live API calls, no network, no shared state between tests.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_client import (
    APIError,
    APIResponse,
    BidAskLevel,
    ContractObject,
    EventData,
    KalshiClient,
    PriceSnapshot,
)
from storage.db import Database
from data_ingestion.ingest import migrate_schema
from data_ingestion.universe import (
    ACTIVE_SPORTS,
    MIN_BOOK_DEPTH_USD,
    MIN_DAYS_TO_RESOLUTION,
    MIN_VOLUME_24H_CONTRACTS,
    MAX_DAYS_TO_RESOLUTION,
    SPORT_SERIES_TICKERS,
    build_universe,
    is_in_universe,
    persist_event_contract,
)


# ---------------------------------------------------------------------------
# Shared timestamps
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
_RES_5D = _NOW + timedelta(days=5)     # 5 days out — comfortably in window
_RES_31D = _NOW + timedelta(days=31)   # 31 days out — too far
_RES_0D = _NOW + timedelta(hours=12)   # resolves today — too soon (.days == 0)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_contract(
    contract_id: str = "C1",
    sport: str = "NBA",
    yes_price: float = 0.55,
    volume_24h: float = 20.0,
    is_resolved: bool = False,
    resolution_date: Optional[datetime] = None,
    series_ticker: str = "KXNBAGAME",
    event_ticker: str = "KXNBAGAME-EVENT1",
) -> ContractObject:
    """Minimal ContractObject that passes all universe filters by default."""
    return ContractObject(
        contract_id=contract_id,
        series_ticker=series_ticker,
        event_ticker=event_ticker,
        sport=sport,
        home_team="TeamA",
        away_team="TeamB",
        game_date=_RES_5D,
        resolution_date=resolution_date if resolution_date is not None else _RES_5D,
        yes_price=yes_price,
        no_price=round(1.0 - yes_price, 4),
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
) -> PriceSnapshot:
    """PriceSnapshot that passes the book_depth_usd filter."""
    return PriceSnapshot(
        contract_id=contract_id,
        timestamp=_NOW,
        yes_price=0.55,
        no_price=0.45,
        volume_fp=1000.0,
        daily_volume=50.0,
        bids=[BidAskLevel(price=0.55, size=100.0)],
        asks=[BidAskLevel(price=0.57, size=80.0)],
        book_depth_usd=book_depth_usd,
    )


def _make_api_response(payload, latency_ms: float = 10.0) -> APIResponse:
    """Wrap any payload in a successful APIResponse."""
    return APIResponse(
        payload=payload,
        http_status_code=200,
        latency_ms=latency_ms,
        endpoint="/test",
        success=True,
    )


def _make_event(
    contracts: List[ContractObject],
    event_ticker: str = "KXNBAGAME-EVENT1",
    series_ticker: str = "KXNBAGAME",
) -> EventData:
    """EventData wrapping a list of ContractObjects."""
    return EventData(
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title="Test Game",
        sub_title="TeamA at TeamB",
        mutually_exclusive=True,
        markets=contracts,
    )


def _make_db(tmp_path) -> Database:
    """Fresh database with schema and migration applied."""
    db = Database(str(tmp_path / "test.db"))
    db.initialise_schema()
    migrate_schema(db)
    return db


def _make_mock_client(
    events_by_ticker: Optional[Dict[str, List[EventData]]] = None,
    snapshots_by_contract: Optional[Dict[str, PriceSnapshot]] = None,
    get_order_book_side_effect=None,
    get_events_side_effect=None,
) -> MagicMock:
    """Mock KalshiClient with configurable get_events and get_order_book."""
    client = MagicMock(spec=KalshiClient)
    events_map = events_by_ticker or {}
    snaps_map = snapshots_by_contract or {}

    if get_events_side_effect is not None:
        client.get_events.side_effect = get_events_side_effect
    else:
        def _get_events(series_ticker: str, status: str = "open"):
            return _make_api_response(events_map.get(series_ticker, []))
        client.get_events.side_effect = _get_events

    if get_order_book_side_effect is not None:
        client.get_order_book.side_effect = get_order_book_side_effect
    else:
        def _get_order_book(contract_id: str):
            snap = snaps_map.get(contract_id, _make_snapshot(contract_id))
            return _make_api_response(snap)
        client.get_order_book.side_effect = _get_order_book

    return client


# ---------------------------------------------------------------------------
# TestIsInUniverse — one test per filter (plus happy path)
# ---------------------------------------------------------------------------

class TestIsInUniverse:
    """is_in_universe() applies seven ordered filters and returns (bool, reason)."""

    def test_happy_path_all_filters_pass(self) -> None:
        """All seven filters pass → (True, "")."""
        c = _make_contract()
        ok, reason = is_in_universe(c, book_depth_usd=15_000.0, now=_NOW)
        assert ok is True
        assert reason == ""

    def test_already_resolved(self) -> None:
        """is_resolved=True → (False, "already_resolved")."""
        c = _make_contract(is_resolved=True)
        ok, reason = is_in_universe(c, 15_000.0, _NOW)
        assert ok is False
        assert reason == "already_resolved"

    def test_sport_not_active(self) -> None:
        """Sport absent from ACTIVE_SPORTS → (False, "sport_not_active")."""
        c = _make_contract(sport="CRICKET")
        ok, reason = is_in_universe(c, 15_000.0, _NOW)
        assert ok is False
        assert reason == "sport_not_active"

    def test_resolution_too_soon(self) -> None:
        """Resolution within < 1 day → (False, "resolution_too_soon")."""
        c = _make_contract(resolution_date=_RES_0D)
        ok, reason = is_in_universe(c, 15_000.0, _NOW)
        assert ok is False
        assert reason == "resolution_too_soon"

    def test_resolution_too_far(self) -> None:
        """Resolution beyond 30 days → (False, "resolution_too_far")."""
        c = _make_contract(resolution_date=_RES_31D)
        ok, reason = is_in_universe(c, 15_000.0, _NOW)
        assert ok is False
        assert reason == "resolution_too_far"

    def test_price_out_of_range_too_low(self) -> None:
        """yes_price below 0.01 → (False, "price_out_of_range")."""
        c = _make_contract(yes_price=0.005)
        ok, reason = is_in_universe(c, 15_000.0, _NOW)
        assert ok is False
        assert reason == "price_out_of_range"

    def test_price_out_of_range_too_high(self) -> None:
        """yes_price above 0.99 → (False, "price_out_of_range")."""
        c = _make_contract(yes_price=0.995)
        ok, reason = is_in_universe(c, 15_000.0, _NOW)
        assert ok is False
        assert reason == "price_out_of_range"

    def test_insufficient_book_depth(self) -> None:
        """book_depth_usd below MIN_BOOK_DEPTH_USD → (False, "insufficient_book_depth")."""
        c = _make_contract()
        ok, reason = is_in_universe(c, book_depth_usd=5_000.0, now=_NOW)
        assert ok is False
        assert reason == "insufficient_book_depth"

    def test_insufficient_volume(self) -> None:
        """volume_24h below MIN_VOLUME_24H_CONTRACTS → (False, "insufficient_volume")."""
        c = _make_contract(volume_24h=3.0)
        ok, reason = is_in_universe(c, book_depth_usd=15_000.0, now=_NOW)
        assert ok is False
        assert reason == "insufficient_volume"

    def test_filter_order_resolved_before_sport(self) -> None:
        """is_resolved check fires before sport check — first failure wins."""
        c = _make_contract(is_resolved=True, sport="CRICKET")
        ok, reason = is_in_universe(c, 15_000.0, _NOW)
        assert reason == "already_resolved"   # not "sport_not_active"

    def test_boundary_exactly_at_min_days(self) -> None:
        """Resolution exactly MIN_DAYS_TO_RESOLUTION days out is admitted."""
        res = _NOW + timedelta(days=MIN_DAYS_TO_RESOLUTION)
        c = _make_contract(resolution_date=res)
        ok, _ = is_in_universe(c, 15_000.0, _NOW)
        assert ok is True

    def test_boundary_exactly_at_max_days(self) -> None:
        """Resolution exactly MAX_DAYS_TO_RESOLUTION days out is admitted."""
        res = _NOW + timedelta(days=MAX_DAYS_TO_RESOLUTION)
        c = _make_contract(resolution_date=res)
        ok, _ = is_in_universe(c, 15_000.0, _NOW)
        assert ok is True


# ---------------------------------------------------------------------------
# TestBuildUniverse — end-to-end with mocked client and real SQLite
# ---------------------------------------------------------------------------

class TestBuildUniverse:
    """build_universe() fetches, filters, persists, and returns admitted contracts."""

    def test_happy_path_two_sports_one_contract_each(self, tmp_path) -> None:
        """Two sports with one liquid contract each → 2 contracts returned, DB has 2 rows."""
        db = _make_db(tmp_path)
        c_nba = _make_contract("NBA_C1", sport="NBA", series_ticker="KXNBAGAME")
        c_nfl = _make_contract(
            "NFL_C1", sport="NFL",
            series_ticker="KXNFLGAME", event_ticker="KXNFLGAME-EVENT1",
        )
        client = _make_mock_client(
            events_by_ticker={
                "KXNBAGAME": [_make_event([c_nba], series_ticker="KXNBAGAME")],
                "KXNFLGAME": [_make_event([c_nfl], event_ticker="KXNFLGAME-EVENT1", series_ticker="KXNFLGAME")],
            },
            snapshots_by_contract={
                "NBA_C1": _make_snapshot("NBA_C1", book_depth_usd=15_000.0),
                "NFL_C1": _make_snapshot("NFL_C1", book_depth_usd=20_000.0),
            },
        )

        result = build_universe(client, db)

        assert len(result) == 2
        ids = {c.contract_id for c in result}
        assert ids == {"NBA_C1", "NFL_C1"}
        count = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        assert count == 2
        snap_count = db._conn.execute("SELECT COUNT(*) FROM contract_prices").fetchone()[0]
        assert snap_count == 2

    def test_one_sport_raises_api_error_other_continues(self, tmp_path) -> None:
        """APIError on one sport does not abort — other sport's contracts are returned."""
        db = _make_db(tmp_path)
        c_nfl = _make_contract(
            "NFL_C1", sport="NFL",
            series_ticker="KXNFLGAME", event_ticker="KXNFLGAME-EVENT1",
        )

        def _get_events(series_ticker: str, status: str = "open"):
            if series_ticker == "KXNBAGAME":
                raise APIError("NBA API down", status_code=503)
            if series_ticker == "KXNFLGAME":
                return _make_api_response(
                    [_make_event([c_nfl], event_ticker="KXNFLGAME-EVENT1", series_ticker="KXNFLGAME")]
                )
            return _make_api_response([])

        client = _make_mock_client(
            get_events_side_effect=_get_events,
            snapshots_by_contract={"NFL_C1": _make_snapshot("NFL_C1")},
        )

        result = build_universe(client, db)

        assert any(c.contract_id == "NFL_C1" for c in result)
        assert all(c.contract_id != "NBA_C1" for c in result)

    def test_contract_order_book_fails_others_admitted(self, tmp_path) -> None:
        """get_order_book() failure on one contract skips that contract, others proceed."""
        db = _make_db(tmp_path)
        c1 = _make_contract("C1")
        c2 = _make_contract("C2")

        order_book_calls = iter([
            RuntimeError("network timeout"),          # C1 fails
            _make_api_response(_make_snapshot("C2")), # C2 succeeds
        ])

        client = _make_mock_client(
            events_by_ticker={"KXNBAGAME": [_make_event([c1, c2])]},
            get_order_book_side_effect=lambda cid: next(order_book_calls),
        )

        result = build_universe(client, db)

        assert len(result) == 1
        assert result[0].contract_id == "C2"
        count = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        assert count == 1

    def test_contract_fails_universe_filter_not_in_db_or_results(self, tmp_path) -> None:
        """A contract that fails is_in_universe is not persisted and not returned."""
        db = _make_db(tmp_path)
        # book_depth_usd below threshold → "insufficient_book_depth"
        c = _make_contract("C1")
        thin_snap = _make_snapshot("C1", book_depth_usd=500.0)

        client = _make_mock_client(
            events_by_ticker={"KXNBAGAME": [_make_event([c])]},
            snapshots_by_contract={"C1": thin_snap},
        )

        result = build_universe(client, db)

        assert result == []
        count = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        assert count == 0

    def test_zero_events_for_sport_no_crash(self, tmp_path) -> None:
        """A sport returning zero events produces no output and does not raise."""
        db = _make_db(tmp_path)
        client = _make_mock_client(events_by_ticker={})  # all sports return []

        result = build_universe(client, db)

        assert result == []
        count = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# TestPersistEventContract — direct unit tests for the persistence bridge
# ---------------------------------------------------------------------------

class TestPersistEventContract:
    """persist_event_contract() writes contract + snapshot to DB without API call."""

    def test_contract_and_snapshot_stored(self, tmp_path) -> None:
        """persist_event_contract writes one contracts row and one contract_prices row."""
        db = _make_db(tmp_path)
        c = _make_contract("C1")
        snap = _make_snapshot("C1", book_depth_usd=15_000.0)

        persist_event_contract(db, c, snap)

        contracts = db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        prices = db._conn.execute("SELECT COUNT(*) FROM contract_prices").fetchone()[0]
        assert contracts == 1
        assert prices == 1

    def test_total_liquidity_maps_to_book_depth_usd(self, tmp_path) -> None:
        """total_liquidity in contract_prices equals snap.book_depth_usd."""
        db = _make_db(tmp_path)
        c = _make_contract("C1")
        snap = _make_snapshot("C1", book_depth_usd=12_345.0)

        persist_event_contract(db, c, snap)

        row = db._conn.execute(
            "SELECT total_liquidity FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()
        assert row is not None
        assert row["total_liquidity"] == pytest.approx(12_345.0)

    def test_ticker_columns_written(self, tmp_path) -> None:
        """series_ticker and event_ticker are written to migrated columns."""
        db = _make_db(tmp_path)
        c = _make_contract("C1", series_ticker="KXNBAGAME", event_ticker="KXNBAGAME-EV1")
        snap = _make_snapshot("C1")

        persist_event_contract(db, c, snap)

        row = db._conn.execute(
            "SELECT series_ticker, event_ticker FROM contracts WHERE contract_id = 'C1'"
        ).fetchone()
        assert row["series_ticker"] == "KXNBAGAME"
        assert row["event_ticker"] == "KXNBAGAME-EV1"
