"""
Tests for data_ingestion/backfill.py — historical contract and price backfill.

All tests use a real SQLite database (temp file via tmp_path) and a fully
mocked KalshiClient. No live API calls, no network, no shared state between tests.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_client import (
    APIResponse,
    BidAskLevel,
    ClientError,
    ContractObject,
    DataValidationError,
    EventData,
    KalshiClient,
    PriceSnapshot,
)
from storage.db import Database
from data_ingestion.ingest import migrate_schema
from data_ingestion.backfill import (
    _DEFAULT_SLEEP_BETWEEN_CONTRACTS,
    _backfill_one_contract,
    _fetch_all_contracts,
    _fetch_candlesticks_with_fallback,
    _store_price_history,
    backfill_all,
    backfill_sport,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
_START_TS = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_CUTOFF_TS = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)  # older than _NOW


def _make_contract(
    contract_id: str = "C1",
    series_ticker: str = "KXNBAGAME",
    event_ticker: str = "KXNBAGAME-EV1",
    sport: str = "NBA",
    is_resolved: bool = True,
    resolution_outcome: Optional[int] = 1,
    resolution_date: Optional[datetime] = None,
) -> ContractObject:
    """Historical ContractObject — resolved by default (backfill operates on closed games)."""
    return ContractObject(
        contract_id=contract_id,
        series_ticker=series_ticker,
        event_ticker=event_ticker,
        sport=sport,
        home_team="TeamA",
        away_team="TeamB",
        game_date=_NOW,
        resolution_date=resolution_date if resolution_date is not None else _NOW,
        yes_price=0.60,
        no_price=0.40,
        volume_fp=5000.0,
        volume_24h=100.0,
        resolution_criteria_text="Resolves YES if TeamA wins.",
        ingestion_timestamp=_NOW,
        open_yes_price=0.50,
        is_resolved=is_resolved,
        resolution_outcome=resolution_outcome,
    )


def _make_snapshot(
    contract_id: str = "C1",
    yes_price: float = 0.60,
    volume_fp: float = 200.0,
    ts_offset_days: int = 0,
) -> PriceSnapshot:
    """Candlestick-style PriceSnapshot (no order book data — bids/asks empty)."""
    ts = datetime(2022, 2, 1 + ts_offset_days, tzinfo=timezone.utc)
    return PriceSnapshot(
        contract_id=contract_id,
        timestamp=ts,
        yes_price=yes_price,
        no_price=round(1.0 - yes_price, 6),
        volume_fp=volume_fp,
        daily_volume=None,
        bids=[],
        asks=[],
        book_depth_usd=0.0,
    )


def _make_api_response(payload, latency_ms: float = 5.0) -> APIResponse:
    """Wrap payload in a successful APIResponse."""
    return APIResponse(
        payload=payload,
        http_status_code=200,
        latency_ms=latency_ms,
        endpoint="/test",
        success=True,
    )


def _make_event(
    contracts: List[ContractObject],
    event_ticker: str = "KXNBAGAME-EV1",
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


def _make_cutoff_response(
    market_settled_ts: Optional[datetime] = None,
) -> APIResponse:
    """APIResponse for get_historical_cutoff().  Omit key to test fallback path."""
    payload: Dict[str, datetime] = {}
    if market_settled_ts is not None:
        payload["market_settled_ts"] = market_settled_ts
    return _make_api_response(payload)


def _make_client(
    contracts: Optional[List[ContractObject]] = None,
    events: Optional[List[EventData]] = None,
    candles: Optional[List[PriceSnapshot]] = None,
    cutoff_ts: Optional[datetime] = None,
    get_historical_markets_side_effect=None,
    get_events_side_effect=None,
    get_price_history_side_effect=None,
    get_historical_candlesticks_side_effect=None,
    get_historical_cutoff_side_effect=None,
) -> MagicMock:
    """Mock KalshiClient with configurable return values for all backfill-relevant methods.

    Defaults:
        get_historical_cutoff → cutoff_ts or no keys (triggers fallback).
        get_historical_markets → contracts or [].
        get_events → events or [].
        get_price_history → candles or [].
        get_historical_candlesticks → candles or [].
    """
    client = MagicMock(spec=KalshiClient)

    # get_historical_cutoff
    if get_historical_cutoff_side_effect is not None:
        client.get_historical_cutoff.side_effect = get_historical_cutoff_side_effect
    else:
        client.get_historical_cutoff.return_value = _make_cutoff_response(cutoff_ts)

    # get_historical_markets
    if get_historical_markets_side_effect is not None:
        client.get_historical_markets.side_effect = get_historical_markets_side_effect
    else:
        client.get_historical_markets.return_value = _make_api_response(contracts or [])

    # get_events
    if get_events_side_effect is not None:
        client.get_events.side_effect = get_events_side_effect
    else:
        _events = events or []
        client.get_events.return_value = _make_api_response(_events)

    # get_price_history
    if get_price_history_side_effect is not None:
        client.get_price_history.side_effect = get_price_history_side_effect
    else:
        client.get_price_history.return_value = _make_api_response(candles or [])

    # get_historical_candlesticks
    if get_historical_candlesticks_side_effect is not None:
        client.get_historical_candlesticks.side_effect = get_historical_candlesticks_side_effect
    else:
        client.get_historical_candlesticks.return_value = _make_api_response(candles or [])

    return client


# ---------------------------------------------------------------------------
# TestFetchAllContracts
# ---------------------------------------------------------------------------

class TestFetchAllContracts:
    """_fetch_all_contracts() merges three sources and deduplicates by contract_id."""

    def test_deduplication_historical_wins(self) -> None:
        """Same contract_id in historical and live → appears once, historical version wins."""
        hist_contract = _make_contract("C1")
        hist_contract = ContractObject(
            **{**hist_contract.__dict__, "yes_price": 0.70}  # unique yes_price to identify source
        )
        live_contract = _make_contract("C1")
        live_contract = ContractObject(
            **{**live_contract.__dict__, "yes_price": 0.50}
        )
        live_event = _make_event([live_contract])

        client = _make_client(
            contracts=[hist_contract],
            events=[live_event],
        )
        result = _fetch_all_contracts(client, "KXNBAGAME", _CUTOFF_TS)

        assert len(result) == 1
        assert result[0].yes_price == pytest.approx(0.70)  # historical version

    def test_counts_logged(self, caplog) -> None:
        """INFO log contains from_historical_api, from_live_settled, from_live_closed fields."""
        hist_contract = _make_contract("C1")
        settled_event = _make_event([_make_contract("C2")], event_ticker="EV2")
        closed_event = _make_event([_make_contract("C3")], event_ticker="EV3")

        call_n = {"n": 0}

        def _get_events(series_ticker: str, status: str = "open"):
            call_n["n"] += 1
            if status == "settled":
                return _make_api_response([settled_event])
            return _make_api_response([closed_event])

        client = _make_client(
            contracts=[hist_contract],
            get_events_side_effect=_get_events,
        )

        with caplog.at_level(logging.INFO, logger="data_ingestion.backfill"):
            result = _fetch_all_contracts(client, "KXNBAGAME", _CUTOFF_TS)

        assert len(result) == 3
        log_records = [r for r in caplog.records if "from_historical_api" in str(r.__dict__)]
        assert len(log_records) == 1
        rec = log_records[0]
        assert rec.__dict__["from_historical_api"] == 1
        assert rec.__dict__["from_live_settled"] == 1
        assert rec.__dict__["from_live_closed"] == 1
        assert rec.__dict__["total_after_dedup"] == 3

    def test_returns_empty_when_all_sources_empty(self) -> None:
        """No contracts from any source → empty list, no error."""
        client = _make_client(contracts=[], events=[])
        result = _fetch_all_contracts(client, "KXNBAGAME", _CUTOFF_TS)
        assert result == []


# ---------------------------------------------------------------------------
# TestStorePriceHistory
# ---------------------------------------------------------------------------

class TestStorePriceHistory:
    """_store_price_history() always tries historical endpoint first, falling back to live on 404."""

    def test_routes_old_contract_to_historical(self, tmp_path) -> None:
        """resolution_date < cutoff_ts → get_historical_candlesticks called, not get_price_history."""
        db = _make_db(tmp_path)
        old_contract = _make_contract(
            "C1", resolution_date=datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
        db.insert_contract(
            contract_id="C1", sport="NBA", home_team="A", away_team="B",
            game_date=_NOW.isoformat(), resolution_date=_NOW.isoformat(),
            ingestion_timestamp=_NOW.isoformat(), open_yes_price=None,
            resolution_outcome=None, is_resolved=0,
        )
        candles = [_make_snapshot("C1")]
        client = _make_client(candles=candles)
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)  # > resolution_date

        _store_price_history(client, db, old_contract, _START_TS, cutoff)

        client.get_historical_candlesticks.assert_called_once()
        client.get_price_history.assert_not_called()

    def test_historical_always_tried_first_regardless_of_resolution_date(self, tmp_path) -> None:
        """Far-future resolution_date (Kalshi placeholder) → historical still tried first, live not called."""
        db = _make_db(tmp_path)
        # resolution_date far in the future — old code would have routed this to live endpoint
        placeholder_contract = _make_contract(
            "C1", resolution_date=datetime(2027, 4, 15, tzinfo=timezone.utc)
        )
        db.insert_contract(
            contract_id="C1", sport="NBA", home_team="A", away_team="B",
            game_date=_NOW.isoformat(), resolution_date=_NOW.isoformat(),
            ingestion_timestamp=_NOW.isoformat(), open_yes_price=None,
            resolution_outcome=None, is_resolved=0,
        )
        candles = [_make_snapshot("C1")]
        client = _make_client(candles=candles)
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)  # < resolution_date

        _store_price_history(client, db, placeholder_contract, _START_TS, cutoff)

        client.get_historical_candlesticks.assert_called_once()
        client.get_price_history.assert_not_called()

    def test_writes_candles_to_db(self, tmp_path) -> None:
        """5 candles → 5 contract_prices rows for the contract."""
        db = _make_db(tmp_path)
        contract = _make_contract("C1")
        db.insert_contract(
            contract_id="C1", sport="NBA", home_team="A", away_team="B",
            game_date=_NOW.isoformat(), resolution_date=_NOW.isoformat(),
            ingestion_timestamp=_NOW.isoformat(), open_yes_price=None,
            resolution_outcome=None, is_resolved=0,
        )
        candles = [_make_snapshot("C1", ts_offset_days=i) for i in range(5)]
        client = _make_client(candles=candles)

        count = _store_price_history(client, db, contract, _START_TS, _CUTOFF_TS)

        assert count == 5
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()[0]
        assert rows == 5

    def test_total_liquidity_is_zero_for_candles(self, tmp_path) -> None:
        """total_liquidity must be 0.0 for candlestick rows (no order book in candles)."""
        db = _make_db(tmp_path)
        db.insert_contract(
            contract_id="C1", sport="NBA", home_team="A", away_team="B",
            game_date=_NOW.isoformat(), resolution_date=_NOW.isoformat(),
            ingestion_timestamp=_NOW.isoformat(), open_yes_price=None,
            resolution_outcome=None, is_resolved=0,
        )
        client = _make_client(candles=[_make_snapshot("C1", volume_fp=300.0)])

        _store_price_history(client, db, _make_contract("C1"), _START_TS, _CUTOFF_TS)

        row = db._conn.execute(
            "SELECT total_liquidity, daily_volume FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()
        assert row["total_liquidity"] == pytest.approx(0.0)
        assert row["daily_volume"] == pytest.approx(300.0)

    def test_returns_zero_on_empty_price_history(self, tmp_path) -> None:
        """Empty price history → 0 candles written, no crash."""
        db = _make_db(tmp_path)
        db.insert_contract(
            contract_id="C1", sport="NBA", home_team="A", away_team="B",
            game_date=_NOW.isoformat(), resolution_date=_NOW.isoformat(),
            ingestion_timestamp=_NOW.isoformat(), open_yes_price=None,
            resolution_outcome=None, is_resolved=0,
        )
        client = _make_client(candles=[])

        count = _store_price_history(client, db, _make_contract("C1"), _START_TS, _CUTOFF_TS)

        assert count == 0
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()[0]
        assert rows == 0


# ---------------------------------------------------------------------------
# TestFetchCandlesticksWithFallback
# ---------------------------------------------------------------------------

class TestFetchCandlesticksWithFallback:
    """_fetch_candlesticks_with_fallback() tries historical first, falls back to live on 404."""

    def test_fetch_candlesticks_historical_succeeds(self) -> None:
        """Historical returns candles → live endpoint never called, returns historical candles."""
        snaps = [_make_snapshot("C1", ts_offset_days=i) for i in range(3)]
        client = _make_client(candles=snaps)
        contract = _make_contract("C1")

        result = _fetch_candlesticks_with_fallback(client, contract, _START_TS)

        assert result == snaps
        client.get_historical_candlesticks.assert_called_once_with(
            contract.contract_id, contract.series_ticker, _START_TS
        )
        client.get_price_history.assert_not_called()

    def test_fetch_candlesticks_historical_404_falls_back_to_live(self) -> None:
        """Historical 404 → live endpoint called and its candles returned."""
        live_snaps = [_make_snapshot("C1")]
        contract = _make_contract("C1")

        client = _make_client(
            get_historical_candlesticks_side_effect=ClientError(
                "Not found", status_code=404
            ),
            candles=live_snaps,
        )
        # Override live to return candles (already set by _make_client via candles=live_snaps)

        result = _fetch_candlesticks_with_fallback(client, contract, _START_TS)

        assert result == live_snaps
        client.get_historical_candlesticks.assert_called_once()
        client.get_price_history.assert_called_once()

    def test_fetch_candlesticks_historical_non_404_error_raises(self) -> None:
        """Historical 403 → raises immediately, live endpoint never called."""
        contract = _make_contract("C1")
        client = _make_client(
            get_historical_candlesticks_side_effect=ClientError(
                "Forbidden", status_code=403
            ),
        )

        with pytest.raises(ClientError) as exc_info:
            _fetch_candlesticks_with_fallback(client, contract, _START_TS)

        assert exc_info.value.status_code == 403
        client.get_price_history.assert_not_called()

    def test_fetch_candlesticks_both_fail_raises(self) -> None:
        """Historical 404 then live 404 → ClientError propagates from live endpoint."""
        contract = _make_contract("C1")
        client = _make_client(
            get_historical_candlesticks_side_effect=ClientError(
                "Not found", status_code=404
            ),
            get_price_history_side_effect=ClientError(
                "Not found", status_code=404
            ),
        )

        with pytest.raises(ClientError):
            _fetch_candlesticks_with_fallback(client, contract, _START_TS)

        client.get_historical_candlesticks.assert_called_once()
        client.get_price_history.assert_called_once()

    def test_store_price_history_writes_all_snaps(self, tmp_path) -> None:
        """5 snapshots returned → 5 rows written to DB, returns 5."""
        db = _make_db(tmp_path)
        db.insert_contract(
            contract_id="C1", sport="NBA", home_team="A", away_team="B",
            game_date=_NOW.isoformat(), resolution_date=_NOW.isoformat(),
            ingestion_timestamp=_NOW.isoformat(), open_yes_price=None,
            resolution_outcome=None, is_resolved=0,
        )
        snaps = [_make_snapshot("C1", ts_offset_days=i) for i in range(5)]
        client = _make_client(candles=snaps)

        count = _store_price_history(client, db, _make_contract("C1"), _START_TS, _CUTOFF_TS)

        assert count == 5
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()[0]
        assert rows == 5

    def test_store_price_history_empty_snapshot_list(self, tmp_path) -> None:
        """Empty snapshot list → 0 rows written, returns 0."""
        db = _make_db(tmp_path)
        db.insert_contract(
            contract_id="C1", sport="NBA", home_team="A", away_team="B",
            game_date=_NOW.isoformat(), resolution_date=_NOW.isoformat(),
            ingestion_timestamp=_NOW.isoformat(), open_yes_price=None,
            resolution_outcome=None, is_resolved=0,
        )
        client = _make_client(candles=[])

        count = _store_price_history(client, db, _make_contract("C1"), _START_TS, _CUTOFF_TS)

        assert count == 0
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM contract_prices WHERE contract_id = 'C1'"
        ).fetchone()[0]
        assert rows == 0

    def test_backfill_one_contract_price_failure_leaves_contract_row(
        self, tmp_path
    ) -> None:
        """Both candlestick endpoints fail → contract row still in DB, exception propagates."""
        db = _make_db(tmp_path)
        contract = _make_contract("C1")
        client = _make_client(
            cutoff_ts=_CUTOFF_TS,
            get_historical_candlesticks_side_effect=ClientError(
                "Not found", status_code=404
            ),
            get_price_history_side_effect=ClientError(
                "Not found", status_code=404
            ),
        )

        with patch("time.sleep"):
            with pytest.raises(ClientError):
                _backfill_one_contract(
                    client, db, contract, _START_TS, _CUTOFF_TS, sleep_s=0.0
                )

        row = db._conn.execute(
            "SELECT contract_id FROM contracts WHERE contract_id = 'C1'"
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# TestBackfillSport
# ---------------------------------------------------------------------------

class TestBackfillSport:
    """backfill_sport() fetches cutoff, fetches contracts, processes all, logs per sport."""

    def test_calls_cutoff_once(self, tmp_path) -> None:
        """get_historical_cutoff called exactly once per backfill_sport invocation."""
        db = _make_db(tmp_path)
        client = _make_client(cutoff_ts=_CUTOFF_TS, contracts=[], events=[])

        with patch("time.sleep"):
            backfill_sport(client, db, "KXNBAGAME", "NBA", _START_TS)

        client.get_historical_cutoff.assert_called_once()

    def test_cutoff_missing_market_settled_ts_uses_fallback(
        self, tmp_path, caplog
    ) -> None:
        """market_settled_ts absent → fallback used, WARNING logged, backfill continues."""
        db = _make_db(tmp_path)
        client = _make_client(cutoff_ts=None, contracts=[], events=[])

        with caplog.at_level(logging.WARNING, logger="data_ingestion.backfill"):
            with patch("time.sleep"):
                count = backfill_sport(client, db, "KXNBAGAME", "NBA", _START_TS)

        assert count == 0
        warns = [r for r in caplog.records if "fallback" in r.getMessage()]
        assert len(warns) >= 1

    def test_one_contract_fails_others_still_processed(self, tmp_path) -> None:
        """DataValidationError on one contract does not abort others — error logged."""
        db = _make_db(tmp_path)
        # resolution_date before cutoff → routes to get_historical_candlesticks
        old_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        c1 = _make_contract("C1", resolution_date=old_date)
        c2 = _make_contract("C2", resolution_date=old_date)

        def _hist_candlesticks(contract_id, series_ticker, start_ts):
            if contract_id == "C1":
                raise DataValidationError("bad data", contract_id="C1")
            return _make_api_response([_make_snapshot(contract_id)])

        client = _make_client(
            cutoff_ts=_CUTOFF_TS,
            contracts=[c1, c2],
            events=[],
            get_historical_candlesticks_side_effect=_hist_candlesticks,
        )

        with patch("time.sleep"):
            count = backfill_sport(client, db, "KXNBAGAME", "NBA", _START_TS)

        assert count == 1
        rows = db._conn.execute(
            "SELECT contract_id FROM contracts"
        ).fetchall()
        ids = {r["contract_id"] for r in rows}
        assert "C2" in ids

    def test_happy_path_contracts_written_to_db(self, tmp_path) -> None:
        """4 contracts × 3 candles each → 4 contract rows, 12 price rows in DB."""
        db = _make_db(tmp_path)
        old_date = datetime(2025, 1, 1, tzinfo=timezone.utc)  # before _CUTOFF_TS → historical path
        contracts = [_make_contract(f"C{i}", resolution_date=old_date) for i in range(4)]

        def _hist_candlesticks(contract_id, series_ticker, start_ts):
            snaps = [_make_snapshot(contract_id, ts_offset_days=j) for j in range(3)]
            return _make_api_response(snaps)

        client = _make_client(
            cutoff_ts=_CUTOFF_TS,
            contracts=contracts,
            events=[],
            get_historical_candlesticks_side_effect=_hist_candlesticks,
        )

        with patch("time.sleep"):
            count = backfill_sport(client, db, "KXNBAGAME", "NBA", _START_TS)

        assert count == 4
        assert db._conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0] == 4
        assert db._conn.execute("SELECT COUNT(*) FROM contract_prices").fetchone()[0] == 12

    def test_resolved_outcome_written_to_db(self, tmp_path) -> None:
        """Resolved contracts have is_resolved=1 and resolution_outcome written."""
        db = _make_db(tmp_path)
        contract = _make_contract("C1", is_resolved=True, resolution_outcome=1)

        client = _make_client(cutoff_ts=_CUTOFF_TS, contracts=[contract], events=[])

        with patch("time.sleep"):
            backfill_sport(client, db, "KXNBAGAME", "NBA", _START_TS)

        row = db._conn.execute(
            "SELECT is_resolved, resolution_outcome FROM contracts WHERE contract_id = 'C1'"
        ).fetchone()
        assert row["is_resolved"] == 1
        assert row["resolution_outcome"] == 1


# ---------------------------------------------------------------------------
# TestBackfillAll
# ---------------------------------------------------------------------------

class TestBackfillAll:
    """backfill_all() loops all sports; one sport failing does not abort others."""

    def test_one_sport_raises_others_still_processed(self, tmp_path) -> None:
        """RuntimeError on NBA does not abort NFL backfill."""
        db = _make_db(tmp_path)
        nfl_contract = _make_contract(
            "NFL_C1", series_ticker="KXNFLGAME", event_ticker="KXNFLGAME-EV1", sport="NFL"
        )

        def _get_hist_markets(series_ticker: str):
            if series_ticker == "KXNBAGAME":
                raise RuntimeError("NBA API unavailable")
            if series_ticker == "KXNFLGAME":
                return _make_api_response([nfl_contract])
            return _make_api_response([])

        client = _make_client(
            cutoff_ts=_CUTOFF_TS,
            get_historical_markets_side_effect=_get_hist_markets,
            events=[],
            candles=[],
        )

        with patch("time.sleep"):
            backfill_all(client, db, start_date="2021-01-01", sleep_between_contracts=0.0)

        row = db._conn.execute(
            "SELECT contract_id FROM contracts WHERE contract_id = 'NFL_C1'"
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# TestRunBackfill (CLI entry point)
# ---------------------------------------------------------------------------

class TestRunBackfill:
    """run_backfill() reads env vars, parses CLI args, routes to correct sport."""

    _PATCHES = [
        "kalshi_client.KalshiConfig.from_env",
        "kalshi_client.KalshiClient",
        "storage.db.Database",
        "data_ingestion.ingest.migrate_schema",
        "dotenv.load_dotenv",
    ]

    def _patch_all(self):
        from contextlib import ExitStack
        stack = ExitStack()
        mocks = {name: stack.enter_context(patch(name)) for name in self._PATCHES}
        mocks["kalshi_client.KalshiConfig.from_env"].return_value = MagicMock()
        mocks["kalshi_client.KalshiClient"].return_value = MagicMock()
        mocks["storage.db.Database"].return_value = MagicMock()
        return stack, mocks

    def test_run_backfill_single_sport_flag(self, tmp_path, monkeypatch) -> None:
        """--sport MLB → only MLB backfilled, other sports not called."""
        monkeypatch.setenv("AXIOM_DB_PATH", str(tmp_path / "backfill.db"))
        monkeypatch.setattr(sys, "argv", ["backfill", "--sport", "MLB"])

        stack, _ = self._patch_all()
        with stack:
            with (
                patch("data_ingestion.backfill.backfill_sport") as mock_sport,
                patch("data_ingestion.backfill.backfill_all") as mock_all,
            ):
                mock_sport.return_value = 0
                from data_ingestion.backfill import run_backfill
                run_backfill()

        mock_sport.assert_called_once()
        args = mock_sport.call_args[0]
        assert args[3] == "MLB"    # sport positional arg
        mock_all.assert_not_called()

    def test_sport_arg_routes_to_single_sport(self, tmp_path, monkeypatch) -> None:
        """--sport NBA calls backfill_sport for NBA only."""
        monkeypatch.setenv("AXIOM_DB_PATH", str(tmp_path / "backfill.db"))
        monkeypatch.setattr(sys, "argv", ["backfill", "--sport", "NBA", "--start-date", "2021-01-01"])

        stack, _ = self._patch_all()
        with stack:
            with (
                patch("data_ingestion.backfill.backfill_sport") as mock_sport,
                patch("data_ingestion.backfill.backfill_all") as mock_all,
            ):
                mock_sport.return_value = 0
                from data_ingestion.backfill import run_backfill
                run_backfill()

        mock_sport.assert_called_once()
        args = mock_sport.call_args[0]
        assert args[2] == "KXNBAGAME"   # series_ticker positional arg
        assert args[3] == "NBA"          # sport positional arg
        mock_all.assert_not_called()

    def test_no_sport_arg_calls_backfill_all(self, tmp_path, monkeypatch) -> None:
        """Omitting --sport calls backfill_all, not backfill_sport."""
        monkeypatch.setenv("AXIOM_DB_PATH", str(tmp_path / "backfill2.db"))
        monkeypatch.setattr(sys, "argv", ["backfill", "--start-date", "2021-01-01"])

        stack, _ = self._patch_all()
        with stack:
            with (
                patch("data_ingestion.backfill.backfill_sport") as mock_sport,
                patch("data_ingestion.backfill.backfill_all") as mock_all,
            ):
                mock_all.return_value = None
                from data_ingestion.backfill import run_backfill
                run_backfill()

        mock_all.assert_called_once()
        mock_sport.assert_not_called()

    def test_missing_db_path_exits_with_code_1(self, monkeypatch) -> None:
        """Missing AXIOM_DB_PATH exits immediately with code 1."""
        monkeypatch.delenv("AXIOM_DB_PATH", raising=False)
        monkeypatch.setattr(sys, "argv", ["backfill"])

        from data_ingestion.backfill import run_backfill
        with patch("dotenv.load_dotenv"), pytest.raises(SystemExit) as exc_info:
            run_backfill()

        assert exc_info.value.code == 1

    def test_invalid_sport_arg_exits_with_code_1(self, tmp_path, monkeypatch) -> None:
        """Unknown --sport value exits with code 1."""
        monkeypatch.setenv("AXIOM_DB_PATH", str(tmp_path / "x.db"))
        monkeypatch.setattr(sys, "argv", ["backfill", "--sport", "CRICKET"])

        stack, _ = self._patch_all()
        with stack:
            from data_ingestion.backfill import run_backfill
            with pytest.raises(SystemExit) as exc_info:
                run_backfill()

        assert exc_info.value.code == 1
