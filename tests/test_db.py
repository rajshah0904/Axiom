"""
Tests for storage/db.py.

All tests use an in-memory SQLite database or a temp file so no disk
artefacts are left behind. No network calls. No shared mutable state
between tests.
"""

import json
import sqlite3
import tempfile
from typing import Optional

import pytest

from storage.db import (
    Database,
    PerformanceRow,
    PositionRow,
    SignalScoreRow,
    _row_to_performance,
    _row_to_position,
    _row_to_signal_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    """In-memory database with schema initialised."""
    d = Database(":memory:")
    d.initialise_schema()
    return d


@pytest.fixture
def db_with_contract(db: Database) -> Database:
    """Database that already has one contract row inserted."""
    db.insert_contract(
        contract_id="KXNBA-C1",
        sport="NBA",
        home_team="OKC Thunder",
        away_team="SAS Spurs",
        game_date="2026-05-28T00:00:00+00:00",
        resolution_date="2026-05-29T00:00:00+00:00",
        ingestion_timestamp="2026-05-26T12:00:00+00:00",
        open_yes_price=0.55,
    )
    return db


# ---------------------------------------------------------------------------
# TestSchemaInit
# ---------------------------------------------------------------------------

class TestSchemaInit:
    """Schema initialisation creates all expected tables."""

    _EXPECTED_TABLES = {
        "contracts",
        "contract_prices",
        "signal_scores",
        "positions",
        "elo_ratings",
        "public_betting",
        "performance_attribution",
    }

    def test_all_tables_created(self, db: Database) -> None:
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert self._EXPECTED_TABLES.issubset(tables)

    def test_schema_idempotent(self, db: Database) -> None:
        """Calling initialise_schema a second time does not raise."""
        db.initialise_schema()  # should not raise

    def test_context_manager_closes_connection(self) -> None:
        with Database(":memory:") as d:
            d.initialise_schema()
        # After exit the connection is closed; any further access raises.
        with pytest.raises(Exception):
            d._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# TestContracts
# ---------------------------------------------------------------------------

class TestContracts:
    """Insert and resolve contract rows."""

    def test_insert_contract_roundtrip(self, db: Database) -> None:
        db.insert_contract(
            contract_id="C1",
            sport="NBA",
            home_team="TeamA",
            away_team="TeamB",
            game_date="2026-06-01T00:00:00+00:00",
            resolution_date="2026-06-02T00:00:00+00:00",
            ingestion_timestamp="2026-05-26T08:00:00+00:00",
            open_yes_price=0.60,
        )
        row = db._conn.execute(
            "SELECT * FROM contracts WHERE contract_id = 'C1'"
        ).fetchone()
        assert row is not None
        assert row["sport"] == "NBA"
        assert row["home_team"] == "TeamA"
        assert row["open_yes_price"] == pytest.approx(0.60)
        assert row["is_resolved"] == 0

    def test_insert_contract_ignores_duplicate(self, db: Database) -> None:
        """Second insert with same contract_id is silently skipped."""
        for _ in range(2):
            db.insert_contract(
                contract_id="DUP",
                sport="NFL",
                home_team="A",
                away_team="B",
                game_date="2026-09-01T00:00:00+00:00",
                resolution_date="2026-09-02T00:00:00+00:00",
                ingestion_timestamp="2026-05-26T00:00:00+00:00",
            )
        count = db._conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE contract_id = 'DUP'"
        ).fetchone()[0]
        assert count == 1

    def test_insert_contract_null_open_price(self, db: Database) -> None:
        db.insert_contract(
            contract_id="NULLPRICE",
            sport="MLB",
            home_team="X",
            away_team="Y",
            game_date="2026-07-04T00:00:00+00:00",
            resolution_date="2026-07-05T00:00:00+00:00",
            ingestion_timestamp="2026-05-26T00:00:00+00:00",
        )
        row = db._conn.execute(
            "SELECT open_yes_price FROM contracts WHERE contract_id = 'NULLPRICE'"
        ).fetchone()
        assert row["open_yes_price"] is None

    def test_mark_contract_resolved_yes(self, db_with_contract: Database) -> None:
        db_with_contract.mark_contract_resolved("KXNBA-C1", resolution_outcome=1)
        row = db_with_contract._conn.execute(
            "SELECT is_resolved, resolution_outcome FROM contracts WHERE contract_id = 'KXNBA-C1'"
        ).fetchone()
        assert row["is_resolved"] == 1
        assert row["resolution_outcome"] == 1

    def test_mark_contract_resolved_no(self, db_with_contract: Database) -> None:
        db_with_contract.mark_contract_resolved("KXNBA-C1", resolution_outcome=0)
        row = db_with_contract._conn.execute(
            "SELECT resolution_outcome FROM contracts WHERE contract_id = 'KXNBA-C1'"
        ).fetchone()
        assert row["resolution_outcome"] == 0

    def test_mark_contract_resolved_invalid_outcome_raises(
        self, db_with_contract: Database
    ) -> None:
        with pytest.raises(ValueError, match="must be 0 or 1"):
            db_with_contract.mark_contract_resolved("KXNBA-C1", resolution_outcome=2)

    def test_count_resolved_contracts(self, db: Database) -> None:
        assert db.count_resolved_contracts() == 0
        db.insert_contract(
            contract_id="R1",
            sport="NBA",
            home_team="A",
            away_team="B",
            game_date="2026-05-28T00:00:00+00:00",
            resolution_date="2026-05-29T00:00:00+00:00",
            ingestion_timestamp="2026-05-26T00:00:00+00:00",
            is_resolved=1,
            resolution_outcome=1,
        )
        assert db.count_resolved_contracts() == 1


# ---------------------------------------------------------------------------
# TestContractPrices
# ---------------------------------------------------------------------------

class TestContractPrices:
    """Price snapshot insertion and retrieval."""

    def test_insert_price_snapshot(self, db_with_contract: Database) -> None:
        db_with_contract.insert_price_snapshot(
            contract_id="KXNBA-C1",
            timestamp="2026-05-26T12:30:00+00:00",
            yes_price=0.57,
            no_price=0.43,
            total_liquidity=15000.0,
            daily_volume=1200.0,
        )
        row = db_with_contract._conn.execute(
            "SELECT * FROM contract_prices WHERE contract_id = 'KXNBA-C1'"
        ).fetchone()
        assert row is not None
        assert row["yes_price"] == pytest.approx(0.57)
        assert row["daily_volume"] == pytest.approx(1200.0)

    def test_insert_price_snapshot_with_depths(
        self, db_with_contract: Database
    ) -> None:
        bid = [{"price": 0.55, "size": 100}]
        ask = [{"price": 0.57, "size": 80}]
        db_with_contract.insert_price_snapshot(
            contract_id="KXNBA-C1",
            timestamp="2026-05-26T13:00:00+00:00",
            yes_price=0.56,
            no_price=0.44,
            total_liquidity=20000.0,
            bid_depth=bid,
            ask_depth=ask,
        )
        row = db_with_contract._conn.execute(
            "SELECT bid_depth_json, ask_depth_json FROM contract_prices "
            "WHERE contract_id = 'KXNBA-C1'"
        ).fetchone()
        assert json.loads(row["bid_depth_json"]) == bid
        assert json.loads(row["ask_depth_json"]) == ask

    def test_price_freshness_reflected_in_get_data_freshness(
        self, db_with_contract: Database
    ) -> None:
        db_with_contract.insert_price_snapshot(
            contract_id="KXNBA-C1",
            timestamp="2026-05-26T14:00:00+00:00",
            yes_price=0.58,
            no_price=0.42,
            total_liquidity=10000.0,
        )
        freshness = db_with_contract.get_data_freshness()
        assert freshness["prices"] == "2026-05-26T14:00:00+00:00"


# ---------------------------------------------------------------------------
# TestSignalScores
# ---------------------------------------------------------------------------

class TestSignalScores:
    """Signal score insertion and retrieval."""

    def test_insert_and_retrieve_signal_scores(
        self, db_with_contract: Database
    ) -> None:
        db_with_contract.insert_signal_scores(
            contract_id="KXNBA-C1",
            computation_timestamp="2026-05-26T08:00:00+00:00",
            momentum_fade_raw=0.12,
            momentum_fade_std=1.4,
            composite_score=0.9,
            p_baseline=0.60,
            p_model=0.65,
            ev=0.05,
        )
        rows = db_with_contract.get_latest_signal_scores(limit=5)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, SignalScoreRow)
        assert row.contract_id == "KXNBA-C1"
        assert row.momentum_fade_raw == pytest.approx(0.12)
        assert row.ev == pytest.approx(0.05)

    def test_get_latest_signal_scores_empty(self, db: Database) -> None:
        rows = db.get_latest_signal_scores()
        assert rows == []

    def test_null_signal_fields_stored_as_none(
        self, db_with_contract: Database
    ) -> None:
        db_with_contract.insert_signal_scores(
            contract_id="KXNBA-C1",
            computation_timestamp="2026-05-26T09:00:00+00:00",
        )
        rows = db_with_contract.get_latest_signal_scores()
        assert rows[0].momentum_fade_raw is None
        assert rows[0].public_fade_raw is None


# ---------------------------------------------------------------------------
# TestPositions
# ---------------------------------------------------------------------------

class TestPositions:
    """Position open/close lifecycle."""

    _POS_ID = "pos-uuid-001"

    def _open(self, db: Database) -> None:
        db.insert_contract(
            contract_id="C-POS",
            sport="NBA",
            home_team="A",
            away_team="B",
            game_date="2026-05-28T00:00:00+00:00",
            resolution_date="2026-05-29T00:00:00+00:00",
            ingestion_timestamp="2026-05-26T00:00:00+00:00",
        )
        db.insert_position(
            position_id=self._POS_ID,
            contract_id="C-POS",
            direction="yes",
            entry_price=0.55,
            entry_timestamp="2026-05-26T08:00:00+00:00",
            size_dollars=500.0,
        )

    def test_insert_position_and_read_active(self, db: Database) -> None:
        self._open(db)
        active = db.get_active_positions()
        assert len(active) == 1
        pos = active[0]
        assert isinstance(pos, PositionRow)
        assert pos.direction == "yes"
        assert pos.size_dollars == pytest.approx(500.0)
        assert pos.exit_timestamp is None

    def test_active_position_not_in_closed(self, db: Database) -> None:
        self._open(db)
        assert db.get_closed_positions() == []

    def test_close_position(self, db: Database) -> None:
        self._open(db)
        db.close_position(
            position_id=self._POS_ID,
            exit_price=0.80,
            exit_timestamp="2026-05-29T10:00:00+00:00",
            realized_ev=0.12,
            slippage=0.003,
        )
        active = db.get_active_positions()
        assert active == []
        closed = db.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].realized_ev == pytest.approx(0.12)
        assert closed[0].exit_price == pytest.approx(0.80)

    def test_invalid_direction_raises(self, db: Database) -> None:
        db.insert_contract(
            contract_id="C-DIR",
            sport="NBA",
            home_team="A",
            away_team="B",
            game_date="2026-05-28T00:00:00+00:00",
            resolution_date="2026-05-29T00:00:00+00:00",
            ingestion_timestamp="2026-05-26T00:00:00+00:00",
        )
        with pytest.raises(ValueError, match="direction must be"):
            db.insert_position(
                position_id="bad-dir",
                contract_id="C-DIR",
                direction="long",
                entry_price=0.55,
                entry_timestamp="2026-05-26T08:00:00+00:00",
                size_dollars=100.0,
            )

    def test_get_total_deployed_capital(self, db: Database) -> None:
        self._open(db)
        assert db.get_total_deployed_capital() == pytest.approx(500.0)

    def test_deployed_capital_zero_when_no_open_positions(self, db: Database) -> None:
        assert db.get_total_deployed_capital() == pytest.approx(0.0)

    def test_get_total_realized_ev(self, db: Database) -> None:
        self._open(db)
        db.close_position(
            position_id=self._POS_ID,
            exit_price=0.80,
            exit_timestamp="2026-05-29T10:00:00+00:00",
            realized_ev=0.05,
            slippage=0.002,
        )
        assert db.get_total_realized_ev() == pytest.approx(0.05)

    def test_realized_ev_zero_when_no_closed_positions(self, db: Database) -> None:
        assert db.get_total_realized_ev() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestEloRatings
# ---------------------------------------------------------------------------

class TestEloRatings:
    """Elo rating insertion and freshness tracking."""

    def test_insert_elo_rating_appears_in_freshness(self, db: Database) -> None:
        db.insert_elo_rating(
            team="Oklahoma City Thunder",
            sport="NBA",
            rating=1548.0,
            as_of_date="2026-05-25",
        )
        freshness = db.get_data_freshness()
        assert freshness["elo"] == "2026-05-25"

    def test_multiple_elo_inserts_stored(self, db: Database) -> None:
        for i, rating in enumerate([1500.0, 1520.0]):
            db.insert_elo_rating(
                team="TeamX",
                sport="NBA",
                rating=rating,
                as_of_date=f"2026-05-{20 + i}",
            )
        count = db._conn.execute(
            "SELECT COUNT(*) FROM elo_ratings WHERE team = 'TeamX'"
        ).fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# TestPublicBetting
# ---------------------------------------------------------------------------

class TestPublicBetting:
    """Public betting insertion and freshness tracking."""

    def test_insert_public_betting(self, db_with_contract: Database) -> None:
        db_with_contract.insert_public_betting(
            contract_id="KXNBA-C1",
            timestamp="2026-05-26T10:00:00+00:00",
            bet_pct_home=0.72,
            dollar_pct_home=0.48,
            sharp_money_indicator=-0.24,
            source="action_network",
        )
        row = db_with_contract._conn.execute(
            "SELECT * FROM public_betting WHERE contract_id = 'KXNBA-C1'"
        ).fetchone()
        assert row["bet_pct_home"] == pytest.approx(0.72)
        assert row["sharp_money_indicator"] == pytest.approx(-0.24)

    def test_public_betting_freshness(self, db_with_contract: Database) -> None:
        db_with_contract.insert_public_betting(
            contract_id="KXNBA-C1",
            timestamp="2026-05-26T11:00:00+00:00",
            bet_pct_home=None,
            dollar_pct_home=None,
            sharp_money_indicator=None,
            source=None,
        )
        freshness = db_with_contract.get_data_freshness()
        assert freshness["public_betting"] == "2026-05-26T11:00:00+00:00"


# ---------------------------------------------------------------------------
# TestPerformanceAttribution
# ---------------------------------------------------------------------------

class TestPerformanceAttribution:
    """Performance attribution insertion and per-signal retrieval."""

    def test_insert_and_retrieve_attribution(self, db: Database) -> None:
        db.insert_performance_attribution(
            date="2026-05-26",
            signal_name="momentum_fade",
            resolved_contract_count=12,
            mean_ev=0.045,
            newey_west_t=2.3,
            rolling_4w_ev=0.040,
            rolling_4w_ir=0.55,
        )
        rows = db.get_recent_performance_attribution("momentum_fade")
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, PerformanceRow)
        assert row.mean_ev == pytest.approx(0.045)
        assert row.newey_west_t == pytest.approx(2.3)

    def test_get_realized_ev_by_signal_with_data(self, db: Database) -> None:
        db.insert_performance_attribution(
            date="2026-05-26",
            signal_name="momentum_fade",
            resolved_contract_count=5,
            mean_ev=0.038,
            newey_west_t=1.8,
            rolling_4w_ev=0.035,
            rolling_4w_ir=0.45,
        )
        ev_map = db.get_realized_ev_by_signal()
        assert ev_map["momentum_fade"] == pytest.approx(0.038)
        assert ev_map["value_reversal"] is None
        assert ev_map["public_money_fade"] is None

    def test_get_realized_ev_by_signal_empty(self, db: Database) -> None:
        ev_map = db.get_realized_ev_by_signal()
        assert all(v is None for v in ev_map.values())


# ---------------------------------------------------------------------------
# TestDataFreshness
# ---------------------------------------------------------------------------

class TestDataFreshness:
    """get_data_freshness returns None for empty tables."""

    def test_all_none_when_tables_empty(self, db: Database) -> None:
        freshness = db.get_data_freshness()
        assert freshness["prices"] is None
        assert freshness["elo"] is None
        assert freshness["public_betting"] is None
        assert freshness["contracts"] is None

    def test_contracts_freshness_reflects_ingestion_timestamp(
        self, db_with_contract: Database
    ) -> None:
        freshness = db_with_contract.get_data_freshness()
        assert freshness["contracts"] == "2026-05-26T12:00:00+00:00"
