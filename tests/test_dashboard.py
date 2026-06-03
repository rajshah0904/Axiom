"""
Tests for monitoring/dashboard.py.

All tests use an in-memory SQLite database. No network calls. No Kalshi
API calls. No disk artefacts outside a temporary directory.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from monitoring.dashboard import (
    ALERT_CRITICAL,
    ALERT_ERROR,
    ALERT_WARNING,
    AlertRecord,
    Dashboard,
    _age_seconds,
    _format_age,
    _read_recent_log_errors,
    _terminal_width,
)
from storage.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    """Empty in-memory database with schema."""
    d = Database(":memory:")
    d.initialise_schema()
    return d


@pytest.fixture
def dash(db: Database) -> Dashboard:
    """Dashboard with no Kalshi client and no log file."""
    return Dashboard(db, log_file_path=None, kalshi_client=None)


def _insert_contract(db: Database, contract_id: str = "C1") -> None:
    """Helper: insert a minimal contract row."""
    db.insert_contract(
        contract_id=contract_id,
        sport="NBA",
        home_team="TeamA",
        away_team="TeamB",
        game_date="2026-06-01T00:00:00+00:00",
        resolution_date="2026-06-02T00:00:00+00:00",
        ingestion_timestamp="2026-05-26T12:00:00+00:00",
        open_yes_price=0.55,
    )


def _insert_position(
    db: Database,
    position_id: str = "p1",
    contract_id: str = "C1",
    size_dollars: float = 500.0,
    closed: bool = False,
    realized_ev: float = 0.05,
) -> None:
    """Helper: insert an open or closed position."""
    db.insert_position(
        position_id=position_id,
        contract_id=contract_id,
        direction="yes",
        entry_price=0.55,
        entry_timestamp="2026-05-26T08:00:00+00:00",
        size_dollars=size_dollars,
    )
    if closed:
        db.close_position(
            position_id=position_id,
            exit_price=0.80,
            exit_timestamp="2026-05-29T10:00:00+00:00",
            realized_ev=realized_ev,
            slippage=0.002,
        )


# ---------------------------------------------------------------------------
# TestRender — smoke tests that render() does not crash on various states
# ---------------------------------------------------------------------------

class TestRender:
    """render() must not raise regardless of database state."""

    def test_render_empty_db_no_crash(self, dash: Dashboard) -> None:
        """render() on a fully empty database should not raise."""
        with patch("monitoring.dashboard._clear_terminal"):
            dash.render()  # no assertion — just must not raise

    def test_render_with_active_positions(self, db: Database) -> None:
        _insert_contract(db)
        _insert_position(db)
        dash = Dashboard(db, log_file_path=None)
        with patch("monitoring.dashboard._clear_terminal"):
            dash.render()

    def test_render_with_closed_positions(self, db: Database) -> None:
        _insert_contract(db)
        _insert_position(db, closed=True)
        dash = Dashboard(db, log_file_path=None)
        with patch("monitoring.dashboard._clear_terminal"):
            dash.render()

    def test_render_with_no_client_shows_not_configured(
        self, db: Database, capsys: Any
    ) -> None:
        dash = Dashboard(db, log_file_path=None, kalshi_client=None)
        with patch("monitoring.dashboard._clear_terminal"):
            dash.render()
        out = capsys.readouterr().out
        assert "not configured" in out

    def test_render_with_missing_log_file(self, db: Database, tmp_path: Any) -> None:
        dash = Dashboard(db, log_file_path=os.path.join(str(tmp_path), "does_not_exist.log"))
        with patch("monitoring.dashboard._clear_terminal"):
            dash.render()  # should not raise — file not found is handled


# ---------------------------------------------------------------------------
# TestCheckAlerts — alert detection logic
# ---------------------------------------------------------------------------

class TestCheckAlerts:
    """check_alerts() returns the correct AlertRecord objects."""

    def test_no_alerts_when_db_empty_and_fresh(self, dash: Dashboard) -> None:
        """Empty tables produce no staleness alerts (no data yet is OK)."""
        alerts = dash.check_alerts()
        # Only capital and fee-erosion checks can fire on an empty db.
        # Capital is 0, fee erosion needs ≥2 closed positions.
        capital_alerts = [a for a in alerts if a.source == "capital_limits"]
        assert capital_alerts == []

    def test_staleness_alert_when_data_is_old(self, db: Database) -> None:
        """ERROR alert fires when a data source is stale beyond the 2-hour threshold."""
        # Insert a price snapshot with a timestamp 3 hours old.
        stale_ts = (
            datetime.now(timezone.utc) - timedelta(hours=3)
        ).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        _insert_contract(db)
        db.insert_price_snapshot(
            contract_id="C1",
            timestamp=stale_ts,
            yes_price=0.55,
            no_price=0.45,
            total_liquidity=15000.0,
        )
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        staleness_alerts = [a for a in alerts if a.source == "data_freshness"]
        assert len(staleness_alerts) >= 1
        assert all(a.level == ALERT_ERROR for a in staleness_alerts)
        assert all(not a.auto_halt for a in staleness_alerts)

    def test_no_staleness_alert_when_data_is_fresh(self, db: Database) -> None:
        """No staleness alert when all populated data sources were updated recently."""
        fresh_ts = (
            datetime.now(timezone.utc) - timedelta(minutes=15)
        ).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        # Insert contract with a fresh ingestion timestamp so the "contracts"
        # data source is not flagged as stale.
        db.insert_contract(
            contract_id="C1",
            sport="NBA",
            home_team="TeamA",
            away_team="TeamB",
            game_date="2026-06-01T00:00:00+00:00",
            resolution_date="2026-06-02T00:00:00+00:00",
            ingestion_timestamp=fresh_ts,
            open_yes_price=0.55,
        )
        db.insert_price_snapshot(
            contract_id="C1",
            timestamp=fresh_ts,
            yes_price=0.55,
            no_price=0.45,
            total_liquidity=15000.0,
        )
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        staleness_alerts = [a for a in alerts if a.source == "data_freshness"]
        assert staleness_alerts == []

    def test_capital_warning_fires_above_4000(self, db: Database) -> None:
        _insert_contract(db, "C1")
        _insert_contract(db, "C2")
        _insert_position(db, "p1", "C1", size_dollars=2500.0)
        _insert_position(db, "p2", "C2", size_dollars=2000.0)
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        capital_alerts = [a for a in alerts if a.source == "capital_limits"]
        assert len(capital_alerts) == 1
        assert capital_alerts[0].level == ALERT_WARNING
        assert not capital_alerts[0].auto_halt

    def test_no_capital_warning_below_threshold(self, db: Database) -> None:
        _insert_contract(db)
        _insert_position(db, size_dollars=500.0)
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        capital_alerts = [a for a in alerts if a.source == "capital_limits"]
        assert capital_alerts == []

    def test_fee_erosion_warning_when_ev_negative(self, db: Database) -> None:
        """WARNING fires when rolling_4w_ev is negative for 2 consecutive attribution rows."""
        for date in ["2026-05-19", "2026-05-26"]:
            db.insert_performance_attribution(
                date=date,
                signal_name="momentum_fade",
                resolved_contract_count=10,
                mean_ev=0.050,
                newey_west_t=2.1,
                rolling_4w_ev=-0.010,
                rolling_4w_ir=-0.20,
            )
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        erosion_alerts = [a for a in alerts if a.source == "fee_erosion"]
        assert len(erosion_alerts) == 1
        assert erosion_alerts[0].level == ALERT_WARNING

    def test_no_fee_erosion_when_ev_positive(self, db: Database) -> None:
        """No fee erosion alert when rolling_4w_ev is positive across 2 consecutive rows."""
        for date in ["2026-05-19", "2026-05-26"]:
            db.insert_performance_attribution(
                date=date,
                signal_name="momentum_fade",
                resolved_contract_count=10,
                mean_ev=0.050,
                newey_west_t=2.1,
                rolling_4w_ev=0.030,
                rolling_4w_ir=0.40,
            )
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        erosion_alerts = [a for a in alerts if a.source == "fee_erosion"]
        assert erosion_alerts == []

    def test_signal_decay_warning_fires(self, db: Database) -> None:
        """WARNING fires when rolling 4w EV is below 40% of mean for 2 rows."""
        for i, date in enumerate(["2026-05-19", "2026-05-26"]):
            db.insert_performance_attribution(
                date=date,
                signal_name="momentum_fade",
                resolved_contract_count=10,
                mean_ev=0.050,          # historical mean = 0.050
                newey_west_t=2.1,
                rolling_4w_ev=0.010,    # 10% of mean — well below 40% threshold
                rolling_4w_ir=0.20,
            )
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        decay_alerts = [a for a in alerts if "signal_decay" in a.source]
        assert len(decay_alerts) >= 1
        assert decay_alerts[0].level == ALERT_WARNING

    def test_signal_decay_does_not_fire_with_only_one_window(
        self, db: Database
    ) -> None:
        """Decay requires two consecutive bad windows — one is not enough."""
        db.insert_performance_attribution(
            date="2026-05-26",
            signal_name="momentum_fade",
            resolved_contract_count=10,
            mean_ev=0.050,
            newey_west_t=2.1,
            rolling_4w_ev=0.010,
            rolling_4w_ir=0.20,
        )
        dash = Dashboard(db, log_file_path=None)
        alerts = dash.check_alerts()
        decay_alerts = [a for a in alerts if "signal_decay" in a.source]
        assert decay_alerts == []

    def test_api_critical_alert_on_auth_failure(self, db: Database) -> None:
        """CRITICAL auto-halt alert when Kalshi client raises AuthenticationError."""
        from kalshi_client import AuthenticationError

        mock_client = MagicMock()
        mock_client.get_balance.side_effect = AuthenticationError(
            "invalid credentials", status_code=401
        )
        dash = Dashboard(db, log_file_path=None, kalshi_client=mock_client)
        alerts = dash.check_alerts()
        api_alerts = [a for a in alerts if a.source == "kalshi_api"]
        assert len(api_alerts) == 1
        assert api_alerts[0].level == ALERT_CRITICAL
        assert api_alerts[0].auto_halt is True

    def test_api_error_alert_on_non_auth_failure(self, db: Database) -> None:
        """ERROR (no auto-halt) alert when client raises non-auth exception."""
        mock_client = MagicMock()
        mock_client.get_balance.side_effect = ConnectionError("timeout")
        dash = Dashboard(db, log_file_path=None, kalshi_client=mock_client)
        alerts = dash.check_alerts()
        api_alerts = [a for a in alerts if a.source == "kalshi_api"]
        assert len(api_alerts) == 1
        assert api_alerts[0].level == ALERT_ERROR
        assert not api_alerts[0].auto_halt

    def test_no_api_alert_when_no_client(self, db: Database) -> None:
        dash = Dashboard(db, log_file_path=None, kalshi_client=None)
        alerts = dash.check_alerts()
        api_alerts = [a for a in alerts if a.source == "kalshi_api"]
        assert api_alerts == []


# ---------------------------------------------------------------------------
# TestHandleCriticalAlerts — auto-halt behaviour
# ---------------------------------------------------------------------------

class TestHandleCriticalAlerts:
    """_handle_critical_alerts raises SystemExit on CRITICAL auto-halt."""

    def _make_critical(self) -> AlertRecord:
        return AlertRecord(
            level=ALERT_CRITICAL,
            source="kalshi_api",
            message="auth failure",
            timestamp=datetime.now(timezone.utc),
            auto_halt=True,
        )

    def _make_warning(self) -> AlertRecord:
        return AlertRecord(
            level=ALERT_WARNING,
            source="fee_erosion",
            message="net EV negative",
            timestamp=datetime.now(timezone.utc),
            auto_halt=False,
        )

    def test_critical_alert_raises_system_exit(self, dash: Dashboard) -> None:
        with pytest.raises(SystemExit) as exc_info:
            dash._handle_critical_alerts([self._make_critical()])
        assert exc_info.value.code == 1

    def test_warning_only_does_not_raise(self, dash: Dashboard) -> None:
        dash._handle_critical_alerts([self._make_warning()])  # no raise

    def test_empty_alerts_does_not_raise(self, dash: Dashboard) -> None:
        dash._handle_critical_alerts([])  # no raise

    def test_mix_of_warning_and_critical_raises(self, dash: Dashboard) -> None:
        alerts = [self._make_warning(), self._make_critical()]
        with pytest.raises(SystemExit):
            dash._handle_critical_alerts(alerts)


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------

class TestAgeSeconds:
    """_age_seconds parses ISO 8601 and returns float seconds."""

    def test_fresh_timestamp(self) -> None:
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        age = _age_seconds(ts, now)
        assert age is not None
        assert 29.0 <= age <= 31.0

    def test_unparseable_returns_none(self) -> None:
        assert _age_seconds("not-a-date", datetime.now(timezone.utc)) is None

    def test_date_only_format(self) -> None:
        now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
        age = _age_seconds("2026-05-26", now)
        assert age is not None
        assert age == pytest.approx(12 * 3600, abs=1)


class TestFormatAge:
    """_format_age returns human-readable string and colour."""

    def test_fresh_returns_minutes(self) -> None:
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        age_str, _ = _format_age(ts, now)
        assert "m ago" in age_str

    def test_stale_returns_stale_label(self) -> None:
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        age_str, _ = _format_age(ts, now)
        assert "STALE" in age_str


class TestReadRecentLogErrors:
    """_read_recent_log_errors parses JSON log and returns error entries."""

    def _write_log(self, tmp_path: Any, entries: List[dict]) -> str:
        path = os.path.join(tmp_path, "test.log")
        with open(path, "w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
        return path

    def test_returns_only_error_levels(self, tmp_path: Any) -> None:
        entries = [
            {"levelname": "DEBUG", "asctime": "2026-05-26 08:00:00", "message": "ok"},
            {"levelname": "INFO", "asctime": "2026-05-26 08:01:00", "message": "info"},
            {"levelname": "WARNING", "asctime": "2026-05-26 08:02:00", "message": "warn"},
            {"levelname": "ERROR", "asctime": "2026-05-26 08:03:00", "message": "err"},
        ]
        log_path = self._write_log(tmp_path, entries)
        results = _read_recent_log_errors(log_path, scan_lines=50, max_results=10)
        assert len(results) == 2
        levels = {r["levelname"] for r in results}
        assert levels == {"WARNING", "ERROR"}

    def test_missing_file_returns_empty_list(self, tmp_path: Any) -> None:
        results = _read_recent_log_errors(
            os.path.join(tmp_path, "nonexistent.log"), 50, 10
        )
        assert results == []

    def test_non_json_lines_skipped(self, tmp_path: Any) -> None:
        path = os.path.join(tmp_path, "mixed.log")
        with open(path, "w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps({"levelname": "ERROR", "message": "real"}) + "\n")
        results = _read_recent_log_errors(path, scan_lines=50, max_results=10)
        assert len(results) == 1
        assert results[0]["message"] == "real"

    def test_respects_max_results(self, tmp_path: Any) -> None:
        entries = [
            {"levelname": "ERROR", "asctime": f"t{i}", "message": f"e{i}"}
            for i in range(20)
        ]
        log_path = self._write_log(tmp_path, entries)
        results = _read_recent_log_errors(log_path, scan_lines=500, max_results=5)
        assert len(results) == 5


class TestTerminalWidth:
    """_terminal_width returns a positive integer."""

    def test_returns_positive_int(self) -> None:
        width = _terminal_width()
        assert isinstance(width, int)
        assert width > 0
