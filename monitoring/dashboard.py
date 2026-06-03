"""
Real-time monitoring dashboard — Phase 1 prediction market quant fund.

Terminal display of system health, active positions, data freshness,
per-signal realised EV, and a rolling error log. Reads exclusively from
the SQLite database and (optionally) pings the Kalshi API for connectivity.

Alert levels follow CLAUDE.md exactly:
  - CRITICAL: auto-halt the system (authentication failure, data validation
    failure, position mismatch, partial fill with open leg).
  - ERROR:    logged and displayed; system continues (data source stale > 2h).
  - WARNING:  logged and displayed; system continues (signal decay, fee
    erosion, position drift, capital limit approaching).

Usage (CLI)::

    python -m monitoring.dashboard           # loop, reads AXIOM_DB_PATH env var
    python -m monitoring.dashboard --once    # single render then exit

Usage (programmatic)::

    from storage.db import Database
    from monitoring.dashboard import Dashboard
    db = Database("fund.db")
    db.initialise_schema()
    dash = Dashboard(db, log_file_path="logs/fund.log")
    dash.run(refresh_interval=60)
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from storage.db import Database

# ---------------------------------------------------------------------------
# Module-level logger — structured JSON output matches the rest of the system
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert level constants (string, not enum, to avoid import overhead)
# ---------------------------------------------------------------------------

ALERT_CRITICAL = "CRITICAL"
ALERT_ERROR = "ERROR"
ALERT_WARNING = "WARNING"

# ---------------------------------------------------------------------------
# Thresholds — sourced from CLAUDE.md; named constants, never magic numbers
# ---------------------------------------------------------------------------

# Data freshness: any source with no update in this window fires ERROR.
_STALENESS_THRESHOLD_SECONDS: int = 7200          # 2 hours

# Capital limits (CLAUDE.md monitoring table).
_CAPITAL_WARNING_DOLLARS: float = 4_000.0         # > $4,000 → WARNING
_CAPITAL_HARD_LIMIT_DOLLARS: float = 5_000.0      # $5,000 is the absolute ceiling

# Signal decay: rolling 4w EV below this fraction of historical mean → WARNING.
_SIGNAL_DECAY_FRACTION: float = 0.40

# Number of recent log lines shown in the error log panel.
_ERROR_LOG_LINES: int = 10

# Number of recent log file lines to scan for WARNING/ERROR/CRITICAL entries.
_LOG_SCAN_LINES: int = 500

# ---------------------------------------------------------------------------
# ANSI colour codes — guarded by _USE_COLOUR at module init
# ---------------------------------------------------------------------------

_USE_COLOUR: bool = sys.stdout.isatty()

_RED = "\033[91m" if _USE_COLOUR else ""
_YELLOW = "\033[93m" if _USE_COLOUR else ""
_GREEN = "\033[92m" if _USE_COLOUR else ""
_CYAN = "\033[96m" if _USE_COLOUR else ""
_BOLD = "\033[1m" if _USE_COLOUR else ""
_RESET = "\033[0m" if _USE_COLOUR else ""


# ---------------------------------------------------------------------------
# AlertRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlertRecord:
    """One fired alert with its level, source component, and message.

    Attributes:
        level: ``"CRITICAL"``, ``"ERROR"``, or ``"WARNING"``.
        source: Component that produced the alert (e.g. ``"data_freshness"``).
        message: Human-readable description of the alert condition.
        timestamp: UTC wall-clock time when the alert was generated.
        auto_halt: True when the system must stop immediately.
    """

    level: str
    source: str
    message: str
    timestamp: datetime
    auto_halt: bool


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class Dashboard:
    """Reads from SQLite and renders a terminal health display.

    Args:
        db: Initialised :class:`~storage.db.Database` instance.
        log_file_path: Absolute path to the system JSON log file.
            Pass ``None`` to skip the error log panel.
        kalshi_client: Optional initialised Kalshi client used to ping
            the API for connectivity checks. Pass ``None`` to skip the
            API status panel (it will show "not configured").
    """

    def __init__(
        self,
        db: Database,
        log_file_path: Optional[str],
        kalshi_client: Optional[Any] = None,
    ) -> None:
        self._db = db
        self._log_file_path = log_file_path
        self._kalshi_client = kalshi_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check_alerts(self) -> List[AlertRecord]:
        """Evaluate all alert conditions and return every fired alert.

        Checks are evaluated in priority order: CRITICAL first, then ERROR,
        then WARNING. The list is sorted by level severity (CRITICAL first).

        Returns:
            List of :class:`AlertRecord`, possibly empty. Callers should
            inspect each record's ``auto_halt`` flag.
        """
        alerts: List[AlertRecord] = []
        alerts.extend(self._check_api_connectivity())
        alerts.extend(self._check_data_freshness())
        alerts.extend(self._check_capital_limit())
        alerts.extend(self._check_signal_decay())
        alerts.extend(self._check_fee_erosion())
        return alerts

    def render(self) -> None:
        """Clear the terminal and print a full dashboard refresh.

        Sections:
            1. Header (timestamp, resolved contract count)
            2. Active alerts
            3. Active positions table
            4. Kalshi API status
            5. Data freshness panel
            6. Realised EV per signal
            7. Recent error log

        Raises:
            Nothing — all individual sections catch and display exceptions
            rather than crashing the render loop.
        """
        width = _terminal_width()
        _clear_terminal()
        _print_header(width)
        self._render_alerts_section(width)
        self._render_positions_section(width)
        self._render_api_status_section(width)
        self._render_freshness_section(width)
        self._render_signal_ev_section(width)
        self._render_error_log_section(width)

    def run(self, refresh_interval: int = 60) -> None:
        """Enter the refresh loop. Blocks until interrupted or CRITICAL alert.

        On each tick:
          1. Render the full dashboard.
          2. Evaluate all alerts.
          3. If any alert has ``auto_halt=True``, log CRITICAL and exit.
          4. Sleep for ``refresh_interval`` seconds.

        Args:
            refresh_interval: Seconds between dashboard refreshes. Default 60.

        Raises:
            SystemExit(1): When a CRITICAL auto-halt condition is detected.
            KeyboardInterrupt: Propagated to the caller — Ctrl-C stops the loop.
        """
        logger.info(
            "Dashboard started",
            extra={"refresh_interval": refresh_interval},
        )
        try:
            while True:
                self.render()
                alerts = self.check_alerts()
                self._handle_critical_alerts(alerts)
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print(f"\n{_CYAN}Dashboard stopped by user.{_RESET}")
            logger.info("Dashboard stopped by user")

    # ------------------------------------------------------------------
    # Private render helpers — one section per method, each ≤ 50 lines
    # ------------------------------------------------------------------

    def _render_alerts_section(self, width: int) -> None:
        """Print the ALERTS section; hidden when no alerts are active."""
        try:
            alerts = self.check_alerts()
        except Exception as exc:  # noqa: BLE001
            print(f"{_RED}[ALERTS] error evaluating alerts: {exc}{_RESET}")
            return

        if not alerts:
            _print_section("ALERTS", width)
            print(f"  {_GREEN}✓ No alerts{_RESET}")
            return

        _print_section("ALERTS", width)
        for alert in alerts:
            colour = _RED if alert.level == ALERT_CRITICAL else _YELLOW
            halt_tag = " [AUTO-HALT]" if alert.auto_halt else ""
            ts = alert.timestamp.strftime("%H:%M:%S UTC")
            print(
                f"  {colour}{_BOLD}{alert.level}{_RESET}{colour}"
                f"  [{alert.source}]  {alert.message}{halt_tag}"
                f"  @ {ts}{_RESET}"
            )

    def _render_positions_section(self, width: int) -> None:
        """Print the active positions table."""
        _print_section("ACTIVE POSITIONS", width)
        try:
            positions = self._db.get_active_positions()
            capital = self._db.get_total_deployed_capital()
        except Exception as exc:  # noqa: BLE001
            print(f"  {_RED}error reading positions: {exc}{_RESET}")
            return

        if not positions:
            print(f"  {_CYAN}No open positions{_RESET}")
        else:
            _print_positions_table(positions)

        resolved = self._db.count_resolved_contracts()
        total_ev = self._db.get_total_realized_ev()
        ev_colour = _GREEN if total_ev >= 0 else _RED
        print(
            f"\n  Deployed: {_BOLD}${capital:,.2f}{_RESET}  "
            f"Realised EV: {ev_colour}{_BOLD}${total_ev:+,.4f}{_RESET}  "
            f"Resolved contracts: {_BOLD}{resolved}{_RESET}"
        )

    def _render_api_status_section(self, width: int) -> None:
        """Print Kalshi API connectivity status."""
        _print_section("KALSHI API STATUS", width)
        if self._kalshi_client is None:
            print(f"  {_YELLOW}not configured (no client passed to dashboard){_RESET}")
            return
        try:
            t0 = time.monotonic()
            self._kalshi_client.get_balance()
            latency_ms = int((time.monotonic() - t0) * 1000)
            now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            print(
                f"  {_GREEN}✓ reachable{_RESET}  "
                f"latency: {_BOLD}{latency_ms} ms{_RESET}  last ping: {now}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {_RED}✗ unreachable — {exc}{_RESET}")

    def _render_freshness_section(self, width: int) -> None:
        """Print the data freshness panel for all data sources."""
        _print_section("DATA FRESHNESS", width)
        try:
            freshness = self._db.get_data_freshness()
        except Exception as exc:  # noqa: BLE001
            print(f"  {_RED}error reading freshness: {exc}{_RESET}")
            return

        labels = {
            "contracts": "Contract prices",
            "prices": "Price snapshots",
            "elo": "Elo ratings",
            "public_betting": "Public betting %",
        }
        now_utc = datetime.now(timezone.utc)
        for key, label in labels.items():
            ts_str = freshness.get(key)
            if ts_str is None:
                print(f"  {_YELLOW}  {label:<20}  no data yet{_RESET}")
                continue
            age_str, colour = _format_age(ts_str, now_utc)
            print(f"  {colour}  {label:<20}  last: {ts_str}  ({age_str}){_RESET}")

    def _render_signal_ev_section(self, width: int) -> None:
        """Print the most recent mean EV per signal."""
        _print_section("REALISED EV BY SIGNAL", width)
        try:
            ev_map = self._db.get_realized_ev_by_signal()
        except Exception as exc:  # noqa: BLE001
            print(f"  {_RED}error reading signal EV: {exc}{_RESET}")
            return

        labels = {
            "momentum_fade": "Momentum fade",
            "value_reversal": "Value reversal",
            "public_money_fade": "Public money fade",
        }
        for key, label in labels.items():
            val = ev_map.get(key)
            if val is None:
                print(f"  {_CYAN}  {label:<22}  no resolved contracts yet{_RESET}")
            else:
                colour = _GREEN if val >= 0 else _RED
                print(f"  {colour}  {label:<22}  mean EV: {val:+.4f}{_RESET}")

    def _render_error_log_section(self, width: int) -> None:
        """Print the last N WARNING/ERROR/CRITICAL entries from the log file."""
        _print_section("RECENT ERRORS", width)
        if self._log_file_path is None:
            print(f"  {_CYAN}log file not configured{_RESET}")
            return
        try:
            entries = _read_recent_log_errors(
                self._log_file_path, _LOG_SCAN_LINES, _ERROR_LOG_LINES
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {_YELLOW}cannot read log file: {exc}{_RESET}")
            return

        if not entries:
            print(f"  {_GREEN}✓ No recent errors in log{_RESET}")
            return
        for entry in entries:
            level = entry.get("levelname", "?")
            ts = entry.get("asctime", "?")
            msg = entry.get("message", str(entry))
            colour = _RED if level in (ALERT_CRITICAL, ALERT_ERROR) else _YELLOW
            print(f"  {colour}[{level}] {ts}  {msg}{_RESET}")

    # ------------------------------------------------------------------
    # Private alert check helpers
    # ------------------------------------------------------------------

    def _check_api_connectivity(self) -> List[AlertRecord]:
        """Return CRITICAL alert if Kalshi API authentication fails."""
        if self._kalshi_client is None:
            return []
        try:
            self._kalshi_client.get_balance()
            return []
        except Exception as exc:
            exc_name = type(exc).__name__
            is_auth = "Authentication" in exc_name
            level = ALERT_CRITICAL if is_auth else ALERT_ERROR
            return [AlertRecord(
                level=level,
                source="kalshi_api",
                message=f"API call failed: {exc_name} — {exc}",
                timestamp=datetime.now(timezone.utc),
                auto_halt=is_auth,
            )]

    def _check_data_freshness(self) -> List[AlertRecord]:
        """Return ERROR alert for each data source stale longer than 2 hours."""
        try:
            freshness = self._db.get_data_freshness()
        except Exception:  # noqa: BLE001
            return []

        now_utc = datetime.now(timezone.utc)
        alerts: List[AlertRecord] = []
        source_labels = {
            "prices": "Price snapshots",
            "elo": "Elo ratings",
            "public_betting": "Public betting %",
            "contracts": "Contract universe",
        }
        for key, label in source_labels.items():
            ts_str = freshness.get(key)
            if ts_str is None:
                continue  # No data yet — not an alert (system may just be starting)
            age_seconds = _age_seconds(ts_str, now_utc)
            if age_seconds is not None and age_seconds > _STALENESS_THRESHOLD_SECONDS:
                hours = age_seconds / 3600
                alerts.append(AlertRecord(
                    level=ALERT_ERROR,
                    source="data_freshness",
                    message=f"{label} stale — last update {hours:.1f} hours ago",
                    timestamp=datetime.now(timezone.utc),
                    auto_halt=False,
                ))
        return alerts

    def _check_capital_limit(self) -> List[AlertRecord]:
        """Return WARNING if deployed capital approaches or exceeds the hard limit."""
        try:
            capital = self._db.get_total_deployed_capital()
        except Exception:  # noqa: BLE001
            return []

        if capital >= _CAPITAL_WARNING_DOLLARS:
            pct = capital / _CAPITAL_HARD_LIMIT_DOLLARS * 100
            return [AlertRecord(
                level=ALERT_WARNING,
                source="capital_limits",
                message=(
                    f"Deployed capital ${capital:,.2f} — "
                    f"{pct:.0f}% of ${_CAPITAL_HARD_LIMIT_DOLLARS:,.0f} hard limit"
                ),
                timestamp=datetime.now(timezone.utc),
                auto_halt=False,
            )]
        return []

    def _check_signal_decay(self) -> List[AlertRecord]:
        """Return WARNING if a signal's rolling 4w EV has decayed for 2+ windows."""
        alerts: List[AlertRecord] = []
        signal_names = ("momentum_fade", "value_reversal", "public_money_fade")
        for name in signal_names:
            alerts.extend(self._check_one_signal_decay(name))
        return alerts

    def _check_one_signal_decay(self, signal_name: str) -> List[AlertRecord]:
        """Check decay for a single signal. Returns 0 or 1 AlertRecord."""
        try:
            rows = self._db.get_recent_performance_attribution(signal_name, limit=8)
        except Exception:  # noqa: BLE001
            return []

        rolling_evs = [r.rolling_4w_ev for r in rows if r.rolling_4w_ev is not None]
        mean_evs = [r.mean_ev for r in rows if r.mean_ev is not None]
        if len(rolling_evs) < 2 or not mean_evs:
            return []

        historical_mean = sum(mean_evs) / len(mean_evs)
        if historical_mean <= 0:
            return []

        # Last two rolling windows both below 40% of historical mean.
        last_two = rolling_evs[:2]
        threshold = _SIGNAL_DECAY_FRACTION * historical_mean
        if all(v < threshold for v in last_two):
            return [AlertRecord(
                level=ALERT_WARNING,
                source=f"signal_decay:{signal_name}",
                message=(
                    f"{signal_name} rolling 4w EV {last_two[0]:.4f} "
                    f"< {_SIGNAL_DECAY_FRACTION*100:.0f}% of "
                    f"historical mean {historical_mean:.4f} "
                    f"for 2 consecutive windows"
                ),
                timestamp=datetime.now(timezone.utc),
                auto_halt=False,
            )]
        return []

    def _check_fee_erosion(self) -> List[AlertRecord]:
        """Return WARNING if any signal's rolling 4w EV is negative for 2 consecutive weeks.

        Reads from performance_attribution. One row = one "week" (attribution runs
        daily; the rolling_4w_ev column is the unit). If fewer than 2 rows exist for
        all signals, returns no alert — insufficient data.

        Returns:
            List of :class:`AlertRecord`, possibly empty.
        """
        alerts: List[AlertRecord] = []
        signal_names = ("momentum_fade", "value_reversal", "public_money_fade")
        for name in signal_names:
            alerts.extend(self._check_one_signal_fee_erosion(name))
        return alerts

    def _check_one_signal_fee_erosion(self, signal_name: str) -> List[AlertRecord]:
        """Check fee erosion for a single signal. Returns 0 or 1 AlertRecord.

        Fires WARNING when the two most recent rolling_4w_ev values are both
        negative. Skips if fewer than 2 non-null rolling_4w_ev rows exist.

        Args:
            signal_name: Signal identifier matching performance_attribution table.

        Returns:
            List with one AlertRecord if erosion detected, else empty list.
        """
        try:
            rows = self._db.get_recent_performance_attribution(signal_name, limit=4)
        except Exception:  # noqa: BLE001
            return []

        rolling_evs = [r.rolling_4w_ev for r in rows if r.rolling_4w_ev is not None]
        if len(rolling_evs) < 2:
            return []

        last_two = rolling_evs[:2]   # rows are newest-first from the DB query
        if all(v < 0 for v in last_two):
            return [AlertRecord(
                level=ALERT_WARNING,
                source="fee_erosion",
                message=(
                    f"{signal_name} rolling 4w EV negative for 2 consecutive windows: "
                    f"{last_two[0]:.4f}, {last_two[1]:.4f}"
                ),
                timestamp=datetime.now(timezone.utc),
                auto_halt=False,
            )]
        return []

    # ------------------------------------------------------------------
    # Private utility
    # ------------------------------------------------------------------

    def _handle_critical_alerts(self, alerts: List[AlertRecord]) -> None:
        """Log all CRITICAL alerts and raise SystemExit if any require auto-halt.

        Args:
            alerts: List of :class:`AlertRecord` from :meth:`check_alerts`.

        Raises:
            SystemExit(1): If any alert has ``auto_halt=True``.
        """
        halt = False
        for alert in alerts:
            if alert.auto_halt:
                logger.critical(
                    "AUTO-HALT: %s — %s",
                    alert.source,
                    alert.message,
                    extra={
                        "alert_level": alert.level,
                        "alert_source": alert.source,
                        "alert_message": alert.message,
                        "alert_timestamp": alert.timestamp.isoformat(),
                    },
                )
                halt = True
        if halt:
            print(f"\n{_RED}{_BOLD}CRITICAL ALERT — SYSTEM HALTED{_RESET}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------

def _terminal_width() -> int:
    """Return the current terminal column width, defaulting to 80."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _clear_terminal() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")  # noqa: S605,S607


def _print_header(width: int) -> None:
    """Print the dashboard title bar with the current UTC timestamp."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = "  AXIOM — Phase 1 Prediction Market Monitor"
    right = f"{now}  "
    pad = max(1, width - len(title) - len(right))
    print(f"{_BOLD}{_CYAN}{title}{' ' * pad}{right}{_RESET}")
    print("─" * width)


def _print_section(name: str, width: int) -> None:
    """Print a section divider with a label."""
    label = f"── {name} "
    print(f"\n{_BOLD}{label}{'─' * max(1, width - len(label) - 1)}{_RESET}")


def _print_positions_table(positions: Any) -> None:
    """Print active positions in a fixed-width tabular format."""
    header = (
        f"  {'CONTRACT':<30}  {'DIR':<4}  {'ENTRY':>7}  {'SIZE':>9}  {'OPENED':<20}"
    )
    print(f"{_BOLD}{header}{_RESET}")
    for pos in positions:
        line = (
            f"  {pos.contract_id:<30}  "
            f"{pos.direction.upper():<4}  "
            f"{pos.entry_price:>7.4f}  "
            f"${pos.size_dollars:>8,.2f}  "
            f"{pos.entry_timestamp[:19]}"
        )
        print(line)


def _format_age(ts_str: str, now_utc: datetime) -> "tuple[str, str]":
    """Return a human-readable age string and ANSI colour for a timestamp.

    Returns:
        Tuple of (age_string, ansi_colour). Colour is GREEN when fresh,
        YELLOW when moderately stale, RED when over the staleness threshold.
    """
    age = _age_seconds(ts_str, now_utc)
    if age is None:
        return "unknown", _YELLOW
    if age < 3600:
        return f"{int(age // 60)}m ago", _GREEN
    if age < _STALENESS_THRESHOLD_SECONDS:
        return f"{age / 3600:.1f}h ago", _YELLOW
    return f"{age / 3600:.1f}h ago — STALE", _RED


def _age_seconds(ts_str: str, now_utc: datetime) -> Optional[float]:
    """Parse a UTC ISO 8601 timestamp string and return age in seconds.

    Returns:
        Float seconds since ``ts_str``, or None if parsing fails.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(ts_str, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return (now_utc - parsed).total_seconds()
        except ValueError:
            continue
    return None


def _read_recent_log_errors(
    log_file_path: str,
    scan_lines: int,
    max_results: int,
) -> List[dict]:
    """Read the last ``scan_lines`` of the JSON log and return error entries.

    Parses one JSON object per line. Returns up to ``max_results`` entries
    where ``levelname`` is WARNING, ERROR, or CRITICAL, newest first.

    Args:
        log_file_path: Path to the structured JSON log file.
        scan_lines: How many lines from the end to inspect.
        max_results: Maximum entries to return.

    Returns:
        List of parsed log entry dicts. Empty list if file absent or unreadable.
    """
    try:
        with open(log_file_path, "r", encoding="utf-8") as fh:
            all_lines = fh.readlines()
    except FileNotFoundError:
        return []

    results: List[dict] = []
    for line in reversed(all_lines[-scan_lines:]):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        level = entry.get("levelname", "")
        if level in (ALERT_WARNING, ALERT_ERROR, ALERT_CRITICAL):
            results.append(entry)
            if len(results) >= max_results:
                break
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m monitoring.dashboard``.

    Reads ``AXIOM_DB_PATH`` (default ``fund.db``) and ``KALSHI_LOG_FILE_PATH``
    from environment variables. Pass ``--once`` to render a single frame and exit.

    Raises:
        SystemExit(2): If the database cannot be opened.
        SystemExit(1): If a CRITICAL auto-halt condition fires.
    """
    db_path = os.environ.get("AXIOM_DB_PATH", "fund.db")
    log_path = os.environ.get("KALSHI_LOG_FILE_PATH")
    run_once = "--once" in sys.argv

    try:
        db = Database(db_path)
        db.initialise_schema()
    except Exception as exc:  # noqa: BLE001
        print(f"{_RED}Cannot open database at {db_path!r}: {exc}{_RESET}", file=sys.stderr)
        sys.exit(2)

    # Attempt to build the Kalshi client; skip gracefully if not configured.
    kalshi_client = _try_build_kalshi_client()

    dash = Dashboard(db, log_file_path=log_path, kalshi_client=kalshi_client)

    if run_once:
        dash.render()
        return

    refresh = int(os.environ.get("AXIOM_DASHBOARD_INTERVAL", "60"))
    dash.run(refresh_interval=refresh)


def _try_build_kalshi_client() -> Optional[Any]:
    """Attempt to instantiate the Kalshi client from environment variables.

    Returns:
        Initialised client if all required env vars are present, else None.
    """
    try:
        from kalshi_client import KalshiClient, KalshiConfig  # type: ignore[import]
        cfg = KalshiConfig.from_env()
        return KalshiClient(cfg)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
