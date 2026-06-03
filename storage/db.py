"""
SQLite storage layer — Phase 1 prediction market quant fund.

Schema matches CLAUDE.md exactly. All timestamps stored as UTC ISO 8601
strings. All monetary values in USD.

Point-in-time discipline: historical rows are never updated (except the
specific ``mark_contract_resolved`` path). The resolved contracts table is
append-only: rows are never deleted.

Usage::

    from storage.db import Database
    db = Database("/path/to/fund.db")
    db.initialise_schema()
    db.insert_contract(...)
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL — exactly matches CLAUDE.md
# ---------------------------------------------------------------------------

_DDL_CONTRACTS = """
CREATE TABLE IF NOT EXISTS contracts (
    contract_id         TEXT PRIMARY KEY,
    sport               TEXT NOT NULL,
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    game_date           TEXT NOT NULL,
    resolution_date     TEXT NOT NULL,
    open_yes_price      REAL,
    resolution_outcome  INTEGER,
    ingestion_timestamp TEXT NOT NULL,
    is_resolved         INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_CONTRACT_PRICES = """
CREATE TABLE IF NOT EXISTS contract_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id     TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    yes_price       REAL NOT NULL,
    no_price        REAL NOT NULL,
    total_liquidity REAL NOT NULL,
    daily_volume    REAL,
    bid_depth_json  TEXT,
    ask_depth_json  TEXT,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);
"""

_DDL_SIGNAL_SCORES = """
CREATE TABLE IF NOT EXISTS signal_scores (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id           TEXT NOT NULL,
    computation_timestamp TEXT NOT NULL,
    momentum_fade_raw     REAL,
    momentum_fade_std     REAL,
    value_raw             REAL,
    value_std             REAL,
    public_fade_raw       REAL,
    public_fade_std       REAL,
    composite_score       REAL,
    p_baseline            REAL,
    p_model               REAL,
    ev                    REAL,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);
"""

_DDL_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    contract_id     TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    entry_timestamp TEXT NOT NULL,
    size_dollars    REAL NOT NULL,
    exit_price      REAL,
    exit_timestamp  TEXT,
    realized_ev     REAL,
    slippage        REAL,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);
"""

_DDL_ELO_RATINGS = """
CREATE TABLE IF NOT EXISTS elo_ratings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    team       TEXT NOT NULL,
    sport      TEXT NOT NULL,
    rating     REAL NOT NULL,
    as_of_date TEXT NOT NULL
);
"""

_DDL_PUBLIC_BETTING = """
CREATE TABLE IF NOT EXISTS public_betting (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id           TEXT NOT NULL,
    timestamp             TEXT NOT NULL,
    bet_pct_home          REAL,
    dollar_pct_home       REAL,
    sharp_money_indicator REAL,
    source                TEXT,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);
"""

_DDL_PERFORMANCE_ATTRIBUTION = """
CREATE TABLE IF NOT EXISTS performance_attribution (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    date                    TEXT NOT NULL,
    signal_name             TEXT NOT NULL,
    resolved_contract_count INTEGER,
    mean_ev                 REAL,
    newey_west_t            REAL,
    rolling_4w_ev           REAL,
    rolling_4w_ir           REAL
);
"""

_ALL_DDL: List[str] = [
    _DDL_CONTRACTS,
    _DDL_CONTRACT_PRICES,
    _DDL_SIGNAL_SCORES,
    _DDL_POSITIONS,
    _DDL_ELO_RATINGS,
    _DDL_PUBLIC_BETTING,
    _DDL_PERFORMANCE_ATTRIBUTION,
]

# ---------------------------------------------------------------------------
# Named row types returned by read methods
# ---------------------------------------------------------------------------

@dataclass
class PositionRow:
    """Active or closed position as stored in the database."""

    position_id: str
    contract_id: str
    direction: str            # "yes" or "no"
    entry_price: float
    entry_timestamp: str      # UTC ISO 8601
    size_dollars: float
    exit_price: Optional[float]
    exit_timestamp: Optional[str]   # None while still open
    realized_ev: Optional[float]
    slippage: Optional[float]


@dataclass
class SignalScoreRow:
    """Signal score record for one contract at one computation timestamp."""

    contract_id: str
    computation_timestamp: str    # UTC ISO 8601
    momentum_fade_raw: Optional[float]
    momentum_fade_std: Optional[float]
    value_raw: Optional[float]
    value_std: Optional[float]
    public_fade_raw: Optional[float]
    public_fade_std: Optional[float]
    composite_score: Optional[float]
    p_baseline: Optional[float]
    p_model: Optional[float]
    ev: Optional[float]


@dataclass
class PerformanceRow:
    """One row from the performance_attribution table."""

    date: str
    signal_name: str
    resolved_contract_count: Optional[int]
    mean_ev: Optional[float]
    newey_west_t: Optional[float]
    rolling_4w_ev: Optional[float]
    rolling_4w_ir: Optional[float]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """SQLite storage layer for the prediction market quant fund.

    Opens a persistent connection at construction time. Call ``close()``
    when done, or use as a context manager.

    Args:
        db_path: Absolute or relative path to the SQLite file.
                 Created automatically if it does not exist.

    Raises:
        sqlite3.Error: If the database file cannot be opened or created.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL mode improves concurrent read throughput.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        # Enforce foreign key constraints at the SQLite level.
        self._conn.execute("PRAGMA foreign_keys=ON;")
        logger.info("Database opened", extra={"db_path": db_path})

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
        logger.debug("Database connection closed", extra={"db_path": self._db_path})

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def initialise_schema(self) -> None:
        """Create all tables if they do not already exist.

        Safe to call on an existing database — uses ``CREATE TABLE IF NOT EXISTS``.

        Raises:
            sqlite3.Error: On schema creation failure.
        """
        with self._conn:
            for ddl in _ALL_DDL:
                self._conn.execute(ddl)
        logger.info("Schema initialised", extra={"db_path": self._db_path})

    # ------------------------------------------------------------------
    # Write methods — contracts
    # ------------------------------------------------------------------

    def insert_contract(
        self,
        contract_id: str,
        sport: str,
        home_team: str,
        away_team: str,
        game_date: str,
        resolution_date: str,
        ingestion_timestamp: str,
        open_yes_price: Optional[float] = None,
        resolution_outcome: Optional[int] = None,
        is_resolved: int = 0,
    ) -> None:
        """Insert a new contract row. Silently skips if ``contract_id`` exists.

        Point-in-time discipline: first ingestion wins. Use
        ``mark_contract_resolved`` to record resolution, not this method.

        Args:
            contract_id: Kalshi internal ticker (primary key).
            sport: Sport code — NBA, NFL, MLB, NHL, or EPL.
            home_team: Home team name.
            away_team: Away team name.
            game_date: UTC ISO 8601 game date string.
            resolution_date: UTC ISO 8601 resolution date string.
            ingestion_timestamp: UTC ISO 8601 timestamp when this was ingested.
            open_yes_price: YES price at contract listing, or None if unknown.
            resolution_outcome: 1 = YES, 0 = NO, None = unresolved.
            is_resolved: 1 if already resolved at ingestion time, else 0.

        Raises:
            sqlite3.Error: On database write failure.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO contracts
                  (contract_id, sport, home_team, away_team,
                   game_date, resolution_date, open_yes_price,
                   resolution_outcome, ingestion_timestamp, is_resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id, sport, home_team, away_team,
                    game_date, resolution_date, open_yes_price,
                    resolution_outcome, ingestion_timestamp, is_resolved,
                ),
            )
        logger.debug("Contract inserted", extra={"contract_id": contract_id})

    def mark_contract_resolved(
        self,
        contract_id: str,
        resolution_outcome: int,
    ) -> None:
        """Mark an existing contract as resolved with its outcome.

        This is the only permitted in-place update to the contracts table.
        The row itself is preserved permanently — this is the validation dataset.

        Args:
            contract_id: Kalshi internal ticker.
            resolution_outcome: 1 if resolved YES, 0 if resolved NO.

        Raises:
            ValueError: If ``resolution_outcome`` is not 0 or 1.
            sqlite3.Error: On database write failure.
        """
        if resolution_outcome not in (0, 1):
            raise ValueError(
                f"resolution_outcome must be 0 or 1, got {resolution_outcome!r}"
            )
        with self._conn:
            self._conn.execute(
                """
                UPDATE contracts
                SET is_resolved = 1, resolution_outcome = ?
                WHERE contract_id = ?
                """,
                (resolution_outcome, contract_id),
            )
        logger.debug(
            "Contract resolved",
            extra={"contract_id": contract_id, "resolution_outcome": resolution_outcome},
        )

    # ------------------------------------------------------------------
    # Write methods — contract_prices
    # ------------------------------------------------------------------

    def insert_price_snapshot(
        self,
        contract_id: str,
        timestamp: str,
        yes_price: float,
        no_price: float,
        total_liquidity: float,
        daily_volume: Optional[float] = None,
        bid_depth: Optional[List[Dict[str, float]]] = None,
        ask_depth: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Insert one price snapshot for a contract.

        Args:
            contract_id: Kalshi internal ticker.
            timestamp: UTC ISO 8601 snapshot timestamp.
            yes_price: YES price in [0.01, 0.99].
            no_price: NO price in [0.01, 0.99].
            total_liquidity: Total liquidity in USD.
            daily_volume: Daily volume in USD, or None.
            bid_depth: List of ``{"price": float, "size": float}`` dicts, or None.
            ask_depth: List of ``{"price": float, "size": float}`` dicts, or None.

        Raises:
            sqlite3.Error: On database write failure.
        """
        bid_json = json.dumps(bid_depth) if bid_depth is not None else None
        ask_json = json.dumps(ask_depth) if ask_depth is not None else None
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO contract_prices
                  (contract_id, timestamp, yes_price, no_price,
                   total_liquidity, daily_volume, bid_depth_json, ask_depth_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id, timestamp, yes_price, no_price,
                    total_liquidity, daily_volume, bid_json, ask_json,
                ),
            )
        logger.debug(
            "Price snapshot inserted",
            extra={"contract_id": contract_id, "timestamp": timestamp},
        )

    # ------------------------------------------------------------------
    # Write methods — signal_scores
    # ------------------------------------------------------------------

    def insert_signal_scores(
        self,
        contract_id: str,
        computation_timestamp: str,
        momentum_fade_raw: Optional[float] = None,
        momentum_fade_std: Optional[float] = None,
        value_raw: Optional[float] = None,
        value_std: Optional[float] = None,
        public_fade_raw: Optional[float] = None,
        public_fade_std: Optional[float] = None,
        composite_score: Optional[float] = None,
        p_baseline: Optional[float] = None,
        p_model: Optional[float] = None,
        ev: Optional[float] = None,
    ) -> None:
        """Insert one signal score record for a contract.

        All signal fields are optional — pass None for any signal that was
        not computed (e.g. public_fade when betting data is unavailable).

        Args:
            contract_id: Kalshi internal ticker.
            computation_timestamp: UTC ISO 8601 timestamp of this computation.
            momentum_fade_raw: Raw momentum fade signal score, or None.
            momentum_fade_std: Standardised momentum fade score, or None.
            value_raw: Raw value reversal score, or None.
            value_std: Standardised value reversal score, or None.
            public_fade_raw: Raw public money fade score, or None.
            public_fade_std: Standardised public money fade score, or None.
            composite_score: Equal-weighted composite of standardised scores, or None.
            p_baseline: Structural win probability from Elo model, or None.
            p_model: Signal-adjusted win probability, or None.
            ev: Expected value after 2% fee, or None.

        Raises:
            sqlite3.Error: On database write failure.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO signal_scores
                  (contract_id, computation_timestamp,
                   momentum_fade_raw, momentum_fade_std,
                   value_raw, value_std,
                   public_fade_raw, public_fade_std,
                   composite_score, p_baseline, p_model, ev)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id, computation_timestamp,
                    momentum_fade_raw, momentum_fade_std,
                    value_raw, value_std,
                    public_fade_raw, public_fade_std,
                    composite_score, p_baseline, p_model, ev,
                ),
            )
        logger.debug("Signal scores inserted", extra={"contract_id": contract_id})

    # ------------------------------------------------------------------
    # Write methods — positions
    # ------------------------------------------------------------------

    def insert_position(
        self,
        position_id: str,
        contract_id: str,
        direction: str,
        entry_price: float,
        entry_timestamp: str,
        size_dollars: float,
    ) -> None:
        """Record a new open position at the time of execution fill.

        Args:
            position_id: UUID string uniquely identifying this position.
            contract_id: Kalshi internal ticker.
            direction: ``"yes"`` or ``"no"``.
            entry_price: Actual fill price (not signal-time price).
            entry_timestamp: UTC ISO 8601 fill confirmation timestamp.
            size_dollars: Position size in USD.

        Raises:
            ValueError: If ``direction`` is not ``"yes"`` or ``"no"``.
            sqlite3.Error: On database write failure.
        """
        if direction not in ("yes", "no"):
            raise ValueError(f"direction must be 'yes' or 'no', got {direction!r}")
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO positions
                  (position_id, contract_id, direction,
                   entry_price, entry_timestamp, size_dollars)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    position_id, contract_id, direction,
                    entry_price, entry_timestamp, size_dollars,
                ),
            )
        logger.debug(
            "Position opened",
            extra={"position_id": position_id, "contract_id": contract_id},
        )

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_timestamp: str,
        realized_ev: float,
        slippage: float,
    ) -> None:
        """Record exit fill for an open position. The row is never deleted.

        Args:
            position_id: UUID of the position to close.
            exit_price: Actual fill price at exit.
            exit_timestamp: UTC ISO 8601 exit fill timestamp.
            realized_ev: Realised EV in dollars after the 2% Kalshi fee.
            slippage: Actual fill price minus signal-time mid price.

        Raises:
            sqlite3.Error: On database write failure.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE positions
                SET exit_price = ?, exit_timestamp = ?,
                    realized_ev = ?, slippage = ?
                WHERE position_id = ?
                """,
                (exit_price, exit_timestamp, realized_ev, slippage, position_id),
            )
        logger.debug(
            "Position closed",
            extra={"position_id": position_id, "realized_ev": realized_ev},
        )

    # ------------------------------------------------------------------
    # Write methods — elo_ratings
    # ------------------------------------------------------------------

    def insert_elo_rating(
        self,
        team: str,
        sport: str,
        rating: float,
        as_of_date: str,
    ) -> None:
        """Append a point-in-time Elo rating snapshot. Never updates past rows.

        Args:
            team: Team name — must be consistent with contracts.home_team / away_team.
            sport: Sport code — NBA, NFL, MLB, NHL, or EPL.
            rating: Elo rating value (e.g. 1500.0 is league average).
            as_of_date: UTC ISO 8601 date for which this rating applies.

        Raises:
            sqlite3.Error: On database write failure.
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO elo_ratings (team, sport, rating, as_of_date) VALUES (?, ?, ?, ?)",
                (team, sport, rating, as_of_date),
            )
        logger.debug(
            "Elo rating inserted",
            extra={"team": team, "sport": sport, "as_of_date": as_of_date},
        )

    # ------------------------------------------------------------------
    # Write methods — public_betting
    # ------------------------------------------------------------------

    def insert_public_betting(
        self,
        contract_id: str,
        timestamp: str,
        bet_pct_home: Optional[float],
        dollar_pct_home: Optional[float],
        sharp_money_indicator: Optional[float],
        source: Optional[str],
    ) -> None:
        """Append a public betting percentage snapshot. Every call is a new row.

        Args:
            contract_id: Kalshi internal ticker.
            timestamp: UTC ISO 8601 snapshot timestamp.
            bet_pct_home: Fraction of bet count on home team, 0–1, or None.
            dollar_pct_home: Fraction of dollar volume on home team, 0–1, or None.
            sharp_money_indicator: ``dollar_pct_home - bet_pct_home``, or None.
            source: Data provider name (e.g. ``"action_network"``).

        Raises:
            sqlite3.Error: On database write failure.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO public_betting
                  (contract_id, timestamp, bet_pct_home, dollar_pct_home,
                   sharp_money_indicator, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id, timestamp, bet_pct_home,
                    dollar_pct_home, sharp_money_indicator, source,
                ),
            )
        logger.debug("Public betting inserted", extra={"contract_id": contract_id})

    # ------------------------------------------------------------------
    # Write methods — performance_attribution
    # ------------------------------------------------------------------

    def insert_performance_attribution(
        self,
        date: str,
        signal_name: str,
        resolved_contract_count: Optional[int],
        mean_ev: Optional[float],
        newey_west_t: Optional[float],
        rolling_4w_ev: Optional[float],
        rolling_4w_ir: Optional[float],
    ) -> None:
        """Append one daily performance attribution row.

        Args:
            date: UTC ISO 8601 date string (YYYY-MM-DD).
            signal_name: ``"momentum_fade"``, ``"value_reversal"``, or ``"public_money_fade"``.
            resolved_contract_count: Resolved contracts this signal fired on, or None.
            mean_ev: Mean realised EV per resolved contract, or None.
            newey_west_t: Newey-West corrected t-statistic, or None.
            rolling_4w_ev: 4-week rolling mean EV, or None.
            rolling_4w_ir: 4-week rolling information ratio, or None.

        Raises:
            sqlite3.Error: On database write failure.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO performance_attribution
                  (date, signal_name, resolved_contract_count,
                   mean_ev, newey_west_t, rolling_4w_ev, rolling_4w_ir)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date, signal_name, resolved_contract_count,
                    mean_ev, newey_west_t, rolling_4w_ev, rolling_4w_ir,
                ),
            )
        logger.debug(
            "Attribution inserted",
            extra={"date": date, "signal_name": signal_name},
        )

    # ------------------------------------------------------------------
    # Read methods — consumed by the monitoring dashboard
    # ------------------------------------------------------------------

    def get_active_positions(self) -> List[PositionRow]:
        """Return all open positions (no exit timestamp), newest entry first.

        Returns:
            List of :class:`PositionRow`. Empty list if no open positions.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM positions
            WHERE exit_timestamp IS NULL
            ORDER BY entry_timestamp DESC
            """
        )
        return [_row_to_position(r) for r in cursor.fetchall()]

    def get_closed_positions(self) -> List[PositionRow]:
        """Return all closed positions, newest exit first.

        Returns:
            List of :class:`PositionRow` with exit fields populated.
            Empty list if no closed positions.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM positions
            WHERE exit_timestamp IS NOT NULL
            ORDER BY exit_timestamp DESC
            """
        )
        return [_row_to_position(r) for r in cursor.fetchall()]

    def get_total_realized_ev(self) -> float:
        """Return sum of ``realized_ev`` across all closed positions.

        Returns:
            Total realised EV in dollars. 0.0 if no closed positions.
        """
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(realized_ev), 0.0) FROM positions WHERE realized_ev IS NOT NULL"
        )
        return float(cursor.fetchone()[0])

    def get_total_deployed_capital(self) -> float:
        """Return total USD currently deployed in open positions.

        Returns:
            Sum of ``size_dollars`` for all open positions. 0.0 if none.
        """
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(size_dollars), 0.0) FROM positions WHERE exit_timestamp IS NULL"
        )
        return float(cursor.fetchone()[0])

    def get_latest_signal_scores(self, limit: int = 20) -> List[SignalScoreRow]:
        """Return the most recent signal score rows, newest computation first.

        Args:
            limit: Maximum rows to return. Default 20.

        Returns:
            List of :class:`SignalScoreRow`. Empty list if no scores yet.
        """
        cursor = self._conn.execute(
            """
            SELECT contract_id, computation_timestamp,
                   momentum_fade_raw, momentum_fade_std,
                   value_raw, value_std,
                   public_fade_raw, public_fade_std,
                   composite_score, p_baseline, p_model, ev
            FROM signal_scores
            ORDER BY computation_timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [_row_to_signal_score(r) for r in cursor.fetchall()]

    def get_realized_ev_by_signal(self) -> Dict[str, Optional[float]]:
        """Return the most recent ``mean_ev`` per signal from performance_attribution.

        Returns:
            Dict mapping signal name to latest ``mean_ev`` (float or None).
            Keys: ``"momentum_fade"``, ``"value_reversal"``, ``"public_money_fade"``.
        """
        signal_names = ("momentum_fade", "value_reversal", "public_money_fade")
        result: Dict[str, Optional[float]] = {s: None for s in signal_names}
        for name in signal_names:
            cursor = self._conn.execute(
                """
                SELECT mean_ev FROM performance_attribution
                WHERE signal_name = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (name,),
            )
            row = cursor.fetchone()
            if row is not None:
                result[name] = row[0]
        return result

    def get_data_freshness(self) -> Dict[str, Optional[str]]:
        """Return the latest ingestion timestamp for each data source.

        Returns:
            Dict with keys ``"prices"``, ``"elo"``, ``"public_betting"``,
            ``"contracts"``. Values are UTC ISO 8601 strings or None if the
            corresponding table has no rows yet.
        """
        queries: Dict[str, str] = {
            "prices": "SELECT MAX(timestamp) FROM contract_prices",
            "elo": "SELECT MAX(as_of_date) FROM elo_ratings",
            "public_betting": "SELECT MAX(timestamp) FROM public_betting",
            "contracts": "SELECT MAX(ingestion_timestamp) FROM contracts",
        }
        freshness: Dict[str, Optional[str]] = {}
        for key, sql in queries.items():
            cursor = self._conn.execute(sql)
            row = cursor.fetchone()
            freshness[key] = row[0] if (row and row[0]) else None
        return freshness

    def get_recent_performance_attribution(
        self, signal_name: str, limit: int = 8
    ) -> List[PerformanceRow]:
        """Return recent attribution rows for one signal, newest first.

        Args:
            signal_name: ``"momentum_fade"``, ``"value_reversal"``, or
                ``"public_money_fade"``.
            limit: Maximum rows to return. Default 8 (~two 4-week windows).

        Returns:
            List of :class:`PerformanceRow`. Empty list if no data for this signal.
        """
        cursor = self._conn.execute(
            """
            SELECT date, signal_name, resolved_contract_count,
                   mean_ev, newey_west_t, rolling_4w_ev, rolling_4w_ir
            FROM performance_attribution
            WHERE signal_name = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (signal_name, limit),
        )
        return [_row_to_performance(r) for r in cursor.fetchall()]

    def count_resolved_contracts(self) -> int:
        """Return count of resolved contracts (permanent validation dataset).

        Returns:
            Integer count. 0 if no contracts have resolved yet.
        """
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM contracts WHERE is_resolved = 1"
        )
        return int(cursor.fetchone()[0])


# ---------------------------------------------------------------------------
# Private row-to-dataclass helpers
# ---------------------------------------------------------------------------

def _row_to_position(row: sqlite3.Row) -> PositionRow:
    """Convert a sqlite3.Row from the positions table to a PositionRow."""
    return PositionRow(
        position_id=row["position_id"],
        contract_id=row["contract_id"],
        direction=row["direction"],
        entry_price=row["entry_price"],
        entry_timestamp=row["entry_timestamp"],
        size_dollars=row["size_dollars"],
        exit_price=row["exit_price"],
        exit_timestamp=row["exit_timestamp"],
        realized_ev=row["realized_ev"],
        slippage=row["slippage"],
    )


def _row_to_signal_score(row: sqlite3.Row) -> SignalScoreRow:
    """Convert a sqlite3.Row from signal_scores to a SignalScoreRow."""
    return SignalScoreRow(
        contract_id=row["contract_id"],
        computation_timestamp=row["computation_timestamp"],
        momentum_fade_raw=row["momentum_fade_raw"],
        momentum_fade_std=row["momentum_fade_std"],
        value_raw=row["value_raw"],
        value_std=row["value_std"],
        public_fade_raw=row["public_fade_raw"],
        public_fade_std=row["public_fade_std"],
        composite_score=row["composite_score"],
        p_baseline=row["p_baseline"],
        p_model=row["p_model"],
        ev=row["ev"],
    )


def _row_to_performance(row: sqlite3.Row) -> PerformanceRow:
    """Convert a sqlite3.Row from performance_attribution to a PerformanceRow."""
    return PerformanceRow(
        date=row["date"],
        signal_name=row["signal_name"],
        resolved_contract_count=row["resolved_contract_count"],
        mean_ev=row["mean_ev"],
        newey_west_t=row["newey_west_t"],
        rolling_4w_ev=row["rolling_4w_ev"],
        rolling_4w_ir=row["rolling_4w_ir"],
    )
