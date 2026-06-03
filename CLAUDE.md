# CLAUDE.md — Prediction Market Quant Fund

Read this file completely before writing any code. Every session starts here.

---

## What This System Is

A systematic quantitative trading system for sports prediction market contracts on Kalshi (Phase 1) and later Polymarket (Phase 2). The system ingests contract data, computes rule-based alpha signals derived from behavioral finance research (primarily Moskowitz 2021 on sports betting as an asset pricing laboratory), sizes positions using a simple EV-based rule, executes trades via the Kalshi API, and tracks realized EV per signal for validation.

This is a live trading system that will handle real capital. Code correctness and failure transparency are more important than cleverness or performance optimization.

---

## Current Phase

**Phase 1 — Kalshi Sports Only**

Single venue. Single category. Three rule-based signals. Simple sizing. No optimizer. No AI agents.

Do not build Phase 2 or later components unless explicitly instructed. The explicit exclusion list below is law.

---

## Project Structure

```
/
├── CLAUDE.md                    # This file
├── PROGRESS.md                  # Phase 1 build checklist and status
├── .env.example                 # Required environment variables
├── .env                         # Never committed, never shown in code
├── kalshi_client.py             # Kalshi API client — typed objects only
├── data_ingestion/
│   ├── elo.py                   # Elo rating ingestion and computation
│   ├── schedule.py              # Schedule and rest differential data
│   └── public_betting.py        # Public bet % and dollar % ingestion
├── models/
│   └── baseline.py              # P_baseline computation (Elo + rest + season position)
├── signals/
│   ├── momentum_fade.py         # Signal 1
│   ├── value_reversal.py        # Signal 2
│   └── public_money_fade.py     # Signal 3
├── construction/
│   └── standardize.py           # Winsorization, cross-sectional standardization, logit transform
├── sizing/
│   └── rule.py                  # Simple proportional sizing, position constraints
├── execution/
│   └── executor.py              # Order routing, fill handling, position reconciliation
├── storage/
│   └── db.py                    # SQLite storage layer
├── monitoring/
│   └── dashboard.py             # Real-time system health and realized EV display
├── attribution/
│   └── tracker.py               # Per-signal realized EV, calibration, benchmarks
├── validation/
│   └── gates.py                 # Five-gate signal validation framework
├── config.py                    # Environment variable loading and validation
└── tests/
    ├── test_kalshi_client.py
    ├── test_signals.py
    ├── test_baseline.py
    ├── test_sizing.py
    └── test_execution.py
```

---

## Tech Stack

**Language**: Python 3.11+. Synchronous only in Phase 1. No async, no threading, no multiprocessing unless explicitly instructed.

**HTTP**: `requests` library. Synchronous. Explicit retry logic with exponential backoff. Never use `httpx` or `aiohttp` in Phase 1.

**Data**: `pandas` for dataframes, `numpy` for numerical computation. No `polars` unless explicitly instructed.

**Statistics**: `statsmodels` for Newey-West, `scikit-learn` for RidgeCV. Standard library only for signal math where possible.

**Storage**: SQLite via `sqlite3` standard library in Phase 1. No SQLAlchemy. No Postgres. No Redis. No external database dependencies.

**Testing**: `pytest`. Mock HTTP with `responses` library. No live API calls in tests ever.

**Logging**: Python standard `logging` module only. JSON structured output. No `loguru`, no `structlog`, no third-party logging libraries.

**Config**: `python-dotenv` for loading `.env`. All config in a typed dataclass. No `pydantic` unless explicitly instructed.

**Typing**: Full type hints on every function signature and dataclass field. No `Any` types. Use `Optional[X]` not `X | None` for Python 3.11 compatibility.

**Not in Phase 1**: `asyncio`, `fastapi`, `celery`, `redis`, `sqlalchemy`, `cvxpy`, `scipy.optimize`, `langchain`, `anthropic` SDK, ZeroMQ, websockets, C extensions.

---

## Explicit Exclusion List

These are banned in Phase 1. If a prompt seems to require one of these, ask for clarification rather than building it.

- Polymarket integration of any kind
- Cross-venue arbitrage logic
- AI agents of any kind (no LLM API calls)
- Black-Litterman implementation
- CVXPY or any portfolio optimizer
- Covariance matrix estimation
- Fama-MacBeth regression (gammas are equal-weighted in Phase 1)
- Crypto signals or data sources
- Political, macro, or geopolitical contract categories
- Spread contracts or over/under contracts (game winner / moneyline only)
- WebSocket connections (REST polling only)
- Async code of any kind
- Any signal not in the three defined below
- Caching layers
- Message queues
- Docker configuration
- Any deployment infrastructure

---

## The Three Alpha Signals

These are the only signals in Phase 1. Do not add signals. Do not modify signal definitions without being explicitly asked.

**Signal 1 — Momentum Fade**
Measures cumulative price drift from contract open to now, weighted toward the most recent game result. Positive score means contract is overpriced due to hot hand fallacy — take NO position. Negative score means underpriced — take YES. Season-conditioned: strongest early in season (multiply by 1.4 before game 20% of season), weakest late (multiply by 0.7 after game 75% of season).

**Signal 2 — Long-Term Value Reversal**
Measures teams underperforming relative to Elo over 1-2 seasons. Contrarian — teams that have done badly get underpriced. Season-conditioned: opposite of momentum, strongest late in season (multiply by 1.3 after game 75%), weakest early (multiply by 0.6 before game 25%).

**Signal 3 — Public Money Fade**
Measures retail sentiment distortion. Fires when public bet count percentage exceeds 70% on one side and the line has moved toward rather than away from the public. The signal is the gap between bet count percentage and dollar percentage — sharp money going opposite to public count is the core indicator. Zero the signal if public betting data unavailable for that contract.

---

## Baseline Model

P_baseline is the structural win probability before any signal. It is computed from Elo rating differential plus home court/field advantage plus rest adjustment plus season position modulator.

It is not a signal. It never enters signal computation. It is the prior that signals adjust from.

Home court advantage constants:
- NBA: +65 Elo points
- NFL: +55 Elo points
- MLB: +24 Elo points
- NHL: +30 Elo points

Rest adjustment per day differential:
- NBA: 0.012 Elo points per day
- NFL: 0.008
- MLB: 0.005
- NHL: 0.010

P_baseline must always be between 0.05 and 0.95. Flag and exclude contracts outside this range.

---

## Sizing Rule

Simple. No optimizer. No Black-Litterman. This is the complete sizing logic for Phase 1:

```
if EV > 0.03:
    position_size = min(EV × portfolio_value × 10, 0.10 × portfolio_value)
    direction = YES if P_model > current_yes_price else NO
else:
    position_size = 0
```

Hard constraints that override the formula:
- Maximum 10% of portfolio in any single contract
- Maximum 40% of portfolio in any single sport
- Minimum position $100 or 0.5% of portfolio, whichever is larger — if below this after sizing, set to zero
- Maximum 5% of that contract's total liquidity pool
- No position in contracts resolving within 24 hours
- Only rebalance if new target differs from current by more than 2 percentage points

Scaling constant for EV computation: 0.15. Do not change this without being explicitly instructed.

---

## Core Architectural Rules

These rules apply to every file in the project. Violating them requires explicit instruction.

**Rule 1 — Loud failures over silent failures.**
A function that raises an exception on bad data is always better than a function that returns a plausible-looking wrong answer. Never swallow exceptions silently. Never return a default value when the correct behavior is to fail. If something is wrong, the system should stop and say so clearly.

**Rule 2 — Point-in-time discipline.**
Every piece of data must be stamped with the timestamp at which it was received, not the timestamp at which it was requested. When computing signals for a contract at time T, only data with ingestion_timestamp < T may be used. This rule applies everywhere — in signal computation, in backtesting, in validation. A single lookahead violation makes the entire backtest invalid.

**Rule 3 — Typed objects at every boundary.**
Raw JSON dictionaries never cross a module boundary. The Kalshi API client parses all responses into typed dataclasses before returning. Every function that passes data between modules uses typed objects. Use `@dataclass` with full type annotations.

**Rule 4 — Structured logging on every external call.**
Every call to an external API or data source produces exactly one JSON structured log entry regardless of success or failure. The log entry always includes: timestamp (UTC ISO), endpoint or source name, contract ID if applicable, success boolean, latency in milliseconds, error type and message if failed, retry count. Use Python's standard `logging` module. Log level: DEBUG for success, WARNING for retried calls that eventually succeed, ERROR for terminal failures, CRITICAL for authentication failures and data validation failures.

**Rule 5 — Configuration from environment only.**
No hardcoded values for API keys, URLs, timeouts, thresholds, or file paths. Everything configurable comes from environment variables loaded at startup. The config module validates all required variables are present at startup and raises a clear error if any are missing. Fail at startup, not at runtime.

**Rule 6 — Tests with every module.**
Every new module ships with a corresponding test file. Tests use mocked external dependencies — no live API calls, no network calls, no file system writes outside a temp directory. Every public function has at least one test for the happy path and one test for each distinct failure mode.

**Rule 7 — Resolved contracts are permanent.**
The resolved contracts table in SQLite is append-only. Nothing is ever deleted or updated. Every resolved contract is stored with its full signal scores, position taken if any, and resolution outcome. This is the validation dataset — its integrity is non-negotiable.

---

## Data Integrity Checks

These checks run on every contract object parsed from the Kalshi API. Any failure raises a data validation error (CRITICAL severity), excludes the contract from the universe, and logs the raw response.

- YES price + NO price must equal 1.00 within ±0.005
- YES price and NO price must both be between 0.01 and 0.99 inclusive
- Resolution date must be in the future at ingestion time
- Contract must have a non-null open price (pull from price history on first ingestion)
- Total liquidity must be positive

---

## Error Type Hierarchy

```python
PredictionMarketError          # Base
├── APIError                   # Base for all API errors
│   ├── AuthenticationError    # Never retry. CRITICAL log. Human intervention required.
│   ├── RateLimitError         # Retry with backoff. WARNING log.
│   ├── ServerError            # Retry with backoff. WARNING log.
│   ├── TimeoutError           # Retry with backoff. WARNING log.
│   └── ClientError            # Never retry. ERROR log.
├── DataValidationError        # Response was 200 but payload is wrong. Never retry. CRITICAL log.
├── ExecutionError             # Base for execution failures
│   ├── PartialFillError       # One leg filled, other didn't. Requires immediate close of filled leg.
│   └── PositionMismatchError  # Internal ledger doesn't match Kalshi reported position.
└── ConfigurationError         # Missing or invalid config at startup.
```

---

## Retry Policy

Maximum 3 retries. Exponential backoff with ±20% jitter.

| Attempt | Base delay | With jitter range |
|---|---|---|
| 1 (first retry) | 1 second | 0.8 – 1.2 seconds |
| 2 | 4 seconds | 3.2 – 4.8 seconds |
| 3 | 16 seconds | 12.8 – 19.2 seconds |

After 3 retries: raise terminal error, log at ERROR, do not continue.

Authentication errors and client errors (4xx except 429): do not retry, raise immediately.

---

## SQLite Schema

```sql
CREATE TABLE contracts (
    contract_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_date TEXT NOT NULL,
    resolution_date TEXT NOT NULL,
    open_yes_price REAL,
    resolution_outcome INTEGER,
    ingestion_timestamp TEXT NOT NULL,
    is_resolved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE contract_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    yes_price REAL NOT NULL,
    no_price REAL NOT NULL,
    total_liquidity REAL NOT NULL,
    daily_volume REAL,
    bid_depth_json TEXT,
    ask_depth_json TEXT,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);

CREATE TABLE signal_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id TEXT NOT NULL,
    computation_timestamp TEXT NOT NULL,
    momentum_fade_raw REAL,
    momentum_fade_std REAL,
    value_raw REAL,
    value_std REAL,
    public_fade_raw REAL,
    public_fade_std REAL,
    composite_score REAL,
    p_baseline REAL,
    p_model REAL,
    ev REAL,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);

CREATE TABLE positions (
    position_id TEXT PRIMARY KEY,
    contract_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_timestamp TEXT NOT NULL,
    size_dollars REAL NOT NULL,
    exit_price REAL,
    exit_timestamp TEXT,
    realized_ev REAL,
    slippage REAL,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);

CREATE TABLE elo_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    sport TEXT NOT NULL,
    rating REAL NOT NULL,
    as_of_date TEXT NOT NULL
);

CREATE TABLE public_betting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    bet_pct_home REAL,
    dollar_pct_home REAL,
    sharp_money_indicator REAL,
    source TEXT,
    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
);

CREATE TABLE performance_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    signal_name TEXT NOT NULL,
    resolved_contract_count INTEGER,
    mean_ev REAL,
    newey_west_t REAL,
    rolling_4w_ev REAL,
    rolling_4w_ir REAL
);
```

All timestamps stored as UTC ISO 8601 strings. All monetary values in USD. No foreign currency handling in Phase 1.

---

## Phase 1 Exit Criteria

Do not begin Phase 2 work until all of these are confirmed:

- 50+ resolved contracts accumulated on live capital
- Momentum fade signal: mean EV > 0.03, Newey-West t-stat > 2.0
- Net realized EV positive after all fees
- No unresolved data integrity issues flagged in monitoring
- Written compliance confirmation from Apollo covering Kalshi trading
- Monitoring dashboard has caught at least one real system issue
- Naive Elo benchmark beaten on realized EV

---

## Monitoring Alerts — Priority Order

When any alert fires, it logs at the specified level and displays prominently in the dashboard. The system does not halt automatically on warnings — it halts automatically on critical alerts.

| Alert | Level | Threshold | Auto-halt? |
|---|---|---|---|
| Kalshi API authentication failure | CRITICAL | Any failure | Yes |
| Data validation failure | CRITICAL | Any instance | Yes |
| Position mismatch (ledger vs Kalshi) | CRITICAL | Any mismatch | Yes |
| Partial fill with open leg | CRITICAL | Any instance | Yes |
| Any data source stale | ERROR | > 2 hours | No |
| Signal decay | WARNING | Rolling 4w EV < 40% of historical mean for 2 consecutive windows | No |
| Position drift | WARNING | Any position > 1.5x target weight | No |
| Fee erosion | WARNING | Net realized EV negative for 2 consecutive weeks | No |
| Capital limit approaching | WARNING | Deployed capital > $4,000 (hard limit $5,000) | No |

---

## What Good Code Looks Like in This Project

Every function has a docstring with: what it does, what it takes, what it returns, and what exceptions it can raise.

Every dataclass field has a type annotation and an inline comment if the field name is not self-explanatory.

No function is longer than 50 lines. If it would be longer, split it.

No deeply nested conditionals. Maximum 3 levels of nesting. Flatten with early returns.

No magic numbers. Every threshold, constant, and default value is named and comes from config or a named constant at the top of the module.

Every test is independent. No test depends on state from another test. No shared mutable state between tests.

---

## Things That Will Be Built Later — Do Not Build Now

- Phase 2: Polymarket integration, cross-venue arb book, WebSocket order book
- Phase 2: CVXPY optimizer, Ledoit-Wolf covariance estimation
- Phase 2: Black-Litterman posterior probability computation
- Phase 2: Fama-MacBeth cross-sectional regression with RidgeCV
- Phase 3: Crypto category signals (Deribit options, funding rates, on-chain flows)
- Phase 3: C++ execution core with ZeroMQ bridge
- Phase 3: Intraday rebalancing for injury speed signal
- Phase 4: Political category signals (polling, FEC data)
- Phase 4: AI agent swarm (Resolution Research, News Calibration, Narrative Sentiment)
- Phase 5: Macro category signals (Fed funds futures, economic surprise index)
- Later: Probability Feed API product layer

---

*This file is the source of truth for project architecture and constraints. When in doubt, this file wins over any prompt.*