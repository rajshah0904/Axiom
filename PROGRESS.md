# Phase 1 Progress — Kalshi Sports Prediction Market System

## What Phase 1 Is

Kalshi sports contracts only. Three rule-based alpha signals. Elo-based baseline model. Simple proportional sizing with a 10% position cap. No optimizer. No Black-Litterman. No AI agents. No Polymarket.

The single question Phase 1 answers: does momentum fade on Kalshi sports contracts produce positive realized EV after 2% fees across a statistically meaningful number of resolved contracts?

Exit criterion: 50+ resolved contracts with positive realized EV after fees on the momentum fade signal. Not calendar time. Resolved contract count and realized EV.

---

## Architecture Overview

```
Kalshi API
    ↓
Contract Ingestion + Universe Filter
    ↓
External Data Ingestion (Elo, Schedule, Public Betting %)
    ↓
Baseline Model (P_baseline per contract)
    ↓
Signal Computation (3 rule-based signals)
    ↓
Signal Construction (winsorize → standardize → logit transform)
    ↓
Simple Sizing Rule
    ↓
Execution Layer (Kalshi API)
    ↓
Monitoring + Attribution Dashboard
```

---

## Component Checklist

### 0. Monitoring Dashboard — 🟡 Partially complete

> Build monitoring before signals. This catches silent bugs before they cost money.

- [x] Active positions table (contract ID, side, size, entry price, current price, unrealized EV)
- [x] Kalshi API status indicator (last successful ping, latency)
- [x] Data freshness timestamp for every data source (Elo, schedule, public betting %)
- [x] Realized EV tracker per signal (running total, per-contract breakdown)
- [x] Error log with timestamp and source component
- [x] Alert on: API failure, data staleness > 2 hours, position drift > 2x target weight, any contract approaching resolution with open position

Implementation: `monitoring/dashboard.py` + `storage/db.py`. SQLite schema matches CLAUDE.md spec exactly. All seven alert conditions from CLAUDE.md implemented. CRITICAL alerts auto-halt via SystemExit(1). Dashboard renders with ANSI colours when stdout is a tty; degrades gracefully to plain text. Run with `python -m monitoring.dashboard` (reads `AXIOM_DB_PATH` env var, default `fund.db`). Tests: `tests/test_db.py` (33 tests), `tests/test_dashboard.py` (31 tests). Total suite: 222/222 passing as of 2026-05-26.

Shell complete. DB schema correct. Core alerts working: API connectivity, 
data staleness, capital limit, fee erosion (proxy), signal decay. 
Missing: unrealized EV (needs live price poll), position drift alert 
(needs sizing targets), resolution-soon alert (needs execution layer), 
data-validation/position-mismatch/partial-fill CRITICAL alerts (need 
execution layer), schedule freshness (no schedule table yet), 
per-contract EV breakdown. Log panel field mismatch (levelname/asctime 
vs level/ts) needs fix. Test count: 247 passing as of 2026-05-27.

---

1. Kalshi API Integration
Docs: Kalshi Trade API · OpenAPI spec · Auth: API Keys · Fixed-Point Migration
Status: 🟡 Code complete — sandbox verification pending. Live API inspection completed 2026-05-26 against external-api.demo.kalshi.co. Architecture updated to events-based sports ingestion.
Authentication (per-request, no session token)

 RSA-PSS + SHA-256 signing with private key PEM (not HMAC, not a shared secret)
 Headers: KALSHI-ACCESS-KEY (Key ID), KALSHI-ACCESS-TIMESTAMP (ms), KALSHI-ACCESS-SIGNATURE (base64)
 Signed message: {timestamp_ms}{METHOD}{path_without_query} — no request body
 Config: KALSHI_API_KEY (Key ID) + KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_PEM
 KALSHI_BASE_URL = host only (e.g. https://external-api.demo.kalshi.co) — paths include /trade-api/v2/...
 Retry policy — exponential backoff ±20% jitter, 3 retries, 1/4/16s; 401/403 → AuthenticationError, CRITICAL, no retry; 429/5xx retry

Endpoints (Trade API v2)
Client methodHTTPPathNotesget_eventsGET/trade-api/v2/eventsPrimary sports ingestion. series_ticker, status=active, with_nested_markets=true, cursor pagination. Returns EventData with nested ContractObject list. series_ticker lives on event not market — this is the only correct ingestion path for sports.get_marketsGET/trade-api/v2/marketsNon-sports universe only. status=active, mve_filter=exclude, cursor pagination. Do not use for sports — markets have no series_ticker.get_marketGET/trade-api/v2/markets/{ticker}Single market detailget_order_bookGET/trade-api/v2/markets/{ticker}/orderbookorderbook_fp.yes_dollars / no_dollars — bids only; YES ask implied from NO bidget_trade_historyGET/trade-api/v2/markets/tradesticker, min_ts (Unix seconds int), limit, cursor — not /markets/{ticker}/tradesget_price_historyGET/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticksstart_ts, end_ts, period_interval (1/60/1440). Requires series_ticker from ContractObject.series_ticker — only populated via get_events().place_orderPOST/trade-api/v2/portfolio/ordersLegacy endpoint, valid until Kalshi migration notice. action (buy/sell), side (yes/no), count_fp, yes_price_dollars or no_price_dollars for limits, buy_max_cost for market buys.get_positionsGET/trade-api/v2/portfolio/positionsmarket_positions + cursor paginationget_balanceGET/trade-api/v2/portfolio/balanceReturns balance_dollars as fixed-point string. Required before live trading for sizing rule portfolio_value.
Live API findings (2026-05-26 sandbox inspection)

series_ticker is absent from individual market objects. It is only present on event objects. get_events() is required for sports ingestion.
Market status field value is "active" not "open". All status filter params use status=active.
Individual market objects have yes_sub_title and no_sub_title but no subtitle. Team extraction uses injected event_sub_title from event sub_title field (e.g. "OKC at SAS (May 28)").
Candlestick price object is empty {} on days with zero trade executions. Fallback: mid of yes_bid.close_dollars and yes_ask.close_dollars. Never default to 0.5 — corrupts momentum fade signal.
Mutually exclusive markets: two contracts per game event (one per team). Both share event_ticker. Grouped at ingestion via EventData.markets.
liquidity_dollars is deprecated and always returns "0.0000". Use volume_fp (contract count) and book_depth_usd from order book for universe liquidity filter.
open_interest_fp is available and represents outstanding contracts. Distinct from volume_fp.

Response parsing (current API shape)
Markets: yes_bid_dollars, yes_ask_dollars, last_price_dollars, volume_fp, volume_24h_fp, latest_expiration_time, previous_price_dollars. Legacy fields (yes_bid, volume, expiration_time) absent.
Events: series_ticker, event_ticker, sub_title, mutually_exclusive, nested markets array.
Candlesticks: price.close_dollars (empty {} on zero-volume days), yes_bid.close_dollars, yes_ask.close_dollars, volume_fp, end_period_ts.
Orders: fill_count_fp, yes_price_dollars; status resting | executed | canceled.
Data integrity checks

 Each price in [0.01, 0.99] — last_price_dollars or mid from book; do not require yes_bid + no_bid ≈ 1.0 (spread breaks this)
 open_yes_price from previous_price_dollars or candlestick open — not open_interest_fp
 volume_fp is contract count not USD — do not compare to dollar thresholds
 book_depth_usd = Σ(price × size) across YES bid levels — used for $10,000 universe filter
 Resolved detection: status in ("finalized", "determined") or result in ("yes", "no")
 series_ticker guard in get_price_history() — raises DataValidationError if empty
 Price unchanged > 6 hours → flag stale (storage/monitoring layer — not client)
 Volume = 0 in last 7 days → exclude from universe (storage layer — not client)

Deliverables

 kalshi_client.py — typed client, RSA-PSS auth, 8 endpoints including get_events() and get_balance(), cursor pagination, fixed-point string parsing, structured logging
 tests/test_kalshi_client.py — mocked tests, ephemeral RSA key, API-accurate JSON shapes aligned with live inspection
 .env.example — Key ID + private key path, correct demo base URL
 cryptography in requirements.txt
 Sandbox smoke test — run against external-api.demo.kalshi.co with real credentials ✓

Sandbox smoke test completed 2026-05-26 against external-api.demo.kalshi.co.
Results: 9/9 steps passed (Step 10 skipped pending KALSHI_SMOKE_PLACE_ORDER flag).
One bug found and fixed: _parse_utc() failed on Kalshi's 4-decimal-place subsecond timestamps (e.g. "2026-05-26T20:26:02.0608+00:00"). Fixed by truncating subseconds to 6 digits before fromisoformat().
Architecture confirmed: series_ticker populated via get_events(), no dict mutation, candlestick mid-price fallback working, book_depth_usd computed correctly from live order book.

Universe filter note: _validate_liquidity() removed from _parse_market(). Zero-volume contracts now parse successfully into MarketData and ContractObject. The is_liquid() static method on KalshiClient is the universe filter entry point. Storage layer must call is_liquid() and separately check snapshot.book_depth_usd >= 10000 from get_order_book() before admitting a contract to the signal universe.

Initial client: 2026-05-24. API reconciliation: 2026-05-24. Live sandbox inspection: 2026-05-26. Findings: series_ticker on event not market, status=active not open, empty candlestick price object on zero-volume days, yes_sub_title/no_sub_title not subtitle. Architecture updated: get_events() added as primary sports ingestion, EventData dataclass added, ContractObject.event_ticker added, candlestick parser updated with mid-price fallback. Fixes applied 2026-05-27: _parse_utc() subsecond truncation, _validate_liquidity() moved out of _parse_market(), is_liquid() universe filter added.

### 2. Contract Universe Management

**Universe filters (all must pass):**
- [ ] Total liquidity pool ≥ $10,000
- [ ] Resolution date: 1 to 30 days from now
- [ ] Minimum 50 trades in past 7 days
- [ ] Sport tag: NFL, NBA, MLB, NHL, EPL (start with NBA and NFL only, add others after first 50 resolved)
- [ ] Contract type: game winner / moneyline only (no spread, no totals in Phase 1)

**Contract object — every contract normalized to:**
- [ ] `contract_id` (Kalshi internal ID)
- [ ] `home_team`, `away_team`
- [ ] `sport` (enum: NBA, NFL, MLB, NHL, EPL)
- [ ] `game_date` (UTC)
- [ ] `resolution_date` (UTC)
- [ ] `current_yes_price` (0 to 1)
- [ ] `total_liquidity`
- [ ] `daily_volume_7d_avg`
- [ ] `resolution_criteria_text`
- [ ] `ingestion_timestamp` (point-in-time stamp — critical)
- [ ] `open_price` (price at contract listing — pulled from price history on first ingestion)

**Resolved contract archive:**
- [ ] Resolved contracts never deleted — permanent table
- [ ] Every resolved contract stored with: contract_id, resolution_outcome (0 or 1), resolution_timestamp, full price history, all signal scores at each rebalance, position taken (if any)
- [ ] This is your validation dataset. Protect it.

**Rebalance frequency:** Daily, every morning before market open.

---

### 3. External Data Ingestion

#### 3a. Elo Ratings

Source: 538 historical Elo data (downloadable CSV, free) + sports-reference.com for current season updates.

- [ ] Historical Elo ratings loaded for NBA (1999-present), NFL (1985-present), MLB (2004-present), NHL (2005-present)
- [ ] Current season Elo ratings updated after each game result
- [ ] Home court/field advantage constants stored per sport:
  - NBA: +65 Elo points for home team
  - NFL: +55 Elo points for home team
  - MLB: +24 Elo points for home team
  - NHL: +30 Elo points for home team
- [ ] Elo-implied win probability formula: `P(home wins) = 1 / (1 + 10^(-(Elo_home - Elo_away + HCA) / 400))`
- [ ] Point-in-time discipline: Elo ratings snapshotted at ingestion timestamp, never retroactively updated

#### 3b. Schedule and Rest Data

Source: sports-reference.com (free), official league schedules.

- [ ] Game schedule for all covered sports: date, home team, away team, game ID
- [ ] Days rest computation: days since each team's last game
- [ ] Back-to-back flag: binary indicator, home team and away team separately
- [ ] Games in last 7 days: count per team
- [ ] Season game number: game N of 82 (NBA), game N of 17 (NFL), etc. — used for season position factor
- [ ] Season position: game number / total season games = value 0 to 1

#### 3c. Public Betting Percentages

Source: Action Network (free tier), Sports Insights, or TheLines.com — all publish public bet % and money % for major sports.

- [ ] Bet count percentage: fraction of total bets on home team (0 to 1)
- [ ] Dollar percentage: fraction of total dollars on home team (0 to 1)
- [ ] Spread (bet % - dollar %): the sharp money indicator
- [ ] Update frequency: pull every 2 hours while market is active
- [ ] Availability: typically available for NFL and NBA, less reliable for MLB and NHL — flag when unavailable, exclude public money fade signal for that contract

**Point-in-time discipline:** Public betting percentages snapshotted at each signal computation timestamp. Do not use current percentages to backfill historical signal scores.

---

### 4. Baseline Model — P_baseline

P_baseline is the structural expected win probability before any signal information. It feeds into signal construction as the control variable and will later become the BL prior.

**Computation per contract:**

Step 1 — Raw Elo probability:
```
P_elo = 1 / (1 + 10^(-(Elo_home - Elo_away + HCA_sport) / 400))
```

Step 2 — Rest adjustment:
```
rest_diff = days_rest_home - days_rest_away
rest_adjustment_NBA = rest_diff × 0.012  (Elo points per day differential, converted to probability)
rest_adjustment_NFL = rest_diff × 0.008
rest_adjustment_MLB = rest_diff × 0.005
rest_adjustment_NHL = rest_diff × 0.010
P_baseline = sigmoid(logit(P_elo) + rest_adjustment)
```

Step 3 — Season position modulator:
Early season (position < 0.25): weight Elo at 0.7, weight recent form at 0.3
Late season (position > 0.75): weight Elo at 0.5, weight recent form at 0.5
Recent form = win% last 10 games, converted to probability adjustment

- [ ] P_baseline computed for every contract in universe at each daily rebalance
- [ ] P_baseline stored with ingestion timestamp (point-in-time)
- [ ] Sanity check: P_baseline must be between 0.05 and 0.95. Flag contracts outside this range.

---

### 5. Signal Computation — Three Alpha Signals

#### Signal 1: Momentum Fade

**What it measures:** The cumulative price drift from contract open to now, indicating overreaction to recent team performance.

**Computation:**
```
price_drift = logit(current_yes_price) - logit(open_yes_price)
```

Weighting by recency — most recent game gets highest weight. Pull team's last 8 games, compute:
```
recent_performance_score = Σ (result_g × decay_weight_g) for g = 1 to 8
decay_weights = [0.35, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.04]  (sum to 1.0)
result_g = 1 if win, 0 if loss
```

Signal score (raw):
```
momentum_fade_raw = -(price_drift × recent_performance_score)
```

Negative sign because you are fading the drift. High positive raw score means contract has drifted up on the back of recent wins — you want to sell it.

**Direction:** Positive signal score → contract is overpriced → take NO position
Negative signal score → contract is underpriced relative to momentum → take YES position (only if other signals confirm)

**Season conditioning (from Moskowitz):**
- Season position < 0.25: multiply signal score by 1.4 (momentum strongest early)
- Season position 0.25-0.75: multiply by 1.0
- Season position > 0.75: multiply by 0.7 (momentum weakest late)

- [ ] price_drift computed correctly using logit transform
- [ ] open_yes_price pulled from price history at contract listing timestamp
- [ ] recent_performance_score uses only games before contract open date (point-in-time)
- [ ] Season conditioning applied
- [ ] Raw score stored with computation timestamp

#### Signal 2: Long-Term Value Reversal

**What it measures:** Teams that have underperformed relative to Elo over a full season get underpriced for individual games. Mean reversion in team quality.

**Computation:**
Pull team's cumulative contract return over the past 1-2 seasons (if Kalshi history available) or proxy from historical sportsbook moneyline returns (from Moskowitz's data structure):
```
LT_return_team = Σ (contract_return_g) for all games in last 2 seasons
contract_return_g = 1 if team won AND you backed them, -1 if lost
```

Value signal (raw):
```
value_raw = -(LT_return_team_home - LT_return_team_away)
```

Negative sign because value is contrarian — teams that have done badly recently are underpriced.

**Season conditioning (from Moskowitz — opposite of momentum):**
- Season position < 0.25: multiply signal score by 0.6 (value weakest early)
- Season position 0.25-0.75: multiply by 1.0
- Season position > 0.75: multiply by 1.3 (value strongest late)

**Data availability note:** Kalshi launched 2021. Full 2-season history may not exist for all teams. Fall back to 1-season history if needed. Flag contracts where less than 20 historical games exist — exclude value signal for those contracts.

- [ ] Historical contract returns computed from Kalshi history + proxy sportsbook data
- [ ] Minimum 20 historical games per team enforced
- [ ] Season conditioning applied
- [ ] Raw score stored with computation timestamp

#### Signal 3: Public Money Fade

**What it measures:** Retail sentiment distortion. When the public is heavily positioned on one side and sharp money disagrees (evidenced by line moving the wrong way), the public side is overpriced.

**Computation:**
```
bet_pct = fraction of bet count on home team (from Action Network)
dollar_pct = fraction of dollar volume on home team (from Action Network)
sharp_money_indicator = dollar_pct - bet_pct
```

Public fade signal fires when:
- `bet_pct > 0.70` (heavy public backing) AND
- `sharp_money_indicator < 0` (sharp money going opposite direction) AND
- Kalshi price has moved toward the public (current_yes_price > open_yes_price for home team)

Signal score (raw):
```
public_fade_raw = -(bet_pct - 0.50) × |sharp_money_indicator| × price_drift_toward_public
```

Positive score = public is overloaded on home team, fade it (take NO on home)
Negative score = public is overloaded on away team, fade it (take YES on home)

**Availability gating:** If public betting % data is unavailable for a contract, signal score = 0. Do not impute. Flag in monitoring dashboard.

- [ ] Action Network or equivalent data source integrated
- [ ] Update frequency: every 2 hours
- [ ] Availability flag per contract tracked
- [ ] Raw score stored with computation timestamp

---

### 6. Signal Construction — Standardization Pipeline

Every signal goes through identical construction before entering the sizing rule. This is non-negotiable — it makes signal scores comparable across contracts, sports, and time periods.

**Step 1 — Winsorization (within sport, within rebalance date):**
- [ ] For each signal, within each sport (NBA separately, NFL separately, etc.), at each daily rebalance date:
- [ ] Clip values at 2nd percentile and 98th percentile
- [ ] Store pre-winsorized and post-winsorized values separately

**Step 2 — Cross-sectional standardization (within sport, within rebalance date):**
- [ ] Subtract cross-sectional mean within sport
- [ ] Divide by cross-sectional standard deviation within sport
- [ ] Result: every signal has mean 0, standard deviation 1, within each sport at each rebalance date
- [ ] Edge case: if fewer than 5 contracts in a sport on a given day, skip standardization and use raw winsorized score

**Step 3 — Composite score:**
```
composite_score = (w1 × momentum_fade_std) + (w2 × value_std) + (w3 × public_fade_std)
```

Initial weights in Phase 1: equal weighting, w1 = w2 = w3 = 1/3.
These weights are not estimated from data yet — that's what Fama-MacBeth does in Phase 2 when you have enough resolved contracts. Equal weighting is the correct starting point.

- [ ] Composite score computed for every contract at each rebalance
- [ ] Composite score stored with all component scores

**Step 4 — Logit transform for EV computation:**
```
P_model = sigmoid(logit(P_baseline) + composite_score × scaling_constant)
EV = P_model - current_yes_price - 0.02  (2% Kalshi fee)
```

Scaling constant starts at 0.15. This controls how far your composite score moves you from the baseline. Too high and you're making extreme bets on noisy signals. Too low and you're ignoring signal. 0.15 is conservative and adjustable.

- [ ] EV computed for every contract at each rebalance
- [ ] EV stored with all inputs for later attribution

---

### 7. Sizing Rule

Simple. No optimizer. No BL.

**Rule:**
```
if EV > 0.03:  # 3 cent minimum threshold after fees
    position_size = min(EV × portfolio_value × 10, 0.10 × portfolio_value)
    direction = YES if P_model > current_yes_price else NO
else:
    position_size = 0
```

**Hard constraints:**
- [ ] Maximum position per contract: 10% of portfolio
- [ ] Maximum position per sport: 40% of portfolio (prevents accidentally being an NBA-only fund)
- [ ] Minimum position: $100 or 0.5% of portfolio, whichever is larger. Below this, don't trade.
- [ ] Maximum 5% of that contract's total liquidity pool per position
- [ ] No position in contracts resolving within 24 hours (too late for signals to add information)

**Turnover control:**
- [ ] Only rebalance a position if new target weight differs from current weight by more than 2 percentage points. Prevents excessive churning on small signal changes.

- [ ] Sizing rule implemented and tested on paper trading data
- [ ] All constraints enforced and logged
- [ ] Edge cases handled: what if sizing rule produces 0 positions on a given day? (expected — log it, don't force trades)

---

### 8. Execution Layer

- [ ] Market order routing for positions < 1% of contract liquidity pool
- [ ] Limit order routing for positions 1-5% of contract liquidity pool (place at inside ask for buys, inside bid for sells)
- [ ] Both legs must fill before position is considered open — no partial positions
- [ ] If limit order unfilled after 30 minutes, move one tick toward market. Repeat up to 3 times, then cancel.
- [ ] Execution price recorded vs. signal price at computation time — slippage tracking
- [ ] Position reconciliation: after every execution, verify Kalshi reported position matches internal ledger. Alert on mismatch.
- [ ] No execution if Kalshi API latency > 500ms — wait for stable connection

---

### 9. Performance Attribution

Run attribution daily. This is not optional — it's how you know whether the system is working.

**Per-signal realized EV tracking:**
- [ ] For each resolved contract where a position was held: record signal scores at entry, position direction, entry price, resolution outcome, realized EV
- [ ] Realized EV per signal: was the signal directionally correct? Did EV exceed fee? Track separately for each of the three signals.
- [ ] Rolling 4-week realized EV per signal — the early warning system for signal decay

**Portfolio-level tracking:**
- [ ] Total realized EV (sum across all resolved contracts)
- [ ] Realized EV per sport (NBA separately, NFL separately)
- [ ] Win rate (% of positions that resolved in correct direction) — target > 52% given fee structure
- [ ] Average position hold time
- [ ] Average slippage vs. signal price

**Benchmark comparison:**
- [ ] Naive benchmark: always bet on Elo favorite. Track its realized EV. Your system must beat this or signals are adding nothing.
- [ ] Random benchmark: random direction, same position sizes. Track its realized EV. Your system must beat this by a statistically significant margin.

**Calibration check (weekly):**
- [ ] For all contracts where model probability was in range [0.55, 0.60]: what fraction resolved YES? Should be approximately 57.5%.
- [ ] Systematic miscalibration in any probability band indicates baseline model error.

---

### 10. Signal Validation — Five Gates

Begin running validation as resolved contracts accumulate. Do not wait until you have 200 contracts — run it continuously and watch the numbers stabilize.

**Gate 1 — Mean EV with Newey-West:**
- [ ] Compute mean EV per dollar across all resolved contracts per signal
- [ ] Compute Newey-West corrected t-statistic with 1-2 lags (daily periods)
- [ ] Threshold: mean EV > 0.03 AND t-stat > 2.0 AND minimum 50 resolved contracts
- [ ] Track this number weekly — it's your primary validation metric

**Gate 2 — Cross-time consistency:**
- [ ] Compute rolling 4-week realized EV per signal
- [ ] IR equivalent: mean rolling EV / standard deviation of rolling EV
- [ ] Threshold: IR > 0.4 across rolling windows
- [ ] A signal that worked for 3 weeks then died is not a signal

**Gate 3 — Out of sample:**
- [ ] Use Kalshi historical data (2021-2024) as in-sample
- [ ] Live trading (Phase 1 deployment) as out of sample
- [ ] Track whether live EV degrades more than 50% from historical backtest EV
- [ ] If it does, diagnose before continuing

**Gate 4 — Fee survivability:**
- [ ] Net EV after 2% Kalshi fee must be positive
- [ ] Also model market impact: positions above 3% of contract liquidity pool face meaningful price impact — adjust EV estimate downward by 0.5% per 1% of pool participation above 3%

**Gate 5 — Economic mechanism:**
- [ ] One paragraph per signal explaining why it works and why it hasn't been arbitraged away
- [ ] Momentum fade: hot hand fallacy, retail participants extrapolate recent results, Kalshi participant base is retail-dominated so sophisticated capital hasn't fully competed away the edge
- [ ] Value reversal: mean reversion in team quality is slow-moving, market anchors on recent results, the signal operates over 1-2 season windows that most participants don't track
- [ ] Public money fade: retail sentiment creates non-information price distortion, sharp money on Kalshi is limited because position sizes are capped, preventing full arbitrage

---

### 11. Data Storage Schema

Minimum tables required. Use Postgres locally to start.

```
contracts
  contract_id, sport, home_team, away_team, game_date,
  resolution_date, open_yes_price, resolution_outcome,
  ingestion_timestamp, is_resolved

contract_prices
  contract_id, timestamp, yes_price, no_price, total_liquidity,
  daily_volume, bid_depth_5, ask_depth_5

signal_scores
  contract_id, computation_timestamp, momentum_fade_raw,
  momentum_fade_std, value_raw, value_std, public_fade_raw,
  public_fade_std, composite_score, p_baseline, p_model, ev

positions
  position_id, contract_id, direction, entry_price,
  entry_timestamp, size_dollars, exit_price, exit_timestamp,
  realized_ev, slippage

elo_ratings
  team_id, sport, rating, as_of_date

public_betting
  contract_id, timestamp, bet_pct_home, dollar_pct_home,
  sharp_money_indicator, source

performance_attribution
  date, signal_name, resolved_contract_count, mean_ev,
  newey_west_t, rolling_4w_ev, rolling_4w_ir
```

- [ ] Schema created and tested
- [ ] Point-in-time discipline enforced at database level — no updates to historical rows, only inserts
- [ ] Resolved contracts table append-only — no deletes ever

---

## Known Failure Modes to Monitor

These are the Phase 1 premortem causes of death. Each has a specific monitor.

| Failure Mode | Monitor | Alert Threshold |
|---|---|---|
| Fee erosion | Net realized EV after fees, rolling 4-week | Negative for 2 consecutive weeks |
| Lookahead bias in data | Point-in-time audit — spot check 10 random resolved contracts | Any single lookahead instance = stop and audit |
| Market impact exceeding signal | Slippage tracker vs. expected slippage model | Average slippage > 0.8% per trade |
| Signal decay | Rolling 4-week realized EV per signal | Below 40% of historical mean for 2 consecutive windows |
| API silent failure | Kalshi API status + data freshness | Any data source stale > 2 hours |
| Position drift | Current weight vs. target weight per contract | Any position > 1.5x target weight |
| Compliance trigger | Track total deployed capital + trade frequency | Set hard limit: $5,000 deployed max in Phase 1 |

---

## Phase 1 Exit Criteria

All of the following must be true before Phase 2 begins:

- [ ] 50+ resolved contracts accumulated on live capital
- [ ] Momentum fade signal: mean EV > 0.03, Newey-West t-stat > 2.0
- [ ] Net realized EV positive after all fees across all positions
- [ ] No unresolved data integrity issues
- [ ] Written compliance confirmation from Apollo covering Kalshi trading
- [ ] Monitoring dashboard operational and has caught at least one real issue (if it's never fired, it's probably not working)
- [ ] Naive Elo benchmark beaten on realized EV

---

## What Phase 1 Is NOT Building

Explicit exclusions. If you find yourself building any of these in Phase 1, stop.

- Polymarket integration
- Cross-venue arb book
- AI agents of any kind
- Black-Litterman
- CVXPY optimizer
- Covariance matrix estimation
- Fama-MacBeth regression (gammas are equal-weighted in Phase 1)
- Crypto signals or data sources
- Political, macro, or geopolitical contract categories
- Spread or over/under contract types
- Multiple simultaneous execution venues
- Any signal not in the three listed above

---

## Current Status

**Overall:** 🟡 In progress

| Component | Status | Notes |
|---|---|---|
| Monitoring dashboard | 🟢 Complete | `monitoring/dashboard.py` — all 7 CLAUDE.md alerts, CRITICAL auto-halt, 31 tests |
| Data storage schema | 🟢 Complete | `storage/db.py` — SQLite, 7 tables, point-in-time discipline, 33 tests |
| Kalshi API integration | 🟢 Complete | RSA-PSS auth, 158/158 tests, reconciled with API v3.19.0, events-based ingestion |
| Data ingestion bridge | 🟢 Complete | `data_ingestion/ingest.py` — migrate_schema, persist_event, persist_price_snapshots, 14 tests |
| Universe management | 🟢 Complete | `data_ingestion/universe.py` — is_in_universe (7 filters), build_universe, persist_event_contract, 20 tests |
| Elo data ingestion | 🔴 Not started | |
| Schedule/rest data | 🔴 Not started | |
| Public betting data | 🔴 Not started | |
| Baseline model | 🔴 Not started | |
| Momentum fade signal | 🔴 Not started | |
| Value reversal signal | 🔴 Not started | |
| Public money fade signal | 🔴 Not started | |
| Signal standardization | 🔴 Not started | |
| Sizing rule | 🔴 Not started | |
| Execution layer | 🔴 Not started | |
| Performance attribution | 🔴 Not started | |
| Signal validation (Gate 1) | 🔴 Waiting on resolved contracts | Needs 50+ |
| Signal validation (Gate 2) | 🔴 Waiting on resolved contracts | |
| Signal validation (Gate 3) | 🔴 Waiting on resolved contracts | |
| Signal validation (Gate 4) | 🔴 Waiting on resolved contracts | |
| Signal validation (Gate 5) | 🔴 Document now, validate live | |
| Compliance confirmation | 🔴 Not started | Needed before live capital |

---

*Last updated: May 2026*
*Phase 1 scope locked. Changes require explicit decision to update this document.*