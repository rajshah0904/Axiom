"""
Kalshi API Client — Phase 1 Sports Prediction Market Trading System

Synchronous REST client with RSA-PSS auth (SHA-256 / MGF1-SHA256), typed dataclasses
at every boundary, structured JSON logging on every API call, exponential backoff retry,
and data integrity checks on every parsed contract. No async, no caching, no websockets.

Required environment variables:
    KALSHI_API_KEY          Key ID from https://kalshi.com/account/api-keys
    KALSHI_PRIVATE_KEY_PEM  RSA private key (inline PEM); or use KALSHI_PRIVATE_KEY_PATH
    KALSHI_PRIVATE_KEY_PATH Path to RSA private key .pem file (alt to KALSHI_PRIVATE_KEY_PEM)
    KALSHI_BASE_URL         Host only, e.g. https://external-api.demo.kalshi.co
    KALSHI_LOG_FILE_PATH    Path for rotating log file
    KALSHI_LOG_LEVEL        DEBUG | INFO | WARNING | ERROR  (default DEBUG)
    KALSHI_TIMEOUT_SECONDS  Float seconds per request       (default 10)
    KALSHI_MAX_RETRIES      Integer retries on transient err (default 3)

Usage:
    from kalshi_client import KalshiClient, KalshiConfig
    cfg = KalshiConfig.from_env()
    client = KalshiClient(cfg)
    resp = client.get_markets(sport_filter="NBAWIN")
    contracts = resp.payload  # List[ContractObject]
"""

from __future__ import annotations

import base64
import json
import logging
import logging.handlers
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

KALSHI_API_VERSION = "v2"
_DEFAULT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_RETRIES = 3
_RETRY_BASE_DELAYS = [1.0, 4.0, 16.0]   # seconds; len must equal _DEFAULT_MAX_RETRIES
_RETRY_JITTER_FACTOR = 0.20              # ±20% applied to each base delay
_PRICE_MIN = 0.01
_PRICE_MAX = 0.99
_ORDER_BOOK_DEPTH = 5                    # top N bid/ask levels to capture
_DEFAULT_CANDLE_INTERVAL = 1440          # 1-day candles for price history

# Documented Kalshi REST API hosts (Trade API environments page).
# KalshiClient.__init__() warns when KALSHI_BASE_URL is not in this set.
_DOCUMENTED_KALSHI_HOSTS: frozenset = frozenset({
    "https://external-api.demo.kalshi.co",   # demo / paper trading
    "https://external-api.kalshi.com",       # production
})


# ---------------------------------------------------------------------------
# Error hierarchy  (matches CLAUDE.md spec exactly)
# ---------------------------------------------------------------------------

class PredictionMarketError(Exception):
    """Base exception for all system errors."""


class APIError(PredictionMarketError):
    """Base class for all Kalshi API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 response_body: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(APIError):
    """Auth failure — never retry, log CRITICAL, require human intervention."""


class RateLimitError(APIError):
    """HTTP 429 — retry with backoff; carry Retry-After header value."""

    def __init__(self, message: str, retry_after: Optional[int] = None,
                 **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServerError(APIError):
    """HTTP 5xx — retry with backoff."""


class TimeoutError(APIError):   # noqa: A001 — intentional shadow of built-in
    """Request timed out — retry with backoff."""

    def __init__(self, message: str, configured_timeout: float,
                 **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.configured_timeout = configured_timeout


class ClientError(APIError):
    """4xx except 401/403/429 — never retry, log ERROR."""


class DataValidationError(PredictionMarketError):
    """HTTP 200 but payload failed schema/integrity checks — never retry, log CRITICAL."""

    def __init__(self, message: str, raw_response: Optional[str] = None,
                 contract_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.contract_id = contract_id


class ExecutionError(PredictionMarketError):
    """Base for trade execution failures."""


class PartialFillError(ExecutionError):
    """Order partially filled (resting with fill_count < initial_count). Close filled leg."""

    def __init__(self, message: str, filled_order_id: str,
                 filled_count: float, contract_id: str) -> None:
        super().__init__(message)
        self.filled_order_id = filled_order_id
        self.filled_count = filled_count
        self.contract_id = contract_id


class PositionMismatchError(ExecutionError):
    """Internal ledger does not match Kalshi-reported position."""

    def __init__(self, message: str, contract_id: str,
                 internal_size: float, reported_size: float) -> None:
        super().__init__(message)
        self.contract_id = contract_id
        self.internal_size = internal_size
        self.reported_size = reported_size


class ConfigurationError(PredictionMarketError):
    """Missing or invalid configuration at startup — fail loud and early."""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_dollars(s: str) -> float:
    """
    Parse a Kalshi fixed-point dollar string (e.g. '0.5500') to float 0–1.
    Uses Decimal to avoid float parsing edge cases.

    Raises:
        InvalidOperation: if s is not a valid decimal string.
    """
    return float(Decimal(s))


def _parse_count(s: str) -> float:
    """
    Parse a Kalshi fixed-point contract count string (e.g. '10.00') to float.

    Raises:
        InvalidOperation: if s is not a valid decimal string.
    """
    return float(Decimal(s))


def _strip_query(path: str) -> str:
    """Strip query string from path before signing — Kalshi signs path only."""
    return path.split("?")[0]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KalshiConfig:
    """
    All runtime configuration loaded from environment variables.
    Instantiate via KalshiConfig.from_env() — never pass secrets as arguments.

    Fields:
        api_key:         Key ID from Kalshi (KALSHI_API_KEY).
        private_key_pem: RSA private key in PEM format as a string.
        base_url:        Host only, no trailing slash, no /trade-api/v2 suffix.
        timeout_seconds: Per-request HTTP timeout.
        max_retries:     Max retry attempts on transient failures.
        log_file_path:   Destination for the rotating JSON log file.
        log_level:       DEBUG | INFO | WARNING | ERROR | CRITICAL.

    Raises:
        ConfigurationError: if any required variable is absent or invalid.
    """

    api_key: str             # KALSHI_API_KEY — Key ID sent in KALSHI-ACCESS-KEY header
    private_key_pem: str     # RSA private key PEM content (from env or file)
    base_url: str            # KALSHI_BASE_URL — host only
    timeout_seconds: float
    max_retries: int
    log_file_path: str
    log_level: str

    @classmethod
    def from_env(cls) -> "KalshiConfig":
        """
        Load and validate all configuration from environment variables.

        Supports inline PEM via KALSHI_PRIVATE_KEY_PEM, or a file path via
        KALSHI_PRIVATE_KEY_PATH. Exactly one must be set.

        Returns:
            Fully populated KalshiConfig.

        Raises:
            ConfigurationError: on missing/invalid variables or unreadable key file.
        """
        import os

        _required_always = {
            "KALSHI_API_KEY": "RSA key ID from Kalshi account settings",
            "KALSHI_BASE_URL": "API host, e.g. https://external-api.demo.kalshi.co",
            "KALSHI_LOG_FILE_PATH": "Path for rotating log file",
        }
        missing = [k for k in _required_always if not os.environ.get(k)]
        if missing:
            details = "; ".join(f"{k} ({_required_always[k]})" for k in missing)
            raise ConfigurationError(
                f"Missing required environment variables: {details}. "
                "Set all variables before starting the system."
            )

        pem = _load_pem_from_env(os.environ)

        timeout = _parse_positive_float(
            os.environ.get("KALSHI_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)),
            "KALSHI_TIMEOUT_SECONDS",
        )
        max_retries = _parse_positive_int(
            os.environ.get("KALSHI_MAX_RETRIES", str(_DEFAULT_MAX_RETRIES)),
            "KALSHI_MAX_RETRIES",
        )
        log_level = os.environ.get("KALSHI_LOG_LEVEL", "DEBUG").upper()
        _validate_log_level(log_level)

        base_url = os.environ["KALSHI_BASE_URL"].rstrip("/")

        return cls(
            api_key=os.environ["KALSHI_API_KEY"],
            private_key_pem=pem,
            base_url=base_url,
            timeout_seconds=timeout,
            max_retries=max_retries,
            log_file_path=os.environ["KALSHI_LOG_FILE_PATH"],
            log_level=log_level,
        )


def _load_pem_from_env(env: Dict[str, str]) -> str:
    """
    Load RSA private key PEM from KALSHI_PRIVATE_KEY_PEM (inline) or
    KALSHI_PRIVATE_KEY_PATH (file path). At least one must be set.

    Raises:
        ConfigurationError: if neither is set or the file is unreadable.
    """
    import os

    inline = env.get("KALSHI_PRIVATE_KEY_PEM", "")
    path = env.get("KALSHI_PRIVATE_KEY_PATH", "")
    if not inline and not path:
        raise ConfigurationError(
            "Set KALSHI_PRIVATE_KEY_PEM (inline PEM) or "
            "KALSHI_PRIVATE_KEY_PATH (path to .pem file) before starting."
        )
    if inline:
        return inline
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read KALSHI_PRIVATE_KEY_PATH={path!r}: {exc}"
        ) from exc


def _parse_positive_float(value: str, name: str) -> float:
    """Parse and validate a positive float config value."""
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got: {value!r}") from exc


def _parse_positive_int(value: str, name: str) -> int:
    """Parse and validate a non-negative integer config value."""
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got: {value!r}") from exc


def _validate_log_level(level: str) -> None:
    """Raise ConfigurationError if level is not a valid Python log level name."""
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError(
            f"KALSHI_LOG_LEVEL must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL, "
            f"got: {level!r}"
        )


# ---------------------------------------------------------------------------
# Typed response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BidAskLevel:
    """Single price level in an order book (YES-centric view)."""

    price: float    # YES contract price, 0–1
    size: float     # contract count available at this level


@dataclass
class ContractObject:
    """
    Fully parsed Kalshi sports contract. Raw JSON never crosses module boundaries.

    yes_price is derived from last_price_dollars (last execution price), which is the
    best single number for signal computation. no_price = 1 - yes_price by binary
    contract definition. Bid/ask prices from the order book differ due to spread.
    """

    contract_id: str                        # Kalshi ticker, e.g. NBAWIN-LAK-GSW-20260101
    series_ticker: str                      # Parent series, e.g. NBAWIN — needed for candlesticks
    event_ticker: str                       # Parent event, e.g. KXNBAGAME-26MAY28OKCSAS — groups mutually exclusive markets
    sport: str                              # NBA | NFL | MLB | NHL | EPL | UNKNOWN
    home_team: str
    away_team: str
    game_date: datetime                     # UTC (close_time from API)
    resolution_date: datetime               # UTC (latest_expiration_time from API)
    yes_price: float                        # last_price_dollars, 0–1
    no_price: float                         # 1 - yes_price
    volume_fp: float                        # volume_fp as float (contract count; NOT USD)
    volume_24h: float                       # volume_24h_fp as float (contracts, NOT USD)
    resolution_criteria_text: str
    ingestion_timestamp: datetime           # UTC moment this object was parsed
    open_yes_price: Optional[float]         # price at contract listing; None until pulled
    is_resolved: bool
    resolution_outcome: Optional[int]       # 1 = YES resolved, 0 = NO resolved


@dataclass
class MarketData:
    """
    Parsed representation of a single Kalshi market object as returned by the API.

    Contains ONLY fields that the Kalshi API actually returns on individual market
    objects. No injected fields (series_ticker, event_sub_title) and no computed
    fields (sport, home_team, away_team). Those are populated by _build_contract()
    using explicit event-level arguments, not by mutating this object.

    Use _parse_market() to construct. Use _build_contract() to upgrade to ContractObject.
    """

    ticker: str                              # Kalshi market ticker
    event_ticker: str                        # event_ticker from the market dict (may be empty)
    yes_price: float                         # derived from last_price_dollars / bid / ask
    no_price: float                          # 1 - yes_price
    volume_fp: float                         # contract count (NOT USD)
    volume_24h: float                        # 24-hour contract volume (NOT USD)
    resolution_date: datetime                # UTC (latest_expiration_time)
    game_date: datetime                      # UTC (close_time)
    status: str                              # lifecycle: active | finalized | determined
    result: str                              # "yes" | "no" | "" (empty until resolved)
    resolution_criteria_text: str            # rules_primary
    yes_sub_title: str                       # describes the YES outcome team
    no_sub_title: str                        # describes the NO outcome team
    title: str                               # market-level title, e.g. "San Antonio at Golden State Winner?"
    previous_price_dollars: Optional[float]  # last closing price; None if absent
    open_interest_fp: float                  # outstanding contracts
    ingestion_timestamp: datetime            # UTC moment this object was parsed


@dataclass
class EventData:
    """
    Parsed Kalshi event object wrapping the nested sports markets.

    An event corresponds to one game (e.g. "Game 6: OKC at SAS"). It contains
    two mutually exclusive ContractObject markets — one for each team — both sharing
    the same event_ticker. series_ticker lives at the event level only; individual
    market objects do not carry it.

    Use get_events() to obtain these objects. Do not use get_markets() for sports
    — it cannot populate series_ticker and therefore cannot call get_price_history().
    """

    event_ticker: str                       # e.g. KXNBAGAME-26MAY28OKCSAS
    series_ticker: str                      # e.g. KXNBAGAME — parent series
    title: str                              # e.g. "Game 6: Oklahoma City at San Antonio"
    sub_title: str                          # e.g. "OKC at SAS (May 28)" — matchup description
    mutually_exclusive: bool                # True: exactly one market resolves YES
    markets: List[ContractObject]           # parsed contracts nested under this event


@dataclass
class PriceSnapshot:
    """
    Point-in-time order book snapshot. Bids sorted best-first (highest price first).
    Asks are implied YES asks derived from NO bids (ask_price = 1 - no_bid_price),
    sorted lowest first.
    """

    contract_id: str
    timestamp: datetime                     # UTC, moment snapshot was received
    yes_price: float                        # best YES bid, 0–1
    no_price: float                         # best NO bid, 0–1
    volume_fp: float                        # volume_fp for the market (contract count; not USD)
    daily_volume: Optional[float]
    bids: List[BidAskLevel]                 # YES bids, best first
    asks: List[BidAskLevel]                 # implied YES asks from NO bids, cheapest first
    book_depth_usd: float                   # Σ(price × size) across YES bid levels in dollars


@dataclass
class Trade:
    """A single completed trade on a Kalshi contract."""

    contract_id: str
    timestamp: datetime                     # UTC
    direction: str                          # "YES" or "NO" (taker_side)
    size_contracts: float                   # count_fp — number of contracts, NOT dollars
    price: float                            # yes_price_dollars execution price, 0–1
    is_aggressive: bool                     # True = taker (hit bid / lifted ask)


@dataclass
class Position:
    """
    Current open position snapshot from get_positions reconciliation call.
    NOT an accounting record — entry price and P&L come from position_fp and
    market_exposure_dollars in the API response. Use this for ledger reconciliation.
    """

    contract_id: str                        # market_ticker
    direction: str                          # "YES" if position_fp > 0, "NO" if < 0
    size_contracts: float                   # abs(position_fp)
    market_exposure: float                  # market_exposure_dollars — dollar notional
    realized_pnl: float                     # realized_pnl_dollars
    last_updated_ts: datetime               # UTC, last_updated_ts from API


@dataclass
class OrderConfirmation:
    """Result of a placed order."""

    order_id: str
    contract_id: str
    direction: str                          # "YES" or "NO"
    action: str                             # "buy" or "sell"
    requested_count: float                  # initial_count_fp
    filled_count: float                     # fill_count_fp
    remaining_count: float                  # remaining_count_fp
    filled_price: Optional[float]           # yes_price_dollars, None if unfilled
    status: str                             # "resting" | "canceled" | "executed"
    created_timestamp: datetime             # UTC


T = TypeVar("T")


@dataclass
class APIResponse(Generic[T]):
    """Wrapper around every API response. Payload is always a typed object."""

    payload: T
    http_status_code: int
    latency_ms: float
    endpoint: str
    success: bool


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_level: str, log_file_path: str) -> logging.Logger:
    """
    Configure the kalshi_client logger for JSON output to stdout + rotating file.

    Rotates daily, retains 7 days of history.

    Returns:
        Configured logger.

    Raises:
        OSError: if log_file_path is not writable.
    """

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            payload: Dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            _SKIP = {
                "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "message",
            }
            for key, val in record.__dict__.items():
                if key not in _SKIP:
                    payload[key] = val
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    logger = logging.getLogger("kalshi_client")
    logger.setLevel(getattr(logging, log_level))
    logger.propagate = False
    logger.handlers.clear()

    fmt = _JsonFormatter()
    stdout_h = logging.StreamHandler()
    stdout_h.setFormatter(fmt)
    logger.addHandler(stdout_h)

    file_h = logging.handlers.TimedRotatingFileHandler(
        log_file_path, when="D", interval=1, backupCount=7,
        encoding="utf-8", utc=True,
    )
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)
    return logger


def _log_api_call(
    logger: logging.Logger,
    *,
    level: int,
    endpoint: str,
    contract_id: Optional[str],
    http_status: Optional[int],
    latency_ms: float,
    success: bool,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    retry_count: int = 0,
) -> None:
    """Emit exactly one structured log entry for an API call."""
    logger.log(
        level,
        f"kalshi_api_call endpoint={endpoint} success={success}",
        extra={
            "endpoint": endpoint,
            "contract_id": contract_id,
            "http_status": http_status,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "error_type": error_type,
            "error_message": error_message,
            "retry_count": retry_count,
        },
    )


# ---------------------------------------------------------------------------
# KalshiClient
# ---------------------------------------------------------------------------

class KalshiClient:
    """
    Synchronous Kalshi REST API client (API v2, RSA-PSS auth).

    All public methods return APIResponse[T]. Raw JSON never leaves this class.

    Endpoints:
        get_events                  GET  /trade-api/v2/events                                         (paginated) — primary sports ingestion
        get_markets                 GET  /trade-api/v2/markets                                        (paginated) — non-sports only
        get_market                  GET  /trade-api/v2/markets/{ticker}
        get_order_book              GET  /trade-api/v2/markets/{ticker}/orderbook
        get_trade_history           GET  /trade-api/v2/markets/trades                                 (paginated)
        get_price_history           GET  /trade-api/v2/series/{s}/markets/{t}/candlesticks
        place_order                 POST /trade-api/v2/portfolio/orders
        get_positions               GET  /trade-api/v2/portfolio/positions                            (paginated)
        get_balance                 GET  /trade-api/v2/portfolio/balance
        get_historical_cutoff       GET  /trade-api/v2/historical/cutoff
        get_historical_markets      GET  /trade-api/v2/historical/markets                             (paginated)
        get_historical_candlesticks GET  /trade-api/v2/historical/markets/{ticker}/candlesticks

    Raises at construction:
        ConfigurationError: if config is invalid.
        ValueError: if private_key_pem is not a valid RSA key.
        OSError: if log_file_path is not writable.
    """

    def __init__(self, config: KalshiConfig) -> None:
        self._cfg = config
        self._logger = _setup_logging(config.log_level, config.log_file_path)
        self._private_key: RSAPrivateKey = _load_rsa_key(config.private_key_pem)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if config.base_url not in _DOCUMENTED_KALSHI_HOSTS:
            self._logger.warning(
                "KALSHI_BASE_URL does not match documented Kalshi API hosts",
                extra={
                    "actual_url": config.base_url,
                    "documented_hosts": sorted(_DOCUMENTED_KALSHI_HOSTS),
                },
            )
        self._logger.info(
            "KalshiClient initialised",
            extra={"base_url": config.base_url, "log_level": config.log_level},
        )

    # ------------------------------------------------------------------
    # RSA-PSS authentication
    # ------------------------------------------------------------------

    def _utc_timestamp_ms(self) -> str:
        """Return current UTC time as milliseconds-since-epoch string."""
        return str(int(time.time() * 1000))

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """
        RSA-PSS sign a Kalshi request.

        Message = {timestamp_ms}{METHOD}{path_without_query}
        Algorithm: PSS / SHA-256 / MGF1-SHA256 / salt_length = digest_size (32 bytes)
        Returns Base64-encoded signature (not hex).

        Args:
            timestamp_ms: milliseconds-since-epoch string.
            method: HTTP verb, uppercased.
            path: URL path; query string stripped before signing.

        Returns:
            Base64-encoded signature string.
        """
        path_no_query = _strip_query(path)
        message = f"{timestamp_ms}{method.upper()}{path_no_query}".encode("utf-8")
        raw_sig = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(raw_sig).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Build the three Kalshi auth headers for a single request.
        Message does NOT include request body per Kalshi API spec.

        Returns:
            Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE.
        """
        ts = self._utc_timestamp_ms()
        return {
            "KALSHI-ACCESS-KEY": self._cfg.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
        }

    # ------------------------------------------------------------------
    # Low-level HTTP with retry
    # ------------------------------------------------------------------

    def _jittered_delay(self, attempt_index: int) -> float:
        """Return seconds to sleep before retry attempt_index (0-indexed)."""
        base = _RETRY_BASE_DELAYS[attempt_index]
        jitter = base * _RETRY_JITTER_FACTOR
        return base + random.uniform(-jitter, jitter)

    def _classify_error(self, response: requests.Response) -> APIError:
        """Map HTTP error response to the correct typed exception. Never raises."""
        code = response.status_code
        body = response.text
        if code in (401, 403):
            return AuthenticationError(
                f"Authentication failed ({code}): {body[:200]}",
                status_code=code, response_body=body,
            )
        if code == 429:
            retry_after: Optional[int] = None
            raw_ra = response.headers.get("Retry-After")
            if raw_ra and raw_ra.isdigit():
                retry_after = int(raw_ra)
            return RateLimitError(
                f"Rate limited (429). Retry-After={retry_after}s",
                retry_after=retry_after, status_code=code, response_body=body,
            )
        if code >= 500:
            return ServerError(
                f"Server error ({code}): {body[:200]}",
                status_code=code, response_body=body,
            )
        return ClientError(
            f"Client error ({code}): {body[:400]}",
            status_code=code, response_body=body,
        )

    def _raw_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[requests.Response, float]:
        """
        Execute one HTTP request, measure latency, return (response, latency_ms).
        Does not retry. Raises TimeoutError on requests.Timeout.
        Auth signature is computed here; body is NOT included in the signed message.
        """
        url = f"{self._cfg.base_url}{path}"
        body_str = json.dumps(body) if body else None
        # Sign path only (no body, no query string per Kalshi spec)
        headers = self._auth_headers(method, path)

        t_start = time.monotonic()
        try:
            resp = self._session.request(
                method=method, url=url, headers=headers,
                params=params, data=body_str,
                timeout=self._cfg.timeout_seconds,
            )
        except requests.Timeout as exc:
            latency_ms = (time.monotonic() - t_start) * 1000
            raise TimeoutError(
                f"Request to {path} timed out after {self._cfg.timeout_seconds}s",
                configured_timeout=self._cfg.timeout_seconds,
            ) from exc
        return resp, (time.monotonic() - t_start) * 1000

    def _handle_timeout_attempt(
        self, attempt: int, exc: TimeoutError,
        endpoint_label: str, contract_id: Optional[str],
    ) -> None:
        """Log timeout attempt. Caller decides whether to retry or raise."""
        is_final = attempt >= self._cfg.max_retries
        level = logging.ERROR if is_final else logging.WARNING
        _log_api_call(
            self._logger, level=level, endpoint=endpoint_label,
            contract_id=contract_id,
            http_status=None, latency_ms=self._cfg.timeout_seconds * 1000,
            success=False, error_type="TimeoutError",
            error_message=str(exc), retry_count=attempt,
        )

    def _handle_error_status(
        self, attempt: int, resp: requests.Response, latency_ms: float,
        endpoint_label: str, contract_id: Optional[str],
    ) -> Tuple[str, APIError]:
        """
        Classify HTTP error and return (action, exc).
        action is one of: "raise_auth", "raise_client", "retry".
        """
        exc = self._classify_error(resp)
        code = resp.status_code
        if code in (401, 403):
            _log_api_call(
                self._logger, level=logging.CRITICAL, endpoint=endpoint_label,
                contract_id=contract_id, http_status=code, latency_ms=latency_ms,
                success=False, error_type="AuthenticationError",
                error_message=str(exc), retry_count=attempt,
            )
            return "raise_auth", exc
        if 400 <= code < 500 and code not in (429,):
            _log_api_call(
                self._logger, level=logging.ERROR, endpoint=endpoint_label,
                contract_id=contract_id, http_status=code, latency_ms=latency_ms,
                success=False, error_type=type(exc).__name__,
                error_message=str(exc), retry_count=attempt,
            )
            return "raise_client", exc
        # 429 / 5xx — both retry with backoff
        is_final = attempt >= self._cfg.max_retries
        level = logging.ERROR if is_final else logging.WARNING
        _log_api_call(
            self._logger, level=level, endpoint=endpoint_label,
            contract_id=contract_id, http_status=code, latency_ms=latency_ms,
            success=False, error_type=type(exc).__name__,
            error_message=str(exc), retry_count=attempt,
        )
        return "retry", exc

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        contract_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, int]:
        """
        Execute request with retry, logging, and error classification.

        Returns:
            (parsed_json, latency_ms, http_status_code) on success.

        Raises:
            AuthenticationError | ClientError: never retried.
            DataValidationError: non-JSON 200 response.
            RateLimitError | ServerError | TimeoutError: after max retries.
        """
        last_exc: Optional[APIError] = None
        endpoint_label = f"{method} {path}"

        for attempt in range(self._cfg.max_retries + 1):
            try:
                resp, latency_ms = self._raw_request(method, path, params, body)
            except TimeoutError as exc:
                self._handle_timeout_attempt(attempt, exc, endpoint_label, contract_id)
                last_exc = exc
                if attempt < self._cfg.max_retries:
                    time.sleep(self._jittered_delay(attempt))
                    continue
                raise

            if resp.status_code >= 400:
                action, exc = self._handle_error_status(
                    attempt, resp, latency_ms, endpoint_label, contract_id
                )
                if action in ("raise_auth", "raise_client"):
                    raise exc
                # "retry"
                last_exc = exc
                if attempt < self._cfg.max_retries:
                    delay = self._jittered_delay(attempt)
                    if isinstance(exc, RateLimitError) and exc.retry_after:
                        delay = max(delay, float(exc.retry_after))
                    time.sleep(delay)
                    continue
                raise exc

            # 2xx — parse JSON
            try:
                parsed = resp.json()
            except ValueError as exc:
                raw_body = resp.text
                _log_api_call(
                    self._logger, level=logging.CRITICAL, endpoint=endpoint_label,
                    contract_id=contract_id, http_status=resp.status_code,
                    latency_ms=latency_ms, success=False,
                    error_type="DataValidationError",
                    error_message=f"Non-JSON response: {raw_body[:200]}",
                    retry_count=attempt,
                )
                raise DataValidationError(
                    f"Response body is not valid JSON from {path}",
                    raw_response=raw_body, contract_id=contract_id,
                ) from exc

            level = logging.WARNING if attempt > 0 else logging.DEBUG
            _log_api_call(
                self._logger, level=level, endpoint=endpoint_label,
                contract_id=contract_id, http_status=resp.status_code,
                latency_ms=latency_ms, success=True, retry_count=attempt,
            )
            return parsed, latency_ms, resp.status_code

        assert last_exc is not None
        raise last_exc

    def _request_paginated(
        self,
        path: str,
        results_key: str,
        params: Optional[Dict[str, Any]] = None,
        contract_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        """
        Paginate through all pages using cursor until cursor is empty or absent.

        Args:
            path: API path.
            results_key: JSON key containing the list (e.g. "markets", "trades").
            params: base query parameters; cursor is added per page.
            contract_id: forwarded for logging.

        Returns:
            (all_items, total_latency_ms, final_http_status).
        """
        all_items: List[Dict[str, Any]] = []
        total_latency = 0.0
        cursor: Optional[str] = None
        status_code = 200

        while True:
            page_params = dict(params or {})
            if cursor:
                page_params["cursor"] = cursor
            raw, latency_ms, status_code = self._request(
                "GET", path, params=page_params, contract_id=contract_id
            )
            total_latency += latency_ms
            all_items.extend(raw.get(results_key, []))
            cursor = raw.get("cursor") or None
            if not cursor:
                break

        return all_items, total_latency, status_code

    # ------------------------------------------------------------------
    # Data integrity checks
    # ------------------------------------------------------------------

    def _validate_individual_prices(
        self, yes_price: float, no_price: float,
        contract_id: str, raw: str,
    ) -> None:
        """
        Validate each price is in [_PRICE_MIN, _PRICE_MAX].

        NOTE: We do NOT require YES + NO = 1.0 here. Bid/ask spread means
        bid prices don't sum to 1.0. For ContractObject we derive no_price
        as 1 - last_price_dollars, which does sum to 1.0 by construction.

        Raises:
            DataValidationError: if either price is out of range.
        """
        for label, price in (("YES", yes_price), ("NO", no_price)):
            if not (_PRICE_MIN <= price <= _PRICE_MAX):
                raise DataValidationError(
                    f"Contract {contract_id}: {label} price {price} "
                    f"out of range [{_PRICE_MIN}, {_PRICE_MAX}]",
                    raw_response=raw, contract_id=contract_id,
                )

    def _validate_resolution_date(
        self, resolution_date: datetime, contract_id: str,
        is_resolved: bool, ingestion_ts: datetime, _raw: str,
    ) -> None:
        """Warn (not raise) on past resolution date for unresolved contracts."""
        if not is_resolved and resolution_date <= ingestion_ts:
            self._logger.warning(
                "Contract has past resolution date but is not marked resolved",
                extra={
                    "contract_id": contract_id,
                    "resolution_date": resolution_date.isoformat(),
                    "ingestion_ts": ingestion_ts.isoformat(),
                },
            )

    def _validate_liquidity(
        self, volume_fp: float, contract_id: str, raw: str,
    ) -> None:
        """Raise DataValidationError if liquidity is non-positive.

        Called only from _parse_contract() (the non-sports path via get_markets()
        and get_market()). Not called from _parse_market() — zero-volume sports
        contracts must parse successfully and are filtered by is_liquid() instead.
        """
        if volume_fp <= 0:
            raise DataValidationError(
                f"Contract {contract_id}: non-positive liquidity {volume_fp}",
                raw_response=raw, contract_id=contract_id,
            )

    @staticmethod
    def is_liquid(
        contract: "ContractObject",
        min_book_depth_usd: float = 10_000.0,
        min_volume_24h: float = 8.0,
    ) -> bool:
        """Universe filter entry point: check whether a contract meets minimum liquidity.

        Returns True when ``contract.volume_24h >= min_volume_24h``.

        The ``min_book_depth_usd`` parameter documents the second required check
        but cannot be evaluated here because book depth requires a live order-book
        snapshot from ``get_order_book()``.  The storage layer must separately verify::

            snapshot = client.get_order_book(contract.contract_id).payload
            snapshot.book_depth_usd >= min_book_depth_usd

        before admitting a contract to the signal universe.

        Args:
            contract: ContractObject from ``get_events()`` or ``get_markets()``.
            min_book_depth_usd: Minimum USD book depth required. Default 10,000.
                Must be verified separately from a live ``get_order_book()`` call.
            min_volume_24h: Minimum 24-hour volume in contracts. Default 8.

        Returns:
            True if ``contract.volume_24h >= min_volume_24h``, False otherwise.
        """
        return contract.volume_24h >= min_volume_24h

    # ------------------------------------------------------------------
    # Parsers — raw dict → typed dataclass
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_utc(ts_str: str) -> datetime:
        """
        Parse ISO 8601 timestamp string to UTC-aware datetime.

        Kalshi API returns variable subsecond precision — sometimes 4 digits
        (e.g. '2026-05-26T20:26:02.0608+00:00'), which fromisoformat() rejects
        on Python < 3.11. Truncate subseconds to at most 6 digits before parsing.

        Handles: no subseconds, 1–6 digit subseconds, 7+ digit subseconds,
        Z-suffix, and naive datetime strings (timezone assumed UTC).

        Args:
            ts_str: ISO 8601 timestamp string from the Kalshi API.

        Raises:
            ValueError: If unparseable after subsecond truncation.
        """
        ts_str = ts_str.replace("Z", "+00:00")
        # Normalise subsecond component to exactly 6 digits.
        # Python < 3.11 fromisoformat() only accepts 0 or 6 subsecond digits.
        # Pad short values (1-5 digits) with trailing zeros; truncate 7+ digits.
        ts_str = re.sub(
            r"\.(\d+)",
            lambda m: "." + (m.group(1) + "000000")[:6],
            ts_str,
        )
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _extract_yes_price(self, raw: Dict[str, Any], contract_id: str) -> float:
        """
        Extract best available YES price from market dict.
        Preference: last_price_dollars → yes_bid_dollars → yes_ask_dollars.

        Raises:
            DataValidationError: if no valid price found.
        """
        for field in ("last_price_dollars", "yes_bid_dollars", "yes_ask_dollars"):
            val = raw.get(field, "")
            if not val:
                continue
            try:
                p = _parse_dollars(str(val))
                if _PRICE_MIN <= p <= _PRICE_MAX:
                    return p
            except (InvalidOperation, ValueError):
                continue
        raise DataValidationError(
            f"Contract {contract_id}: no valid YES price in "
            f"last_price_dollars / yes_bid_dollars / yes_ask_dollars",
            raw_response=json.dumps(raw), contract_id=contract_id,
        )

    def _parse_contract(self, raw_dict: Dict[str, Any]) -> ContractObject:
        """
        Parse a single market dict into a ContractObject.
        Ingestion timestamp recorded at parse time (point-in-time discipline).

        Raises:
            DataValidationError: on integrity failure.
            KeyError | ValueError: on missing or malformed required fields.
        """
        ingestion_ts = datetime.now(timezone.utc)   # record NOW, not request time
        raw_str = json.dumps(raw_dict)
        contract_id = str(raw_dict.get("ticker", raw_dict.get("id", "")))

        yes_price = self._extract_yes_price(raw_dict, contract_id)
        no_price = round(1.0 - yes_price, 6)
        self._validate_individual_prices(yes_price, no_price, contract_id, raw_str)

        volume_raw = raw_dict.get("volume_fp", raw_dict.get("volume", "0"))
        total_liquidity = _safe_parse_count(str(volume_raw))
        self._validate_liquidity(total_liquidity, contract_id, raw_str)

        resolution_date = self._parse_utc(
            raw_dict.get("latest_expiration_time", raw_dict.get("expiration_time", ""))
        )
        # Market status enum: open | finalized | determined (not a "finalized" boolean)
        status_raw = str(raw_dict.get("status", "")).lower()
        result_raw = raw_dict.get("result", "")
        is_resolved = (
            status_raw in ("finalized", "determined")
            or result_raw in ("yes", "no", "scalar")
        )
        self._validate_resolution_date(resolution_date, contract_id, is_resolved, ingestion_ts, raw_str)

        resolution_outcome: Optional[int] = None
        if result_raw == "yes":
            resolution_outcome = 1
        elif result_raw == "no":
            resolution_outcome = 0

        series_ticker = str(raw_dict.get("series_ticker", ""))
        event_ticker = str(raw_dict.get("event_ticker", ""))
        # event_sub_title is injected by get_events() from the event-level sub_title field
        # (e.g. "OKC at SAS (May 28)"). It contains the matchup — yes_sub_title /
        # no_sub_title on individual markets both describe only one team's outcome.
        subtitle = (
            raw_dict.get("event_sub_title")         # injected from event level
            or raw_dict.get("subtitle")             # present on non-sports markets
            or raw_dict.get("title", "")            # final fallback
        )
        sport, home_team, away_team = _extract_sport_teams(subtitle, raw_dict)

        game_date = self._parse_utc(
            raw_dict.get("close_time", raw_dict.get("latest_expiration_time", ""))
        )

        vol_24h = raw_dict.get("volume_24h_fp", raw_dict.get("volume_24h", "0"))
        daily_vol = _safe_parse_count(str(vol_24h))

        open_raw = raw_dict.get("previous_price_dollars") or raw_dict.get("open_price_dollars")
        open_yes_price: Optional[float] = None
        if open_raw:
            try:
                open_yes_price = _parse_dollars(str(open_raw))
            except (InvalidOperation, ValueError):
                pass

        return ContractObject(
            contract_id=contract_id,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            sport=sport,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            resolution_date=resolution_date,
            yes_price=yes_price,
            no_price=no_price,
            volume_fp=total_liquidity,
            volume_24h=daily_vol,
            resolution_criteria_text=raw_dict.get("rules_primary", ""),
            ingestion_timestamp=ingestion_ts,
            open_yes_price=open_yes_price,
            is_resolved=is_resolved,
            resolution_outcome=resolution_outcome,
        )

    def _parse_market(self, raw: Dict[str, Any]) -> MarketData:
        """
        Parse a raw Kalshi market dict into a MarketData object.

        Performs the same price extraction and integrity validation as _parse_contract()
        but does NOT accept or use any injected fields. series_ticker, event_ticker,
        and event_sub_title are absent from individual market API objects and must not
        be written into raw dicts before calling this method.

        Records ingestion_timestamp at parse time for point-in-time discipline.

        Args:
            raw: market dict directly from the Kalshi API response.

        Returns:
            MarketData with all API-returned fields parsed to typed values.

        Raises:
            DataValidationError: on price or liquidity integrity failure.
            KeyError | ValueError: on missing or malformed required fields.
        """
        ingestion_ts = datetime.now(timezone.utc)
        raw_str = json.dumps(raw)
        ticker = str(raw.get("ticker", raw.get("id", "")))

        yes_price = self._extract_yes_price(raw, ticker)
        no_price = round(1.0 - yes_price, 6)
        self._validate_individual_prices(yes_price, no_price, ticker, raw_str)

        volume_raw = raw.get("volume_fp", raw.get("volume", "0"))
        vol_fp = _safe_parse_count(str(volume_raw))
        # Liquidity validation intentionally omitted here: zero-volume contracts
        # must parse successfully so the universe filter (is_liquid()) can decide
        # what to admit.  _validate_liquidity() is still called in _parse_contract()
        # for the non-sports path.

        resolution_date = self._parse_utc(
            raw.get("latest_expiration_time", raw.get("expiration_time", ""))
        )
        game_date = self._parse_utc(
            raw.get("close_time", raw.get("latest_expiration_time", ""))
        )

        status_raw = str(raw.get("status", "")).lower()
        result_raw = str(raw.get("result", ""))
        is_resolved = (
            status_raw in ("finalized", "determined")
            or result_raw in ("yes", "no", "scalar")
        )
        self._validate_resolution_date(
            resolution_date, ticker, is_resolved, ingestion_ts, raw_str
        )

        vol_24h_raw = raw.get("volume_24h_fp", raw.get("volume_24h", "0"))
        vol_24h = _safe_parse_count(str(vol_24h_raw))

        prev_raw = raw.get("previous_price_dollars")
        previous_price: Optional[float] = None
        if prev_raw:
            try:
                previous_price = _parse_dollars(str(prev_raw))
            except (InvalidOperation, ValueError):
                pass

        open_interest = _safe_parse_count(str(raw.get("open_interest_fp", "0")))

        return MarketData(
            ticker=ticker,
            event_ticker=str(raw.get("event_ticker", "")),
            yes_price=yes_price,
            no_price=no_price,
            volume_fp=vol_fp,
            volume_24h=vol_24h,
            resolution_date=resolution_date,
            game_date=game_date,
            status=status_raw,
            result=result_raw,
            resolution_criteria_text=str(raw.get("rules_primary", "")),
            yes_sub_title=str(raw.get("yes_sub_title", "")),
            no_sub_title=str(raw.get("no_sub_title", "")),
            title=str(raw.get("title", "")),
            previous_price_dollars=previous_price,
            open_interest_fp=open_interest,
            ingestion_timestamp=ingestion_ts,
        )

    def _build_contract(
        self,
        market: MarketData,
        series_ticker: str,
        event_ticker: str,
        event_sub_title: str,
    ) -> ContractObject:
        """
        Construct a ContractObject from a parsed MarketData plus event-level fields.

        series_ticker, event_ticker, and event_sub_title are not present on individual
        market API objects — they live on the parent event. They must be passed
        explicitly from the event dict; they are never read from the market dict.

        Args:
            market: parsed market data (output of _parse_market).
            series_ticker: parent series, e.g. "KXNBAGAME" — needed for candlesticks.
            event_ticker: parent event, e.g. "KXNBAGAME-26MAY28OKCSAS".
            event_sub_title: matchup description, e.g. "OKC at SAS (May 28)" —
                             used for sport and team extraction.

        Returns:
            ContractObject with all fields populated.
        """
        sport, home_team, away_team = _extract_sport_teams(
            event_sub_title,
            {
                "series_ticker": series_ticker,
                "yes_sub_title": market.yes_sub_title,
                "no_sub_title": market.no_sub_title,
                "title": market.title,
            },
        )

        is_resolved = (
            market.status in ("finalized", "determined")
            or market.result in ("yes", "no", "scalar")
        )
        resolution_outcome: Optional[int] = None
        if market.result == "yes":
            resolution_outcome = 1
        elif market.result == "no":
            resolution_outcome = 0

        return ContractObject(
            contract_id=market.ticker,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            sport=sport,
            home_team=home_team,
            away_team=away_team,
            game_date=market.game_date,
            resolution_date=market.resolution_date,
            yes_price=market.yes_price,
            no_price=market.no_price,
            volume_fp=market.volume_fp,
            volume_24h=market.volume_24h,
            resolution_criteria_text=market.resolution_criteria_text,
            ingestion_timestamp=market.ingestion_timestamp,
            open_yes_price=market.previous_price_dollars,
            is_resolved=is_resolved,
            resolution_outcome=resolution_outcome,
        )

    def _parse_ob_levels(self, levels: List[Any]) -> List[BidAskLevel]:
        """Parse a list of [price_str, count_str] pairs into BidAskLevel objects."""
        out: List[BidAskLevel] = []
        for lvl in (levels or [])[:_ORDER_BOOK_DEPTH]:
            try:
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    out.append(BidAskLevel(
                        price=_parse_dollars(str(lvl[0])),
                        size=_parse_count(str(lvl[1])),
                    ))
            except (InvalidOperation, ValueError, IndexError):
                continue
        return out

    def _parse_price_snapshot(
        self, raw_dict: Dict[str, Any], contract_id: str,
    ) -> PriceSnapshot:
        """
        Parse an orderbook_fp response into a PriceSnapshot.

        YES bids come from orderbook_fp.yes_dollars.
        Implied YES asks are derived from NO bids: ask_price = 1.0 - no_bid_price.
        yes_price = best YES bid; no_price = best NO bid.

        Raises:
            DataValidationError: if prices are out of range.
        """
        ingestion_ts = datetime.now(timezone.utc)
        raw_str = json.dumps(raw_dict)

        ob_fp = raw_dict.get("orderbook_fp", raw_dict.get("orderbook", {}))
        raw_yes = ob_fp.get("yes_dollars", ob_fp.get("yes", []))
        raw_no = ob_fp.get("no_dollars", ob_fp.get("no", []))

        bids = sorted(self._parse_ob_levels(raw_yes), key=lambda x: x.price, reverse=True)
        no_bids = sorted(self._parse_ob_levels(raw_no), key=lambda x: x.price, reverse=True)
        asks = sorted(
            [BidAskLevel(price=round(1.0 - b.price, 6), size=b.size) for b in no_bids],
            key=lambda x: x.price,
        )

        yes_price = bids[0].price if bids else _safe_parse_dollars_field(raw_dict, "last_price_dollars", 0.5)
        no_price = no_bids[0].price if no_bids else round(1.0 - yes_price, 6)
        self._validate_individual_prices(yes_price, no_price, contract_id, raw_str)

        vol = _safe_parse_count(str(raw_dict.get("volume_fp", raw_dict.get("volume", "0"))))
        vol_24h_raw = raw_dict.get("volume_24h_fp", raw_dict.get("volume_24h", ""))
        daily_vol: Optional[float] = _safe_parse_count(str(vol_24h_raw)) if vol_24h_raw else None
        book_depth_usd = sum(b.price * b.size for b in bids)

        return PriceSnapshot(
            contract_id=contract_id, timestamp=ingestion_ts,
            yes_price=yes_price, no_price=no_price,
            volume_fp=vol, daily_volume=daily_vol,
            bids=bids, asks=asks,
            book_depth_usd=book_depth_usd,
        )

    @staticmethod
    def _parse_trade(raw_dict: Dict[str, Any], contract_id: str) -> Trade:
        """
        Parse a single trade record from /markets/trades.
        Field names: count_fp, yes_price_dollars, taker_side, created_time.

        Raises:
            KeyError | ValueError | InvalidOperation: on malformed fields.
        """
        # Default to "" (not "yes") so missing taker_side gives is_aggressive=False
        taker_raw = raw_dict.get("taker_side", raw_dict.get("taker_outcome_side", ""))
        taker = str(taker_raw).lower()
        direction = "YES" if taker == "yes" else "NO"
        count_raw = raw_dict.get("count_fp", raw_dict.get("count", "0"))
        price_raw = raw_dict.get("yes_price_dollars", raw_dict.get("yes_price", "0.5"))
        return Trade(
            contract_id=contract_id,
            timestamp=KalshiClient._parse_utc(raw_dict["created_time"]),
            direction=direction,
            size_contracts=_parse_count(str(count_raw)),
            price=_parse_dollars(str(price_raw)),
            is_aggressive=bool(taker),   # False only when taker_side absent from payload
        )

    @staticmethod
    def _parse_position(raw_dict: Dict[str, Any]) -> Position:
        """
        Parse a position record from /portfolio/positions.
        Derives direction from sign of position_fp (+YES, −NO).
        Uses market_exposure_dollars, realized_pnl_dollars, last_updated_ts.

        Raises:
            KeyError | ValueError | InvalidOperation: on malformed fields.
        """
        ticker = str(raw_dict.get("market_ticker", raw_dict.get("ticker", "")))
        pos_fp_raw = str(raw_dict.get("position_fp", raw_dict.get("position", "0")))
        pos_fp = _parse_count(pos_fp_raw)
        direction = "YES" if pos_fp >= 0 else "NO"
        size_contracts = abs(pos_fp)

        exposure_raw = raw_dict.get("market_exposure_dollars", raw_dict.get("market_exposure", "0"))
        market_exposure = _safe_parse_dollars_field_str(str(exposure_raw))

        pnl_raw = raw_dict.get("realized_pnl_dollars", raw_dict.get("realized_pnl", "0"))
        realized_pnl = _safe_parse_dollars_field_str(str(pnl_raw))

        ts_raw = raw_dict.get("last_updated_ts", raw_dict.get("updated_at", ""))
        try:
            last_updated = KalshiClient._parse_utc(ts_raw) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            last_updated = datetime.now(timezone.utc)

        return Position(
            contract_id=ticker,
            direction=direction,
            size_contracts=size_contracts,
            market_exposure=market_exposure,
            realized_pnl=realized_pnl,
            last_updated_ts=last_updated,
        )

    @staticmethod
    def _parse_order_confirmation(
        raw: Dict[str, Any], contract_id: str,
    ) -> OrderConfirmation:
        """
        Parse order response. Status is one of: resting | canceled | executed.
        Partial fill detection: status == resting AND fill_count < requested_count.

        Raises:
            KeyError | ValueError | InvalidOperation: on malformed fields.
        """
        order_id = str(raw.get("order_id", raw.get("client_order_id", str(uuid.uuid4()))))
        status_raw = str(raw.get("status", "resting")).lower()
        status_map = {
            "executed": "executed", "filled": "executed",
            "resting": "resting", "open": "resting",
            "canceled": "canceled", "cancelled": "canceled",
        }
        status = status_map.get(status_raw, "resting")

        requested = _parse_count(str(raw.get("initial_count_fp", raw.get("count_fp", "0"))))
        filled = _parse_count(str(raw.get("fill_count_fp", raw.get("filled_count", "0"))))
        remaining = _parse_count(str(raw.get("remaining_count_fp", str(max(0.0, requested - filled)))))

        price_raw = raw.get("yes_price_dollars", raw.get("yes_price"))
        filled_price: Optional[float] = None
        if price_raw:
            try:
                filled_price = _parse_dollars(str(price_raw))
            except (InvalidOperation, ValueError):
                pass

        direction_raw = str(raw.get("side", "yes")).upper()
        action_raw = str(raw.get("action", "buy")).lower()

        ts_raw = raw.get("created_time", "")
        try:
            created_ts = KalshiClient._parse_utc(ts_raw) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            created_ts = datetime.now(timezone.utc)

        return OrderConfirmation(
            order_id=order_id, contract_id=contract_id,
            direction=direction_raw, action=action_raw,
            requested_count=requested, filled_count=filled, remaining_count=remaining,
            filled_price=filled_price, status=status, created_timestamp=created_ts,
        )

    @staticmethod
    def _candlestick_to_snapshot(
        candle: Dict[str, Any], contract_id: str,
    ) -> Optional[PriceSnapshot]:
        """
        Map a single candlestick record to a PriceSnapshot.

        Price source priority:
          1. price.close_dollars (OpenAPI MarketCandlestick field)
          2. price.close (legacy compat)
          3. Mid of yes_bid.close_dollars and yes_ask.close_dollars — used on
             zero-volume days where price object is empty {}
          4. Returns None if no price data available — caller must skip.

        IMPORTANT: Never fall back to 0.5. A 0.5 anchor in price history silently
        corrupts the momentum fade signal's price drift computation.

        Returns:
            PriceSnapshot, or None if no price data is available.

        Raises:
            KeyError | ValueError | InvalidOperation: on malformed fields.
        """
        price_obj = candle.get("price", {})
        close_raw = price_obj.get("close_dollars") or price_obj.get("close")

        if close_raw:
            yes_price = _parse_dollars(str(close_raw))
        else:
            # Zero-volume day: price object is empty {}. Use bid/ask mid.
            bid_close = candle.get("yes_bid", {}).get("close_dollars")
            ask_close = candle.get("yes_ask", {}).get("close_dollars")
            if bid_close and ask_close:
                yes_price = round(
                    (_parse_dollars(str(bid_close)) + _parse_dollars(str(ask_close))) / 2.0,
                    4,
                )
            else:
                return None   # no price data — caller skips this candle

        no_price = round(1.0 - yes_price, 6)
        ts = datetime.fromtimestamp(int(candle["end_period_ts"]), tz=timezone.utc)
        # volume_fp is contracts (not USD); fall back to volume for compat
        vol = _safe_parse_count(str(candle.get("volume_fp", candle.get("volume", "0"))))
        return PriceSnapshot(
            contract_id=contract_id, timestamp=ts,
            yes_price=yes_price, no_price=no_price,
            volume_fp=vol, daily_volume=None,
            bids=[], asks=[],
            book_depth_usd=0.0,   # no order book data in candlestick responses
        )

    # ------------------------------------------------------------------
    # Public API endpoints
    # ------------------------------------------------------------------

    def get_events(
        self,
        series_ticker: str,
        status: str = "open",
    ) -> "APIResponse[List[EventData]]":
        """
        Fetch active sports events with nested markets from Kalshi.
        This is the correct and only valid ingestion path for sports contracts.

        get_markets() does NOT return series_ticker on individual market objects and
        therefore cannot be used to build sports contract universes or call
        get_price_history(). Always use get_events() for sports.

        Args:
            series_ticker: series filter, e.g. "KXNBAGAME". Required for sports.
            status: event status query filter; API enum: unopened | open | closed | settled.
                    Default "open" for tradable events. (Market object status field may
                    read "active" — that is not the same as this query param.)

        Returns:
            APIResponse wrapping List[EventData]. Each EventData contains event-level
            metadata (series_ticker, event_ticker, sub_title, mutually_exclusive) and
            a List[ContractObject] for the nested markets. series_ticker and event_ticker
            are injected into each ContractObject from the parent event.
            Invalid contracts are excluded and logged; they do not raise.

        Raises:
            AuthenticationError, RateLimitError, ServerError, TimeoutError,
            DataValidationError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/events"
        params: Dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": status,
            "with_nested_markets": "true",
            "limit": 100,
        }

        events_raw, latency_ms, status_code = self._request_paginated(
            path, "events", params=params
        )

        all_events: List[EventData] = []
        for event in events_raw:
            evt_ticker = str(event.get("event_ticker", ""))
            evt_series = str(event.get("series_ticker", series_ticker))
            sub_title = str(event.get("sub_title", ""))
            title = str(event.get("title", ""))
            mutually_exclusive = bool(event.get("mutually_exclusive", False))

            contracts: List[ContractObject] = []
            for market_raw in (event.get("markets") or []):
                # series_ticker, event_ticker, and event_sub_title live on the event
                # object, not on individual market dicts. Pass them as explicit
                # arguments to _build_contract() — never write them into market_raw.
                try:
                    market_data = self._parse_market(market_raw)
                    contract = self._build_contract(
                        market_data,
                        series_ticker=evt_series,
                        event_ticker=evt_ticker,
                        event_sub_title=sub_title,
                    )
                    contracts.append(contract)
                except DataValidationError as exc:
                    self._logger.error(
                        "Excluding contract from event due to integrity failure",
                        extra={
                            "event_ticker": evt_ticker,
                            "contract_id": exc.contract_id,
                            "error": str(exc),
                            "raw_fragment": (exc.raw_response or "")[:300],
                        },
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    self._logger.error(
                        "Excluding contract from event due to parse error",
                        extra={
                            "event_ticker": evt_ticker,
                            "error": str(exc),
                            "market_keys": list(market_raw.keys()),
                        },
                    )

            all_events.append(EventData(
                event_ticker=evt_ticker,
                series_ticker=evt_series,
                title=title,
                sub_title=sub_title,
                mutually_exclusive=mutually_exclusive,
                markets=contracts,
            ))

        return APIResponse(
            payload=all_events, http_status_code=status_code,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_markets(
        self, sport_filter: Optional[str] = None,
    ) -> APIResponse[List[ContractObject]]:
        """
        Fetch all active markets from Kalshi with cursor pagination.
        Used for daily universe construction.

        Args:
            sport_filter: optional series_ticker prefix, e.g. "NBAWIN", "NFLWIN".
                          Passed as series_ticker query parameter.

        Returns:
            APIResponse wrapping List[ContractObject]. Invalid contracts excluded (logged).

        Raises:
            AuthenticationError, RateLimitError, ServerError, TimeoutError,
            DataValidationError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/markets"
        params: Dict[str, Any] = {"status": "open", "limit": 1000, "mve_filter": "exclude"}
        if sport_filter:
            params["series_ticker"] = sport_filter

        markets_raw, latency_ms, status = self._request_paginated(
            path, "markets", params=params
        )
        contracts: List[ContractObject] = []
        for market in markets_raw:
            try:
                contracts.append(self._parse_contract(market))
            except DataValidationError as exc:
                self._logger.error(
                    "Excluding contract due to integrity failure",
                    extra={"contract_id": exc.contract_id, "error": str(exc),
                           "raw_fragment": (exc.raw_response or "")[:300]},
                )
            except (KeyError, ValueError, TypeError) as exc:
                self._logger.error(
                    "Excluding contract due to parse error",
                    extra={"error": str(exc), "market_keys": list(market.keys())},
                )
        return APIResponse(
            payload=contracts, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_market(self, contract_id: str) -> APIResponse[ContractObject]:
        """
        Fetch full detail for a single contract including resolution criteria text.
        Called on first ingestion of any new contract in the universe.

        Args:
            contract_id: Kalshi ticker.

        Returns:
            APIResponse wrapping ContractObject.

        Raises:
            AuthenticationError, ClientError (404 = not found), DataValidationError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/markets/{contract_id}"
        raw, latency_ms, status = self._request("GET", path, contract_id=contract_id)
        contract = self._parse_contract(raw.get("market", raw))
        return APIResponse(
            payload=contract, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_order_book(self, contract_id: str) -> APIResponse[PriceSnapshot]:
        """
        Fetch top-5 bid/ask snapshot for a contract. Called every 15 min for universe.

        Args:
            contract_id: Kalshi ticker.

        Returns:
            APIResponse wrapping PriceSnapshot. Bids best-first; asks cheapest-first.

        Raises:
            AuthenticationError, ClientError, DataValidationError,
            RateLimitError, ServerError, TimeoutError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/markets/{contract_id}/orderbook"
        params: Dict[str, Any] = {"depth": _ORDER_BOOK_DEPTH}
        raw, latency_ms, status = self._request(
            "GET", path, params=params, contract_id=contract_id
        )
        return APIResponse(
            payload=self._parse_price_snapshot(raw, contract_id),
            http_status_code=status, latency_ms=latency_ms,
            endpoint=path, success=True,
        )

    def get_trade_history(
        self, contract_id: str, lookback_hours: float,
    ) -> APIResponse[List[Trade]]:
        """
        Fetch trades for a contract via the global trades endpoint with cursor pagination.
        Endpoint: GET /trade-api/v2/markets/trades?ticker=...&min_ts=<unix_int>

        Args:
            contract_id: Kalshi ticker.
            lookback_hours: hours to look back from now.

        Returns:
            APIResponse wrapping List[Trade] sorted oldest-first.

        Raises:
            AuthenticationError, ClientError, DataValidationError,
            RateLimitError, ServerError, TimeoutError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/markets/trades"
        min_ts_unix = int(datetime.now(timezone.utc).timestamp() - lookback_hours * 3600)
        params: Dict[str, Any] = {"ticker": contract_id, "min_ts": min_ts_unix, "limit": 1000}

        trades_raw, latency_ms, status = self._request_paginated(
            path, "trades", params=params, contract_id=contract_id
        )
        trades: List[Trade] = []
        for t in trades_raw:
            try:
                trades.append(self._parse_trade(t, contract_id))
            except (KeyError, ValueError, InvalidOperation) as exc:
                self._logger.warning(
                    "Skipping malformed trade record",
                    extra={"contract_id": contract_id, "error": str(exc)},
                )
        trades.sort(key=lambda x: x.timestamp)
        return APIResponse(
            payload=trades, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_price_history(
        self,
        contract_id: str,
        series_ticker: str,
        start_timestamp: datetime,
        period_interval: int = _DEFAULT_CANDLE_INTERVAL,
    ) -> APIResponse[List[PriceSnapshot]]:
        """
        Fetch price history via the candlesticks endpoint (not /history, which does not exist).
        Endpoint: GET /trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks

        Point-in-time discipline: start_timestamp must be the moment data was requested,
        not a future or retroactively adjusted time.

        Args:
            contract_id: Kalshi ticker.
            series_ticker: parent series, e.g. "NBAWIN". Available on ContractObject.series_ticker.
            start_timestamp: UTC datetime; candles at or after this time are returned.
            period_interval: candle size in minutes. Valid values: 1, 60, 1440 (default 1440).

        Returns:
            APIResponse wrapping List[PriceSnapshot] sorted oldest-first.
            Snapshots from candles have empty bids/asks (no order book data in candles).

        Raises:
            ValueError: if period_interval is not 1, 60, or 1440.
            AuthenticationError, ClientError, DataValidationError,
            RateLimitError, ServerError, TimeoutError.
        """
        if not series_ticker:
            raise DataValidationError(
                f"series_ticker is required for candlestick endpoint "
                f"but was empty for contract {contract_id}",
                contract_id=contract_id,
            )

        if period_interval not in (1, 60, 1440):
            raise ValueError(f"period_interval must be 1, 60, or 1440; got {period_interval}")

        path = (
            f"/trade-api/{KALSHI_API_VERSION}/series/{series_ticker}"
            f"/markets/{contract_id}/candlesticks"
        )
        params: Dict[str, Any] = {
            "start_ts": int(start_timestamp.timestamp()),
            "end_ts": int(datetime.now(timezone.utc).timestamp()),
            "period_interval": period_interval,
        }
        raw, latency_ms, status = self._request(
            "GET", path, params=params, contract_id=contract_id
        )
        candles_raw: List[Dict[str, Any]] = raw.get("candlesticks", [])
        snapshots: List[PriceSnapshot] = []
        for candle in candles_raw:
            try:
                snap = self._candlestick_to_snapshot(candle, contract_id)
                if snap is not None:
                    snapshots.append(snap)
                else:
                    self._logger.debug(
                        "Skipping zero-volume candlestick: no price data available",
                        extra={
                            "contract_id": contract_id,
                            "end_period_ts": candle.get("end_period_ts"),
                        },
                    )
            except (KeyError, ValueError, InvalidOperation) as exc:
                self._logger.warning(
                    "Skipping malformed candlestick",
                    extra={"contract_id": contract_id, "error": str(exc)},
                )
        snapshots.sort(key=lambda x: x.timestamp)
        return APIResponse(
            payload=snapshots, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def place_order(
        self,
        contract_id: str,
        direction: str,
        action: str,
        count: float,
        limit_price: Optional[float] = None,
        buy_max_cost: Optional[int] = None,
    ) -> APIResponse[OrderConfirmation]:
        """
        Place an order on Kalshi. Expects HTTP 201 on success.

        DEPRECATION NOTE: This uses the legacy /portfolio/orders endpoint, valid
        until May 2026 per Kalshi API deprecation schedule. Migrate when notified.

        Args:
            contract_id: Kalshi ticker.
            direction: "YES" or "NO" — maps to API field `side`.
            action: "buy" or "sell" — required by Kalshi API.
            count: number of contracts (not dollars). Use count_fp string internally.
            limit_price: Price expressed as a YES price (0–1). For YES orders → sent as
                         yes_price_dollars. For NO orders → sent as no_price_dollars = 1 -
                         limit_price. If None → market buy via buy_max_cost (sell without
                         limit_price raises ValueError).
            buy_max_cost: max cents to spend for market buy. Defaults to int(count * 100)
                          (maximum $1 per contract). Ignored for limit orders and sell orders.

        Returns:
            APIResponse[OrderConfirmation] with HTTP status 201.

        Raises:
            ValueError: on invalid direction/action.
            PartialFillError: if status == resting and fill < initial (partial fill detected).
            AuthenticationError, ClientError, DataValidationError.
        """
        direction = direction.upper()
        action = action.lower()
        if direction not in ("YES", "NO"):
            raise ValueError(f"direction must be 'YES' or 'NO', got {direction!r}")
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")

        path = f"/trade-api/{KALSHI_API_VERSION}/portfolio/orders"
        count_fp_str = f"{count:.2f}"
        body: Dict[str, Any] = {
            "ticker": contract_id,
            "side": direction.lower(),
            "action": action,
            "count_fp": count_fp_str,
            "client_order_id": str(uuid.uuid4()),
        }
        if limit_price is not None:
            if direction == "YES":
                body["yes_price_dollars"] = f"{limit_price:.4f}"
            else:
                # NO limit: API expects no_price_dollars.
                # limit_price is always expressed as a YES price, so NO price = 1 - limit_price.
                body["no_price_dollars"] = f"{round(1.0 - limit_price, 4):.4f}"
            body["time_in_force"] = "good_till_cancelled"
        else:
            if action == "sell":
                raise ValueError(
                    "Market sell orders are not supported — Kalshi has no sell-side "
                    "buy_max_cost equivalent. Pass limit_price for a fill_or_kill limit "
                    "sell, or close by buying the opposite side."
                )
            effective_max_cost = buy_max_cost if buy_max_cost is not None else int(count * 100)
            body["buy_max_cost"] = effective_max_cost
            body["time_in_force"] = "fill_or_kill"

        raw, latency_ms, status = self._request(
            "POST", path, body=body, contract_id=contract_id
        )
        confirmation = self._parse_order_confirmation(raw.get("order", raw), contract_id)
        self._check_partial_fill(confirmation, contract_id)

        return APIResponse(
            payload=confirmation, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def _check_partial_fill(
        self, conf: OrderConfirmation, contract_id: str,
    ) -> None:
        """
        Detect partial fills: status == resting AND fill_count < requested_count.
        Logs CRITICAL and raises PartialFillError so caller can close the filled leg.
        """
        is_partial = (
            conf.status == "resting"
            and conf.filled_count > 0
            and conf.filled_count < conf.requested_count
        )
        if not is_partial:
            return
        self._logger.critical(
            "Partial fill detected — caller must close filled leg immediately",
            extra={
                "contract_id": contract_id,
                "order_id": conf.order_id,
                "filled_count": conf.filled_count,
                "requested_count": conf.requested_count,
            },
        )
        raise PartialFillError(
            f"Partial fill on {contract_id}: {conf.filled_count:.2f} "
            f"of {conf.requested_count:.2f} contracts filled",
            filled_order_id=conf.order_id,
            filled_count=conf.filled_count,
            contract_id=contract_id,
        )

    def get_positions(self) -> APIResponse[List[Position]]:
        """
        Fetch all current open positions with cursor pagination.
        Run after every execution for ledger reconciliation.

        Returns:
            APIResponse wrapping List[Position].

        Raises:
            AuthenticationError, ClientError, DataValidationError,
            RateLimitError, ServerError, TimeoutError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/portfolio/positions"
        positions_raw, latency_ms, status = self._request_paginated(
            path, "market_positions"
        )
        positions: List[Position] = []
        for p in positions_raw:
            try:
                positions.append(self._parse_position(p))
            except (KeyError, ValueError, InvalidOperation) as exc:
                self._logger.error(
                    "Failed to parse position record",
                    extra={"error": str(exc), "raw_keys": list(p.keys())},
                )
        return APIResponse(
            payload=positions, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_balance(self) -> "APIResponse[float]":
        """
        Fetch the current portfolio cash balance in dollars.
        Endpoint: GET /trade-api/v2/portfolio/balance

        Returns:
            APIResponse[float] where payload is balance_dollars as a Python float.
            Logs WARNING and returns 0.0 if balance_dollars is absent in the response.

        Raises:
            DataValidationError: if balance_dollars is present but malformed.
            AuthenticationError, ClientError, RateLimitError, ServerError, TimeoutError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/portfolio/balance"
        raw, latency_ms, status = self._request("GET", path)
        raw_balance = raw.get("balance_dollars")
        if raw_balance is None:
            self._logger.warning(
                "get_balance: balance_dollars absent in response; returning 0.0",
                extra={"endpoint": path, "raw_keys": list(raw.keys())},
            )
            balance: float = 0.0
        else:
            try:
                balance = _parse_dollars(str(raw_balance))
            except (InvalidOperation, ValueError) as exc:
                raise DataValidationError(
                    f"get_balance: malformed balance_dollars field: {raw_balance!r}",
                    raw_response=json.dumps(raw),
                ) from exc
        return APIResponse(
            payload=balance, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_historical_cutoff(self) -> "APIResponse[Dict[str, datetime]]":
        """
        Fetch the Kalshi historical data cutoff timestamps.
        Endpoint: GET /trade-api/v2/historical/cutoff

        Extracts market_settled_ts, trades_created_ts, orders_updated_ts — each
        parsed to UTC datetime. Keys absent from the response are omitted rather
        than raising. One structured log entry is emitted regardless of which keys
        are present.

        Returns:
            APIResponse[Dict[str, datetime]] with up to three keys.

        Raises:
            AuthenticationError, ClientError, RateLimitError, ServerError, TimeoutError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/historical/cutoff"
        raw, latency_ms, status = self._request("GET", path)

        _CUTOFF_KEYS = ("market_settled_ts", "trades_created_ts", "orders_updated_ts")
        cutoffs: Dict[str, datetime] = {}
        for key in _CUTOFF_KEYS:
            raw_val = raw.get(key)
            if raw_val is None:
                continue
            try:
                cutoffs[key] = self._parse_utc(str(raw_val))
            except ValueError:
                self._logger.warning(
                    "get_historical_cutoff: skipping unparseable timestamp",
                    extra={"key": key, "raw_value": str(raw_val)},
                )

        return APIResponse(
            payload=cutoffs, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_historical_markets(
        self, series_ticker: str,
    ) -> "APIResponse[List[ContractObject]]":
        """
        Fetch all historical markets for a series from the Kalshi historical API.
        Endpoint: GET /trade-api/v2/historical/markets  (paginated)

        Returns flat ContractObject instances, not EventData wrappers.
        Uses _parse_market() + _build_contract() — NOT _parse_contract() —
        so zero-volume historical contracts parse successfully.

        sport is resolved from series_ticker via SERIES_TICKER_TO_SPORT;
        event_sub_title is unavailable on flat historical market objects so "" is
        injected (the keyword lookup in _extract_sport_teams handles it).

        Args:
            series_ticker: Kalshi series ticker, e.g. "KXNBAGAME".

        Returns:
            APIResponse wrapping List[ContractObject]. Invalid markets excluded (logged).

        Raises:
            AuthenticationError, RateLimitError, ServerError, TimeoutError.
        """
        path = f"/trade-api/{KALSHI_API_VERSION}/historical/markets"
        params: Dict[str, Any] = {"series_ticker": series_ticker, "limit": 1000}

        markets_raw, latency_ms, status_code = self._request_paginated(
            path, "markets", params=params
        )

        contracts: List[ContractObject] = []
        for market_raw in markets_raw:
            try:
                market_data = self._parse_market(market_raw)
                contract = self._build_contract(
                    market_data,
                    series_ticker=series_ticker,
                    event_ticker=market_data.event_ticker,
                    event_sub_title="",
                )
                contracts.append(contract)
            except DataValidationError as exc:
                self._logger.error(
                    "Excluding historical market due to integrity failure",
                    extra={
                        "series_ticker": series_ticker,
                        "contract_id": exc.contract_id,
                        "error": str(exc),
                    },
                )
            except (KeyError, ValueError, TypeError) as exc:
                self._logger.error(
                    "Excluding historical market due to parse error",
                    extra={
                        "series_ticker": series_ticker,
                        "error": str(exc),
                        "market_keys": list(market_raw.keys()),
                    },
                )

        return APIResponse(
            payload=contracts, http_status_code=status_code,
            latency_ms=latency_ms, endpoint=path, success=True,
        )

    def get_historical_candlesticks(
        self,
        contract_id: str,
        series_ticker: str,
        start_timestamp: datetime,
        period_interval: int = _DEFAULT_CANDLE_INTERVAL,
    ) -> "APIResponse[List[PriceSnapshot]]":
        """
        Fetch historical candlesticks for a contract via the historical API.
        Endpoint: GET /trade-api/v2/historical/markets/{ticker}/candlesticks

        series_ticker is NOT used in the URL path — only contract_id (ticker).
        It is accepted for API consistency but is not sent in the request.

        Args:
            contract_id: Kalshi ticker.
            series_ticker: Accepted for API consistency; not used in the path.
            start_timestamp: UTC datetime; candles at or after this time returned.
            period_interval: Candle size in minutes. Valid: 1, 60, 1440 (default).

        Returns:
            APIResponse wrapping List[PriceSnapshot] sorted oldest-first.

        Raises:
            ValueError: if period_interval is not 1, 60, or 1440.
            DataValidationError: if contract_id is empty.
            AuthenticationError, ClientError, RateLimitError, ServerError, TimeoutError.
        """
        if not contract_id:
            raise DataValidationError(
                "contract_id is required for historical candlestick endpoint but was empty",
                contract_id=contract_id,
            )
        if period_interval not in (1, 60, 1440):
            raise ValueError(
                f"period_interval must be 1, 60, or 1440; got {period_interval}"
            )

        path = (
            f"/trade-api/{KALSHI_API_VERSION}/historical/markets"
            f"/{contract_id}/candlesticks"
        )
        params: Dict[str, Any] = {
            "start_ts": int(start_timestamp.timestamp()),
            "end_ts": int(datetime.now(timezone.utc).timestamp()),
            "period_interval": period_interval,
        }
        raw, latency_ms, status = self._request(
            "GET", path, params=params, contract_id=contract_id
        )
        candles_raw: List[Dict[str, Any]] = raw.get("candlesticks", [])
        snapshots: List[PriceSnapshot] = []
        for candle in candles_raw:
            try:
                snap = self._candlestick_to_snapshot(candle, contract_id)
                if snap is not None:
                    snapshots.append(snap)
                else:
                    self._logger.debug(
                        "Skipping zero-volume historical candlestick: no price data",
                        extra={
                            "contract_id": contract_id,
                            "end_period_ts": candle.get("end_period_ts"),
                        },
                    )
            except (KeyError, ValueError, InvalidOperation) as exc:
                self._logger.warning(
                    "Skipping malformed historical candlestick",
                    extra={"contract_id": contract_id, "error": str(exc)},
                )
        snapshots.sort(key=lambda x: x.timestamp)
        return APIResponse(
            payload=snapshots, http_status_code=status,
            latency_ms=latency_ms, endpoint=path, success=True,
        )


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------

def _load_rsa_key(pem: str) -> RSAPrivateKey:
    """
    Load and validate RSA private key from PEM string.

    Raises:
        ValueError: if pem is not a valid RSA private key.
        ConfigurationError: wrapped around cryptography errors.
    """
    try:
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    except Exception as exc:
        raise ConfigurationError(
            f"KALSHI_PRIVATE_KEY_PEM / KALSHI_PRIVATE_KEY_PATH is not a valid "
            f"RSA private key: {exc}"
        ) from exc
    if not isinstance(key, RSAPrivateKey):
        raise ConfigurationError(
            "Key is not an RSA private key. Kalshi requires RSA-PSS."
        )
    return key


def _safe_parse_count(s: str) -> float:
    """Parse count string; return 0.0 on error (used for optional volume fields)."""
    try:
        return _parse_count(s)
    except (InvalidOperation, ValueError):
        return 0.0


def _safe_parse_dollars_field(raw: Dict[str, Any], field: str, default: float) -> float:
    """Parse a dollar field from dict; return default on error."""
    val = raw.get(field, "")
    if not val:
        return default
    try:
        return _parse_dollars(str(val))
    except (InvalidOperation, ValueError):
        return default


def _safe_parse_dollars_field_str(s: str) -> float:
    """Parse a dollar string; return 0.0 on error."""
    try:
        return _parse_dollars(s)
    except (InvalidOperation, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Sport/team extraction (module-level)
# ---------------------------------------------------------------------------

_SPORT_KEYWORDS: Dict[str, str] = {
    "nba": "NBA", "nfl": "NFL", "mlb": "MLB", "nhl": "NHL", "epl": "EPL",
    "basketball": "NBA", "football": "NFL", "baseball": "MLB",
    "hockey": "NHL", "soccer": "EPL",
}

# Primary sport lookup keyed by Kalshi series ticker.
# Checked before subtitle keyword scan — authoritative for known Kalshi series.
SERIES_TICKER_TO_SPORT: Dict[str, str] = {
    "KXNBAGAME":   "NBA",
    "KXNFLGAME":   "NFL",
    "KXMLBGAME":   "MLB",
    "KXNHLGAME":   "NHL",
    "KXNCAAMBGAME": "NCAAB",
    "KXNCAAFGAME": "NCAAF",
    "KXUFCFIGHT":  "UFC",
}


def _extract_ufc_opponent(this_fighter: str, title: str) -> str:
    """Extract the opponent's name from a UFC market title string.

    UFC titles follow the pattern:
        "Will {Fighter} win the {LastA} vs {LastB} professional MMA fight..."

    The matchup segment is the text between "the " and " professional".
    Split on " vs " to get the two last-name fragments, then match against
    yes_sub_title to identify which side this contract is on, and return
    the full name of the other fighter from yes_sub_title of the sibling
    contract — but since we only have one contract here, return the raw
    opponent fragment from the title as the best available approximation.

    Args:
        this_fighter: Full name of the contract's fighter (from yes_sub_title).
        title: Full market title string.

    Returns:
        Opponent name string, or "UNKNOWN" if parsing fails.
    """
    if not title or " vs " not in title:
        return "UNKNOWN"
    try:
        # Extract "LastA vs LastB" segment between "the " and " professional"
        after_the = title.split(" the ", 1)[1]
        matchup = after_the.split(" professional")[0].strip()
        parts = matchup.split(" vs ", 1)
        if len(parts) != 2:
            return "UNKNOWN"
        frag_a = parts[0].strip()
        frag_b = parts[1].strip()
        # Match by word overlap between the title fragment and this_fighter's full name.
        # This handles compound surnames (e.g. "Nascimento de Souza") correctly where
        # the fragment starts with a middle surname rather than the final surname word.
        name_words = set(this_fighter.lower().split())
        if set(frag_a.lower().split()) & name_words:
            return frag_b
        if set(frag_b.lower().split()) & name_words:
            return frag_a
        # Fallback: this fighter is frag_a side, opponent is frag_b
        return frag_b
    except (IndexError, AttributeError):
        return "UNKNOWN"


def _extract_sport_teams(
    subtitle: str, raw: Dict[str, Any],
) -> Tuple[str, str, str]:
    """Extract sport, home team, and away team from market metadata.

    Uses series_ticker as the primary sport signal (via SERIES_TICKER_TO_SPORT),
    falling back to subtitle keyword scan only when unrecognised. UFC contracts
    source fighter names from yes_sub_title / no_sub_title rather than subtitle
    separator parsing. Returns ("UNKNOWN", …) fields rather than raising on failure.

    Args:
        subtitle: Event-level subtitle, e.g. "OKC at SAS (May 28)".
        raw: Raw market dict. Relevant keys: series_ticker, yes_sub_title, no_sub_title.

    Returns:
        Tuple (sport, home_team, away_team).
    """
    # 1. Series-ticker primary lookup — authoritative for known Kalshi series.
    series_ticker = str(raw.get("series_ticker", "")).upper()
    sport = SERIES_TICKER_TO_SPORT.get(series_ticker, "UNKNOWN")

    # 2. Subtitle keyword scan — fallback only when series_ticker is unrecognised.
    if sport == "UNKNOWN":
        sub_lower = subtitle.lower()
        for keyword, tag in _SPORT_KEYWORDS.items():
            if keyword in sub_lower or keyword in series_ticker.lower():
                sport = tag
                break

    # 3. Team extraction: UFC sources home_team from yes_sub_title (this contract's fighter)
    #    and derives the opponent from the title's matchup segment via _extract_ufc_opponent.
    #    yes_sub_title == no_sub_title on real UFC contracts (both name the same fighter),
    #    so no_sub_title cannot be used for the opponent.
    if sport == "UFC":
        this_fighter = str(raw.get("yes_sub_title", "")).strip()
        title_str = str(raw.get("title", ""))
        opponent = _extract_ufc_opponent(this_fighter, title_str)
        home_team = this_fighter or "UNKNOWN"
        away_team = opponent
    else:
        home_team = "UNKNOWN"
        away_team = "UNKNOWN"
        # Historical API markets have title (e.g. "San Antonio at Golden State Winner?")
        # but no event_sub_title. Strip the trailing suffix to get a clean matchup string,
        # then try it before the subtitle argument. Take the full right part from a cleaned
        # title (no trailing garbage); take only the first word from a raw subtitle (which
        # may carry a " (date)" suffix like "SAS (May 28)").
        title_raw = str(raw.get("title", ""))
        title_clean = re.sub(r"\s*Winner\?$|\?$", "", title_raw).strip()
        # subtitle (event_sub_title from live events path) wins when non-empty.
        # title_clean is the fallback for historical flat market objects where subtitle is "".
        # full_right=False for subtitle: first word only to skip " (date)" suffixes.
        # full_right=True for title_clean: no trailing garbage after stripping "Winner?".
        for candidate, full_right in ((subtitle, False), (title_clean, True)):
            if not candidate:
                continue
            for sep in (" vs ", " at ", " @ ", " v "):
                if sep in candidate:
                    parts = candidate.split(sep, 1)
                    away_team = parts[0].strip()
                    home_team = parts[1].strip() if full_right else parts[1].strip().split(" ")[0]
                    break
            if home_team != "UNKNOWN":
                break

    # 4. Warn on any UNKNOWN field — loud failures over silent failures.
    unknown_fields = [
        name for name, val in (
            ("sport", sport), ("home_team", home_team), ("away_team", away_team)
        )
        if val == "UNKNOWN"
    ]
    if unknown_fields:
        import logging as _logging
        _logging.getLogger("kalshi_client").warning(
            "Could not extract all sport/team fields from market metadata",
            extra={
                "unknown_fields": unknown_fields,
                "subtitle": subtitle,
                "series_ticker": str(raw.get("series_ticker", "")),
            },
        )

    return sport, home_team, away_team
