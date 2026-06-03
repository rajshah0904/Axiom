"""
Tests for kalshi_client.py — Phase 1 Kalshi API layer.

All tests are offline: no live API calls, no network, no file system writes outside
pytest tmp_path. HTTP is mocked with the `responses` library. RSA keys are generated
ephemerally per session using the cryptography library.

Every public method has at minimum:
  - one happy-path test
  - one test per distinct failure mode

Tests are fully independent; no shared mutable state.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
import responses as responses_mock

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# Ensure the parent directory is importable without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import kalshi_client as kc
from kalshi_client import (
    APIResponse,
    AuthenticationError,
    ClientError,
    ConfigurationError,
    ContractObject,
    DataValidationError,
    EventData,
    KalshiClient,
    KalshiConfig,
    MarketData,
    OrderConfirmation,
    PartialFillError,
    Position,
    PositionMismatchError,
    PriceSnapshot,
    RateLimitError,
    ServerError,
    Trade,
    _extract_sport_teams,
    _extract_ufc_opponent,
)
# is_liquid is a static method accessed via KalshiClient.is_liquid()


# ---------------------------------------------------------------------------
# Constants shared across all tests
# ---------------------------------------------------------------------------

BASE_URL = "https://external-api.demo.kalshi.co"
API_ROOT = f"{BASE_URL}/trade-api/v2"
CONTRACT_ID = "NBAWIN-LAK-GSW-20260601"
SERIES_TICKER = "NBAWIN"
FUTURE_TS = "2026-12-01T12:00:00Z"   # well past today (2026-05-24)
CLOSE_TS = "2026-11-30T22:00:00Z"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rsa_private_key_pem() -> str:
    """
    Generate an ephemeral RSA-2048 private key for all tests.
    Session-scoped so key generation runs once per test session (~0.1 s).
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


@pytest.fixture
def tmp_log(tmp_path) -> str:
    """Return a writable temp file path for the rotating log handler."""
    return str(tmp_path / "test_kalshi.log")


@pytest.fixture
def env_vars(tmp_log, rsa_private_key_pem, monkeypatch) -> None:
    """Set all required environment variables using the ephemeral RSA key."""
    monkeypatch.setenv("KALSHI_API_KEY", "test-key-id-abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM", rsa_private_key_pem)
    monkeypatch.setenv("KALSHI_BASE_URL", BASE_URL)
    monkeypatch.setenv("KALSHI_LOG_FILE_PATH", tmp_log)
    monkeypatch.setenv("KALSHI_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("KALSHI_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("KALSHI_MAX_RETRIES", "3")


@pytest.fixture
def client(env_vars) -> KalshiClient:
    """Return a fully configured KalshiClient with the ephemeral RSA key."""
    cfg = KalshiConfig.from_env()
    return KalshiClient(cfg)


# ---------------------------------------------------------------------------
# Helper factories — produce minimal valid dicts using Kalshi API v3 field names
# ---------------------------------------------------------------------------

def make_market(ticker: str = CONTRACT_ID, series: str = SERIES_TICKER) -> Dict[str, Any]:
    """Minimal valid market dict. All dollar amounts are fixed-point strings."""
    return {
        "ticker": ticker,
        "series_ticker": series,
        "subtitle": "Los Angeles Lakers vs Golden State Warriors",
        "last_price_dollars": "0.5500",
        "yes_bid_dollars": "0.5400",
        "yes_ask_dollars": "0.5600",
        "volume_fp": "1000.00",
        "volume_24h_fp": "50.00",
        "latest_expiration_time": FUTURE_TS,
        "close_time": CLOSE_TS,
        "status": "open",   # market lifecycle: open | finalized | determined
        "result": "",
        "rules_primary": "Resolves YES if the Lakers win.",
    }


def make_order_response(
    status: str = "executed",
    fill_count: str = "5.00",
    initial_count: str = "5.00",
    order_id: str = "ord-abc123",
) -> Dict[str, Any]:
    """Kalshi POST /portfolio/orders response envelope."""
    remaining = str(round(float(initial_count) - float(fill_count), 2))
    return {
        "order": {
            "order_id": order_id,
            "status": status,
            "initial_count_fp": initial_count,
            "fill_count_fp": fill_count,
            "remaining_count_fp": remaining,
            "yes_price_dollars": "0.5500",
            "side": "yes",
            "action": "buy",
            "created_time": "2026-05-24T10:00:00Z",
        }
    }


def make_position_dict(
    ticker: str = CONTRACT_ID,
    pos_fp: str = "10.00",
) -> Dict[str, Any]:
    """Kalshi position record from /portfolio/positions."""
    return {
        "market_ticker": ticker,
        "position_fp": pos_fp,
        "market_exposure_dollars": "5.50",
        "realized_pnl_dollars": "0.25",
        "last_updated_ts": "2026-05-24T10:00:00Z",
    }


def make_trade_dict(
    count: str = "5.00",
    price: str = "0.5500",
    side: str = "yes",
    created_time: str = "2026-05-24T10:00:00Z",
) -> Dict[str, Any]:
    """Single trade record from /markets/trades."""
    return {
        "count_fp": count,
        "yes_price_dollars": price,
        "taker_side": side,
        "created_time": created_time,
    }


def make_candle(
    end_ts: int = 1748736000,
    close: Optional[str] = "0.5500",
) -> Dict[str, Any]:
    """
    Single candlestick record from the candlesticks endpoint.
    Uses OpenAPI MarketCandlestick field names: close_dollars and volume_fp.

    Pass close=None to simulate a zero-volume day (price object is empty {}).
    On zero-volume days the API still provides yes_bid and yes_ask close prices.
    """
    return {
        "end_period_ts": end_ts,
        "price": {"close_dollars": close} if close else {},
        "volume_fp": "200" if close else "0",
        "yes_bid": {"close_dollars": "0.5300", "open_dollars": "0.5300"},
        "yes_ask": {"close_dollars": "0.5700", "open_dollars": "0.5900"},
    }


def make_event(
    event_ticker: str = "KXNBAGAME-26MAY28OKCSAS",
    series_ticker: str = "KXNBAGAME",
    sub_title: str = "OKC at SAS (May 28)",
) -> Dict[str, Any]:
    """
    Single event dict from GET /trade-api/v2/events?with_nested_markets=true.
    Nested market objects intentionally have empty series_ticker to match the real
    API — series_ticker lives on the event, not on individual market objects.
    """
    return {
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
        "title": "Game 6: Oklahoma City at San Antonio",
        "sub_title": sub_title,
        "mutually_exclusive": True,
        "markets": [
            make_market(
                ticker=f"{event_ticker}-SAS",
                series="",   # intentionally empty — injected from event level
            )
        ],
    }


def make_market_raw(ticker: str = CONTRACT_ID) -> Dict[str, Any]:
    """
    Raw market dict as returned by the Kalshi API when nested inside an event.

    Deliberately omits series_ticker, event_ticker (at event level only), and
    event_sub_title (not an API field). Includes yes_sub_title and no_sub_title,
    which ARE returned on individual market objects but describe only one team each.

    Use this factory to test _parse_market() and to verify get_events() does not
    mutate market dicts by injecting keys that don't belong there.
    """
    return {
        "ticker": ticker,
        "last_price_dollars": "0.5500",
        "yes_bid_dollars": "0.5400",
        "yes_ask_dollars": "0.5600",
        "volume_fp": "1000.00",
        "volume_24h_fp": "50.00",
        "latest_expiration_time": FUTURE_TS,
        "close_time": CLOSE_TS,
        "status": "active",
        "result": "",
        "rules_primary": "Resolves YES if the team wins.",
        "yes_sub_title": "Oklahoma City Thunder",
        "no_sub_title": "San Antonio Spurs",
        "previous_price_dollars": "0.4800",
        "open_interest_fp": "250.00",
        # Note: no series_ticker — lives on event, not market
        # Note: no event_ticker — lives on event, not market
        # Note: no event_sub_title — not an API field; injected by old bad code
    }


# ===========================================================================
# Configuration tests
# ===========================================================================

class TestConfig:
    def test_loads_all_fields_from_env(self, env_vars):
        cfg = KalshiConfig.from_env()
        assert cfg.api_key == "test-key-id-abc123"
        assert cfg.base_url == BASE_URL
        assert cfg.timeout_seconds == 10.0
        assert cfg.max_retries == 3
        assert cfg.log_level == "DEBUG"
        assert "BEGIN" in cfg.private_key_pem   # confirms PEM header present

    def test_missing_api_key_raises(self, env_vars, monkeypatch):
        monkeypatch.delenv("KALSHI_API_KEY")
        with pytest.raises(ConfigurationError, match="KALSHI_API_KEY"):
            KalshiConfig.from_env()

    def test_missing_base_url_raises(self, env_vars, monkeypatch):
        monkeypatch.delenv("KALSHI_BASE_URL")
        with pytest.raises(ConfigurationError, match="KALSHI_BASE_URL"):
            KalshiConfig.from_env()

    def test_missing_log_file_path_raises(self, env_vars, monkeypatch):
        monkeypatch.delenv("KALSHI_LOG_FILE_PATH")
        with pytest.raises(ConfigurationError, match="KALSHI_LOG_FILE_PATH"):
            KalshiConfig.from_env()

    def test_missing_key_material_raises(self, env_vars, monkeypatch):
        monkeypatch.delenv("KALSHI_PRIVATE_KEY_PEM", raising=False)
        monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
        with pytest.raises(ConfigurationError, match="KALSHI_PRIVATE_KEY"):
            KalshiConfig.from_env()

    def test_invalid_timeout_raises(self, env_vars, monkeypatch):
        monkeypatch.setenv("KALSHI_TIMEOUT_SECONDS", "not-a-number")
        with pytest.raises(ConfigurationError, match="KALSHI_TIMEOUT_SECONDS"):
            KalshiConfig.from_env()

    def test_invalid_max_retries_raises(self, env_vars, monkeypatch):
        monkeypatch.setenv("KALSHI_MAX_RETRIES", "abc")
        with pytest.raises(ConfigurationError, match="KALSHI_MAX_RETRIES"):
            KalshiConfig.from_env()

    def test_invalid_log_level_raises(self, env_vars, monkeypatch):
        monkeypatch.setenv("KALSHI_LOG_LEVEL", "VERBOSE")
        with pytest.raises(ConfigurationError, match="KALSHI_LOG_LEVEL"):
            KalshiConfig.from_env()

    def test_private_key_path_loads_from_file(
        self, tmp_log, rsa_private_key_pem, monkeypatch, tmp_path
    ):
        key_file = tmp_path / "test.pem"
        key_file.write_text(rsa_private_key_pem)
        monkeypatch.setenv("KALSHI_API_KEY", "key-id")
        monkeypatch.setenv("KALSHI_BASE_URL", BASE_URL)
        monkeypatch.setenv("KALSHI_LOG_FILE_PATH", tmp_log)
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_file))
        monkeypatch.delenv("KALSHI_PRIVATE_KEY_PEM", raising=False)
        cfg = KalshiConfig.from_env()
        assert "BEGIN" in cfg.private_key_pem

    def test_private_key_path_missing_file_raises(
        self, env_vars, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("KALSHI_PRIVATE_KEY_PEM", raising=False)
        monkeypatch.setenv(
            "KALSHI_PRIVATE_KEY_PATH", str(tmp_path / "nonexistent.pem")
        )
        with pytest.raises(ConfigurationError, match="Cannot read"):
            KalshiConfig.from_env()

    def test_base_url_trailing_slash_stripped(self, env_vars, monkeypatch):
        monkeypatch.setenv("KALSHI_BASE_URL", f"{BASE_URL}/")
        cfg = KalshiConfig.from_env()
        assert not cfg.base_url.endswith("/")

    def test_invalid_rsa_key_pem_raises_at_client_init(self, env_vars, monkeypatch):
        """Bad PEM passes config parsing but raises ConfigurationError in KalshiClient()."""
        monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM", "not-a-valid-pem")
        cfg = KalshiConfig.from_env()
        with pytest.raises(ConfigurationError, match="RSA"):
            KalshiClient(cfg)


# ===========================================================================
# Authentication tests
# ===========================================================================

class TestAuthentication:
    def test_all_three_auth_headers_present(self, client):
        headers = client._auth_headers("GET", "/trade-api/v2/markets")
        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers

    def test_access_key_matches_config(self, client):
        headers = client._auth_headers("GET", "/trade-api/v2/markets")
        assert headers["KALSHI-ACCESS-KEY"] == "test-key-id-abc123"

    def test_timestamp_is_milliseconds_since_epoch(self, client):
        before_ms = int(time.time() * 1000)
        headers = client._auth_headers("GET", "/trade-api/v2/markets")
        after_ms = int(time.time() * 1000)
        ts = int(headers["KALSHI-ACCESS-TIMESTAMP"])
        assert before_ms <= ts <= after_ms

    def test_signature_is_valid_base64(self, client):
        headers = client._auth_headers("GET", "/trade-api/v2/markets")
        sig = headers["KALSHI-ACCESS-SIGNATURE"]
        decoded = base64.b64decode(sig)   # raises if not valid base64
        assert len(decoded) == 256        # RSA-2048 raw signature length

    def test_signature_verifiable_with_matching_public_key(
        self, client, rsa_private_key_pem
    ):
        """Signature produced by _sign() must verify under the corresponding public key."""
        private_key = serialization.load_pem_private_key(
            rsa_private_key_pem.encode(), password=None
        )
        public_key = private_key.public_key()

        ts = "1748736000000"
        method = "GET"
        path = "/trade-api/v2/markets"
        sig_b64 = client._sign(ts, method, path)
        sig_bytes = base64.b64decode(sig_b64)
        message = f"{ts}{method}{path}".encode("utf-8")

        # Raises InvalidSignature on failure — no assertion needed
        public_key.verify(
            sig_bytes,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )

    def test_query_string_excluded_from_signed_message(
        self, client, rsa_private_key_pem
    ):
        """
        Path query string must be stripped before signing.
        The signature must verify against the path-only message, not path+query.
        """
        private_key = serialization.load_pem_private_key(
            rsa_private_key_pem.encode(), password=None
        )
        public_key = private_key.public_key()

        ts = "1748736000000"
        method = "GET"
        path_with_query = "/trade-api/v2/markets?status=open&limit=1000"
        path_no_query = "/trade-api/v2/markets"

        sig_b64 = client._sign(ts, method, path_with_query)
        sig_bytes = base64.b64decode(sig_b64)
        # Correct message: no query string
        message = f"{ts}{method}{path_no_query}".encode("utf-8")

        public_key.verify(
            sig_bytes,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )

    def test_method_uppercased_in_signed_message(
        self, client, rsa_private_key_pem
    ):
        """Method must be uppercased when building the signature message."""
        private_key = serialization.load_pem_private_key(
            rsa_private_key_pem.encode(), password=None
        )
        public_key = private_key.public_key()

        ts = "1748736000000"
        path = "/trade-api/v2/portfolio/orders"
        # Pass lowercase method — _sign must uppercase it internally
        sig_b64 = client._sign(ts, "post", path)
        sig_bytes = base64.b64decode(sig_b64)
        message = f"{ts}POST{path}".encode("utf-8")   # uppercase POST

        public_key.verify(
            sig_bytes,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )


# ===========================================================================
# Retry / error handling tests
# ===========================================================================

class TestRetryAndErrors:
    @responses_mock.activate
    def test_429_retried_twice_then_succeeds(self, client):
        url = f"{API_ROOT}/markets"
        responses_mock.add(responses_mock.GET, url, json={"error": "rate limited"}, status=429)
        responses_mock.add(responses_mock.GET, url, json={"error": "rate limited"}, status=429)
        responses_mock.add(responses_mock.GET, url, json={"markets": [], "cursor": ""}, status=200)

        with patch("time.sleep") as mock_sleep:
            result = client.get_markets()
        assert result.success is True
        assert mock_sleep.call_count == 2

    @responses_mock.activate
    def test_500_retried_once_then_succeeds(self, client):
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        responses_mock.add(responses_mock.GET, url, json={"error": "server error"}, status=500)
        responses_mock.add(responses_mock.GET, url, json={"market": make_market()}, status=200)

        with patch("time.sleep") as mock_sleep:
            result = client.get_market(CONTRACT_ID)
        assert result.success is True
        assert mock_sleep.call_count == 1

    @responses_mock.activate
    def test_401_raises_authentication_error_without_retry(self, client):
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        responses_mock.add(responses_mock.GET, url, json={"error": "unauthorized"}, status=401)

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(AuthenticationError):
                client.get_market(CONTRACT_ID)
        mock_sleep.assert_not_called()

    @responses_mock.activate
    def test_403_raises_authentication_error_without_retry(self, client):
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        responses_mock.add(responses_mock.GET, url, json={"error": "forbidden"}, status=403)

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(AuthenticationError):
                client.get_market(CONTRACT_ID)
        mock_sleep.assert_not_called()

    @responses_mock.activate
    def test_404_raises_client_error_without_retry(self, client):
        url = f"{API_ROOT}/markets/UNKNOWN-TICKER"
        responses_mock.add(responses_mock.GET, url, json={"error": "not found"}, status=404)

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError):
                client.get_market("UNKNOWN-TICKER")
        mock_sleep.assert_not_called()

    @responses_mock.activate
    def test_422_raises_client_error_without_retry(self, client):
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        responses_mock.add(responses_mock.GET, url, json={"error": "unprocessable"}, status=422)

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError):
                client.get_market(CONTRACT_ID)
        mock_sleep.assert_not_called()

    @responses_mock.activate
    def test_429_exhausted_raises_rate_limit_error(self, client):
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        for _ in range(4):   # initial attempt + 3 retries
            responses_mock.add(
                responses_mock.GET, url, json={"error": "rate limited"}, status=429
            )
        with patch("time.sleep"):
            with pytest.raises(RateLimitError):
                client.get_market(CONTRACT_ID)

    @responses_mock.activate
    def test_500_exhausted_raises_server_error(self, client):
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        for _ in range(4):
            responses_mock.add(responses_mock.GET, url, json={"error": "oops"}, status=500)

        with patch("time.sleep"):
            with pytest.raises(ServerError):
                client.get_market(CONTRACT_ID)

    @responses_mock.activate
    def test_backoff_delays_match_spec(self, client):
        """Retry delays: base 1s, 4s, 16s ±20% jitter."""
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        for _ in range(4):
            responses_mock.add(responses_mock.GET, url, json={}, status=500)

        sleep_calls: list = []
        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            with pytest.raises(ServerError):
                client.get_market(CONTRACT_ID)

        assert len(sleep_calls) == 3
        assert 0.8 <= sleep_calls[0] <= 1.2     # 1s ±20%
        assert 3.2 <= sleep_calls[1] <= 4.8     # 4s ±20%
        assert 12.8 <= sleep_calls[2] <= 19.2   # 16s ±20%

    @responses_mock.activate
    def test_429_retry_after_header_respected(self, client):
        """When server returns Retry-After: 30, sleep must be >= 30 seconds."""
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        responses_mock.add(
            responses_mock.GET, url,
            json={"error": "rate limited"}, status=429,
            headers={"Retry-After": "30"},
        )
        responses_mock.add(responses_mock.GET, url, json={"market": make_market()}, status=200)

        sleep_calls: list = []
        with patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            result = client.get_market(CONTRACT_ID)
        assert result.success is True
        assert sleep_calls[0] >= 30.0

    @responses_mock.activate
    def test_non_json_success_response_raises_data_validation_error(self, client):
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        responses_mock.add(
            responses_mock.GET, url,
            body="<html>Error 502</html>",
            content_type="text/html",
            status=200,
        )
        with pytest.raises(DataValidationError):
            client.get_market(CONTRACT_ID)

    @responses_mock.activate
    def test_timeout_retried_then_raises_kalshi_timeout_error(self, client):
        import requests as req_lib
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        for _ in range(4):
            responses_mock.add(responses_mock.GET, url, body=req_lib.Timeout())

        with patch("time.sleep"):
            with pytest.raises(kc.TimeoutError):
                client.get_market(CONTRACT_ID)


# ===========================================================================
# Data integrity / validation tests
# ===========================================================================

class TestDataValidation:
    def test_yes_price_below_minimum_raises(self, client):
        raw = make_market()
        raw["last_price_dollars"] = "0.0001"
        raw["yes_bid_dollars"] = "0.0001"
        raw["yes_ask_dollars"] = "0.0001"
        with pytest.raises(DataValidationError):
            client._parse_contract(raw)

    def test_yes_price_above_maximum_raises(self, client):
        raw = make_market()
        raw["last_price_dollars"] = "0.9999"
        raw["yes_bid_dollars"] = "0.9999"
        raw["yes_ask_dollars"] = "0.9999"
        with pytest.raises(DataValidationError):
            client._parse_contract(raw)

    def test_spread_bids_not_summing_to_one_is_valid(self, client):
        """
        YES bid 0.54 + NO bid 0.44 = 0.98 ≠ 1.0 — valid spread scenario.
        We must NOT check bid sum; only individual price bounds matter.
        """
        raw = {
            "orderbook_fp": {
                "yes_dollars": [["0.5400", "100.00"]],
                "no_dollars": [["0.4400", "100.00"]],
            },
            "volume_fp": "1000.00",
        }
        snapshot = client._parse_price_snapshot(raw, CONTRACT_ID)
        assert snapshot.yes_price == pytest.approx(0.54)
        assert snapshot.no_price == pytest.approx(0.44)

    def test_zero_liquidity_raises(self, client):
        raw = make_market()
        raw["volume_fp"] = "0.00"
        with pytest.raises(DataValidationError, match="non-positive liquidity"):
            client._parse_contract(raw)

    def test_negative_liquidity_raises(self, client):
        raw = make_market()
        raw["volume_fp"] = "-1.00"
        with pytest.raises(DataValidationError, match="non-positive liquidity"):
            client._parse_contract(raw)

    def test_no_valid_price_field_raises(self, client):
        raw = make_market()
        del raw["last_price_dollars"]
        del raw["yes_bid_dollars"]
        del raw["yes_ask_dollars"]
        with pytest.raises(DataValidationError, match="no valid YES price"):
            client._parse_contract(raw)

    def test_past_resolution_date_warns_not_raises(self, client, caplog):
        """Unresolved contract with past expiry emits WARNING, does not raise."""
        raw = make_market()
        raw["latest_expiration_time"] = "2020-01-01T00:00:00Z"
        raw["close_time"] = "2020-01-01T00:00:00Z"
        # status=open means unresolved; make_market already sets this

        kalshi_logger = logging.getLogger("kalshi_client")
        kalshi_logger.addHandler(caplog.handler)
        caplog.set_level(logging.WARNING, logger="kalshi_client")
        try:
            contract = client._parse_contract(raw)
        finally:
            kalshi_logger.removeHandler(caplog.handler)

        assert isinstance(contract, ContractObject)
        assert any(
            "past" in r.message.lower() or "resolution" in r.message.lower()
            for r in caplog.records
        )

    def test_ingestion_timestamp_recorded_at_parse_time(self, client):
        """Point-in-time discipline: ingestion_timestamp must be set to parse time."""
        before = datetime.now(timezone.utc)
        raw = make_market()
        contract = client._parse_contract(raw)
        after = datetime.now(timezone.utc)
        assert before <= contract.ingestion_timestamp <= after


# ===========================================================================
# Parser unit tests
# ===========================================================================

class TestParsers:
    def test_parse_contract_core_fields(self, client):
        raw = make_market()
        c = client._parse_contract(raw)
        assert c.contract_id == CONTRACT_ID
        assert c.series_ticker == SERIES_TICKER
        assert c.sport == "NBA"
        assert c.yes_price == pytest.approx(0.55)
        assert c.no_price == pytest.approx(0.45)
        assert c.volume_fp == pytest.approx(1000.0)
        assert c.is_resolved is False
        assert c.resolution_outcome is None

    def test_parse_contract_resolved_yes(self, client):
        raw = make_market()
        raw["status"] = "finalized"   # market lifecycle status, not a boolean field
        raw["result"] = "yes"
        c = client._parse_contract(raw)
        assert c.is_resolved is True
        assert c.resolution_outcome == 1

    def test_parse_contract_resolved_no(self, client):
        raw = make_market()
        raw["result"] = "no"
        c = client._parse_contract(raw)
        assert c.resolution_outcome == 0

    def test_parse_contract_resolved_by_status_field(self, client):
        """is_resolved must derive from status enum, not a non-existent finalized boolean."""
        raw = make_market()
        raw["status"] = "determined"   # another valid resolved status
        c = client._parse_contract(raw)
        assert c.is_resolved is True

    def test_parse_contract_open_status_is_not_resolved(self, client):
        raw = make_market()
        raw["status"] = "open"
        c = client._parse_contract(raw)
        assert c.is_resolved is False

    def test_parse_contract_falls_back_to_yes_bid_when_no_last_price(self, client):
        raw = make_market()
        del raw["last_price_dollars"]
        c = client._parse_contract(raw)
        assert c.yes_price == pytest.approx(0.54)   # yes_bid_dollars

    def test_parse_trade_yes_direction(self):
        raw = make_trade_dict(count="5.00", price="0.5500", side="yes")
        trade = KalshiClient._parse_trade(raw, CONTRACT_ID)
        assert trade.contract_id == CONTRACT_ID
        assert trade.size_contracts == pytest.approx(5.0)
        assert trade.price == pytest.approx(0.55)
        assert trade.direction == "YES"
        assert trade.is_aggressive is True

    def test_parse_trade_no_direction(self):
        raw = make_trade_dict(count="3.00", price="0.4500", side="no")
        trade = KalshiClient._parse_trade(raw, CONTRACT_ID)
        assert trade.direction == "NO"
        assert trade.size_contracts == pytest.approx(3.0)

    def test_parse_trade_is_aggressive_false_when_taker_side_absent(self):
        """Missing taker_side → is_aggressive=False (not erroneously True via default 'yes')."""
        raw = make_trade_dict()
        del raw["taker_side"]
        trade = KalshiClient._parse_trade(raw, CONTRACT_ID)
        assert trade.is_aggressive is False

    def test_trade_has_size_contracts_not_size_dollars(self):
        """Renamed field: Trade.size_contracts (not .size_dollars)."""
        raw = make_trade_dict()
        trade = KalshiClient._parse_trade(raw, CONTRACT_ID)
        assert hasattr(trade, "size_contracts")
        assert not hasattr(trade, "size_dollars")

    def test_parse_position_yes_direction(self):
        pos = KalshiClient._parse_position(make_position_dict(pos_fp="10.00"))
        assert pos.direction == "YES"
        assert pos.size_contracts == pytest.approx(10.0)
        assert pos.market_exposure == pytest.approx(5.5)
        assert pos.realized_pnl == pytest.approx(0.25)

    def test_parse_position_no_direction(self):
        pos = KalshiClient._parse_position(make_position_dict(pos_fp="-7.00"))
        assert pos.direction == "NO"
        assert pos.size_contracts == pytest.approx(7.0)

    def test_parse_order_confirmation_executed(self):
        raw = make_order_response(status="executed")["order"]
        conf = KalshiClient._parse_order_confirmation(raw, CONTRACT_ID)
        assert conf.status == "executed"
        assert conf.action == "buy"
        assert conf.requested_count == pytest.approx(5.0)
        assert conf.filled_count == pytest.approx(5.0)
        assert conf.remaining_count == pytest.approx(0.0)
        assert conf.filled_price == pytest.approx(0.55)

    def test_parse_order_confirmation_resting_unfilled(self):
        raw = make_order_response(status="resting", fill_count="0.00")["order"]
        conf = KalshiClient._parse_order_confirmation(raw, CONTRACT_ID)
        assert conf.status == "resting"
        assert conf.filled_count == pytest.approx(0.0)

    def test_parse_order_status_alias_filled_maps_to_executed(self):
        """Legacy 'filled' status should map to 'executed'."""
        raw = make_order_response()["order"]
        raw["status"] = "filled"
        conf = KalshiClient._parse_order_confirmation(raw, CONTRACT_ID)
        assert conf.status == "executed"

    def test_parse_order_status_alias_cancelled_maps_to_canceled(self):
        raw = make_order_response()["order"]
        raw["status"] = "cancelled"
        conf = KalshiClient._parse_order_confirmation(raw, CONTRACT_ID)
        assert conf.status == "canceled"

    def test_candlestick_to_snapshot_fields(self):
        candle = make_candle(end_ts=1748736000, close="0.5500")
        snap = KalshiClient._candlestick_to_snapshot(candle, CONTRACT_ID)
        assert snap.contract_id == CONTRACT_ID
        assert snap.yes_price == pytest.approx(0.55)
        assert snap.no_price == pytest.approx(0.45)
        assert snap.volume_fp == pytest.approx(200.0)
        assert snap.book_depth_usd == pytest.approx(0.0)   # no bids in candle snapshots
        assert snap.bids == []
        assert snap.asks == []

    def test_extract_sport_teams_nba(self):
        sport, home, away = _extract_sport_teams(
            "Los Angeles Lakers vs Golden State Warriors",
            {"series_ticker": "NBAWIN"},
        )
        assert sport == "NBA"
        assert away == "Los Angeles Lakers"

    def test_extract_sport_teams_unknown(self):
        sport, home, away = _extract_sport_teams(
            "Some Unknown Event",
            {"series_ticker": "XYZ"},
        )
        assert sport == "UNKNOWN"

    def test_extract_sport_teams_ufc(self):
        """UFC home_team from yes_sub_title; away_team extracted from title matchup segment."""
        sport, home_team, away_team = _extract_sport_teams(
            "",
            {
                "series_ticker": "KXUFCFIGHT",
                "yes_sub_title": "Jon Jones",
                "title": "Will Jon Jones win the Jones vs Miocic professional MMA fight scheduled for Nov 16, 2024?",
            },
        )
        assert sport == "UFC"
        assert home_team == "Jon Jones"
        assert away_team == "Miocic"

    def test_ext_ufc_opponent_basic(self):
        """Radtke side: opponent fragment 'Nascimento de Souza' returned."""
        result = _extract_ufc_opponent(
            "Charles Radtke",
            "Will Charles Radtke win the Radtke vs Nascimento de Souza professional MMA fight scheduled for Apr 4, 2026?",
        )
        assert result == "Nascimento de Souza"

    def test_extract_ufc_opponent_other_side(self):
        """Nascimento de Souza side: opponent fragment 'Radtke' returned."""
        result = _extract_ufc_opponent(
            "Jose Henrique Nascimento de Souza",
            "Will Charles Radtke win the Radtke vs Nascimento de Souza professional MMA fight scheduled for Apr 4, 2026?",
        )
        assert result == "Radtke"

    def test_extract_ufc_opponent_no_vs_in_title(self):
        """Title with no ' vs ' → UNKNOWN."""
        result = _extract_ufc_opponent(
            "Jon Jones",
            "Will Jon Jones win the heavyweight championship bout?",
        )
        assert result == "UNKNOWN"

    def test_extract_ufc_opponent_empty_title(self):
        """Empty title → UNKNOWN."""
        result = _extract_ufc_opponent("Jon Jones", "")
        assert result == "UNKNOWN"

    def test_extract_sport_teams_ufc_full(self):
        """End-to-end: UFC raw dict produces correct home/away via title extraction."""
        sport, home_team, away_team = _extract_sport_teams(
            "",
            {
                "series_ticker": "KXUFCFIGHT",
                "yes_sub_title": "Charles Radtke",
                "title": "Will Charles Radtke win the Radtke vs Nascimento de Souza professional MMA fight scheduled for Apr 4, 2026?",
            },
        )
        assert sport == "UFC"
        assert home_team == "Charles Radtke"
        assert away_team == "Nascimento de Souza"

    def test_extract_sport_teams_ncaaf(self):
        """NCAAF detected via SERIES_TICKER_TO_SPORT, not subtitle keyword scan."""
        sport, home_team, away_team = _extract_sport_teams(
            "Alabama at Georgia",
            {"series_ticker": "KXNCAAFGAME"},
        )
        assert sport == "NCAAF"
        assert home_team == "Georgia"
        assert away_team == "Alabama"

    def test_extract_sport_teams_ncaab(self):
        """NCAAB detected via SERIES_TICKER_TO_SPORT, not subtitle keyword scan."""
        sport, home_team, away_team = _extract_sport_teams(
            "Duke at UNC",
            {"series_ticker": "KXNCAAMBGAME"},
        )
        assert sport == "NCAAB"
        assert home_team == "UNC"
        assert away_team == "Duke"

    def test_extract_sport_teams_from_title_at_separator(self):
        """title with ' at ' separator → away left, home right (full multi-word name)."""
        sport, home_team, away_team = _extract_sport_teams(
            "",
            {"series_ticker": "KXNFLGAME", "title": "Houston at New England Winner?"},
        )
        assert sport == "NFL"
        assert away_team == "Houston"
        assert home_team == "New England"

    def test_extract_sport_teams_from_title_vs_separator(self):
        """title with ' vs ' separator → away left, home right."""
        sport, home_team, away_team = _extract_sport_teams(
            "",
            {"series_ticker": "KXMLBGAME", "title": "Minnesota vs Kansas City Winner?"},
        )
        assert sport == "MLB"
        assert away_team == "Minnesota"
        assert home_team == "Kansas City"

    def test_extract_sport_teams_title_takes_priority_over_subtitle(self):
        """title is tried before subtitle; subtitle='' still yields correct teams from title."""
        sport, home_team, away_team = _extract_sport_teams(
            "",
            {"series_ticker": "KXNBAGAME", "title": "Denver at Utah Winner?"},
        )
        assert sport == "NBA"
        assert away_team == "Denver"
        assert home_team == "Utah"

    def test_extract_sport_teams_ufc_uses_title_for_opponent(self):
        """UFC away_team is extracted from the title matchup segment, not no_sub_title."""
        sport, home_team, away_team = _extract_sport_teams(
            "",
            {
                "series_ticker": "KXUFCFIGHT",
                "title": "Will Charles Radtke win the Radtke vs Nascimento de Souza professional MMA fight scheduled for Apr 4, 2026?",
                "yes_sub_title": "Charles Radtke",
                "no_sub_title": "Charles Radtke",  # real UFC: no_sub_title == yes_sub_title
            },
        )
        assert sport == "UFC"
        assert home_team == "Charles Radtke"
        assert away_team == "Nascimento de Souza"

    def test_extract_sport_teams_title_with_parentheses(self):
        """Team names with parentheses are stored as-is without stripping."""
        sport, home_team, away_team = _extract_sport_teams(
            "",
            {"series_ticker": "KXNCAAFGAME", "title": "Miami (FL) at Ole Miss Winner?"},
        )
        assert sport == "NCAAF"
        assert away_team == "Miami (FL)"
        assert home_team == "Ole Miss"

    def test_parse_contract_scalar_result_is_resolved(self, client):
        """result='scalar' on a finalized contract → is_resolved=True, resolution_outcome=None."""
        raw = make_market()
        raw["status"] = "finalized"
        raw["result"] = "scalar"
        c = client._parse_contract(raw)
        assert c.is_resolved is True
        assert c.resolution_outcome is None


# ===========================================================================
# _parse_market() tests  (Change 2)
# ===========================================================================

class TestParseMarket:
    """
    Tests for KalshiClient._parse_market().

    _parse_market() must parse ONLY fields present in the raw market dict from the
    API. It must not require or use injected fields (series_ticker, event_sub_title).
    It records ingestion_timestamp at parse time.
    """

    def test_returns_market_data_instance(self, client):
        raw = make_market_raw()
        result = client._parse_market(raw)
        assert isinstance(result, MarketData)

    def test_ticker_field_parsed(self, client):
        raw = make_market_raw(ticker="KXNBAGAME-26MAY28OKCSAS-SAS")
        result = client._parse_market(raw)
        assert result.ticker == "KXNBAGAME-26MAY28OKCSAS-SAS"

    def test_yes_and_no_price_derived_from_last_price(self, client):
        raw = make_market_raw()
        result = client._parse_market(raw)
        assert result.yes_price == pytest.approx(0.55)
        assert result.no_price == pytest.approx(0.45)

    def test_volume_fp_parsed_as_contract_count(self, client):
        raw = make_market_raw()
        result = client._parse_market(raw)
        assert result.volume_fp == pytest.approx(1000.0)
        assert result.volume_24h == pytest.approx(50.0)

    def test_yes_sub_title_and_no_sub_title_captured(self, client):
        raw = make_market_raw()
        result = client._parse_market(raw)
        assert result.yes_sub_title == "Oklahoma City Thunder"
        assert result.no_sub_title == "San Antonio Spurs"

    def test_previous_price_dollars_parsed_as_optional_float(self, client):
        raw = make_market_raw()
        result = client._parse_market(raw)
        assert result.previous_price_dollars == pytest.approx(0.48)

    def test_previous_price_dollars_none_when_absent(self, client):
        raw = make_market_raw()
        del raw["previous_price_dollars"]
        result = client._parse_market(raw)
        assert result.previous_price_dollars is None

    def test_open_interest_fp_parsed(self, client):
        raw = make_market_raw()
        result = client._parse_market(raw)
        assert result.open_interest_fp == pytest.approx(250.0)

    def test_ingestion_timestamp_recorded_at_parse_time(self, client):
        before = datetime.now(timezone.utc)
        raw = make_market_raw()
        result = client._parse_market(raw)
        after = datetime.now(timezone.utc)
        assert before <= result.ingestion_timestamp <= after

    def test_works_without_injected_fields(self, client):
        """
        _parse_market() must succeed on a raw dict that has no series_ticker,
        event_ticker, or event_sub_title keys — these belong on the event, not market.
        """
        raw = make_market_raw()
        assert "series_ticker" not in raw
        assert "event_sub_title" not in raw
        result = client._parse_market(raw)
        assert isinstance(result, MarketData)
        # event_ticker on market dict is empty string (API behavior)
        assert result.event_ticker == ""

    def test_parse_market_captures_title(self, client):
        """title field from the raw dict is stored on MarketData for downstream use."""
        raw = make_market_raw()
        raw["title"] = "Houston at New England Winner?"
        result = client._parse_market(raw)
        assert result.title == "Houston at New England Winner?"

    def test_invalid_price_raises_data_validation_error(self, client):
        raw = make_market_raw()
        raw["last_price_dollars"] = "invalid"
        raw["yes_bid_dollars"] = ""
        raw["yes_ask_dollars"] = ""
        with pytest.raises(DataValidationError):
            client._parse_market(raw)

    def test_zero_volume_parses_successfully(self, client):
        """
        _parse_market() must NOT raise on volume_fp=0.  Zero-volume contracts
        are valid API responses; the universe filter (is_liquid()) decides
        what to admit — the parser must faithfully represent the API payload.
        """
        raw = make_market_raw()
        raw["volume_fp"] = "0.00"
        result = client._parse_market(raw)
        assert isinstance(result, MarketData)
        assert result.volume_fp == pytest.approx(0.0)

    def test_parse_market_scalar_result_is_resolved(self, client):
        """result='scalar' on a finalized market → ContractObject.is_resolved=True."""
        raw = make_market_raw()
        raw["status"] = "finalized"
        raw["result"] = "scalar"
        md = client._parse_market(raw)
        contract = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-EV1",
            event_sub_title="OKC at SAS",
        )
        assert contract.is_resolved is True

    def test_parse_market_scalar_result_outcome_is_none(self, client):
        """result='scalar' → resolution_outcome is None, not 0 or 1."""
        raw = make_market_raw()
        raw["status"] = "finalized"
        raw["result"] = "scalar"
        md = client._parse_market(raw)
        contract = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-EV1",
            event_sub_title="OKC at SAS",
        )
        assert contract.resolution_outcome is None


# ===========================================================================
# _build_contract() tests  (Change 3)
# ===========================================================================

class TestBuildContract:
    """
    Tests for KalshiClient._build_contract().

    _build_contract() must source series_ticker, event_ticker, and sport/team fields
    exclusively from its explicit arguments — never from the MarketData object.
    """

    def _make_market_data(self, client: KalshiClient) -> MarketData:
        """Helper: parse a raw market dict into a MarketData for use in tests."""
        return client._parse_market(make_market_raw())

    def test_returns_contract_object(self, client):
        md = self._make_market_data(client)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert isinstance(result, ContractObject)

    def test_series_ticker_comes_from_argument_not_market(self, client):
        """
        The raw market dict has no series_ticker (or an empty one).
        ContractObject.series_ticker must come from the explicit argument.
        """
        md = self._make_market_data(client)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert result.series_ticker == "KXNBAGAME"

    def test_event_ticker_comes_from_argument_not_market(self, client):
        """
        market.event_ticker is "" (no event_ticker on market dicts in live API).
        ContractObject.event_ticker must come from the explicit argument.
        """
        md = self._make_market_data(client)
        assert md.event_ticker == ""   # confirm market dict had no event_ticker
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert result.event_ticker == "KXNBAGAME-26MAY28OKCSAS"

    def test_sport_extracted_from_event_sub_title(self, client):
        """Sport must be derived from event_sub_title + series_ticker argument."""
        md = self._make_market_data(client)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert result.sport == "NBA"

    def test_teams_extracted_from_event_sub_title(self, client):
        """away_team and home_team must come from event_sub_title, not yes/no_sub_title."""
        md = self._make_market_data(client)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        # "OKC at SAS (May 28)" → away=OKC, home=SAS
        assert result.away_team == "OKC"
        assert result.home_team == "SAS"

    def test_price_and_volume_preserved_from_market_data(self, client):
        md = self._make_market_data(client)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert result.yes_price == pytest.approx(md.yes_price)
        assert result.no_price == pytest.approx(md.no_price)
        assert result.volume_fp == pytest.approx(md.volume_fp)

    def test_open_yes_price_comes_from_previous_price_dollars(self, client):
        """previous_price_dollars in MarketData maps to open_yes_price in ContractObject."""
        md = self._make_market_data(client)
        assert md.previous_price_dollars == pytest.approx(0.48)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert result.open_yes_price == pytest.approx(0.48)

    def test_ingestion_timestamp_preserved_from_market_data(self, client):
        """Point-in-time discipline: ingestion_timestamp must not be reset in _build_contract."""
        md = self._make_market_data(client)
        original_ts = md.ingestion_timestamp
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert result.ingestion_timestamp == original_ts

    def test_build_contract_uses_title_for_teams(self, client):
        """MarketData.title is passed to _extract_sport_teams when event_sub_title is empty."""
        raw = make_market_raw()
        raw["title"] = "San Antonio at Golden State Winner?"
        md = client._parse_market(raw)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-EV1",
            event_sub_title="",
        )
        assert result.away_team == "San Antonio"
        assert result.home_team == "Golden State"

    def test_build_contract_event_sub_title_wins_over_title(self, client):
        """event_sub_title takes priority over MarketData.title for team extraction."""
        raw = make_market_raw()
        raw["title"] = "San Antonio at Golden State Winner?"
        md = client._parse_market(raw)
        result = client._build_contract(
            md,
            series_ticker="KXNBAGAME",
            event_ticker="KXNBAGAME-26MAY28OKCSAS",
            event_sub_title="OKC at SAS (May 28)",
        )
        assert result.away_team == "OKC"
        assert result.home_team == "SAS"


# ===========================================================================
# get_markets tests
# ===========================================================================

class TestGetMarkets:
    @responses_mock.activate
    def test_returns_list_of_contract_objects(self, client):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets",
            json={"markets": [make_market()], "cursor": ""},
            status=200,
        )
        result = client.get_markets()
        assert result.success is True
        assert isinstance(result.payload, list)
        assert len(result.payload) == 1
        assert isinstance(result.payload[0], ContractObject)
        assert result.payload[0].contract_id == CONTRACT_ID

    @responses_mock.activate
    def test_sport_filter_sends_series_ticker_query_param(self, client):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets",
            json={"markets": [make_market()], "cursor": ""},
            status=200,
        )
        client.get_markets(sport_filter="NBAWIN")
        req_url = responses_mock.calls[0].request.url
        assert "series_ticker=NBAWIN" in req_url

    @responses_mock.activate
    def test_cursor_pagination_collects_all_pages(self, client):
        url = f"{API_ROOT}/markets"
        responses_mock.add(responses_mock.GET, url, json={
            "markets": [make_market("TICKER-1")],
            "cursor": "page-2-token",
        }, status=200)
        responses_mock.add(responses_mock.GET, url, json={
            "markets": [make_market("TICKER-2")],
            "cursor": "",
        }, status=200)

        result = client.get_markets()
        assert len(result.payload) == 2
        ids = {c.contract_id for c in result.payload}
        assert ids == {"TICKER-1", "TICKER-2"}
        # Second call must include cursor param
        assert "cursor=page-2-token" in responses_mock.calls[1].request.url

    @responses_mock.activate
    def test_invalid_contract_excluded_valid_contract_included(self, client):
        bad = make_market("BAD-TICKER")
        bad["last_price_dollars"] = "invalid-price"
        bad["yes_bid_dollars"] = ""
        bad["yes_ask_dollars"] = ""
        good = make_market("GOOD-TICKER")
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets",
            json={"markets": [bad, good], "cursor": ""},
            status=200,
        )
        result = client.get_markets()
        assert len(result.payload) == 1
        assert result.payload[0].contract_id == "GOOD-TICKER"

    @responses_mock.activate
    def test_401_raises_authentication_error(self, client):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets",
            json={"error": "unauthorized"}, status=401,
        )
        with pytest.raises(AuthenticationError):
            client.get_markets()

    @responses_mock.activate
    def test_sends_status_open_query_param(self, client):
        """
        GET /markets status filter uses API enum open (not market field value active).
        OpenAPI MarketStatusQuery: unopened | open | paused | closed | settled.
        """
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets",
            json={"markets": [], "cursor": ""},
            status=200,
        )
        client.get_markets()
        req_url = responses_mock.calls[0].request.url
        assert "status=open" in req_url
        assert "status=active" not in req_url


# ===========================================================================
# get_market tests
# ===========================================================================

class TestGetMarket:
    @responses_mock.activate
    def test_returns_single_contract_object(self, client):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets/{CONTRACT_ID}",
            json={"market": make_market()},
            status=200,
        )
        result = client.get_market(CONTRACT_ID)
        assert result.success is True
        assert isinstance(result.payload, ContractObject)
        assert result.payload.contract_id == CONTRACT_ID
        assert result.payload.series_ticker == SERIES_TICKER

    @responses_mock.activate
    def test_404_raises_client_error(self, client):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets/UNKNOWN",
            json={"error": "not found"}, status=404,
        )
        with pytest.raises(ClientError):
            client.get_market("UNKNOWN")

    @responses_mock.activate
    def test_response_includes_positive_latency(self, client):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets/{CONTRACT_ID}",
            json={"market": make_market()}, status=200,
        )
        result = client.get_market(CONTRACT_ID)
        assert result.latency_ms >= 0


# ===========================================================================
# get_order_book tests
# ===========================================================================

class TestGetOrderBook:
    _OB_URL = f"{API_ROOT}/markets/{CONTRACT_ID}/orderbook"

    def _ob_body(
        self,
        yes_bids=None,
        no_bids=None,
    ) -> Dict[str, Any]:
        return {
            "orderbook_fp": {
                "yes_dollars": yes_bids or [["0.5400", "100.00"], ["0.5300", "200.00"]],
                "no_dollars": no_bids or [["0.4400", "150.00"], ["0.4300", "250.00"]],
            },
            "volume_fp": "1000.00",
            "volume_24h_fp": "50.00",
        }

    @responses_mock.activate
    def test_returns_price_snapshot(self, client):
        responses_mock.add(responses_mock.GET, self._OB_URL, json=self._ob_body(), status=200)
        result = client.get_order_book(CONTRACT_ID)
        assert result.success is True
        assert isinstance(result.payload, PriceSnapshot)

    @responses_mock.activate
    def test_yes_bids_sorted_best_price_first(self, client):
        responses_mock.add(responses_mock.GET, self._OB_URL, json=self._ob_body(), status=200)
        snap = client.get_order_book(CONTRACT_ID).payload
        assert len(snap.bids) >= 2
        assert snap.bids[0].price >= snap.bids[1].price

    @responses_mock.activate
    def test_asks_implied_from_no_bids_cheapest_first(self, client):
        """Implied YES ask = 1.0 - NO bid price; asks must be sorted cheapest first."""
        responses_mock.add(responses_mock.GET, self._OB_URL, json=self._ob_body(), status=200)
        snap = client.get_order_book(CONTRACT_ID).payload
        # Best NO bid is 0.44 → implied YES ask is 0.56
        assert snap.asks[0].price == pytest.approx(1.0 - 0.44, abs=1e-5)
        assert snap.asks[0].price <= snap.asks[1].price

    @responses_mock.activate
    def test_spread_bid_sum_below_one_is_valid(self, client):
        """YES bid 0.54 + NO bid 0.44 = 0.98 — realistic spread, no DataValidationError."""
        responses_mock.add(
            responses_mock.GET, self._OB_URL,
            json=self._ob_body(
                yes_bids=[["0.5400", "100.00"]],
                no_bids=[["0.4400", "100.00"]],
            ),
            status=200,
        )
        snap = client.get_order_book(CONTRACT_ID).payload
        assert snap.yes_price == pytest.approx(0.54)
        assert snap.no_price == pytest.approx(0.44)

    @responses_mock.activate
    def test_depth_param_sent(self, client):
        responses_mock.add(responses_mock.GET, self._OB_URL, json=self._ob_body(), status=200)
        client.get_order_book(CONTRACT_ID)
        assert "depth=" in responses_mock.calls[0].request.url


# ===========================================================================
# get_trade_history tests
# ===========================================================================

class TestGetTradeHistory:
    _TRADES_URL = f"{API_ROOT}/markets/trades"

    @responses_mock.activate
    def test_hits_markets_trades_path_not_per_market_path(self, client):
        """Correct path: /markets/trades — NOT /markets/{ticker}/trades."""
        responses_mock.add(
            responses_mock.GET, self._TRADES_URL,
            json={"trades": [make_trade_dict()], "cursor": ""},
            status=200,
        )
        client.get_trade_history(CONTRACT_ID, lookback_hours=24)
        req_url = responses_mock.calls[0].request.url
        assert "/markets/trades" in req_url
        # Verify it did NOT hit a per-market URL like /markets/NBAWIN-LAK-GSW.../trades
        assert f"/markets/{CONTRACT_ID}/trades" not in req_url

    @responses_mock.activate
    def test_ticker_query_param_is_contract_id(self, client):
        responses_mock.add(
            responses_mock.GET, self._TRADES_URL,
            json={"trades": [], "cursor": ""},
            status=200,
        )
        client.get_trade_history(CONTRACT_ID, lookback_hours=1)
        req_url = responses_mock.calls[0].request.url
        assert f"ticker={CONTRACT_ID}" in req_url

    @responses_mock.activate
    def test_min_ts_is_unix_integer_at_correct_lookback(self, client):
        responses_mock.add(
            responses_mock.GET, self._TRADES_URL,
            json={"trades": [], "cursor": ""},
            status=200,
        )
        before_call = int(time.time())
        client.get_trade_history(CONTRACT_ID, lookback_hours=24)
        after_call = int(time.time())

        parsed = urllib.parse.parse_qs(
            urllib.parse.urlparse(responses_mock.calls[0].request.url).query
        )
        min_ts = int(parsed["min_ts"][0])
        expected_low = before_call - 24 * 3600 - 5
        expected_high = after_call - 24 * 3600 + 5
        assert expected_low <= min_ts <= expected_high

    @responses_mock.activate
    def test_trades_sorted_oldest_first(self, client):
        t_old = make_trade_dict(created_time="2026-05-24T09:00:00Z")
        t_new = make_trade_dict(created_time="2026-05-24T11:00:00Z")
        responses_mock.add(
            responses_mock.GET, self._TRADES_URL,
            json={"trades": [t_new, t_old], "cursor": ""},
            status=200,
        )
        result = client.get_trade_history(CONTRACT_ID, lookback_hours=24)
        timestamps = [t.timestamp for t in result.payload]
        assert timestamps == sorted(timestamps)

    @responses_mock.activate
    def test_trade_field_is_size_contracts_not_size_dollars(self, client):
        responses_mock.add(
            responses_mock.GET, self._TRADES_URL,
            json={"trades": [make_trade_dict(count="7.00")], "cursor": ""},
            status=200,
        )
        result = client.get_trade_history(CONTRACT_ID, lookback_hours=1)
        trade = result.payload[0]
        assert trade.size_contracts == pytest.approx(7.0)
        assert not hasattr(trade, "size_dollars")

    @responses_mock.activate
    def test_cursor_pagination_collects_all_trades(self, client):
        t1 = make_trade_dict(created_time="2026-05-24T09:00:00Z")
        t2 = make_trade_dict(created_time="2026-05-24T10:00:00Z")
        responses_mock.add(
            responses_mock.GET, self._TRADES_URL,
            json={"trades": [t1], "cursor": "next-page"},
            status=200,
        )
        responses_mock.add(
            responses_mock.GET, self._TRADES_URL,
            json={"trades": [t2], "cursor": ""},
            status=200,
        )
        result = client.get_trade_history(CONTRACT_ID, lookback_hours=24)
        assert len(result.payload) == 2


# ===========================================================================
# get_price_history tests
# ===========================================================================

class TestGetPriceHistory:
    _CANDLES_URL = (
        f"{API_ROOT}/series/{SERIES_TICKER}/markets/{CONTRACT_ID}/candlesticks"
    )

    @responses_mock.activate
    def test_hits_candlesticks_path_with_series_ticker(self, client):
        responses_mock.add(
            responses_mock.GET, self._CANDLES_URL,
            json={"candlesticks": [make_candle()]},
            status=200,
        )
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        client.get_price_history(CONTRACT_ID, SERIES_TICKER, start_timestamp=start)

        req_url = responses_mock.calls[0].request.url
        assert f"/series/{SERIES_TICKER}/markets/{CONTRACT_ID}/candlesticks" in req_url

    def test_invalid_period_interval_raises_value_error(self, client):
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="period_interval"):
            client.get_price_history(
                CONTRACT_ID, SERIES_TICKER,
                start_timestamp=start,
                period_interval=30,   # invalid; valid values: 1, 60, 1440
            )

    @responses_mock.activate
    def test_period_interval_sent_as_query_param(self, client):
        responses_mock.add(
            responses_mock.GET, self._CANDLES_URL,
            json={"candlesticks": [make_candle()]},
            status=200,
        )
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        client.get_price_history(
            CONTRACT_ID, SERIES_TICKER, start_timestamp=start, period_interval=60
        )
        assert "period_interval=60" in responses_mock.calls[0].request.url

    @responses_mock.activate
    def test_snapshots_sorted_oldest_first(self, client):
        c1 = make_candle(end_ts=1748736000, close="0.5000")
        c2 = make_candle(end_ts=1748822400, close="0.6000")
        responses_mock.add(
            responses_mock.GET, self._CANDLES_URL,
            json={"candlesticks": [c2, c1]},   # reversed order from API
            status=200,
        )
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        result = client.get_price_history(CONTRACT_ID, SERIES_TICKER, start_timestamp=start)
        timestamps = [s.timestamp for s in result.payload]
        assert timestamps == sorted(timestamps)

    @responses_mock.activate
    def test_candle_snapshots_have_empty_bids_and_asks(self, client):
        """Candlestick endpoint returns no order book depth — bids/asks must be empty."""
        responses_mock.add(
            responses_mock.GET, self._CANDLES_URL,
            json={"candlesticks": [make_candle()]},
            status=200,
        )
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        result = client.get_price_history(CONTRACT_ID, SERIES_TICKER, start_timestamp=start)
        assert result.payload[0].bids == []
        assert result.payload[0].asks == []

    @responses_mock.activate
    def test_all_three_valid_period_intervals_accepted(self, client):
        for interval in (1, 60, 1440):
            responses_mock.add(
                responses_mock.GET, self._CANDLES_URL,
                json={"candlesticks": [make_candle()]},
                status=200,
            )
            start = datetime(2026, 5, 1, tzinfo=timezone.utc)
            result = client.get_price_history(
                CONTRACT_ID, SERIES_TICKER,
                start_timestamp=start,
                period_interval=interval,
            )
            assert result.success is True

    def test_candlestick_empty_price_uses_mid_of_bid_ask(self, client):
        """
        Zero-volume day: price object is {}, but yes_bid.close_dollars and
        yes_ask.close_dollars are present. yes_price must be their mid.
        Mid of 0.53 and 0.57 = 0.55 exactly.
        """
        candle = make_candle(close=None)   # price={}, yes_bid=0.53, yes_ask=0.57
        snap = KalshiClient._candlestick_to_snapshot(candle, CONTRACT_ID)
        assert snap is not None
        assert snap.yes_price == pytest.approx(0.55)

    def test_candlestick_empty_price_no_bid_ask_returns_none(self, client):
        """
        Empty price object AND no yes_bid/yes_ask → return None.
        Caller must skip; defaulting to 0.5 would corrupt momentum fade signal.
        """
        candle = {
            "end_period_ts": 1748736000,
            "price": {},
            "volume_fp": "0",
            # no yes_bid, no yes_ask
        }
        snap = KalshiClient._candlestick_to_snapshot(candle, CONTRACT_ID)
        assert snap is None

    def test_candlestick_with_close_dollars_uses_close(self, client):
        """Normal candle: price.close_dollars present → use it directly."""
        candle = make_candle(close="0.6200")
        snap = KalshiClient._candlestick_to_snapshot(candle, CONTRACT_ID)
        assert snap is not None
        assert snap.yes_price == pytest.approx(0.62)

    @responses_mock.activate
    def test_get_price_history_skips_none_snapshots(self, client):
        """
        Snapshots where _candlestick_to_snapshot returns None (zero-volume,
        no bid/ask) must be excluded from the returned list — not included as
        sentinel values or raising exceptions.
        """
        good_candle = make_candle(close="0.5500")
        bad_candle = {
            "end_period_ts": 1748822400,
            "price": {},
            "volume_fp": "0",
            # no yes_bid, no yes_ask → returns None
        }
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/series/{SERIES_TICKER}/markets/{CONTRACT_ID}/candlesticks",
            json={"candlesticks": [good_candle, bad_candle]},
            status=200,
        )
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        result = client.get_price_history(CONTRACT_ID, SERIES_TICKER, start_timestamp=start)
        assert len(result.payload) == 1
        assert result.payload[0].yes_price == pytest.approx(0.55)


# ===========================================================================
# place_order tests
# ===========================================================================

class TestPlaceOrder:
    _ORDER_URL = f"{API_ROOT}/portfolio/orders"

    @responses_mock.activate
    def test_limit_order_body_contains_required_fields(self, client):
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(),
            status=201,
        )
        client.place_order(
            CONTRACT_ID, direction="YES", action="buy",
            count=5.0, limit_price=0.55,
        )
        body = json.loads(responses_mock.calls[0].request.body)
        assert body["ticker"] == CONTRACT_ID
        assert body["side"] == "yes"
        assert body["action"] == "buy"
        assert body["count_fp"] == "5.00"
        assert "yes_price_dollars" in body
        assert body["time_in_force"] == "good_till_cancelled"
        assert "buy_max_cost" not in body

    @responses_mock.activate
    def test_market_order_body_uses_buy_max_cost(self, client):
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(),
            status=201,
        )
        client.place_order(
            CONTRACT_ID, direction="YES", action="buy",
            count=5.0, buy_max_cost=600,
        )
        body = json.loads(responses_mock.calls[0].request.body)
        assert body["buy_max_cost"] == 600
        assert body["time_in_force"] == "fill_or_kill"
        assert "yes_price_dollars" not in body

    @responses_mock.activate
    def test_count_fp_is_contracts_not_dollars_times_100(self, client):
        """5 contracts → count_fp='5.00', NOT '500' (old dollar-cent encoding)."""
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(),
            status=201,
        )
        client.place_order(
            CONTRACT_ID, direction="YES", action="buy",
            count=5.0, limit_price=0.55,
        )
        body = json.loads(responses_mock.calls[0].request.body)
        assert body["count_fp"] == "5.00"
        assert body["count_fp"] != "500"

    @responses_mock.activate
    def test_client_order_id_is_uuid4(self, client):
        import re
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(),
            status=201,
        )
        client.place_order(
            CONTRACT_ID, direction="YES", action="buy",
            count=5.0, limit_price=0.55,
        )
        body = json.loads(responses_mock.calls[0].request.body)
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(pattern, body["client_order_id"])

    @responses_mock.activate
    def test_no_limit_order_sends_no_price_dollars(self, client):
        """NO limit at limit_price (YES-expressed) → no_price_dollars = 1 - limit_price."""
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(),
            status=201,
        )
        client.place_order(
            CONTRACT_ID, direction="NO", action="buy",
            count=5.0, limit_price=0.55,
        )
        body = json.loads(responses_mock.calls[0].request.body)
        assert "no_price_dollars" in body
        assert "yes_price_dollars" not in body
        # NO price = 1 - 0.55 = 0.45
        assert abs(float(body["no_price_dollars"]) - 0.45) < 1e-4

    def test_market_sell_without_limit_raises_value_error(self, client):
        """Market sells are unsupported — no buy_max_cost equivalent for the sell side."""
        with pytest.raises(ValueError, match="Market sell"):
            client.place_order(CONTRACT_ID, direction="YES", action="sell", count=5.0)

    @responses_mock.activate
    def test_yes_limit_order_sends_yes_price_dollars(self, client):
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(),
            status=201,
        )
        client.place_order(
            CONTRACT_ID, direction="YES", action="buy",
            count=5.0, limit_price=0.60,
        )
        body = json.loads(responses_mock.calls[0].request.body)
        assert "yes_price_dollars" in body
        assert "no_price_dollars" not in body
        assert body["yes_price_dollars"] == "0.6000"

    def test_invalid_direction_raises_value_error(self, client):
        with pytest.raises(ValueError, match="direction"):
            client.place_order(CONTRACT_ID, direction="BOTH", action="buy", count=5.0)

    def test_invalid_action_raises_value_error(self, client):
        with pytest.raises(ValueError, match="action"):
            client.place_order(CONTRACT_ID, direction="YES", action="hold", count=5.0)

    @responses_mock.activate
    def test_executed_order_returns_order_confirmation(self, client):
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(status="executed"),
            status=201,
        )
        result = client.place_order(
            CONTRACT_ID, direction="YES", action="buy",
            count=5.0, limit_price=0.55,
        )
        assert isinstance(result.payload, OrderConfirmation)
        assert result.payload.status == "executed"
        assert result.payload.action == "buy"
        assert result.payload.filled_count == pytest.approx(5.0)
        assert result.payload.requested_count == pytest.approx(5.0)

    @responses_mock.activate
    def test_partial_fill_raises_partial_fill_error(self, client):
        """status=resting with 0 < fill_count < initial_count → PartialFillError."""
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(status="resting", fill_count="2.00", initial_count="5.00"),
            status=201,
        )
        with pytest.raises(PartialFillError) as exc_info:
            client.place_order(
                CONTRACT_ID, direction="YES", action="buy",
                count=5.0, limit_price=0.55,
            )
        err = exc_info.value
        assert err.filled_count == pytest.approx(2.0)
        assert err.contract_id == CONTRACT_ID
        assert err.filled_order_id == "ord-abc123"

    @responses_mock.activate
    def test_resting_with_zero_fill_does_not_raise_partial_fill_error(self, client):
        """status=resting, fill_count=0 (completely unfilled) — no PartialFillError."""
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(status="resting", fill_count="0.00", initial_count="5.00"),
            status=201,
        )
        result = client.place_order(
            CONTRACT_ID, direction="YES", action="buy",
            count=5.0, limit_price=0.55,
        )
        assert result.payload.status == "resting"

    @responses_mock.activate
    def test_market_order_default_buy_max_cost_equals_count_cents(self, client):
        """If buy_max_cost omitted, default = int(count * 100) cents."""
        responses_mock.add(
            responses_mock.POST, self._ORDER_URL,
            json=make_order_response(),
            status=201,
        )
        client.place_order(CONTRACT_ID, direction="NO", action="buy", count=7.0)
        body = json.loads(responses_mock.calls[0].request.body)
        assert body["buy_max_cost"] == 700   # int(7.0 * 100)


# ===========================================================================
# get_positions tests
# ===========================================================================

class TestGetPositions:
    _POSITIONS_URL = f"{API_ROOT}/portfolio/positions"

    @responses_mock.activate
    def test_returns_list_of_position_objects(self, client):
        responses_mock.add(
            responses_mock.GET, self._POSITIONS_URL,
            json={"market_positions": [make_position_dict()], "cursor": ""},
            status=200,
        )
        result = client.get_positions()
        assert result.success is True
        assert len(result.payload) == 1
        assert isinstance(result.payload[0], Position)

    @responses_mock.activate
    def test_positive_position_fp_maps_to_yes_direction(self, client):
        responses_mock.add(
            responses_mock.GET, self._POSITIONS_URL,
            json={"market_positions": [make_position_dict(pos_fp="10.00")], "cursor": ""},
            status=200,
        )
        pos = client.get_positions().payload[0]
        assert pos.direction == "YES"
        assert pos.size_contracts == pytest.approx(10.0)

    @responses_mock.activate
    def test_negative_position_fp_maps_to_no_direction(self, client):
        responses_mock.add(
            responses_mock.GET, self._POSITIONS_URL,
            json={"market_positions": [make_position_dict(pos_fp="-5.00")], "cursor": ""},
            status=200,
        )
        pos = client.get_positions().payload[0]
        assert pos.direction == "NO"
        assert pos.size_contracts == pytest.approx(5.0)

    @responses_mock.activate
    def test_position_dollar_fields_mapped_correctly(self, client):
        responses_mock.add(
            responses_mock.GET, self._POSITIONS_URL,
            json={"market_positions": [make_position_dict()], "cursor": ""},
            status=200,
        )
        pos = client.get_positions().payload[0]
        assert pos.market_exposure == pytest.approx(5.5)
        assert pos.realized_pnl == pytest.approx(0.25)
        assert pos.contract_id == CONTRACT_ID

    @responses_mock.activate
    def test_cursor_pagination_collects_all_positions(self, client):
        responses_mock.add(
            responses_mock.GET, self._POSITIONS_URL,
            json={"market_positions": [make_position_dict("TICKER-1")], "cursor": "p2"},
            status=200,
        )
        responses_mock.add(
            responses_mock.GET, self._POSITIONS_URL,
            json={"market_positions": [make_position_dict("TICKER-2")], "cursor": ""},
            status=200,
        )
        result = client.get_positions()
        assert len(result.payload) == 2
        tickers = {p.contract_id for p in result.payload}
        assert tickers == {"TICKER-1", "TICKER-2"}

    @responses_mock.activate
    def test_401_raises_authentication_error(self, client):
        responses_mock.add(
            responses_mock.GET, self._POSITIONS_URL,
            json={"error": "unauthorized"}, status=401,
        )
        with pytest.raises(AuthenticationError):
            client.get_positions()


# ===========================================================================
# get_events tests  (Change 2 — primary sports ingestion)
# ===========================================================================

class TestGetEvents:
    _EVENTS_URL = f"{API_ROOT}/events"

    @responses_mock.activate
    def test_returns_list_of_event_data(self, client):
        """Successful response returns APIResponse wrapping List[EventData]."""
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [make_event()], "cursor": ""},
            status=200,
        )
        result = client.get_events(series_ticker="KXNBAGAME")
        assert result.success is True
        assert isinstance(result.payload, list)
        assert len(result.payload) == 1
        assert isinstance(result.payload[0], EventData)

    @responses_mock.activate
    def test_event_data_fields_populated_correctly(self, client):
        """EventData carries event-level metadata from the response."""
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [make_event()], "cursor": ""},
            status=200,
        )
        event = client.get_events("KXNBAGAME").payload[0]
        assert event.event_ticker == "KXNBAGAME-26MAY28OKCSAS"
        assert event.series_ticker == "KXNBAGAME"
        assert event.sub_title == "OKC at SAS (May 28)"
        assert event.mutually_exclusive is True

    @responses_mock.activate
    def test_injects_series_ticker_into_nested_contracts(self, client):
        """
        Market objects nested inside events have empty series_ticker in the real API.
        get_events() must inject series_ticker from the event level so that
        get_price_history() can be called on each returned ContractObject.
        """
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [make_event(series_ticker="KXNBAGAME")], "cursor": ""},
            status=200,
        )
        event = client.get_events("KXNBAGAME").payload[0]
        assert len(event.markets) >= 1
        for contract in event.markets:
            assert contract.series_ticker == "KXNBAGAME"

    @responses_mock.activate
    def test_injects_event_ticker_into_nested_contracts(self, client):
        """Each ContractObject must carry the parent event_ticker for grouping."""
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [make_event(event_ticker="KXNBAGAME-26MAY28OKCSAS")], "cursor": ""},
            status=200,
        )
        event = client.get_events("KXNBAGAME").payload[0]
        for contract in event.markets:
            assert contract.event_ticker == "KXNBAGAME-26MAY28OKCSAS"

    @responses_mock.activate
    def test_cursor_pagination_collects_all_events(self, client):
        """Two-page response: all events from both pages are returned."""
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [make_event("EV-1")], "cursor": "pg2"},
            status=200,
        )
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [make_event("EV-2")], "cursor": ""},
            status=200,
        )
        result = client.get_events("KXNBAGAME")
        assert len(result.payload) == 2
        tickers = {e.event_ticker for e in result.payload}
        assert tickers == {"EV-1", "EV-2"}
        assert "cursor=pg2" in responses_mock.calls[1].request.url

    @responses_mock.activate
    def test_sends_status_open(self, client):
        """GET /events status filter must use API enum open (not market field active)."""
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [], "cursor": ""},
            status=200,
        )
        client.get_events("KXNBAGAME")
        assert "status=open" in responses_mock.calls[0].request.url
        assert "status=active" not in responses_mock.calls[0].request.url

    @responses_mock.activate
    def test_sends_with_nested_markets_true(self, client):
        """with_nested_markets=true must be sent so market objects are included."""
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [], "cursor": ""},
            status=200,
        )
        client.get_events("KXNBAGAME")
        assert "with_nested_markets=true" in responses_mock.calls[0].request.url

    @responses_mock.activate
    def test_401_raises_authentication_error(self, client):
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"error": "unauthorized"}, status=401,
        )
        with pytest.raises(AuthenticationError):
            client.get_events("KXNBAGAME")

    @responses_mock.activate
    def test_invalid_nested_market_excluded_valid_included(self, client):
        """Corrupt market nested inside event is excluded; valid ones are included."""
        event = make_event()
        bad_market = make_market(ticker="BAD-MKT")
        bad_market["last_price_dollars"] = "invalid"
        bad_market["yes_bid_dollars"] = ""
        bad_market["yes_ask_dollars"] = ""
        event["markets"].append(bad_market)
        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [event], "cursor": ""},
            status=200,
        )
        result = client.get_events("KXNBAGAME")
        event_data = result.payload[0]
        contract_ids = {c.contract_id for c in event_data.markets}
        assert "BAD-MKT" not in contract_ids
        assert len(event_data.markets) == 1  # only the valid make_market() contract

    @responses_mock.activate
    def test_get_events_does_not_mutate_raw_market_dicts(self, client):
        """
        get_events() must not write keys into raw market dicts (old implementation
        injected series_ticker, event_ticker, and event_sub_title). Verify by spying
        on _parse_market() and checking that event_sub_title is absent from its arg.

        This validates the architectural boundary: event-level fields are passed as
        explicit arguments to _build_contract(), not smuggled into the market dict.
        """
        seen_raw_keys: List[frozenset] = []
        real_parse_market = client._parse_market

        def spy(raw: Dict[str, Any]) -> MarketData:
            seen_raw_keys.append(frozenset(raw.keys()))
            return real_parse_market(raw)

        responses_mock.add(
            responses_mock.GET, self._EVENTS_URL,
            json={"events": [make_event()], "cursor": ""},
            status=200,
        )
        with patch.object(client, "_parse_market", side_effect=spy):
            client.get_events("KXNBAGAME")

        assert len(seen_raw_keys) >= 1, "Expected _parse_market to be called at least once"
        for keys in seen_raw_keys:
            assert "event_sub_title" not in keys, (
                f"_parse_market received event_sub_title in its arg — "
                f"dict was mutated before calling it. Keys: {keys}"
            )


# ===========================================================================
# Error class structure tests
# ===========================================================================

class TestErrorClasses:
    def test_authentication_error_stores_status_code(self):
        exc = AuthenticationError("auth failed", status_code=401)
        assert exc.status_code == 401
        assert "auth failed" in str(exc)

    def test_rate_limit_error_stores_retry_after(self):
        exc = RateLimitError("rate limited", retry_after=30, status_code=429)
        assert exc.retry_after == 30
        assert exc.status_code == 429

    def test_partial_fill_error_has_filled_count_not_filled_size(self):
        """Renamed field: PartialFillError.filled_count (not .filled_size)."""
        exc = PartialFillError(
            "partial fill",
            filled_order_id="ord-1",
            filled_count=2.0,
            contract_id=CONTRACT_ID,
        )
        assert exc.filled_count == 2.0
        assert exc.contract_id == CONTRACT_ID
        assert exc.filled_order_id == "ord-1"
        assert not hasattr(exc, "filled_size")

    def test_position_mismatch_error_stores_sizes(self):
        exc = PositionMismatchError(
            "mismatch",
            contract_id=CONTRACT_ID,
            internal_size=10.0,
            reported_size=8.0,
        )
        assert exc.contract_id == CONTRACT_ID
        assert exc.internal_size == 10.0
        assert exc.reported_size == 8.0

    def test_data_validation_error_stores_raw_response(self):
        exc = DataValidationError(
            "bad data",
            raw_response='{"err": "x"}',
            contract_id=CONTRACT_ID,
        )
        assert exc.raw_response == '{"err": "x"}'
        assert exc.contract_id == CONTRACT_ID

    def test_full_error_hierarchy(self):
        assert issubclass(AuthenticationError, kc.APIError)
        assert issubclass(RateLimitError, kc.APIError)
        assert issubclass(ServerError, kc.APIError)
        assert issubclass(kc.TimeoutError, kc.APIError)
        assert issubclass(ClientError, kc.APIError)
        assert issubclass(kc.APIError, kc.PredictionMarketError)
        assert issubclass(DataValidationError, kc.PredictionMarketError)
        assert issubclass(PartialFillError, kc.ExecutionError)
        assert issubclass(PositionMismatchError, kc.ExecutionError)
        assert issubclass(kc.ExecutionError, kc.PredictionMarketError)
        assert issubclass(kc.ConfigurationError, kc.PredictionMarketError)


# ===========================================================================
# Logging tests
# ===========================================================================

class TestLogging:
    @responses_mock.activate
    def test_successful_api_call_emits_log_entry(self, client, caplog):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets/{CONTRACT_ID}",
            json={"market": make_market()}, status=200,
        )
        kalshi_logger = logging.getLogger("kalshi_client")
        kalshi_logger.addHandler(caplog.handler)
        caplog.set_level(logging.DEBUG, logger="kalshi_client")
        try:
            client.get_market(CONTRACT_ID)
        finally:
            kalshi_logger.removeHandler(caplog.handler)
        assert len(caplog.records) > 0

    @responses_mock.activate
    def test_retried_success_logs_at_warning(self, client, caplog):
        """When a call fails once then succeeds, the success entry is logged at WARNING
        (because retry_count > 0)."""
        url = f"{API_ROOT}/markets/{CONTRACT_ID}"
        responses_mock.add(responses_mock.GET, url, json={}, status=500)
        responses_mock.add(responses_mock.GET, url, json={"market": make_market()}, status=200)

        kalshi_logger = logging.getLogger("kalshi_client")
        kalshi_logger.addHandler(caplog.handler)
        caplog.set_level(logging.WARNING, logger="kalshi_client")
        try:
            with patch("time.sleep"):
                client.get_market(CONTRACT_ID)
        finally:
            kalshi_logger.removeHandler(caplog.handler)
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1

    @responses_mock.activate
    def test_authentication_failure_logs_at_critical(self, client, caplog):
        responses_mock.add(
            responses_mock.GET, f"{API_ROOT}/markets/{CONTRACT_ID}",
            json={"error": "unauthorized"}, status=401,
        )
        kalshi_logger = logging.getLogger("kalshi_client")
        kalshi_logger.addHandler(caplog.handler)
        caplog.set_level(logging.CRITICAL, logger="kalshi_client")
        try:
            with pytest.raises(AuthenticationError):
                client.get_market(CONTRACT_ID)
        finally:
            kalshi_logger.removeHandler(caplog.handler)
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) >= 1


# ===========================================================================
# get_balance tests  (Fix 1)
# ===========================================================================

class TestGetBalance:
    _BALANCE_URL = f"{API_ROOT}/portfolio/balance"

    @responses_mock.activate
    def test_returns_balance_as_float(self, client):
        """Successful response: balance_dollars parsed to float payload."""
        responses_mock.add(
            responses_mock.GET, self._BALANCE_URL,
            json={"balance_dollars": "1250.7500"},
            status=200,
        )
        result = client.get_balance()
        assert result.success is True
        assert result.payload == pytest.approx(1250.75)
        assert result.http_status_code == 200

    @responses_mock.activate
    def test_auth_failure_raises_authentication_error(self, client):
        """401 from balance endpoint raises AuthenticationError without retry."""
        responses_mock.add(
            responses_mock.GET, self._BALANCE_URL,
            json={"error": "unauthorized"}, status=401,
        )
        with pytest.raises(AuthenticationError):
            client.get_balance()

    @responses_mock.activate
    def test_missing_balance_field_returns_zero_and_warns(self, client, caplog):
        """
        If balance_dollars is absent from the response, payload is 0.0 and a
        WARNING is emitted — not a silent zero or an exception.
        """
        responses_mock.add(
            responses_mock.GET, self._BALANCE_URL,
            json={"something_else": "irrelevant"},
            status=200,
        )
        kalshi_logger = logging.getLogger("kalshi_client")
        kalshi_logger.addHandler(caplog.handler)
        caplog.set_level(logging.WARNING, logger="kalshi_client")
        try:
            result = client.get_balance()
        finally:
            kalshi_logger.removeHandler(caplog.handler)
        assert result.payload == pytest.approx(0.0)
        assert any("balance_dollars" in r.message for r in caplog.records)

    @responses_mock.activate
    def test_malformed_balance_field_raises_data_validation_error(self, client):
        """balance_dollars present but not a valid decimal → DataValidationError."""
        responses_mock.add(
            responses_mock.GET, self._BALANCE_URL,
            json={"balance_dollars": "N/A"},
            status=200,
        )
        with pytest.raises(DataValidationError, match="malformed balance_dollars"):
            client.get_balance()


# ===========================================================================
# Additional targeted tests for Fixes 3–6
# ===========================================================================

class TestFix3BookDepthUsd:
    """Fix 3 — book_depth_usd computed as Σ(price × size) across YES bid levels."""

    _OB_URL = f"{API_ROOT}/markets/{CONTRACT_ID}/orderbook"

    def _ob_body(self, yes_bids=None, no_bids=None) -> Dict[str, Any]:
        return {
            "orderbook_fp": {
                "yes_dollars": yes_bids or [],
                "no_dollars": no_bids or [["0.4400", "100.00"]],
            },
            "volume_fp": "1000.00",
        }

    @responses_mock.activate
    def test_book_depth_usd_is_sum_of_price_times_size(self, client):
        """
        Two YES bid levels: 0.54 × 100 + 0.53 × 200 = 54 + 106 = 160.0
        book_depth_usd must equal exactly that sum.
        """
        responses_mock.add(
            responses_mock.GET, self._OB_URL,
            json=self._ob_body(
                yes_bids=[["0.5400", "100.00"], ["0.5300", "200.00"]],
                no_bids=[["0.4400", "100.00"]],
            ),
            status=200,
        )
        snap = client.get_order_book(CONTRACT_ID).payload
        expected = 0.54 * 100.0 + 0.53 * 200.0
        assert snap.book_depth_usd == pytest.approx(expected)

    @responses_mock.activate
    def test_book_depth_usd_zero_when_no_bids(self, client):
        """Empty YES bid list → book_depth_usd = 0.0."""
        responses_mock.add(
            responses_mock.GET, self._OB_URL,
            json=self._ob_body(yes_bids=[], no_bids=[["0.4400", "100.00"]]),
            status=200,
        )
        snap = client.get_order_book(CONTRACT_ID).payload
        assert snap.book_depth_usd == pytest.approx(0.0)


class TestFix4SeriesTickerGuard:
    """Fix 4 — empty series_ticker raises DataValidationError before URL construction."""

    def test_empty_series_ticker_raises_data_validation_error(self, client):
        """
        Calling get_price_history() with series_ticker='' must raise
        DataValidationError immediately — no HTTP call is made.
        """
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        with pytest.raises(DataValidationError, match="series_ticker is required"):
            client.get_price_history(CONTRACT_ID, "", start_timestamp=start)

    def test_error_message_contains_contract_id(self, client):
        """The DataValidationError message must name the offending contract."""
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        with pytest.raises(DataValidationError, match=CONTRACT_ID):
            client.get_price_history(CONTRACT_ID, "", start_timestamp=start)


class TestFix5BaseUrlWarning:
    """
    Fix 5 — KalshiClient.__init__() emits WARNING when KALSHI_BASE_URL is not
    one of the two documented Kalshi hosts.

    The warning fires AFTER _setup_logging() (which calls logger.handlers.clear()).
    To capture it with caplog we patch _setup_logging to inject caplog.handler
    into the returned logger before control returns to __init__().
    """

    @staticmethod
    def _make_caplog_injector(caplog) -> Any:
        """Return a wrapper around kc._setup_logging that appends caplog.handler."""
        _real_setup = kc._setup_logging

        def _inject(log_level: str, log_file_path: str) -> logging.Logger:
            logger = _real_setup(log_level, log_file_path)
            logger.addHandler(caplog.handler)
            return logger

        return _inject

    def test_nonstandard_base_url_emits_warning(
        self, env_vars, monkeypatch, caplog
    ):
        monkeypatch.setenv("KALSHI_BASE_URL", "https://custom.example.com")
        cfg = KalshiConfig.from_env()
        caplog.set_level(logging.WARNING, logger="kalshi_client")
        with patch("kalshi_client._setup_logging", side_effect=self._make_caplog_injector(caplog)):
            KalshiClient(cfg)
        assert any(
            "does not match" in r.message for r in caplog.records
        ), f"Expected warning not found; records: {[r.message for r in caplog.records]}"

    def test_documented_demo_url_does_not_warn(
        self, env_vars, caplog
    ):
        """Standard demo URL must NOT trigger the 'does not match' warning."""
        cfg = KalshiConfig.from_env()
        caplog.set_level(logging.WARNING, logger="kalshi_client")
        with patch("kalshi_client._setup_logging", side_effect=self._make_caplog_injector(caplog)):
            KalshiClient(cfg)
        url_warning_records = [
            r for r in caplog.records
            if "does not match" in r.message
        ]
        assert len(url_warning_records) == 0


class TestFix6UnknownSportWarning:
    """
    Fix 6 — _extract_sport_teams() emits WARNING when any returned field is
    'UNKNOWN'. The warning includes subtitle, series_ticker, and which fields
    are UNKNOWN in the extra dict.
    """

    def test_unknown_fields_emit_warning_when_no_separator(self, caplog):
        """
        A subtitle with no 'vs' or '@' separator → sport/home/away all UNKNOWN.
        The module-level _extract_sport_teams() function must emit WARNING.
        """
        kalshi_logger = logging.getLogger("kalshi_client")
        kalshi_logger.addHandler(caplog.handler)
        caplog.set_level(logging.WARNING, logger="kalshi_client")
        try:
            sport, home, away = _extract_sport_teams(
                "Some Unknown Event",
                {"series_ticker": "XYZ"},
            )
        finally:
            kalshi_logger.removeHandler(caplog.handler)

        # At least one of the returned fields must be UNKNOWN
        assert "UNKNOWN" in (sport, home, away)
        # A WARNING must have been emitted
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1, (
            "Expected at least one WARNING when sport/team fields are UNKNOWN"
        )

    def test_known_subtitle_does_not_emit_warning(self, caplog):
        """
        A well-formed subtitle like 'Lakers vs Warriors' resolves all fields —
        no WARNING should be emitted.
        """
        kalshi_logger = logging.getLogger("kalshi_client")
        kalshi_logger.addHandler(caplog.handler)
        caplog.set_level(logging.WARNING, logger="kalshi_client")
        try:
            sport, home, away = _extract_sport_teams(
                "Los Angeles Lakers vs Golden State Warriors",
                {"series_ticker": "NBAWIN"},
            )
        finally:
            kalshi_logger.removeHandler(caplog.handler)

        assert sport == "NBA"
        assert "UNKNOWN" not in (sport, home, away)
        unknown_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "UNKNOWN" in r.message.upper()
        ]
        assert len(unknown_warnings) == 0


# ===========================================================================
# _parse_utc() tests  (Fix 1)
# ===========================================================================

class TestParseUtc:
    """
    Tests for KalshiClient._parse_utc().

    The live Kalshi API returns subsecond timestamps with variable precision,
    including 4-digit subseconds such as '2026-05-26T20:26:02.0608+00:00'.
    Python < 3.11's fromisoformat() cannot parse these; _parse_utc() must
    truncate to 6 digits before parsing.
    """

    def test_parse_utc_standard_3_decimal_places(self):
        """Standard 3-decimal-place timestamp parses without modification."""
        result = KalshiClient._parse_utc("2026-05-26T20:26:02.060+00:00")
        assert result == datetime(2026, 5, 26, 20, 26, 2, 60000, tzinfo=timezone.utc)

    def test_parse_4_decimal_places_truncated(self):
        """4-decimal-place subsecond timestamp must parse without raising."""
        result = KalshiClient._parse_utc("2026-05-26T20:26:02.0608+00:00")
        # Truncated to 6 digits → .060800 → 60800 microseconds
        assert result == datetime(2026, 5, 26, 20, 26, 2, 60800, tzinfo=timezone.utc)

    def test_parse_utc_7_decimal_places_truncated(self):
        """7+ digit subseconds are truncated to 6 before parsing."""
        result = KalshiClient._parse_utc("2026-05-26T20:26:02.0608123+00:00")
        # Truncated to .060812 → 60812 microseconds
        assert result == datetime(2026, 5, 26, 20, 26, 2, 60812, tzinfo=timezone.utc)

    def test_parse_utc_no_subseconds(self):
        """Timestamp with no subsecond component parses correctly."""
        result = KalshiClient._parse_utc("2026-05-26T20:26:02+00:00")
        assert result == datetime(2026, 5, 26, 20, 26, 2, tzinfo=timezone.utc)

    def test_parse_utc_z_suffix(self):
        """Z-suffix is normalised to +00:00 and parsed as UTC-aware datetime."""
        result = KalshiClient._parse_utc("2026-05-26T20:26:02.060Z")
        assert result.tzinfo is not None
        assert result == datetime(2026, 5, 26, 20, 26, 2, 60000, tzinfo=timezone.utc)

    def test_parse_utc_naive_datetime_gets_utc(self):
        """Naive datetime string (no timezone) is assumed UTC."""
        result = KalshiClient._parse_utc("2026-05-26T20:26:02.060000")
        assert result.tzinfo is not None
        assert result == datetime(2026, 5, 26, 20, 26, 2, 60000, tzinfo=timezone.utc)

    def test_parse_utc_invalid_string_raises(self):
        """Unparseable string raises ValueError."""
        with pytest.raises(ValueError):
            KalshiClient._parse_utc("not-a-date")


# ===========================================================================
# is_liquid() tests  (Fix 3)
# ===========================================================================

def _make_contract_for_liquidity(volume_24h: float) -> ContractObject:
    """Build a minimal ContractObject with the given 24h volume for is_liquid() tests."""
    now = datetime.now(timezone.utc)
    return ContractObject(
        contract_id="TEST-C1",
        series_ticker="KXNBAGAME",
        event_ticker="KXNBAGAME-26MAY28OKCSAS",
        sport="NBA",
        home_team="TeamA",
        away_team="TeamB",
        game_date=now,
        resolution_date=now,
        yes_price=0.55,
        no_price=0.45,
        volume_fp=100.0,
        volume_24h=volume_24h,
        resolution_criteria_text="Resolves YES if TeamA wins.",
        ingestion_timestamp=now,
        open_yes_price=0.50,
        is_resolved=False,
        resolution_outcome=None,
    )


class TestIsLiquid:
    """
    Tests for KalshiClient.is_liquid().

    is_liquid() is the universe filter entry point. It checks volume_24h
    against a configurable threshold. The book_depth_usd check requires a
    live order-book snapshot and is documented but not enforced here.
    """

    def test_is_liquid_passes_with_sufficient_volume(self):
        """volume_24h=10.0 with default min_volume_24h=8.0 → True."""
        contract = _make_contract_for_liquidity(volume_24h=10.0)
        assert KalshiClient.is_liquid(contract) is True

    def test_is_liquid_fails_with_zero_volume(self):
        """volume_24h=0.0 with default min_volume_24h=8.0 → False."""
        contract = _make_contract_for_liquidity(volume_24h=0.0)
        assert KalshiClient.is_liquid(contract) is False

    def test_is_liquid_custom_threshold(self):
        """volume_24h=5.0 with min_volume_24h=8.0 → False (below threshold)."""
        contract = _make_contract_for_liquidity(volume_24h=5.0)
        assert KalshiClient.is_liquid(contract, min_volume_24h=8.0) is False

    def test_is_liquid_exactly_at_threshold(self):
        """volume_24h exactly equal to min_volume_24h → True (boundary inclusive)."""
        contract = _make_contract_for_liquidity(volume_24h=8.0)
        assert KalshiClient.is_liquid(contract, min_volume_24h=8.0) is True


# ===========================================================================
# Historical API endpoint tests
# ===========================================================================

class TestHistoricalEndpoints:
    """
    Tests for get_historical_cutoff, get_historical_markets,
    and get_historical_candlesticks.
    All tests use mocked HTTP — no live API calls.
    """

    @responses_mock.activate
    def test_get_historical_cutoff_happy_path(self, client):
        """All three timestamp fields present → dict with three datetime values."""
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/historical/cutoff",
            json={
                "market_settled_ts": "2026-05-01T00:00:00Z",
                "trades_created_ts": "2026-05-01T01:00:00Z",
                "orders_updated_ts": "2026-05-01T02:00:00Z",
            },
            status=200,
        )
        resp = client.get_historical_cutoff()
        assert resp.success is True
        assert len(resp.payload) == 3
        assert isinstance(resp.payload["market_settled_ts"], datetime)
        assert isinstance(resp.payload["trades_created_ts"], datetime)
        assert isinstance(resp.payload["orders_updated_ts"], datetime)
        # Values parsed correctly
        assert resp.payload["market_settled_ts"].year == 2026
        assert resp.payload["market_settled_ts"].tzinfo is not None

    @responses_mock.activate
    def test_get_historical_cutoff_missing_field(self, client):
        """One field absent from response → dict has two keys, no raise."""
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/historical/cutoff",
            json={
                "market_settled_ts": "2026-05-01T00:00:00Z",
                "trades_created_ts": "2026-05-01T01:00:00Z",
                # orders_updated_ts intentionally absent
            },
            status=200,
        )
        resp = client.get_historical_cutoff()
        assert resp.success is True
        assert len(resp.payload) == 2
        assert "market_settled_ts" in resp.payload
        assert "trades_created_ts" in resp.payload
        assert "orders_updated_ts" not in resp.payload

    @responses_mock.activate
    def test_get_historical_markets_happy_path(self, client):
        """Two valid markets → two ContractObjects returned, sport from series_ticker."""
        m1 = dict(make_market_raw("KXNBAGAME-EV1-OKC"))
        m2 = dict(make_market_raw("KXNBAGAME-EV1-SAS"))
        m1["event_ticker"] = "KXNBAGAME-EV1"
        m2["event_ticker"] = "KXNBAGAME-EV1"
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/historical/markets",
            json={"markets": [m1, m2], "cursor": ""},
            status=200,
        )
        resp = client.get_historical_markets("KXNBAGAME")
        assert resp.success is True
        assert len(resp.payload) == 2
        for contract in resp.payload:
            assert contract.sport == "NBA"
            assert contract.series_ticker == "KXNBAGAME"

    @responses_mock.activate
    def test_get_historical_markets_zero_volume_contract_parses(self, client):
        """volume_fp=0 does NOT raise — historical contracts frequently have zero volume."""
        m = dict(make_market_raw("KXNBAGAME-EV1-OKC"))
        m["volume_fp"] = "0.00"
        m["event_ticker"] = "KXNBAGAME-EV1"
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/historical/markets",
            json={"markets": [m], "cursor": ""},
            status=200,
        )
        resp = client.get_historical_markets("KXNBAGAME")
        assert resp.success is True
        assert len(resp.payload) == 1
        assert resp.payload[0].volume_fp == pytest.approx(0.0)

    @responses_mock.activate
    def test_get_historical_markets_one_bad_contract(self, client):
        """One market fails price extraction → other contract still returned."""
        good = dict(make_market_raw("KXNBAGAME-EV1-OKC"))
        good["event_ticker"] = "KXNBAGAME-EV1"
        bad = {
            "ticker": "KXNBAGAME-EV1-SAS",
            "event_ticker": "KXNBAGAME-EV1",
            "last_price_dollars": "invalid",
            "yes_bid_dollars": "also-invalid",
            "yes_ask_dollars": "",
            "volume_fp": "100.00",
            "volume_24h_fp": "10.00",
            "latest_expiration_time": FUTURE_TS,
            "close_time": CLOSE_TS,
            "status": "finalized",
            "result": "yes",
            "rules_primary": "",
        }
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/historical/markets",
            json={"markets": [good, bad], "cursor": ""},
            status=200,
        )
        resp = client.get_historical_markets("KXNBAGAME")
        assert len(resp.payload) == 1
        assert resp.payload[0].contract_id == "KXNBAGAME-EV1-OKC"

    @responses_mock.activate
    def test_get_historical_candlesticks_happy_path(self, client):
        """Three candle dicts → three PriceSnapshots sorted oldest-first."""
        candles = [
            make_candle(end_ts=1748822400, close="0.6000"),  # newest
            make_candle(end_ts=1748736000, close="0.5500"),  # middle
            make_candle(end_ts=1748649600, close="0.5000"),  # oldest
        ]
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/historical/markets/{CONTRACT_ID}/candlesticks",
            json={"candlesticks": candles},
            status=200,
        )
        start = datetime(2026, 5, 30, tzinfo=timezone.utc)
        resp = client.get_historical_candlesticks(CONTRACT_ID, SERIES_TICKER, start)
        assert resp.success is True
        assert len(resp.payload) == 3
        # Sorted oldest-first
        assert resp.payload[0].timestamp < resp.payload[1].timestamp
        assert resp.payload[1].timestamp < resp.payload[2].timestamp
        assert resp.payload[0].yes_price == pytest.approx(0.5000)

    @responses_mock.activate
    def test_get_historical_candlesticks_empty_price_object(self, client):
        """Zero-volume candle with bid/ask mid → snapshot returned using mid-price."""
        zero_vol_candle = make_candle(end_ts=1748736000, close=None)
        responses_mock.add(
            responses_mock.GET,
            f"{API_ROOT}/historical/markets/{CONTRACT_ID}/candlesticks",
            json={"candlesticks": [zero_vol_candle]},
            status=200,
        )
        start = datetime(2026, 5, 30, tzinfo=timezone.utc)
        resp = client.get_historical_candlesticks(CONTRACT_ID, SERIES_TICKER, start)
        assert len(resp.payload) == 1
        # Mid of bid 0.5300 and ask 0.5700 = 0.5500
        assert resp.payload[0].yes_price == pytest.approx(0.5500, abs=1e-4)

    def test_get_historical_candlesticks_invalid_interval(self, client):
        """period_interval=30 → ValueError (not in valid set 1, 60, 1440)."""
        start = datetime(2026, 5, 30, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="period_interval"):
            client.get_historical_candlesticks(
                CONTRACT_ID, SERIES_TICKER, start, period_interval=30
            )
