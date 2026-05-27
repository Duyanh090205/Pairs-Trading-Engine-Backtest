"""Alpaca paper broker REST client (auth, account, orders, positions)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    base_url: str
    data_url: str
    paper: bool

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        return cls(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            base_url=os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            data_url=os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets"),
            paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
        )


def build_trading_client(cfg: AlpacaConfig):
    """Returns alpaca.trading.client.TradingClient."""
    from alpaca.trading.client import TradingClient
    return TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)


def build_data_client(cfg: AlpacaConfig):
    """Returns alpaca.data.historical.StockHistoricalDataClient."""
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(cfg.api_key, cfg.secret_key)


def check_auth(cfg: AlpacaConfig) -> dict:
    """Round-trip auth test. Returns account summary dict. Raises on failure.

    `status` is normalized to the enum name (e.g. 'ACTIVE'), not the full
    repr 'AccountStatus.ACTIVE' which str() would produce.
    """
    client = build_trading_client(cfg)
    acct = client.get_account()
    status = getattr(acct.status, "name", None) or getattr(acct.status, "value", None) \
             or str(acct.status).rsplit(".", 1)[-1]
    return {
        "id": str(acct.id),
        "status": status,
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "equity": float(acct.equity),
        "pattern_day_trader": acct.pattern_day_trader,
    }
