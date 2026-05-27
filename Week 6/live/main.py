"""Live engine event loop. Wires all the per-module pieces together.

Run as:
    python live/main.py           # standalone (engine only)
    uvicorn live.dashboard.server:app   # bundled with dashboard (Render mode)

In Render bundled mode, `live.dashboard.server` imports `start_engine_background`
and launches the engine as an asyncio task at FastAPI startup so a single uvicorn
process runs both dashboard + engine, sharing the same SQLite state.

Spec deliverable (Week 6):
  - WebSocket connect Alpaca + authentication
  - Live trades + drift monitor (broker raw vs backtest predicted)
  - 5-min disconnect drill survives (engine reconnects, doesn't crash)
"""
from __future__ import annotations

# CRITICAL: BLAS=1 BEFORE numpy import. Inherited by all spawned tasks.
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import asyncio
import json
import math
import pickle
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Module imports (after sys.path fix)
from live.broker.alpaca_client import AlpacaConfig, build_data_client, build_trading_client  # noqa: E402
from live.broker.trading_stream import TradingStreamHandler  # noqa: E402
from live.broker.websocket_handler import AlpacaStream, ConnState, StreamConfig  # noqa: E402
from live.engine_live.alpha_refit import RefitAlpha, refit_all_pairs  # noqa: E402
from live.engine_live.eom_flatten import flatten_all_open_positions  # noqa: E402
from live.engine_live.live_pair import decide  # noqa: E402
from live.engine_live.sizer import compute_pair_notional  # noqa: E402
from live.engine_live.trading_calendar import is_last_trading_day_of_month, is_trading_day  # noqa: E402
from live.engine_live.z_tracker import ZTracker  # noqa: E402
from live.execution.order_manager import OrderRequest, submit_order  # noqa: E402
from live.execution.reconciliation import reconcile, trip_halt_on_mismatch  # noqa: E402
from live.monitor.alerts import Severity, alert  # noqa: E402
from live.monitor.kill_switch import KillSwitchThresholds, evaluate as evaluate_kill_switch  # noqa: E402
from live.safety import hardstop  # noqa: E402
from live.state.persist import connect, init_db, log_event  # noqa: E402


# ============================================================
# Configuration
# ============================================================

ENTRY_Z = float(_os.environ.get("ENTRY_Z", "3.0"))
HARD_SL_Z = float(_os.environ.get("HARD_SL_Z", "5.0"))
Z_WINDOW = 60
RECONCILE_INTERVAL_S = 600        # 10 minutes
KILL_SWITCH_INTERVAL_S = 300      # 5 minutes
DECISION_BAR_TIMEFRAME = "1Day"
STATE_DB_PATH = Path(_os.environ.get("STATE_DB_PATH", "./live/state/live_state.db"))

# Artifacts (created by build-time discovery)
PAIRS_FP = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"


# ============================================================
# Pair state — per-pair runtime memory
# ============================================================

@dataclass
class PairContext:
    pair_id: str
    ticker_a: str
    ticker_b: str
    alpha: float
    beta: float
    notional: float
    z_tracker: ZTracker
    current_state: int = 0          # -1 / 0 / +1
    last_z: float | None = None
    last_bar_ts: str | None = None


# ============================================================
# Engine
# ============================================================

class LiveEngine:
    """Daily stat-arb engine. One instance per process."""

    def __init__(self, cfg: AlpacaConfig | None = None) -> None:
        self.cfg = cfg or AlpacaConfig.from_env()
        self.pair_contexts: dict[str, PairContext] = {}      # keyed by pair_id
        self.universe_tickers: set[str] = set()
        self.latest_log_close: dict[str, float] = {}         # latest bar per ticker
        self.latest_bar_ts: dict[str, str] = {}
        self._stop = False
        self._session_start_equity: float | None = None

    # ------- Initialization -------

    def initialize(self) -> None:
        """Load discovery artifacts + initialize pair contexts."""
        if not PAIRS_FP.exists():
            raise FileNotFoundError(
                f"discovered_pairs.parquet not found at {PAIRS_FP}. "
                "Run scripts/run_live_discovery.py first."
            )
        if not FACTOR_FP.exists():
            raise FileNotFoundError(f"factor_state.pkl not found at {FACTOR_FP}")

        import pandas as pd
        pairs_df = pd.read_parquet(PAIRS_FP)
        with FACTOR_FP.open("rb") as f:
            factor_state = pickle.load(f)
        resid = factor_state["residual_log_prices"]

        refits = refit_all_pairs(pairs_df, resid, n_lookback=60)
        logger.info(f"Loaded {len(refits)} pairs after alpha refit")

        for r in refits:
            ra = resid[r.ticker_a].dropna()
            rb = resid[r.ticker_b].dropna()
            aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(aligned) < 60:
                continue
            spread = aligned.iloc[:, 0].values - r.alpha_refit - r.beta * aligned.iloc[:, 1].values
            notional = compute_pair_notional(spread)

            ztracker = ZTracker(window=Z_WINDOW, seed=spread.tolist())
            ctx = PairContext(
                pair_id=f"{r.ticker_a}_{r.ticker_b}",
                ticker_a=r.ticker_a, ticker_b=r.ticker_b,
                alpha=r.alpha_refit, beta=r.beta,
                notional=notional,
                z_tracker=ztracker,
            )
            self.pair_contexts[ctx.pair_id] = ctx
            self.universe_tickers.add(r.ticker_a)
            self.universe_tickers.add(r.ticker_b)

        logger.info(f"Engine initialized: {len(self.pair_contexts)} pairs, "
                    f"{len(self.universe_tickers)} unique tickers")

        # Init SQLite state
        init_db(STATE_DB_PATH)

        # Restore session-start equity (or capture if first run)
        client = build_trading_client(self.cfg)
        acct = client.get_account()
        self._session_start_equity = float(acct.equity)
        logger.info(f"Session start equity: ${self._session_start_equity:.2f}")
        with connect(STATE_DB_PATH) as conn:
            log_event(conn, "engine_init", "INFO",
                      f"Loaded {len(self.pair_contexts)} pairs; "
                      f"start equity ${self._session_start_equity:.2f}")

    # ------- Bar handler (market data stream) -------

    async def on_bar(self, bar_payload: dict) -> None:
        """Called for each new bar from Alpaca market data stream.

        bar_payload keys: ticker, ts_utc, open, high, low, close, volume, vwap, n_trades.
        Daily bars: bar at market close. On bar close → trigger decision for relevant pairs.
        """
        ticker = bar_payload["ticker"]
        if ticker not in self.universe_tickers:
            return
        self.latest_log_close[ticker] = math.log(bar_payload["close"])
        self.latest_bar_ts[ticker] = bar_payload["ts_utc"]

        # Find all pairs touching this ticker and check if BOTH legs have fresh bar
        for pair_id, ctx in self.pair_contexts.items():
            if ticker not in (ctx.ticker_a, ctx.ticker_b):
                continue
            if (ctx.ticker_a not in self.latest_log_close or
                ctx.ticker_b not in self.latest_log_close):
                continue
            # Only decide if both bar timestamps are the same (same trading day)
            ts_a = self.latest_bar_ts.get(ctx.ticker_a)
            ts_b = self.latest_bar_ts.get(ctx.ticker_b)
            if ts_a != ts_b:
                continue
            await self._decide_and_maybe_trade(ctx, ts_a)

    async def _decide_and_maybe_trade(self, ctx: PairContext, bar_ts: str) -> None:
        """Compute spread → push to ZTracker → decide → submit order if signal."""
        if bar_ts == ctx.last_bar_ts:
            return  # already processed this bar
        ctx.last_bar_ts = bar_ts

        # Residual not strictly computed live — for simplicity use log_close difference.
        # This is a SIMPLIFICATION; full live would project through PCA loadings (TODO 5).
        # For deliverable, raw spread is sufficient — engine demonstrates wiring.
        a_log = self.latest_log_close[ctx.ticker_a]
        b_log = self.latest_log_close[ctx.ticker_b]
        spread = a_log - ctx.alpha - ctx.beta * b_log

        z = ctx.z_tracker.push(spread)
        if z is None:
            return
        ctx.last_z = z
        d = decide(ctx.current_state, z, entry_z=ENTRY_Z, hard_sl_z=HARD_SL_Z)
        if d.action == "hold":
            return
        await self._execute_action(ctx, d, bar_ts)
        ctx.current_state = d.new_state

    async def _execute_action(self, ctx: PairContext, decision, bar_ts: str) -> None:
        """Submit orders for entry/exit."""
        logger.info(f"{ctx.pair_id}: {decision.action} at Z={ctx.last_z:.3f}")
        trading_client = build_trading_client(self.cfg)

        # Determine sides per leg from action
        if decision.action == "enter_long":
            side_a, side_b = "buy", "sell"
        elif decision.action == "enter_short":
            side_a, side_b = "sell", "buy"
        elif decision.action in ("exit_zero", "exit_hard"):
            # Reverse of current position
            if ctx.current_state == 1:
                side_a, side_b = "sell", "buy"
            elif ctx.current_state == -1:
                side_a, side_b = "buy", "sell"
            else:
                return
        else:
            return

        # Compute qty per leg (notional / decision price)
        price_a = math.exp(self.latest_log_close[ctx.ticker_a])
        price_b = math.exp(self.latest_log_close[ctx.ticker_b])
        qty_a = max(1.0, round(ctx.notional / price_a))
        qty_b = max(1.0, round(ctx.notional * ctx.beta / price_b))

        req_a = OrderRequest(
            pair_id=ctx.pair_id, bar_ts=bar_ts, leg="A",
            ticker=ctx.ticker_a, side=side_a, qty=qty_a, order_type="market",
        )
        req_b = OrderRequest(
            pair_id=ctx.pair_id, bar_ts=bar_ts, leg="B",
            ticker=ctx.ticker_b, side=side_b, qty=qty_b, order_type="market",
        )
        with connect(STATE_DB_PATH) as conn:
            out_a = submit_order(trading_client, conn, req_a, decision_price=price_a)
            out_b = submit_order(trading_client, conn, req_b, decision_price=price_b)
        logger.info(f"  -> orders submitted: A={out_a.status}, B={out_b.status}")
        if out_a.refused_reason or out_b.refused_reason:
            alert(Severity.WARN,
                  f"{ctx.pair_id} {decision.action}: refused "
                  f"(A={out_a.refused_reason}, B={out_b.refused_reason})",
                  dedupe_key=f"refused_{ctx.pair_id}")

    # ------- Periodic loops -------

    async def kill_switch_loop(self) -> None:
        """Every 5 minutes, evaluate kill_switch conditions."""
        while not self._stop:
            try:
                client = build_trading_client(self.cfg)
                acct = client.get_account()
                current_equity = float(acct.equity)
                with connect(STATE_DB_PATH) as conn:
                    reason = evaluate_kill_switch(
                        conn,
                        session_start_equity=self._session_start_equity or current_equity,
                        current_equity=current_equity,
                    )
                    if reason:
                        from live.state.persist import set_halt
                        set_halt(conn, reason)
                        log_event(conn, "kill_switch_triggered", "CRITICAL", reason)
                        alert(Severity.CRITICAL, f"Kill switch: {reason}",
                              dedupe_key=f"ks_{reason}")

                # Hardstop check (file-based, independent)
                if hardstop.HARDSTOP_FLAG_PATH.exists():
                    with connect(STATE_DB_PATH) as conn:
                        st = hardstop.check(conn, self._session_start_equity or current_equity,
                                             current_equity)
                        if st.tripped:
                            alert(Severity.CRITICAL, f"Hardstop: {st.reason}",
                                  dedupe_key=f"hs_{st.reason}")
            except Exception as e:
                logger.error(f"kill_switch_loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(KILL_SWITCH_INTERVAL_S)

    async def reconcile_loop(self) -> None:
        """Every 10 minutes, reconcile broker positions vs local SQLite."""
        while not self._stop:
            try:
                trading_client = build_trading_client(self.cfg)
                with connect(STATE_DB_PATH) as conn:
                    mismatches = reconcile(trading_client, conn, tolerance_shares=1.0)
                    if mismatches:
                        trip_halt_on_mismatch(conn, mismatches)
                        alert(Severity.ERROR,
                              f"Reconcile mismatch: {len(mismatches)} tickers; engine halted",
                              dedupe_key="reconcile_mismatch")
            except Exception as e:
                logger.error(f"reconcile_loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(RECONCILE_INTERVAL_S)

    async def eom_flatten_loop(self) -> None:
        """Check daily if today is last trading day of month; flatten if so."""
        while not self._stop:
            try:
                today = datetime.now(timezone.utc).date()
                if is_trading_day(today) and is_last_trading_day_of_month(today):
                    # Trigger at 19:55 UTC (≈ 15:55 ET DST)
                    now_utc = datetime.now(timezone.utc)
                    if now_utc.hour == 19 and now_utc.minute >= 55:
                        trading_client = build_trading_client(self.cfg)
                        with connect(STATE_DB_PATH) as conn:
                            bar_ts = now_utc.isoformat()
                            actions = flatten_all_open_positions(
                                trading_client, conn, bar_ts,
                            )
                        logger.info(f"EOM flatten triggered: {len(actions)} pairs closed")
                        alert(Severity.INFO, f"EOM flatten: {len(actions)} pairs closed")
                        await asyncio.sleep(86400)   # 1 day cooldown after flatten
                        continue
            except Exception as e:
                logger.error(f"eom_flatten_loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(300)   # check every 5 min

    # ------- Main run -------

    async def run(self) -> None:
        """Main entry. Runs market stream + trading stream + periodic loops concurrently."""
        self.initialize()

        if not self.universe_tickers:
            logger.error("No pairs to trade — engine idle")
            return

        # Trading event stream (fills)
        def _conn_factory():
            c = sqlite3.connect(STATE_DB_PATH, isolation_level=None)
            c.row_factory = sqlite3.Row
            return c

        trading_handler = TradingStreamHandler(self.cfg, conn_factory=_conn_factory)

        # Market data stream
        market_stream = AlpacaStream(
            cfg=self.cfg,
            tickers=sorted(self.universe_tickers),
            stream_cfg=StreamConfig(stale_threshold_s=90.0,
                                     enable_trades_crosscheck=False),
        )

        # Launch all concurrently
        tasks = [
            asyncio.create_task(market_stream.run(self.on_bar), name="market_stream"),
            asyncio.create_task(trading_handler.run(), name="trading_stream"),
            asyncio.create_task(self.kill_switch_loop(), name="kill_switch"),
            asyncio.create_task(self.reconcile_loop(), name="reconcile"),
            asyncio.create_task(self.eom_flatten_loop(), name="eom_flatten"),
        ]
        logger.info(f"Engine running with {len(tasks)} concurrent tasks")
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self._stop = True


# ============================================================
# Public API for dashboard wiring
# ============================================================

_engine_instance: LiveEngine | None = None
_engine_task: asyncio.Task | None = None


def get_engine() -> LiveEngine | None:
    return _engine_instance


async def start_engine_background() -> None:
    """Called from FastAPI startup. Launches engine as concurrent task."""
    global _engine_instance, _engine_task
    if _engine_instance is not None:
        logger.warning("Engine already started, skipping")
        return
    _engine_instance = LiveEngine()
    _engine_task = asyncio.create_task(_engine_instance.run(), name="live_engine")
    logger.info("Engine started as background task")


def configure_logging() -> None:
    log_dir = Path(_os.environ.get("LOG_DIR", "./live/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=_os.environ.get("LOG_LEVEL", "INFO"))
    logger.add(log_dir / "live_{time:YYYYMMDD}.log",
               rotation="00:00", retention="30 days", level="DEBUG")


# ============================================================
# Standalone entry
# ============================================================

async def main_standalone() -> None:
    load_dotenv()
    configure_logging()
    engine = LiveEngine()
    await engine.run()


if __name__ == "__main__":
    try:
        asyncio.run(main_standalone())
    except KeyboardInterrupt:
        logger.info("Engine stopped by user")
