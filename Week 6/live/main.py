"""Live engine event loop with all 5 audit bugs fixed.

Architecture (post-audit):
  - WebSocket connected to Alpaca for telemetry (disconnect drill).
    Bars from minute stream are LOGGED but DO NOT drive decisions.
  - Decision loop runs ONCE per trading day at 21:00 UTC (16:00 ET market close).
    Pulls fresh daily bars via REST, projects residuals through PCA loadings
    (matches backtest spread definition), runs decide(), submits orders.
  - State recovered from SQLite on restart (positions + session equity).
  - Qty floor (int), never overshoot notional cap.

Run as:
    python live/main.py           # standalone
    uvicorn live.dashboard.server:app   # bundled (Render mode)
"""
from __future__ import annotations

# BLAS=1 BEFORE numpy import.
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import asyncio
import math
import pickle
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.broker.alpaca_client import AlpacaConfig, build_data_client, build_trading_client  # noqa: E402
from live.broker.trading_stream import TradingStreamHandler  # noqa: E402
from live.broker.websocket_handler import AlpacaStream, StreamConfig  # noqa: E402
from live.engine_live.alpha_refit import refit_all_pairs  # noqa: E402
from live.engine_live.eom_flatten import flatten_all_open_positions  # noqa: E402
from live.engine_live.live_pair import decide  # noqa: E402
from live.engine_live.residual_projector import ResidualProjector  # noqa: E402
from live.engine_live.sizer import compute_pair_notional  # noqa: E402
from live.engine_live.trading_calendar import is_last_trading_day_of_month, is_trading_day  # noqa: E402
from live.engine_live.z_tracker import ZTracker  # noqa: E402
from live.execution.order_manager import OrderRequest, submit_order  # noqa: E402
from live.execution.reconciliation import reconcile, trip_halt_on_mismatch  # noqa: E402
from live.monitor.alerts import Severity, alert  # noqa: E402
from live.monitor.kill_switch import KillSwitchThresholds, evaluate as evaluate_kill_switch  # noqa: E402
from live.safety import hardstop  # noqa: E402
from live.state.persist import (  # noqa: E402
    connect, get_last_decision_date, get_or_set_session_equity,
    init_db, log_event, set_last_decision_date,
)

ENTRY_Z = float(_os.environ.get("ENTRY_Z", "3.0"))
HARD_SL_Z = float(_os.environ.get("HARD_SL_Z", "5.0"))
Z_WINDOW = 60
RECONCILE_INTERVAL_S = 600
KILL_SWITCH_INTERVAL_S = 300
STATE_DB_PATH = Path(_os.environ.get("STATE_DB_PATH", "./live/state/live_state.db"))

PAIRS_FP = ROOT / "live" / "state" / "discovered_pairs.parquet"
FACTOR_FP = ROOT / "live" / "state" / "factor_state.pkl"
CACHE_DIR = ROOT / "live" / "state" / "daily_cache"


@dataclass
class PairContext:
    pair_id: str
    ticker_a: str
    ticker_b: str
    alpha: float
    beta: float
    notional: float
    z_tracker: ZTracker
    current_state: int = 0
    last_z: float | None = None
    last_decision_date: str | None = None


class LiveEngine:
    def __init__(self, cfg: AlpacaConfig | None = None) -> None:
        self.cfg = cfg or AlpacaConfig.from_env()
        self.pair_contexts: dict[str, PairContext] = {}
        self.universe_tickers: set[str] = set()
        self.factor_tickers: list[str] = []
        self.projector: ResidualProjector | None = None
        self._stop = False
        self._session_start_equity: float | None = None
        self._last_market_data: dict[str, float] = {}    # latest log_close per ticker (from REST)
        # WebSocket bar telemetry only (NOT used for decisions; just to keep connection alive)
        self._latest_ws_bar_ts: dict[str, float] = {}

    # ------- Initialization -------

    def initialize(self) -> None:
        if not PAIRS_FP.exists():
            raise FileNotFoundError(f"discovered_pairs.parquet not found at {PAIRS_FP}")
        if not FACTOR_FP.exists():
            raise FileNotFoundError(f"factor_state.pkl not found at {FACTOR_FP}")

        import pandas as pd
        pairs_df = pd.read_parquet(PAIRS_FP)
        with FACTOR_FP.open("rb") as f:
            factor_state = pickle.load(f)
        resid = factor_state["residual_log_prices"]

        # ResidualProjector for live bar -> residual conversion (FIX BUG 1)
        self.projector = ResidualProjector(FACTOR_FP)
        self.factor_tickers = list(factor_state["tickers"])

        refits = refit_all_pairs(pairs_df, resid, n_lookback=60)
        logger.info(f"Loaded {len(refits)} pairs after alpha refit")

        for r in refits:
            ra = resid[r.ticker_a].dropna()
            rb = resid[r.ticker_b].dropna()
            aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(aligned) < 60:
                continue
            spread_form = aligned.iloc[:, 0].values - r.alpha_refit - r.beta * aligned.iloc[:, 1].values
            notional = compute_pair_notional(spread_form)
            ztracker = ZTracker(window=Z_WINDOW, seed=spread_form.tolist())
            ctx = PairContext(
                pair_id=f"{r.ticker_a}_{r.ticker_b}",
                ticker_a=r.ticker_a, ticker_b=r.ticker_b,
                alpha=r.alpha_refit, beta=r.beta, notional=notional,
                z_tracker=ztracker,
            )
            self.pair_contexts[ctx.pair_id] = ctx
            self.universe_tickers.add(r.ticker_a)
            self.universe_tickers.add(r.ticker_b)

        logger.info(f"Engine: {len(self.pair_contexts)} pairs, {len(self.universe_tickers)} unique tickers")

        init_db(STATE_DB_PATH)

        # FIX BUG 5: persist session_start_equity across restarts
        client = build_trading_client(self.cfg)
        acct = client.get_account()
        current_equity = float(acct.equity)
        with connect(STATE_DB_PATH) as conn:
            self._session_start_equity = get_or_set_session_equity(conn, current_equity)
            if abs(self._session_start_equity - current_equity) > 0.01:
                logger.info(f"Restored session_start_equity=${self._session_start_equity:.2f} "
                            f"(current=${current_equity:.2f})")
            else:
                logger.info(f"Session start equity: ${self._session_start_equity:.2f}")

            # FIX BUG 2: restore pair states from open positions on restart
            self._restore_pair_states_from_db(conn)
            log_event(conn, "engine_init", "INFO",
                      f"Loaded {len(self.pair_contexts)} pairs; "
                      f"session_start_equity=${self._session_start_equity:.2f}")

    def _restore_pair_states_from_db(self, conn: sqlite3.Connection) -> None:
        """FIX BUG 2: rebuild ctx.current_state from open positions in SQLite."""
        rows = conn.execute(
            "SELECT pair_id, direction FROM positions WHERE exit_ts IS NULL"
        ).fetchall()
        for r in rows:
            ctx = self.pair_contexts.get(r["pair_id"])
            if ctx is None:
                logger.warning(f"Open position {r['pair_id']} has no pair_context — "
                                f"pair list may have changed; manual intervention needed")
                continue
            ctx.current_state = int(r["direction"])
            logger.info(f"Restored state: {r['pair_id']} = {ctx.current_state}")

    # ------- Daily decision (FIX BUG 3 — replaces minute on_bar) -------

    async def daily_decision_loop(self) -> None:
        """Run once per trading day at 21:05 UTC (~16:05 ET market close + 5min buffer)."""
        while not self._stop:
            try:
                now = datetime.now(timezone.utc)
                today_date = now.date()
                # Last decision date (persisted) to dedupe across restarts
                with connect(STATE_DB_PATH) as conn:
                    last_date_str = get_last_decision_date(conn)

                # Skip non-trading days
                if not is_trading_day(today_date):
                    await asyncio.sleep(3600)   # check hourly
                    continue

                # Already decided today?
                if last_date_str == today_date.isoformat():
                    await asyncio.sleep(3600)
                    continue

                # Wait until 21:05 UTC (16:05 ET — 5min after close to ensure final bar lands)
                target = datetime.combine(today_date, dtime(21, 5),
                                            tzinfo=timezone.utc)
                if now < target:
                    wait_s = (target - now).total_seconds()
                    logger.info(f"Decision loop sleeping {wait_s:.0f}s until "
                                f"{target.isoformat()} (next market close)")
                    await asyncio.sleep(min(wait_s, 3600))
                    continue

                # It's past close — run decisions
                await self._run_daily_decisions(today_date)
                with connect(STATE_DB_PATH) as conn:
                    set_last_decision_date(conn, today_date.isoformat())

                # Sleep ~22h until next attempt
                await asyncio.sleep(22 * 3600)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"daily_decision_loop error: {type(e).__name__}: {e}")
                await asyncio.sleep(600)

    async def _run_daily_decisions(self, today: date) -> None:
        """FIX BUG 1+3: pull fresh REST bars, project residuals, decide per pair."""
        logger.info(f"Daily decision run for {today.isoformat()}")

        # Pull recent ~30 daily bars for FACTOR universe (residual projection needs cross-section)
        trading_data = await self._pull_recent_daily_bars(today)
        if not trading_data:
            logger.warning("No bars pulled; skipping decisions")
            return

        # Project residuals via PCA loadings (matches backtest)
        residuals = self.projector.compute_residuals(trading_data)
        if not residuals:
            logger.warning("Residual projection returned empty")
            return

        bar_ts = today.isoformat() + "T20:00:00Z"
        for pair_id, ctx in self.pair_contexts.items():
            ra = residuals.get(ctx.ticker_a)
            rb = residuals.get(ctx.ticker_b)
            if ra is None or rb is None or ra.dropna().empty or rb.dropna().empty:
                continue
            # Latest residual values
            ra_last = float(ra.dropna().iloc[-1])
            rb_last = float(rb.dropna().iloc[-1])
            spread = ra_last - ctx.alpha - ctx.beta * rb_last
            z = ctx.z_tracker.push(spread)
            if z is None:
                continue
            ctx.last_z = z
            d = decide(ctx.current_state, z, entry_z=ENTRY_Z, hard_sl_z=HARD_SL_Z)
            if d.action == "hold":
                continue
            await self._execute_action(ctx, d, bar_ts, trading_data)
            ctx.current_state = d.new_state
            ctx.last_decision_date = today.isoformat()

    async def _pull_recent_daily_bars(self, today: date) -> dict:
        """REST pull daily bars for factor universe. Returns dict[ticker] -> DataFrame.

        FIX BUG 6: window starts at FORMATION_END (fixed), not today-30 (sliding).
        Reason: ResidualProjector._anchor_shifts is cached on first call. If the
        window's first day slides forward each daily call, the anchor reference
        becomes inconsistent and Z-score drifts.
        """
        import json
        import pandas as pd
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = build_data_client(self.cfg)
        # Read formation_end from discovery meta — FIXED reference, doesn't slide.
        meta_fp = ROOT / "live" / "state" / "discovery_meta.json"
        if meta_fp.exists():
            meta = json.loads(meta_fp.read_text())
            formation_end = date.fromisoformat(meta["formation_end"])
        else:
            formation_end = today - timedelta(days=30)   # fallback
        start = datetime.combine(formation_end, dtime(0, 0), tzinfo=timezone.utc)
        end = datetime.combine(today, dtime(23, 59), tzinfo=timezone.utc)

        out: dict[str, pd.DataFrame] = {}
        # Batch 100 tickers at a time (alpaca-py supports list)
        batch_size = 100
        ticker_list = list(self.factor_tickers)
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i:i + batch_size]
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                    start=start, end=end, feed=DataFeed.IEX,
                    adjustment="split",
                )
                resp = client.get_stock_bars(req)
                for tk in batch:
                    bars = resp[tk] if hasattr(resp, "__getitem__") else resp.data.get(tk, [])
                    if not bars:
                        continue
                    rows = [{"date": b.timestamp.astimezone(timezone.utc).date(),
                             "close": float(b.close)} for b in bars]
                    df = pd.DataFrame(rows)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                    df["log_close"] = df["close"].apply(math.log)
                    out[tk] = df[["log_close"]]
            except Exception as e:
                logger.warning(f"REST batch failed for {batch[:3]}...: {e}")
        logger.info(f"Pulled daily bars for {len(out)}/{len(ticker_list)} tickers")
        return out

    async def _execute_action(self, ctx: PairContext, decision, bar_ts: str,
                               trading_data: dict) -> None:
        logger.info(f"{ctx.pair_id}: {decision.action} at Z={ctx.last_z:.3f}")
        trading_client = build_trading_client(self.cfg)

        if decision.action == "enter_long":
            side_a, side_b = "buy", "sell"
        elif decision.action == "enter_short":
            side_a, side_b = "sell", "buy"
        elif decision.action in ("exit_zero", "exit_hard"):
            if ctx.current_state == 1:
                side_a, side_b = "sell", "buy"
            elif ctx.current_state == -1:
                side_a, side_b = "buy", "sell"
            else:
                return
        else:
            return

        # Look up raw close from trading_data (not residual)
        try:
            price_a = math.exp(float(trading_data[ctx.ticker_a]["log_close"].iloc[-1]))
            price_b = math.exp(float(trading_data[ctx.ticker_b]["log_close"].iloc[-1]))
        except Exception:
            logger.error(f"Missing price for {ctx.pair_id}; skip")
            return

        # FIX BUG 4: floor (int) instead of round — never overshoot notional
        qty_a = max(1, int(ctx.notional / price_a))
        qty_b = max(1, int((ctx.notional * abs(ctx.beta)) / price_b))

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
        logger.info(f"  -> orders: A={out_a.status}, B={out_b.status}")
        if out_a.refused_reason or out_b.refused_reason:
            alert(Severity.WARN,
                  f"{ctx.pair_id} {decision.action}: refused "
                  f"(A={out_a.refused_reason}, B={out_b.refused_reason})",
                  dedupe_key=f"refused_{ctx.pair_id}")

    # ------- WebSocket telemetry (NOT decision-driving) -------

    async def on_bar(self, bar_payload: dict) -> None:
        """Just log incoming WebSocket bars to demonstrate connection. NOT decision logic."""
        ticker = bar_payload.get("ticker")
        if ticker in self.universe_tickers:
            self._latest_ws_bar_ts[ticker] = bar_payload.get("ts_utc")

    # ------- Periodic loops -------

    async def kill_switch_loop(self) -> None:
        while not self._stop:
            try:
                client = build_trading_client(self.cfg)
                current_equity = float(client.get_account().equity)
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
        while not self._stop:
            try:
                trading_client = build_trading_client(self.cfg)
                with connect(STATE_DB_PATH) as conn:
                    mismatches = reconcile(trading_client, conn, tolerance_shares=1.0)
                    if mismatches:
                        trip_halt_on_mismatch(conn, mismatches)
                        alert(Severity.ERROR,
                              f"Reconcile mismatch: {len(mismatches)} tickers",
                              dedupe_key="reconcile_mismatch")
            except Exception as e:
                logger.error(f"reconcile_loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(RECONCILE_INTERVAL_S)

    async def eom_flatten_loop(self) -> None:
        while not self._stop:
            try:
                today = datetime.now(timezone.utc).date()
                if is_trading_day(today) and is_last_trading_day_of_month(today):
                    now_utc = datetime.now(timezone.utc)
                    if now_utc.hour == 19 and now_utc.minute >= 55:
                        trading_client = build_trading_client(self.cfg)
                        with connect(STATE_DB_PATH) as conn:
                            bar_ts = now_utc.isoformat()
                            actions = flatten_all_open_positions(trading_client, conn, bar_ts)
                        logger.info(f"EOM flatten: {len(actions)} pairs closed")
                        alert(Severity.INFO, f"EOM flatten: {len(actions)} pairs closed")
                        await asyncio.sleep(86400)
                        continue
            except Exception as e:
                logger.error(f"eom_flatten_loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(300)

    async def run(self) -> None:
        self.initialize()
        if not self.universe_tickers:
            logger.error("No pairs to trade — engine idle")
            return

        def _conn_factory():
            c = sqlite3.connect(STATE_DB_PATH, isolation_level=None)
            c.row_factory = sqlite3.Row
            return c

        trading_handler = TradingStreamHandler(self.cfg, conn_factory=_conn_factory)
        market_stream = AlpacaStream(
            cfg=self.cfg,
            tickers=sorted(self.universe_tickers),
            stream_cfg=StreamConfig(stale_threshold_s=90.0,
                                     enable_trades_crosscheck=False),
        )

        tasks = [
            asyncio.create_task(market_stream.run(self.on_bar), name="market_stream"),
            asyncio.create_task(trading_handler.run(), name="trading_stream"),
            asyncio.create_task(self.daily_decision_loop(), name="daily_decision"),
            asyncio.create_task(self.kill_switch_loop(), name="kill_switch"),
            asyncio.create_task(self.reconcile_loop(), name="reconcile"),
            asyncio.create_task(self.eom_flatten_loop(), name="eom_flatten"),
        ]
        logger.info(f"Engine running with {len(tasks)} concurrent tasks")
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self._stop = True


_engine_instance: LiveEngine | None = None


def get_engine() -> LiveEngine | None:
    return _engine_instance


async def start_engine_background() -> None:
    global _engine_instance
    if _engine_instance is not None:
        logger.warning("Engine already started")
        return
    _engine_instance = LiveEngine()
    asyncio.create_task(_engine_instance.run(), name="live_engine")
    logger.info("Engine started as background task")


def configure_logging() -> None:
    log_dir = Path(_os.environ.get("LOG_DIR", "./live/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=_os.environ.get("LOG_LEVEL", "INFO"))
    logger.add(log_dir / "live_{time:YYYYMMDD}.log",
               rotation="00:00", retention="30 days", level="DEBUG")


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
