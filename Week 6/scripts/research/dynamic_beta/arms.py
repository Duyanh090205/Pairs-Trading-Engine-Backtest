"""
Dynamic-β smoke test — arm implementations.

Three arms operate on the SAME pair list from V4 discovery. Only β source differs.
Engine pipeline (Z-window, state machine, sizing, costs) is reused from engine_daily.

A0: V4 baseline β (Johansen, 12-month formation residuals)
A1: OLS β refit on last 60 days of formation residuals
B1: V4 β as init + within-window Kalman per bar (HL=10d, Q = δ·R)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# Engine reuse — same Z, state machine, sizing, cost path as V4
from engine_daily.engine_daily import (
    rolling_z_with_warmup,
    compute_vol_target_notional_daily,
    _state_machine_daily,
    _TC_BPS,
    _BORROW_BPS_YR,
)
from engine_daily.alpha_refit import recompute_alpha


# ---------------------------------------------------------------------------
# β estimators (per arm)
# ---------------------------------------------------------------------------

def beta_a0_baseline(beta_v4: float) -> float:
    """A0: V4 baseline — Johansen β from discovery (12-month formation residuals)."""
    return float(beta_v4)


def beta_a1_short_ols(
    resid_a_form: pd.Series,
    resid_b_form: pd.Series,
    n_lookback: int = 60,
) -> float:
    """
    A1: OLS β on last `n_lookback` formation bars.

    Forces β > 0 to stay in V4's β-positive regime (V3 R5 filter). If short-window
    OLS returns β ≤ 0, fall back to V4 baseline β (caller injects via post-check).
    """
    df = pd.concat([resid_a_form, resid_b_form], axis=1, join="inner").dropna()
    df.columns = ["a", "b"]
    if len(df) < n_lookback:
        # degenerate: not enough bars → return NaN, caller falls back to A0
        return float("nan")
    tail = df.tail(n_lookback)
    a = tail["a"].values.astype(np.float64)
    b = tail["b"].values.astype(np.float64)
    # OLS: β = Cov(a,b) / Var(b)
    a_mean = a.mean()
    b_mean = b.mean()
    cov_ab = float(np.sum((a - a_mean) * (b - b_mean)) / (len(a) - 1))
    var_b = float(np.sum((b - b_mean) ** 2) / (len(b) - 1))
    if var_b <= 1e-18:
        return float("nan")
    return cov_ab / var_b


# ---------------------------------------------------------------------------
# Kalman per-bar β update (B1) — operates on residual log-prices
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _kalman_beta_inner(a: np.ndarray, b: np.ndarray,
                       alpha: float, beta0: float,
                       R: float, delta: float):
    """
    Numba 2-state KF over (alpha_state, beta_state) where ONLY beta_state is treated
    as time-varying for spread computation. We use the same parameterization as V2
    kalman.py: Q = δ·R·I, observation H = [1, b[t]], obs noise R.

    But the alpha state is anchored (we pass in the V4-refit alpha and don't let it
    drift) — this is a deliberate design choice for B1: the smoke test is about β,
    not α. Implementation: we still let alpha state drift internally (so the filter
    is mathematically consistent) but we use ONLY the β state for the time-varying
    spread. Spread at t uses PRIOR β (before observing y[t]) → no look-ahead.

    Returns (spread_prior, beta_prior, beta_post).
    """
    n = len(a)
    Q_scale = delta * R

    P00 = R;  P01 = 0.0
    P10 = 0.0; P11 = R

    th0 = alpha   # alpha state
    th1 = beta0   # beta state

    spread_prior = np.empty(n)
    beta_prior   = np.empty(n)
    beta_post    = np.empty(n)

    for i in range(n):
        lb = b[i]

        # PREDICT: P_pred = P + Q*I
        Pp00 = P00 + Q_scale
        Pp01 = P01
        Pp10 = P10
        Pp11 = P11 + Q_scale

        # Prior spread — uses β BEFORE incorporating today's a[i]
        beta_prior[i]   = th1
        spread_prior[i] = a[i] - th0 - th1 * lb

        # H = [1, lb]
        h0 = 1.0
        h1 = lb

        # Innovation covariance
        S = h0*h0*Pp00 + h0*h1*Pp01 + h1*h0*Pp10 + h1*h1*Pp11 + R
        if S < 1e-300:
            beta_post[i] = th1
            continue

        # Kalman gain
        K0 = (Pp00*h0 + Pp01*h1) / S
        K1 = (Pp10*h0 + Pp11*h1) / S

        # Innovation
        innov = a[i] - th0*h0 - th1*h1

        # Update state
        th0 = th0 + K0 * innov
        th1 = th1 + K1 * innov

        # Joseph stabilized covariance
        IKH00 = 1.0 - K0*h0;  IKH01 = -K0*h1
        IKH10 = -K1*h0;       IKH11 = 1.0 - K1*h1

        M00 = IKH00*Pp00 + IKH01*Pp10
        M01 = IKH00*Pp01 + IKH01*Pp11
        M10 = IKH10*Pp00 + IKH11*Pp10
        M11 = IKH10*Pp01 + IKH11*Pp11

        P00 = M00*IKH00 + M01*IKH01 + R*K0*K0
        P01 = M00*IKH10 + M01*IKH11 + R*K0*K1
        P10 = M10*IKH00 + M11*IKH01 + R*K1*K0
        P11 = M10*IKH10 + M11*IKH11 + R*K1*K1

        beta_post[i] = th1

    return spread_prior, beta_prior, beta_post


@njit(cache=True, fastmath=False)
def _kalman_with_alpha_post(a: np.ndarray, b: np.ndarray,
                            alpha: float, beta0: float,
                            R: float, delta: float):
    """
    Same Kalman as _kalman_beta_inner, but also returns posterior α series
    (needed to lock per-trade α for P&L accounting in B1).

    Returns (spread_prior, beta_prior, beta_post, alpha_post).
    """
    n = len(a)
    Q_scale = delta * R

    P00 = R;  P01 = 0.0
    P10 = 0.0; P11 = R

    th0 = alpha
    th1 = beta0

    spread_prior = np.empty(n)
    beta_prior = np.empty(n)
    beta_post = np.empty(n)
    alpha_post = np.empty(n)

    for i in range(n):
        lb = b[i]

        Pp00 = P00 + Q_scale
        Pp01 = P01
        Pp10 = P10
        Pp11 = P11 + Q_scale

        beta_prior[i] = th1
        spread_prior[i] = a[i] - th0 - th1 * lb

        h0 = 1.0
        h1 = lb
        S = h0*h0*Pp00 + h0*h1*Pp01 + h1*h0*Pp10 + h1*h1*Pp11 + R
        if S < 1e-300:
            beta_post[i] = th1
            alpha_post[i] = th0
            continue

        K0 = (Pp00*h0 + Pp01*h1) / S
        K1 = (Pp10*h0 + Pp11*h1) / S
        innov = a[i] - th0*h0 - th1*h1
        th0 = th0 + K0 * innov
        th1 = th1 + K1 * innov

        IKH00 = 1.0 - K0*h0;  IKH01 = -K0*h1
        IKH10 = -K1*h0;       IKH11 = 1.0 - K1*h1
        M00 = IKH00*Pp00 + IKH01*Pp10
        M01 = IKH00*Pp01 + IKH01*Pp11
        M10 = IKH10*Pp00 + IKH11*Pp10
        M11 = IKH10*Pp01 + IKH11*Pp11
        P00 = M00*IKH00 + M01*IKH01 + R*K0*K0
        P01 = M00*IKH10 + M01*IKH11 + R*K0*K1
        P10 = M10*IKH00 + M11*IKH01 + R*K1*K0
        P11 = M10*IKH10 + M11*IKH11 + R*K1*K1

        beta_post[i] = th1
        alpha_post[i] = th0

    return spread_prior, beta_prior, beta_post, alpha_post


@njit(cache=True, fastmath=False)
def _kalman_with_guards(a: np.ndarray, b: np.ndarray,
                        alpha: float, beta0: float,
                        R: float, delta: float,
                        beta_lo: float, beta_hi: float,
                        innov_k: float):
    """
    Kalman with two guardrails (active when finite, bypassed otherwise):
      - Posterior β clamp: β_post ∈ [beta_lo, beta_hi]. Pass beta_lo=-inf, beta_hi=+inf to disable.
      - Innovation outlier gate: skip state update at bar i if |innov| > innov_k · √S.
        Pass innov_k <= 0 (e.g., -1.0) to disable.

    Note: clamping is applied to the POSTERIOR β only. Prior β at each bar is
    inherited from the previous (already-clamped) posterior, so the constraint
    propagates correctly forward.

    Returns (spread_prior, beta_prior, beta_post, alpha_post, gated_flags)
    where gated_flags[i] = 1.0 if innovation gate fired at bar i, else 0.0.
    """
    n = len(a)
    Q_scale = delta * R

    P00 = R;  P01 = 0.0
    P10 = 0.0; P11 = R

    th0 = alpha
    th1 = beta0

    spread_prior = np.empty(n)
    beta_prior = np.empty(n)
    beta_post = np.empty(n)
    alpha_post = np.empty(n)
    gated_flags = np.zeros(n)

    use_gate = (innov_k > 0.0)

    for i in range(n):
        lb = b[i]

        Pp00 = P00 + Q_scale
        Pp01 = P01
        Pp10 = P10
        Pp11 = P11 + Q_scale

        beta_prior[i] = th1
        spread_prior[i] = a[i] - th0 - th1 * lb

        h0 = 1.0
        h1 = lb
        S = h0*h0*Pp00 + h0*h1*Pp01 + h1*h0*Pp10 + h1*h1*Pp11 + R
        if S < 1e-300:
            beta_post[i] = th1
            alpha_post[i] = th0
            continue

        innov = a[i] - th0*h0 - th1*h1

        # Innovation gate: skip update if outlier
        if use_gate and (innov * innov) > (innov_k * innov_k * S):
            gated_flags[i] = 1.0
            beta_post[i] = th1
            alpha_post[i] = th0
            # Keep P (no update — equivalent to "no observation"). To avoid P→∞,
            # we still advance P_pred (already done above).
            P00 = Pp00; P01 = Pp01; P10 = Pp10; P11 = Pp11
            continue

        K0 = (Pp00*h0 + Pp01*h1) / S
        K1 = (Pp10*h0 + Pp11*h1) / S
        th0 = th0 + K0 * innov
        th1 = th1 + K1 * innov

        # Posterior β clamp
        if th1 < beta_lo:
            th1 = beta_lo
        elif th1 > beta_hi:
            th1 = beta_hi

        IKH00 = 1.0 - K0*h0;  IKH01 = -K0*h1
        IKH10 = -K1*h0;       IKH11 = 1.0 - K1*h1
        M00 = IKH00*Pp00 + IKH01*Pp10
        M01 = IKH00*Pp01 + IKH01*Pp11
        M10 = IKH10*Pp00 + IKH11*Pp10
        M11 = IKH10*Pp01 + IKH11*Pp11
        P00 = M00*IKH00 + M01*IKH01 + R*K0*K0
        P01 = M00*IKH10 + M01*IKH11 + R*K0*K1
        P10 = M10*IKH00 + M11*IKH01 + R*K1*K0
        P11 = M10*IKH10 + M11*IKH11 + R*K1*K1

        beta_post[i] = th1
        alpha_post[i] = th0

    return spread_prior, beta_prior, beta_post, alpha_post, gated_flags


def hl_to_delta(half_life_bars: float) -> float:
    """Map target β half-life (in bars) to δ for Q = δ·R parameterization.

    Steady-state K_β ≈ √δ → HL ≈ ln(2)/√δ → δ = (ln(2)/HL)².
    For HL=10 → δ ≈ 4.8e-3. For HL=30 → δ ≈ 5.3e-4. For HL=60 → δ ≈ 1.3e-4.
    """
    if half_life_bars <= 0:
        raise ValueError(f"half_life_bars must be > 0, got {half_life_bars}")
    K = np.log(2.0) / half_life_bars
    return float(K * K)


# ---------------------------------------------------------------------------
# Engine wrapper — same as engine_daily.run_pair_daily but accepts pre-computed
# spread (so B1 can supply time-varying-β spread). Kept private to this module.
# ---------------------------------------------------------------------------

def _run_engine_with_spread(
    spread_form: pd.Series,
    spread_trade: pd.Series,
    df_trade_index: pd.DatetimeIndex,
    entry_z: float,
    z_window: int,
    hard_sl_z: float,
    cost_data,
    ticker_a: str | None,
    ticker_b: str | None,
) -> pd.DataFrame:
    """
    Engine logic copied from engine_daily.run_pair_daily (flat-cost branch only,
    no carry-forward — matches smoke design). Accepts pre-computed spread so that
    Kalman arms can supply time-varying-β spread.

    Same Z formula, same state machine, same vol-target sizing, same cost convention.
    """
    if len(spread_trade) < 5:
        return pd.DataFrame()

    # Z with formation warmup
    z = rolling_z_with_warmup(spread_trade, spread_form, window=z_window)
    z_vals = z.values.astype(np.float64)

    # State machine + 1-bar execution lag
    signal, exit_code = _state_machine_daily(z_vals, entry_z, hard_sl_z, initial_state=0)
    position = np.zeros(len(signal), dtype=np.int8)
    position[1:] = signal[:-1]

    # Sizing — vol-target on trading-window spread
    notional = compute_vol_target_notional_daily(spread_trade.values)

    # Daily P&L — spread-change × position × notional
    spread_change = np.zeros(len(spread_trade))
    spread_change[1:] = np.diff(spread_trade.values)
    daily_pnl = position.astype(np.float64) * spread_change * notional

    # Entry/exit detection
    pos_prev = np.zeros(len(position), dtype=np.int8)
    pos_prev[1:] = position[:-1]
    is_entry = (position != 0) & (pos_prev == 0)
    is_exit = (position == 0) & (pos_prev != 0)
    n_bars = len(position)

    cost_entry = np.zeros(n_bars, dtype=np.float64)
    cost_exit = np.zeros(n_bars, dtype=np.float64)
    borrow_cost = np.zeros(n_bars, dtype=np.float64)
    cost_spread_dec = np.zeros(n_bars, dtype=np.float64)
    cost_impact_dec = np.zeros(n_bars, dtype=np.float64)
    cost_commission_dec = np.zeros(n_bars, dtype=np.float64)

    use_dynamic = (cost_data is not None and ticker_a is not None and ticker_b is not None)

    notional_per_leg = notional * 0.5
    short_notional_per_leg = notional_per_leg
    flat_commission_per_side = (_TC_BPS / 10000.0) * notional
    daily_borrow_per_short = (_BORROW_BPS_YR / 10000.0) / 365.0 * short_notional_per_leg

    i = 0
    while i < n_bars:
        if is_entry[i]:
            entry_idx = i
            exit_idx = -1
            for j in range(i + 1, n_bars):
                if is_exit[j]:
                    exit_idx = j
                    break
            if exit_idx == -1:
                exit_idx = n_bars - 1

            entry_date = df_trade_index[entry_idx]
            exit_date = df_trade_index[exit_idx]
            side_a = int(position[entry_idx])

            cal_days = (pd.Timestamp(exit_date).normalize().date()
                        - pd.Timestamp(entry_date).normalize().date()).days
            if cal_days > 0:
                borrow_cost[exit_idx] += daily_borrow_per_short * cal_days

            if use_dynamic:
                from engine_daily.cost_engine import compute_pair_trade_cost
                c = compute_pair_trade_cost(
                    cost_data, ticker_a, ticker_b,
                    entry_date=entry_date, exit_date=exit_date,
                    notional_per_leg=notional_per_leg, side_a=side_a,
                    borrow_rate_bps_annual=_BORROW_BPS_YR,
                )
                cost_entry[entry_idx] += (c["spread_entry_$"] + c["impact_entry_$"]
                                          + c["commission_entry_$"])
                cost_exit[exit_idx] += (c["spread_exit_$"] + c["impact_exit_$"]
                                        + c["commission_exit_$"])
                cost_spread_dec[entry_idx] += c["spread_entry_$"]
                cost_spread_dec[exit_idx] += c["spread_exit_$"]
                cost_impact_dec[entry_idx] += c["impact_entry_$"]
                cost_impact_dec[exit_idx] += c["impact_exit_$"]
                cost_commission_dec[entry_idx] += c["commission_entry_$"]
                cost_commission_dec[exit_idx] += c["commission_exit_$"]
            else:
                cost_entry[entry_idx] += flat_commission_per_side
                cost_exit[exit_idx] += flat_commission_per_side

            i = exit_idx + 1
        else:
            i += 1

    daily_pnl_net = daily_pnl - cost_entry - cost_exit - borrow_cost
    cum_pnl = np.cumsum(daily_pnl_net)

    return pd.DataFrame({
        "spread": spread_trade.values,
        "zscore": z_vals,
        "signal": signal,
        "position": position,
        "exit_code": exit_code,
        "daily_pnl_gross": daily_pnl,
        "daily_pnl_net": daily_pnl_net,
        "cum_pnl": cum_pnl,
        "cost_entry": cost_entry,
        "cost_exit": cost_exit,
        "borrow": borrow_cost,
        "notional": np.full(n_bars, notional),
    }, index=df_trade_index)


# ---------------------------------------------------------------------------
# Per-arm runners — call signature matches engine_daily.run_pair_daily inputs
# ---------------------------------------------------------------------------

def run_arm_a0(
    resid_a_form: pd.Series, resid_b_form: pd.Series,
    resid_a_trade: pd.Series, resid_b_trade: pd.Series,
    beta_v4: float, alpha_v4: float,
    R: float,
    entry_z: float, z_window: int, hard_sl_z: float,
    cost_data=None, ticker_a=None, ticker_b=None,
) -> tuple[pd.DataFrame, dict]:
    """A0: V4 baseline. β = β_v4 (Johansen 12-month). Alpha = V4-refit on last 60d."""
    df_form = pd.concat([resid_a_form, resid_b_form], axis=1, join="inner").dropna()
    df_form.columns = ["a", "b"]
    df_trade = pd.concat([resid_a_trade, resid_b_trade], axis=1, join="inner").dropna()
    df_trade.columns = ["a", "b"]
    if len(df_trade) < 5:
        return pd.DataFrame(), {"beta_used": float("nan"), "alpha_used": float("nan")}

    beta = beta_a0_baseline(beta_v4)
    alpha = recompute_alpha(resid_a_form, resid_b_form, beta=beta, n_lookback=60)

    spread_form = pd.Series(df_form["a"].values - alpha - beta * df_form["b"].values,
                            index=df_form.index)
    spread_trade = pd.Series(df_trade["a"].values - alpha - beta * df_trade["b"].values,
                             index=df_trade.index)

    res = _run_engine_with_spread(spread_form, spread_trade, df_trade.index,
                                  entry_z, z_window, hard_sl_z,
                                  cost_data, ticker_a, ticker_b)
    return res, {"beta_used": float(beta), "alpha_used": float(alpha),
                 "beta_min": float(beta), "beta_max": float(beta)}


def run_arm_a1(
    resid_a_form: pd.Series, resid_b_form: pd.Series,
    resid_a_trade: pd.Series, resid_b_trade: pd.Series,
    beta_v4: float, alpha_v4: float,
    R: float,
    entry_z: float, z_window: int, hard_sl_z: float,
    cost_data=None, ticker_a=None, ticker_b=None,
    n_lookback: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """A1: short-window OLS β. Falls back to A0 if β_ols invalid (NaN or ≤0)."""
    df_form = pd.concat([resid_a_form, resid_b_form], axis=1, join="inner").dropna()
    df_form.columns = ["a", "b"]
    df_trade = pd.concat([resid_a_trade, resid_b_trade], axis=1, join="inner").dropna()
    df_trade.columns = ["a", "b"]
    if len(df_trade) < 5:
        return pd.DataFrame(), {"beta_used": float("nan"), "alpha_used": float("nan")}

    beta_ols = beta_a1_short_ols(resid_a_form, resid_b_form, n_lookback=n_lookback)
    fellback = False
    if not np.isfinite(beta_ols) or beta_ols <= 0:
        # Defensible fallback: short-window OLS undefined → use V4 β (matches A0)
        beta = float(beta_v4)
        fellback = True
    else:
        beta = float(beta_ols)

    alpha = recompute_alpha(resid_a_form, resid_b_form, beta=beta, n_lookback=60)

    spread_form = pd.Series(df_form["a"].values - alpha - beta * df_form["b"].values,
                            index=df_form.index)
    spread_trade = pd.Series(df_trade["a"].values - alpha - beta * df_trade["b"].values,
                             index=df_trade.index)

    res = _run_engine_with_spread(spread_form, spread_trade, df_trade.index,
                                  entry_z, z_window, hard_sl_z,
                                  cost_data, ticker_a, ticker_b)
    return res, {"beta_used": float(beta), "alpha_used": float(alpha),
                 "beta_min": float(beta), "beta_max": float(beta),
                 "a1_fellback_to_v4": fellback}


def run_arm_b1(
    resid_a_form: pd.Series, resid_b_form: pd.Series,
    resid_a_trade: pd.Series, resid_b_trade: pd.Series,
    beta_v4: float, alpha_v4: float,
    R: float,
    entry_z: float, z_window: int, hard_sl_z: float,
    cost_data=None, ticker_a=None, ticker_b=None,
    half_life_bars: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """
    B1 (corrected): Kalman SIGNAL spread (per-bar β) + ENTRY-LOCKED P&L spread.

    Why split:
      - Per-bar β in spread for SIGNAL (Z) is the legitimate Kalman use:
        Z absorbs regime changes via the updated β estimate.
      - Per-bar β in spread for P&L is a textbook accounting bug: it books
        phantom P&L from β re-labeling (Δβ · b[t] term) without the actual
        portfolio rebalancing that would require it. Real pairs trader holds
        portfolio at locked hedge ratio between rebalances.

    Fix:
      - spread_signal[t] = a[t] - α_prior[t] - β_prior[t] · b[t]  (Kalman, for Z)
      - At entry bar e: lock (α_lock, β_lock) = (α_post[e-1], β_post[e-1])
                        (using POSTERIOR knowledge as of decision bar t-1)
      - spread_pnl[t] = a[t] - α_lock - β_lock · b[t]  (locked during trade)
      - daily_pnl[t] = position[t] · Δspread_pnl[t] · notional
      - notional = vol_target(spread_v4_trading)  (= A0's notional → comparison parity)
    """
    df_form = pd.concat([resid_a_form, resid_b_form], axis=1, join="inner").dropna()
    df_form.columns = ["a", "b"]
    df_trade = pd.concat([resid_a_trade, resid_b_trade], axis=1, join="inner").dropna()
    df_trade.columns = ["a", "b"]
    if len(df_trade) < 5:
        return pd.DataFrame(), {"beta_used": float("nan"), "alpha_used": float("nan")}

    beta_init = float(beta_v4)
    alpha = recompute_alpha(resid_a_form, resid_b_form, beta=beta_init, n_lookback=60)
    delta = hl_to_delta(half_life_bars)

    # ---- Formation warmup spread: V4 baseline (matches A0 — clean join at t=0) ----
    spread_form = pd.Series(df_form["a"].values - alpha - beta_init * df_form["b"].values,
                            index=df_form.index)

    # ---- Kalman over trading window: produces signal spread + α/β series ----
    a_trade = df_trade["a"].values.astype(np.float64)
    b_trade = df_trade["b"].values.astype(np.float64)
    n = len(a_trade)

    if R <= 0 or not np.isfinite(R):
        # Defensible fallback to A0 behavior if R undefined
        spread_signal = a_trade - alpha - beta_init * b_trade
        alpha_post = np.full(n, alpha)
        beta_prior = np.full(n, beta_init)
        beta_post = np.full(n, beta_init)
    else:
        spread_signal, beta_prior, beta_post, alpha_post = _kalman_with_alpha_post(
            a_trade, b_trade, alpha, beta_init, float(R), float(delta),
        )

    # ---- V4 baseline P&L spread (used for vol-target sizing — parity with A0) ----
    spread_v4_trade_arr = a_trade - alpha - beta_init * b_trade

    # ---- Signal: rolling Z with formation warmup, on SIGNAL spread ----
    spread_signal_s = pd.Series(spread_signal, index=df_trade.index)
    z = rolling_z_with_warmup(spread_signal_s, spread_form, window=z_window)
    z_vals = z.values.astype(np.float64)

    # ---- State machine → positions (1-bar exec lag) ----
    signal, exit_code = _state_machine_daily(z_vals, entry_z, hard_sl_z, initial_state=0)
    position = np.zeros(len(signal), dtype=np.int8)
    position[1:] = signal[:-1]

    # ---- Sizing: V4 baseline spread (same notional as A0) ----
    notional = compute_vol_target_notional_daily(spread_v4_trade_arr)

    # ---- P&L: per-trade entry-locked (α, β) ----
    spread_pnl = np.empty(n, dtype=np.float64)
    # Initialize with init-(α, β) so spread_pnl[0] is well-defined even before any trade
    spread_pnl[:] = a_trade - alpha - beta_init * b_trade

    in_trade = False
    alpha_lock = alpha
    beta_lock = beta_init
    entry_betas: list[float] = []
    entry_alphas: list[float] = []

    pos_prev_arr = np.zeros(n, dtype=np.int8)
    pos_prev_arr[1:] = position[:-1]
    for t in range(n):
        pos_curr = position[t]
        pos_prev = pos_prev_arr[t]
        if not in_trade and pos_curr != 0:
            # Entering at bar t — decision was at t-1, lock (α, β) as of posterior[t-1]
            lock_idx = t - 1 if t >= 1 else 0
            alpha_lock = float(alpha_post[lock_idx])
            beta_lock = float(beta_post[lock_idx])
            entry_alphas.append(alpha_lock)
            entry_betas.append(beta_lock)
            in_trade = True
            # Backfill spread_pnl[t-1] with locked formula so Δspread_pnl[t] is correct
            if t >= 1:
                spread_pnl[t - 1] = a_trade[t - 1] - alpha_lock - beta_lock * b_trade[t - 1]
        if in_trade:
            spread_pnl[t] = a_trade[t] - alpha_lock - beta_lock * b_trade[t]
        if in_trade and pos_curr == 0:
            in_trade = False

    # ---- Daily P&L ----
    spread_change = np.zeros(n)
    spread_change[1:] = np.diff(spread_pnl)
    daily_pnl_gross = position.astype(np.float64) * spread_change * notional

    # ---- Costs (same convention as engine_daily.run_pair_daily) ----
    is_entry = (position != 0) & (pos_prev_arr == 0)
    is_exit = (position == 0) & (pos_prev_arr != 0)
    cost_entry = np.zeros(n, dtype=np.float64)
    cost_exit = np.zeros(n, dtype=np.float64)
    borrow_cost = np.zeros(n, dtype=np.float64)

    use_dynamic = (cost_data is not None and ticker_a is not None and ticker_b is not None)
    notional_per_leg = notional * 0.5
    flat_commission_per_side = (_TC_BPS / 10000.0) * notional
    daily_borrow_per_short = (_BORROW_BPS_YR / 10000.0) / 365.0 * notional_per_leg

    index = df_trade.index
    i = 0
    while i < n:
        if is_entry[i]:
            entry_idx = i
            exit_idx = -1
            for j in range(i + 1, n):
                if is_exit[j]:
                    exit_idx = j
                    break
            if exit_idx == -1:
                exit_idx = n - 1
            entry_date = index[entry_idx]
            exit_date = index[exit_idx]
            side_a = int(position[entry_idx])
            cal_days = (pd.Timestamp(exit_date).normalize().date()
                        - pd.Timestamp(entry_date).normalize().date()).days
            if cal_days > 0:
                borrow_cost[exit_idx] += daily_borrow_per_short * cal_days
            if use_dynamic:
                from engine_daily.cost_engine import compute_pair_trade_cost
                c = compute_pair_trade_cost(
                    cost_data, ticker_a, ticker_b,
                    entry_date=entry_date, exit_date=exit_date,
                    notional_per_leg=notional_per_leg, side_a=side_a,
                    borrow_rate_bps_annual=_BORROW_BPS_YR,
                )
                cost_entry[entry_idx] += (c["spread_entry_$"] + c["impact_entry_$"]
                                          + c["commission_entry_$"])
                cost_exit[exit_idx] += (c["spread_exit_$"] + c["impact_exit_$"]
                                        + c["commission_exit_$"])
            else:
                cost_entry[entry_idx] += flat_commission_per_side
                cost_exit[exit_idx] += flat_commission_per_side
            i = exit_idx + 1
        else:
            i += 1

    daily_pnl_net = daily_pnl_gross - cost_entry - cost_exit - borrow_cost

    res = pd.DataFrame({
        "spread": spread_signal,    # signal spread (Kalman)
        "spread_pnl": spread_pnl,   # P&L spread (entry-locked)
        "zscore": z_vals,
        "signal": signal,
        "position": position,
        "exit_code": exit_code,
        "daily_pnl_gross": daily_pnl_gross,
        "daily_pnl_net": daily_pnl_net,
        "cum_pnl": np.cumsum(daily_pnl_net),
        "cost_entry": cost_entry,
        "cost_exit": cost_exit,
        "borrow": borrow_cost,
        "notional": np.full(n, notional),
        "beta_prior_t": beta_prior,
        "beta_post_t": beta_post,
    }, index=df_trade.index)

    return res, {
        "beta_used": float(beta_init),
        "alpha_used": float(alpha),
        "beta_min": float(np.min(beta_prior)),
        "beta_max": float(np.max(beta_prior)),
        "beta_drift_abs": float(np.max(np.abs(beta_prior - beta_init))),
        "delta": float(delta),
        "half_life_target_bars": float(half_life_bars),
        "n_entries_locked": len(entry_betas),
        "entry_beta_mean": float(np.mean(entry_betas)) if entry_betas else float("nan"),
        "entry_beta_max_abs_drift": (
            float(np.max(np.abs(np.array(entry_betas) - beta_init)))
            if entry_betas else 0.0
        ),
    }


def run_arm_b1_guarded(
    resid_a_form: pd.Series, resid_b_form: pd.Series,
    resid_a_trade: pd.Series, resid_b_trade: pd.Series,
    beta_v4: float, alpha_v4: float,
    R: float,
    entry_z: float, z_window: int, hard_sl_z: float,
    cost_data=None, ticker_a=None, ticker_b=None,
    half_life_bars: float = 10.0,
    clamp_factor: float | None = 3.0,
    innov_k: float | None = 4.0,
    drift_close_threshold: float | None = 0.30,
) -> tuple[pd.DataFrame, dict]:
    """
    B1 with three optional guardrails:

      - clamp_factor: posterior β ∈ [β_init/clamp_factor, β_init·clamp_factor]
        (e.g., 3.0 ⇒ β ∈ [β₀/3, 3β₀]; set to None to disable)
      - innov_k: skip Kalman update if |innovation| > innov_k·√S
        (e.g., 4.0 ⇒ 4-sigma gate; set to None to disable)
      - drift_close_threshold: force-exit a trade when
        |β_post[t] - β_lock| / |β_lock| > drift_close_threshold
        (e.g., 0.30 ⇒ 30% drift triggers close; set to None to disable)

    When all three are None, this is equivalent to run_arm_b1.
    Same signal/P&L split as B1 (Kalman for Z; entry-locked β for P&L).
    """
    df_form = pd.concat([resid_a_form, resid_b_form], axis=1, join="inner").dropna()
    df_form.columns = ["a", "b"]
    df_trade = pd.concat([resid_a_trade, resid_b_trade], axis=1, join="inner").dropna()
    df_trade.columns = ["a", "b"]
    if len(df_trade) < 5:
        return pd.DataFrame(), {"beta_used": float("nan"), "alpha_used": float("nan")}

    beta_init = float(beta_v4)
    alpha = recompute_alpha(resid_a_form, resid_b_form, beta=beta_init, n_lookback=60)
    delta = hl_to_delta(half_life_bars)

    spread_form = pd.Series(df_form["a"].values - alpha - beta_init * df_form["b"].values,
                            index=df_form.index)

    a_trade = df_trade["a"].values.astype(np.float64)
    b_trade = df_trade["b"].values.astype(np.float64)
    n = len(a_trade)

    # ---- Determine effective guard params ----
    if clamp_factor is not None and clamp_factor > 0 and beta_init != 0:
        beta_lo = beta_init / clamp_factor
        beta_hi = beta_init * clamp_factor
        # Order if beta_init < 0
        if beta_lo > beta_hi:
            beta_lo, beta_hi = beta_hi, beta_lo
    else:
        beta_lo = -np.inf
        beta_hi = np.inf

    innov_k_val = float(innov_k) if (innov_k is not None and innov_k > 0) else -1.0

    if R <= 0 or not np.isfinite(R):
        spread_signal = a_trade - alpha - beta_init * b_trade
        alpha_post = np.full(n, alpha)
        beta_prior = np.full(n, beta_init)
        beta_post = np.full(n, beta_init)
        gated_flags = np.zeros(n)
    else:
        spread_signal, beta_prior, beta_post, alpha_post, gated_flags = _kalman_with_guards(
            a_trade, b_trade, alpha, beta_init, float(R), float(delta),
            float(beta_lo), float(beta_hi), innov_k_val,
        )

    spread_v4_trade_arr = a_trade - alpha - beta_init * b_trade

    spread_signal_s = pd.Series(spread_signal, index=df_trade.index)
    z = rolling_z_with_warmup(spread_signal_s, spread_form, window=z_window)
    z_vals = z.values.astype(np.float64)

    signal, exit_code = _state_machine_daily(z_vals, entry_z, hard_sl_z, initial_state=0)
    position = np.zeros(len(signal), dtype=np.int8)
    position[1:] = signal[:-1]

    notional = compute_vol_target_notional_daily(spread_v4_trade_arr)

    # ---- Walk trades, lock (α, β) at entry, optionally force-close on drift ----
    spread_pnl = np.empty(n, dtype=np.float64)
    spread_pnl[:] = a_trade - alpha - beta_init * b_trade

    n_drift_closes = 0
    in_trade = False
    alpha_lock = alpha
    beta_lock = beta_init
    entry_betas: list[float] = []
    use_drift = (drift_close_threshold is not None and drift_close_threshold > 0)

    pos_prev_arr = np.zeros(n, dtype=np.int8)
    pos_prev_arr[1:] = position[:-1]
    for t in range(n):
        pos_curr = position[t]
        if not in_trade and pos_curr != 0:
            lock_idx = t - 1 if t >= 1 else 0
            alpha_lock = float(alpha_post[lock_idx])
            beta_lock = float(beta_post[lock_idx])
            entry_betas.append(beta_lock)
            in_trade = True
            if t >= 1:
                spread_pnl[t - 1] = a_trade[t - 1] - alpha_lock - beta_lock * b_trade[t - 1]
        if in_trade:
            spread_pnl[t] = a_trade[t] - alpha_lock - beta_lock * b_trade[t]
            # Drift-close: if β has drifted past threshold, mark exit at t+1
            if use_drift and abs(beta_lock) > 1e-12:
                drift_pct = abs(beta_post[t] - beta_lock) / abs(beta_lock)
                if drift_pct > drift_close_threshold:
                    # Force-close at t+1: zero positions from t+1 onward until next entry
                    # (the state machine's signal will re-engage cleanly on next threshold cross)
                    for u in range(t + 1, n):
                        if position[u] == 0 and pos_prev_arr[u] != 0:
                            break  # natural exit already happens here
                        position[u] = 0
                        pos_prev_arr[u] = position[u - 1] if u >= 1 else 0
                    # Update pos_prev_arr for the bar AFTER t (it's now zero)
                    if t + 1 < n:
                        pos_prev_arr[t + 1] = position[t]
                    in_trade = False
                    n_drift_closes += 1
                    exit_code[t + 1 if t + 1 < n else t] = 3  # 3 = drift-close
                    continue
        if in_trade and pos_curr == 0:
            in_trade = False

    # Recompute pos_prev_arr after any drift-close override
    pos_prev_arr = np.zeros(n, dtype=np.int8)
    pos_prev_arr[1:] = position[:-1]

    spread_change = np.zeros(n)
    spread_change[1:] = np.diff(spread_pnl)
    daily_pnl_gross = position.astype(np.float64) * spread_change * notional

    is_entry = (position != 0) & (pos_prev_arr == 0)
    is_exit = (position == 0) & (pos_prev_arr != 0)
    cost_entry = np.zeros(n, dtype=np.float64)
    cost_exit = np.zeros(n, dtype=np.float64)
    borrow_cost = np.zeros(n, dtype=np.float64)

    use_dynamic = (cost_data is not None and ticker_a is not None and ticker_b is not None)
    notional_per_leg = notional * 0.5
    flat_commission_per_side = (_TC_BPS / 10000.0) * notional
    daily_borrow_per_short = (_BORROW_BPS_YR / 10000.0) / 365.0 * notional_per_leg

    index = df_trade.index
    i = 0
    while i < n:
        if is_entry[i]:
            entry_idx = i
            exit_idx = -1
            for j in range(i + 1, n):
                if is_exit[j]:
                    exit_idx = j
                    break
            if exit_idx == -1:
                exit_idx = n - 1
            entry_date = index[entry_idx]
            exit_date = index[exit_idx]
            side_a = int(position[entry_idx])
            cal_days = (pd.Timestamp(exit_date).normalize().date()
                        - pd.Timestamp(entry_date).normalize().date()).days
            if cal_days > 0:
                borrow_cost[exit_idx] += daily_borrow_per_short * cal_days
            if use_dynamic:
                from engine_daily.cost_engine import compute_pair_trade_cost
                c = compute_pair_trade_cost(
                    cost_data, ticker_a, ticker_b,
                    entry_date=entry_date, exit_date=exit_date,
                    notional_per_leg=notional_per_leg, side_a=side_a,
                    borrow_rate_bps_annual=_BORROW_BPS_YR,
                )
                cost_entry[entry_idx] += (c["spread_entry_$"] + c["impact_entry_$"]
                                          + c["commission_entry_$"])
                cost_exit[exit_idx] += (c["spread_exit_$"] + c["impact_exit_$"]
                                        + c["commission_exit_$"])
            else:
                cost_entry[entry_idx] += flat_commission_per_side
                cost_exit[exit_idx] += flat_commission_per_side
            i = exit_idx + 1
        else:
            i += 1

    daily_pnl_net = daily_pnl_gross - cost_entry - cost_exit - borrow_cost

    res = pd.DataFrame({
        "spread": spread_signal,
        "spread_pnl": spread_pnl,
        "zscore": z_vals,
        "signal": signal,
        "position": position,
        "exit_code": exit_code,
        "daily_pnl_gross": daily_pnl_gross,
        "daily_pnl_net": daily_pnl_net,
        "cum_pnl": np.cumsum(daily_pnl_net),
        "cost_entry": cost_entry,
        "cost_exit": cost_exit,
        "borrow": borrow_cost,
        "notional": np.full(n, notional),
        "beta_prior_t": beta_prior,
        "beta_post_t": beta_post,
        "innov_gated": gated_flags,
    }, index=df_trade.index)

    return res, {
        "beta_used": float(beta_init),
        "alpha_used": float(alpha),
        "beta_min": float(np.min(beta_prior)),
        "beta_max": float(np.max(beta_prior)),
        "beta_post_min": float(np.min(beta_post)),
        "beta_post_max": float(np.max(beta_post)),
        "delta": float(delta),
        "half_life_target_bars": float(half_life_bars),
        "clamp_lo": float(beta_lo),
        "clamp_hi": float(beta_hi),
        "innov_k": float(innov_k_val),
        "drift_close_threshold": (
            float(drift_close_threshold) if drift_close_threshold is not None else None
        ),
        "n_entries_locked": len(entry_betas),
        "n_innov_gated": int(gated_flags.sum()),
        "n_drift_closes": int(n_drift_closes),
    }
