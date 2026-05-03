"""
S&P 500 Futures Backtest Engine — Buy-the-Dip Strategy
=======================================================
Simulates buying S&P 500 futures contracts when price drops from a rolling high.

Entry trigger modes:
  - "single":    Buy ONCE when price first crosses below threshold. No repeat buys.
  - "step_down": Buy on each additional X% step down (at -5%, -10%, -15%, ...).
  - "per_tick":  Buy every tick while price is below threshold (aggressive averaging).

Execution model:
  - Fill at NEXT minute's price (1-tick delay) to avoid look-ahead bias
  - Optional slippage in basis points applied to fill price
  - Margin call delay (default 30min) to reflect 1987 broker overload
"""


def run_backtest(df, initial_capital=100_000, position_pct=0.25, dip_threshold=0.05,
                 margin_call_delay=30, slippage_bps=50,
                 entry_mode="step_down", martingale=False,
                 block_buys_during_mc=True,
                 multiplier=500, initial_margin=10_000, maintenance_margin=6_000,
                 scenario_name="Default"):
    """
    Args:
        df: DataFrame with Timestamp and SP500_Futures columns
        initial_capital: Starting account equity in dollars
        position_pct: Fraction of equity to deploy per buy signal
        dip_threshold: Buy when price drops this fraction from rolling high (0.05 = 5%)
        margin_call_delay: Minutes before broker executes liquidation (0=instant, 30=1987 realistic)
        slippage_bps: Execution slippage in basis points (50bps = 0.5% worse fill)
        entry_mode: "single" | "step_down" | "per_tick"
        martingale: If True, double contract count on each successive buy
        multiplier: Dollar value per index point per contract ($500 for S&P 500 futures)
        initial_margin: Margin deposit required to open 1 contract
        maintenance_margin: Minimum equity per contract before margin call
        scenario_name: Label for this scenario
    """
    prices_list = list(df['SP500_Futures'])
    timestamps_list = list(df['Timestamp'])
    n = len(prices_list)

    positions = []
    realized_pnl = 0.0
    portfolio_history = []
    buy_events = []
    bankruptcy_time = None
    margin_call_time = None
    margin_call_count = 0
    max_contracts_held = 0
    max_notional_exposure = 0.0
    peak_equity = initial_capital

    current_date = None
    rolling_daily_high = 0
    liquidated = False

    # Delayed margin call state
    pending_margin_call = None
    margin_call_trigger_time = None

    # Entry mode state
    # "single": has the single buy fired today?
    single_fired_today = False
    # "step_down": track which step levels have been triggered (multiples of dip_threshold)
    last_step_level = 0  # how many steps down we've bought at

    # Martingale state
    last_contracts_bought = 0

    # Pending buy order (next-minute fill)
    pending_buy = None  # dict with contracts_to_buy, signal_price, equity_at_signal

    for i in range(n):
        timestamp = timestamps_list[i]
        price = prices_list[i]

        # ── Execute pending buy from previous tick (next-minute fill) ──
        if pending_buy is not None and not liquidated:
            # Apply slippage: buyer gets a worse price (higher)
            slip = price * (slippage_bps / 10_000)
            fill_price = price + slip

            contracts_to_buy = pending_buy['contracts_to_buy']

            positions.append({
                'entry_price': fill_price,
                'contracts': contracts_to_buy
            })
            buy_events.append({
                'timestamp': timestamp,
                'price': fill_price,
                'signal_price': pending_buy['signal_price'],
                'contracts': contracts_to_buy,
                'equity_at_buy': pending_buy['equity_at_signal'],
                'slippage': slip * contracts_to_buy * multiplier
            })
            last_contracts_bought = contracts_to_buy
            pending_buy = None

        # ── Track rolling daily high (resets each calendar day) ──
        if current_date != timestamp.date():
            current_date = timestamp.date()
            rolling_daily_high = price
            single_fired_today = False
            last_step_level = 0
        rolling_daily_high = max(rolling_daily_high, price)

        # ── Mark-to-market ──
        unrealized_pnl = sum(
            (price - pos['entry_price']) * multiplier * pos['contracts']
            for pos in positions
        )
        equity = initial_capital + realized_pnl + unrealized_pnl
        total_contracts = sum(pos['contracts'] for pos in positions)
        notional_exposure = total_contracts * price * multiplier

        max_contracts_held = max(max_contracts_held, total_contracts)
        max_notional_exposure = max(max_notional_exposure, notional_exposure)
        peak_equity = max(peak_equity, equity)

        # ── Margin call logic ──
        if total_contracts > 0 and equity < total_contracts * maintenance_margin:
            if margin_call_delay == 0:
                # Instant liquidation
                realized_pnl += unrealized_pnl
                unrealized_pnl = 0
                equity = initial_capital + realized_pnl
                positions = []
                total_contracts = 0
                margin_call_time = margin_call_time or timestamp
                margin_call_count += 1
                liquidated = True
                pending_buy = None  # cancel any pending order
                if equity <= 0 and bankruptcy_time is None:
                    bankruptcy_time = timestamp
            else:
                # Delayed: record when margin call was first triggered
                if pending_margin_call is None:
                    pending_margin_call = timestamp
                    margin_call_trigger_time = timestamp

        # Execute delayed margin call after delay period
        if pending_margin_call is not None and margin_call_delay > 0:
            minutes_elapsed = (timestamp - pending_margin_call).total_seconds() / 60
            if minutes_elapsed >= margin_call_delay:
                # Liquidate at CURRENT price (much worse after 30min of freefall)
                unrealized_pnl = sum(
                    (price - pos['entry_price']) * multiplier * pos['contracts']
                    for pos in positions
                )
                realized_pnl += unrealized_pnl
                unrealized_pnl = 0
                equity = initial_capital + realized_pnl
                positions = []
                total_contracts = 0
                margin_call_time = margin_call_time or margin_call_trigger_time
                margin_call_count += 1
                liquidated = True
                pending_margin_call = None
                pending_buy = None
                if equity <= 0 and bankruptcy_time is None:
                    bankruptcy_time = timestamp

        # Check bankruptcy without margin call (positions still open)
        if equity <= 0 and bankruptcy_time is None:
            bankruptcy_time = timestamp

        # ── Buy trigger ──
        # No new orders if: already liquidated, bankrupt, or under margin.
        # block_buys_during_mc: if True (sophisticated), broker blocks new risk
        #   during pending margin call. If False (naive bot), keeps buying.
        can_buy = (not liquidated
                   and bankruptcy_time is None
                   and equity >= initial_margin)
        if block_buys_during_mc and pending_margin_call is not None:
            can_buy = False
        # Cancel pending buy if we just got liquidated
        if liquidated and pending_buy is not None:
            pending_buy = None

        if can_buy and pending_buy is None:
            drop_pct = (rolling_daily_high - price) / rolling_daily_high if rolling_daily_high > 0 else 0
            should_buy = False

            if entry_mode == "single":
                # Buy ONCE per day, first time price crosses threshold
                if drop_pct >= dip_threshold and not single_fired_today:
                    should_buy = True
                    single_fired_today = True

            elif entry_mode == "step_down":
                # Buy at each additional dip_threshold step: -5%, -10%, -15%, ...
                current_step = int(drop_pct / dip_threshold)
                if current_step > last_step_level and current_step >= 1:
                    should_buy = True
                    last_step_level = current_step

            elif entry_mode == "per_tick":
                # Buy every tick while below threshold (original aggressive mode)
                if drop_pct >= dip_threshold:
                    should_buy = True

            if should_buy:
                if martingale:
                    contracts_to_buy = max(1, last_contracts_bought * 2)
                    max_affordable = max(1, int(equity / initial_margin))
                    contracts_to_buy = min(contracts_to_buy, max_affordable)
                else:
                    contracts_to_buy = max(1, int((equity * position_pct) / initial_margin))

                # Queue order for NEXT minute execution (no look-ahead)
                pending_buy = {
                    'contracts_to_buy': contracts_to_buy,
                    'signal_price': price,
                    'equity_at_signal': equity
                }

        # ── Record state ──
        portfolio_history.append({
            'timestamp': timestamp,
            'price': price,
            'equity': equity,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_contracts': total_contracts
        })

    stop_loss_pct = calculate_stop_loss(portfolio_history)

    return {
        'scenario_name': scenario_name,
        'portfolio_history': portfolio_history,
        'buy_events': buy_events,
        'bankruptcy_time': bankruptcy_time,
        'margin_call_time': margin_call_time,
        'margin_call_count': margin_call_count,
        'final_value': portfolio_history[-1]['equity'] if portfolio_history else initial_capital,
        'max_contracts_held': max_contracts_held,
        'max_notional_exposure': max_notional_exposure,
        'peak_equity': peak_equity,
        'stop_loss_pct': stop_loss_pct,
        'config': {
            'initial_capital': initial_capital,
            'position_pct': position_pct,
            'dip_threshold': dip_threshold,
            'margin_call_delay': margin_call_delay,
            'slippage_bps': slippage_bps,
            'entry_mode': entry_mode,
            'martingale': martingale,
            'block_buys_during_mc': block_buys_during_mc,
            'multiplier': multiplier,
            'initial_margin': initial_margin,
            'maintenance_margin': maintenance_margin,
        }
    }


def calculate_stop_loss(portfolio_history):
    if not portfolio_history:
        return 0
    initial_equity = portfolio_history[0]['equity']
    min_equity = min(p['equity'] for p in portfolio_history)
    max_drawdown_pct = (1 - min_equity / initial_equity) * 100
    if min_equity <= 0 or max_drawdown_pct >= 90:
        return 20.0
    return 15.0
