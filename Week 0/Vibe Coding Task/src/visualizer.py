import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_chart(portfolio_history, buy_events, bankruptcy_time, output_path, halt_periods=None, margin_call_time=None):
    """
    Generate two-panel chart: top panel shows account equity crash, bottom shows S&P 500 price.
    Uses integer index x-axis to eliminate weekend gaps. Displays trading hours only (9:30-16:00).
    Lines break at trading halts (NaN inserted at halt boundaries).
    """
    fig, (ax_equity, ax_price) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 1.2],
                                                sharex=True, gridspec_kw={'hspace': 0.08})

    # Extract data
    timestamps = [p['timestamp'] for p in portfolio_history]
    prices = [p['price'] for p in portfolio_history]
    account_equity = [p['equity'] for p in portfolio_history]
    indices = list(range(len(timestamps)))

    # Build halt-gap-aware plot arrays: insert NaN at halt boundaries so PRICE line breaks
    # Equity stays continuous (account value doesn't disappear during halts)
    prices_plot = list(prices)
    if halt_periods:
        for halt_start, halt_end in halt_periods:
            before_idx = None
            after_idx = None
            for i, ts in enumerate(timestamps):
                if ts < halt_start:
                    before_idx = i
                if after_idx is None and ts > halt_end:
                    after_idx = i
            # NaN out price indices within the visual halt band region
            if before_idx is not None and after_idx is not None:
                halt_duration_min = (halt_end - halt_start).total_seconds() / 60
                margin = int(halt_duration_min / 2)
                nan_start = max(0, before_idx - margin)
                nan_end = min(len(prices_plot) - 1, after_idx + margin)
                for j in range(nan_start, nan_end + 1):
                    prices_plot[j] = float('nan')

    # ── TOP PANEL: Account Equity ──
    ax_equity.plot(indices, account_equity, color='#2ca02c', linewidth=2.5, label='Account Equity', zorder=3)
    # Fill green above zero, red below zero
    ax_equity.fill_between(indices, account_equity, 0,
                           where=[e >= 0 for e in account_equity],
                           alpha=0.15, color='#2ca02c', interpolate=True, zorder=1)
    ax_equity.fill_between(indices, account_equity, 0,
                           where=[e < 0 for e in account_equity],
                           alpha=0.2, color='#d62728', interpolate=True, zorder=1)
    ax_equity.axhline(y=0, color='#d62728', linestyle='-', linewidth=1.5, alpha=0.7, label='Zero Line')
    ax_equity.axhline(y=100_000, color='#999999', linestyle=':', linewidth=1, alpha=0.5, label='Initial Capital ($100K)')
    ax_equity.set_ylabel('Account Equity ($)', fontsize=11, color='#2ca02c', fontweight='bold')
    ax_equity.tick_params(axis='y', labelcolor='#333')
    ax_equity.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_equity.set_axisbelow(True)

    # Annotate final equity (matches report number exactly)
    final_eq = account_equity[-1]
    final_eq_idx = len(account_equity) - 1
    ax_equity.annotate(f'Final: ${final_eq:,.0f}', xy=(final_eq_idx, final_eq),
                       xytext=(final_eq_idx - 120, final_eq + 25000),
                       fontsize=10, fontweight='bold', color='#d62728',
                       arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d62728', alpha=0.9))

    # Mark margin call on equity panel too
    if margin_call_time:
        try:
            mc_idx = next(i for i, p in enumerate(portfolio_history) if p['timestamp'] == margin_call_time)
            ax_equity.axvline(x=mc_idx, color='#d62728', linestyle='--', linewidth=2, alpha=0.8, zorder=5)
            ax_equity.text(mc_idx + 3, max(account_equity) * 0.85, 'MARGIN CALL\nLIQUIDATED',
                          fontsize=10, color='#d62728', ha='left', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#d62728', linewidth=2))
        except StopIteration:
            pass

    # Mark bankruptcy on equity panel
    if bankruptcy_time:
        try:
            bk_idx = next(i for i, p in enumerate(portfolio_history) if p['timestamp'] == bankruptcy_time)
            ax_equity.scatter([bk_idx], [account_equity[bk_idx]], color='#8b0000', s=250, marker='X', zorder=6, label='Bankruptcy')
        except StopIteration:
            pass

    # Equity panel y-axis: show full range including negative
    min_eq_val = min(account_equity)
    eq_min = min(min_eq_val * 1.15 if min_eq_val < 0 else min_eq_val * 0.85, -5000)
    eq_max = max(account_equity) * 1.08
    ax_equity.set_ylim([eq_min, eq_max])

    # Equity legend
    ax_equity.legend(loc='upper left', fontsize=9, framealpha=0.95)

    # ── BOTTOM PANEL: S&P 500 Futures Price ──
    ax_price.plot(indices, prices_plot, color='#1f77b4', linewidth=2.5, label='S&P 500 Futures', zorder=3)
    ax_price.set_ylabel('S&P 500 Index Level', fontsize=11, color='#1f77b4', fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='#1f77b4')
    ax_price.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_price.set_axisbelow(True)

    # Mark margin call on price panel
    if margin_call_time:
        try:
            mc_idx = next(i for i, p in enumerate(portfolio_history) if p['timestamp'] == margin_call_time)
            ax_price.axvline(x=mc_idx, color='#d62728', linestyle='--', linewidth=2.5, alpha=0.8, zorder=5, label='Margin Call')
        except StopIteration:
            pass

    # Mark trading halts on BOTH panels
    halt_patch = mpatches.Patch(color='#ffeb99', alpha=0.4, label='Trading halts')
    if halt_periods:
        for halt_start, halt_end in halt_periods:
            before_idx = None
            after_idx = None
            for i, ts in enumerate(timestamps):
                if ts < halt_start:
                    before_idx = i
                if after_idx is None and ts > halt_end:
                    after_idx = i

            if before_idx is not None and after_idx is not None:
                halt_duration_min = (halt_end - halt_start).total_seconds() / 60
                margin = int(halt_duration_min / 2)

                for ax in [ax_equity, ax_price]:
                    ax.axvspan(before_idx + 0.5 - margin, after_idx - 0.5 + margin,
                              color='#ffeb99', alpha=0.5, zorder=0, edgecolor='#ffd700', linewidth=2)

                # Halt label on price panel only
                mid_idx = (before_idx + after_idx) / 2
                ax_price.text(mid_idx, max(prices) * 0.92, f"HALT\n{int(halt_duration_min)}m",
                             fontsize=9, color='#b8860b', ha='center', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#ffd700'))

    # Mark buy events on price panel
    if buy_events:
        buy_indices = []
        buy_prices_list = []
        for event in buy_events:
            try:
                event_idx = next(i for i, ts in enumerate(timestamps) if ts == event['timestamp'])
                buy_indices.append(event_idx)
                buy_prices_list.append(event['price'])
            except StopIteration:
                pass

        if buy_indices:
            ax_price.scatter(buy_indices, buy_prices_list, color='#ff3333', s=120, marker='v',
                            zorder=5, alpha=0.8, edgecolors='#8b0000', linewidth=1.5, label='Buy Events')
            ax_price.scatter(buy_indices, buy_prices_list, color='none', s=180, marker='o',
                            zorder=4, edgecolors='#ff6666', linewidth=1, alpha=0.5)

    # Price panel y-axis
    ax_price.set_ylim([min(prices) * 0.98, max(prices) * 1.02])

    # X-axis: day boundaries (on bottom panel only since sharex)
    unique_dates = []
    day_start_indices = []
    current_date = None
    for i, ts in enumerate(timestamps):
        if ts.date() != current_date:
            current_date = ts.date()
            unique_dates.append(current_date)
            day_start_indices.append(i)

    ax_price.set_xticks(day_start_indices)
    day_labels = [f"{d.strftime('%a').capitalize()}\n{d.strftime('%b %d')}" for d in unique_dates]
    ax_price.set_xticklabels(day_labels)
    ax_price.set_xlim(-10, len(indices) + 10)
    ax_price.set_xlabel('Trading Day (9:30-16:00 EST)', fontsize=11, color='#333', fontweight='bold')

    # Price panel legend
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    ax_price.legend(price_handles + [halt_patch], price_labels + ['Trading halts'],
                   loc='upper left', fontsize=9, framealpha=0.95)

    # Title on top panel
    ax_equity.set_title('Naive Step-Down Bot — 30min Margin Delay\nS&P 500 Futures — Black Monday 1987',
                       fontsize=13, fontweight='bold', pad=15)

    # Source
    fig.text(0.5, 0.01, 'Source: 1987_crash_market_data.csv | S&P 500 Futures, trading hours only (9:30-16:00 EST, no weekends)',
            ha='center', fontsize=9, style='italic', color='#666')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Chart saved to {output_path}")


def generate_scenario_comparison(all_results, output_path, halt_periods=None):
    """
    Clean 2-panel chart showing 4 key equity curves + results table.
    Only plots: Primary (naive), Control (single-fire), best delay survivor, per-tick.
    Full results table shows all scenarios.
    """
    fig, (ax_eq, ax_price) = plt.subplots(
        2, 1, figsize=(16, 9), height_ratios=[1.5, 1],
        sharex=True, gridspec_kw={'hspace': 0.06})

    base = all_results[0]
    timestamps = [p['timestamp'] for p in base['portfolio_history']]
    prices = [p['price'] for p in base['portfolio_history']]
    indices = list(range(len(timestamps)))

    # ── Halt bands ──
    halt_spans = []
    if halt_periods:
        for halt_start, halt_end in halt_periods:
            before_idx = after_idx = None
            for j, ts in enumerate(timestamps):
                if ts < halt_start:
                    before_idx = j
                if after_idx is None and ts > halt_end:
                    after_idx = j
            if before_idx is not None and after_idx is not None:
                dur = (halt_end - halt_start).total_seconds() / 60
                margin = int(dur / 2)
                halt_spans.append((before_idx, after_idx, margin, dur))

    for ax in [ax_eq, ax_price]:
        for bi, ai, m, d in halt_spans:
            ax.axvspan(bi + 0.5 - m, ai - 0.5 + m, color='#ffeb99', alpha=0.3, zorder=0)

    # ── Select 4 key scenarios to plot ──
    # 1: Primary (naive bot) — always first
    # 2: Control (single-fire) — find by name
    # 3: Best surviving delay scenario (highest final value among delay results)
    # 4: Per-tick (worst case)
    plot_scenarios = [base]  # primary always plotted

    control = next((r for r in all_results if 'Control' in r['scenario_name']), None)
    if control:
        plot_scenarios.append(control)

    # Best surviving delay (instant MC, 0min)
    delay_survived = [r for r in all_results if 'delay' in r['scenario_name'].lower()
                      and not r['bankruptcy_time']]
    if delay_survived:
        best_delay = max(delay_survived, key=lambda r: r['final_value'])
        plot_scenarios.append(best_delay)

    per_tick = next((r for r in all_results if 'per-tick' in r['scenario_name'].lower()), None)
    if per_tick:
        plot_scenarios.append(per_tick)

    # Style map for the 4 lines
    STYLES = {
        0: ('#2ca02c', '-', 3.5, 1.0, 'Naive bot (step-down, 30min delay)'),
        1: ('#e07020', '-', 2.8, 0.9, 'Control: single-fire (survived)'),
        2: ('#1f77b4', '--', 2.0, 0.7, 'Naive 0min delay (survived)'),
        3: ('#7b2d8e', '-.', 2.0, 0.7, 'Per-tick averaging (worst case)'),
    }

    # ── Plot the 4 lines (secondaries first, primary last) ──
    for idx in reversed(range(len(plot_scenarios))):
        result = plot_scenarios[idx]
        history = result['portfolio_history']
        equity = [p['equity'] for p in history]
        if len(equity) < len(indices):
            equity = equity + [equity[-1]] * (len(indices) - len(equity))
        equity = equity[:len(indices)]

        color, style, width, alpha, label = STYLES[idx]
        zorder = 5 if idx == 0 else 3
        ax_eq.plot(indices, equity, color=color, linewidth=width, linestyle=style,
                  zorder=zorder, alpha=alpha, label=label)

        # Mark bankruptcy
        if result['bankruptcy_time']:
            try:
                bk_idx = next(j for j, p in enumerate(history) if p['timestamp'] == result['bankruptcy_time'])
                is_primary = idx == 0
                ax_eq.scatter([bk_idx], [equity[bk_idx]], color=color,
                             s=200 if is_primary else 100,
                             marker='X', zorder=10, edgecolors='black',
                             linewidth=1.5 if is_primary else 1)
                if is_primary:
                    ax_eq.annotate('BANKRUPT', xy=(bk_idx, equity[bk_idx]),
                                  xytext=(bk_idx + 40, equity[bk_idx] - 20000),
                                  fontsize=10, fontweight='bold', color=color,
                                  arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           edgecolor=color, alpha=0.9))
            except StopIteration:
                pass

    # ── Equity panel styling ──
    ax_eq.axhline(y=0, color='#d62728', linestyle='-', linewidth=1.5, alpha=0.5)
    ax_eq.axhline(y=100_000, color='#999', linestyle=':', linewidth=1, alpha=0.3)
    ax_eq.set_ylabel('Account Equity ($)', fontsize=11, fontweight='bold')
    ax_eq.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax_eq.set_axisbelow(True)

    # Legend — only 4 items, clean
    ax_eq.legend(loc='upper right', fontsize=9, framealpha=0.95, handlelength=2.5)

    ax_eq.set_title('Scenario Comparison: What Drives Bankruptcy?\n'
                    'S&P 500 Futures — Buy-the-Dip Strategy — Black Monday 1987',
                    fontsize=13, fontweight='bold', pad=12)

    # ── Results table (bottom-left, all scenarios) ──
    header = f"{'Scenario':<28} {'Final':>9} {'Return':>7} {'Status':<8}"
    divider = "-" * 58
    table_lines = [header, divider]
    for r in all_results:
        cfg = r['config']
        final = r['final_value']
        init = cfg['initial_capital']
        ret = (final - init) / init * 100
        if r['bankruptcy_time']:
            status = "BANKRUPT"
        elif final < 10_000:
            status = "BLOWN UP"
        else:
            status = "survived"
        name = r['scenario_name'][:27]
        arrow = ">> " if r is base else "   "
        table_lines.append(f"{arrow}{name:<25} ${final:>8,.0f} {ret:>+6.1f}% {status}")

    table_text = "\n".join(table_lines)
    ax_eq.text(0.01, 0.02, table_text, transform=ax_eq.transAxes,
              fontsize=7, fontfamily='monospace', verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#999'))

    # ── Bottom panel: S&P 500 price ──
    prices_plot = list(prices)
    for bi, ai, m, d in halt_spans:
        nan_start = max(0, bi - m)
        nan_end = min(len(prices_plot) - 1, ai + m)
        for j in range(nan_start, nan_end + 1):
            prices_plot[j] = float('nan')

    ax_price.plot(indices, prices_plot, color='#1f77b4', linewidth=2, zorder=3)
    ax_price.set_ylabel('S&P 500 Futures', fontsize=11, color='#1f77b4', fontweight='bold')
    ax_price.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax_price.set_axisbelow(True)

    for bi, ai, m, d in halt_spans:
        mid = (bi + ai) / 2
        ax_price.text(mid, max(prices) * 0.93, f"HALT\n{int(d)}m",
                     fontsize=8, color='#b8860b', ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#ffd700'))

    # X-axis
    unique_dates = []
    day_start_indices = []
    current_date = None
    for j, ts in enumerate(timestamps):
        if ts.date() != current_date:
            current_date = ts.date()
            unique_dates.append(current_date)
            day_start_indices.append(j)

    ax_price.set_xticks(day_start_indices)
    day_labels = [f"{d.strftime('%a').capitalize()}\n{d.strftime('%b %d')}" for d in unique_dates]
    ax_price.set_xticklabels(day_labels)
    ax_price.set_xlim(-10, len(indices) + 10)
    ax_price.set_xlabel('Trading Day (9:30-16:00 EST)', fontsize=11, color='#333', fontweight='bold')

    fig.text(0.5, 0.005,
             'Source: 1987_crash_market_data.csv | S&P 500 Futures | Sensitivity analysis',
             ha='center', fontsize=8, style='italic', color='#666')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scenario comparison saved to {output_path}")
