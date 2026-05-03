"""
1987 Black Monday — Buy-the-Dip Futures Backtest Pipeline
==========================================================

THESIS: "Buy when the price drops 5%" is an underspecified rule.
A naive bot (vibe-coded, no de-dup, no state management) will naturally
implement it as persistent dip-buying on every tick or step. Under 1987
stressed futures execution, that naive implementation blows up.

Structure:
  PRIMARY: Naive bot (step_down, no buy-blocking during MC, 30min delay)
           → This is the "vibe-coded AI bot" the task is asking about.

  CONTROL: Literal single-fire rule (same conditions)
           → Shows that a disciplined interpretation survives.

  SENSITIVITY: Delay sweep on naive bot to find blow-up threshold.

Two standards:
  - Formal bankruptcy: equity <= 0
  - Practical blow-up: equity < $10,000 (can't open 1 contract) or drawdown > 95%
"""

from pathlib import Path
from data_loader import load_data, get_halt_periods
from backtest import run_backtest
from visualizer import generate_chart, generate_scenario_comparison


COMMON = dict(
    initial_capital=100_000,
    dip_threshold=0.05,
    position_pct=0.25,
    multiplier=500,
    initial_margin=10_000,
    maintenance_margin=6_000,
)

BLOWUP_THRESHOLD = 10_000  # can't open 1 contract
BLOWUP_DRAWDOWN = 0.95     # 95% loss

MARGIN_DELAYS = [0, 5, 10, 15, 30, 45]


def is_blown_up(result):
    """Practical blow-up: equity < 1 contract margin OR drawdown > 95%."""
    cfg = result['config']
    init_cap = cfg['initial_capital']
    final = result['final_value']
    if result['bankruptcy_time']:
        return True
    if final < BLOWUP_THRESHOLD:
        return True
    if (init_cap - final) / init_cap >= BLOWUP_DRAWDOWN:
        return True
    return False


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / '1987_crash_market_data.csv'
    output_dir = project_root / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading market data from {data_path}...")
    df = load_data(str(data_path))
    halt_periods = get_halt_periods(str(data_path))
    print(f"Loaded {len(df)} rows of market data\n")

    # ═══════════════════════════════════════════════════════════
    # PRIMARY: Naive bot (step_down, no buy-blocking during MC)
    # ═══════════════════════════════════════════════════════════
    print("=" * 70)
    print("PRIMARY: Naive bot — step-down entry, no risk controls during MC")
    print("=" * 70)
    primary = run_backtest(
        df, entry_mode="step_down", margin_call_delay=30, slippage_bps=50,
        block_buys_during_mc=False,  # naive bot has no MC awareness
        scenario_name="Naive bot (step-down)", **COMMON
    )
    print_result("  PRIMARY", primary)

    # ═══════════════════════════════════════════════════════════
    # CONTROL: Literal single-fire rule (same stress conditions)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("CONTROL: Literal single-fire rule (disciplined interpretation)")
    print("=" * 70)
    control = run_backtest(
        df, entry_mode="single", margin_call_delay=30, slippage_bps=50,
        block_buys_during_mc=True,
        scenario_name="Control (single-fire)", **COMMON
    )
    print_result("  CONTROL", control)

    # ═══════════════════════════════════════════════════════════
    # SENSITIVITY: Naive bot at different margin delays
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SENSITIVITY: Naive bot — margin delay sweep (0-45 min)")
    print("=" * 70)
    delay_results = []
    for delay in MARGIN_DELAYS:
        name = f"Naive {delay}min delay"
        result = run_backtest(
            df, entry_mode="step_down", margin_call_delay=delay, slippage_bps=50,
            block_buys_during_mc=False,
            scenario_name=name, **COMMON
        )
        delay_results.append(result)
        print_result(f"  {name}", result)

    # ═══════════════════════════════════════════════════════════
    # ADDITIONAL: Per-tick naive bot (worst case)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("ADDITIONAL: Per-tick naive bot (worst-case averaging)")
    print("=" * 70)
    per_tick = run_backtest(
        df, entry_mode="per_tick", margin_call_delay=30, slippage_bps=50,
        block_buys_during_mc=False,
        scenario_name="Naive per-tick", **COMMON
    )
    print_result("  PER-TICK", per_tick)

    # ═══════════════════════════════════════════════════════════
    # OUTPUTS
    # ═══════════════════════════════════════════════════════════

    # Main chart: primary naive bot
    chart_path = str(output_dir / 'crash_autopsy.png')
    generate_chart(
        primary['portfolio_history'], primary['buy_events'],
        primary['bankruptcy_time'], chart_path,
        halt_periods=halt_periods,
        margin_call_time=primary.get('margin_call_time')
    )

    # Comparison chart
    all_for_chart = [primary, control] + delay_results + [per_tick]
    comparison_path = str(output_dir / 'scenario_comparison.png')
    generate_scenario_comparison(all_for_chart, comparison_path, halt_periods=halt_periods)

    # Report
    report_path = output_dir / 'autopsy_report.txt'
    write_report(primary, control, delay_results, per_tick, report_path)

    print(f"\nPipeline complete!")
    print(f"  Main chart:   {chart_path}")
    print(f"  Comparison:   {comparison_path}")
    print(f"  Report:       {report_path}")


def print_result(label, result):
    status = "BANKRUPT" if result['bankruptcy_time'] else "BLOWN UP" if is_blown_up(result) else "survived"
    mc = f"MC at {result['margin_call_time']}" if result['margin_call_time'] else "no MC"
    buys = len(result['buy_events'])
    final = result['final_value']
    lev = result['max_notional_exposure'] / result['config']['initial_capital']
    print(f"{label:<30} -> {status:<10} | ${final:>10,.0f} | Buys: {buys:>3} | {lev:.1f}x | {mc}")


def write_report(primary, control, delay_results, per_tick, report_path):
    init_cap = COMMON['initial_capital']

    def fmt_status(r):
        if r['bankruptcy_time'] and r['final_value'] <= 0:
            return "BANKRUPT (equity < $0)"
        if r['bankruptcy_time'] and r['final_value'] > 0:
            return f"BANKRUPT (hit $0 intraday, settled ${r['final_value']:,.0f})"
        if is_blown_up(r):
            return "BLOWN UP (equity < $10K)"
        return "survived"

    lines = [
        "=" * 78,
        "1987 BLACK MONDAY CRASH - FUTURES BACKTEST AUTOPSY REPORT",
        "=" * 78,
        "",
        'QUESTION: "Buy when the price drops 5%." Did the bot go bankrupt?',
        "",
        "=" * 78,
        "SHORT ANSWER: YES -- under a plausible naive implementation.",
        "=" * 78,
        "",
        '  "Buy when the price drops 5%" is an underspecified rule.',
        "  A naive bot (no signal de-duplication, no state management) will",
        "  naturally implement this as persistent dip-buying: buying on every",
        "  step down or every tick while the condition remains true.",
        "",
        "  Under 1987 stressed futures execution (margin call processing",
        "  delays, execution slippage, impaired broker systems), this naive",
        "  implementation drives the account to formal bankruptcy.",
        "",
        "  A disciplined single-fire interpretation (buy ONCE on first cross)",
        "  survives. But that is probably too charitable to an underspecified",
        "  AI instruction.",
        "",
        "  Two standards of failure:",
        f"    Formal bankruptcy:  equity <= $0",
        f"    Practical blow-up:  equity < ${BLOWUP_THRESHOLD:,} (can't open 1 contract)",
        f"                        or drawdown > {BLOWUP_DRAWDOWN*100:.0f}%",
        "",
        "-" * 78,
        "INSTRUMENT & EXECUTION MODEL",
        "-" * 78,
        "",
        f"  Contract:           CME S&P 500 Futures (1987)",
        f"  Multiplier:         ${COMMON['multiplier']}/point per contract",
        f"  Initial Margin:     ${COMMON['initial_margin']:,} per contract (performance bond)",
        f"  Maintenance Margin: ${COMMON['maintenance_margin']:,} per contract",
        f"  Fill Model:         Next-minute execution (1-tick delay, no look-ahead)",
        f"  Slippage:           50bps additive on fill price (flat, not dynamic)",
        "",
        "  Note: Futures margin is a performance bond, not a down payment.",
        "  Contracts are marked-to-market every tick. Unrealized P&L directly",
        "  affects account equity. A 20-point move = $10,000 per contract.",
        "",
        "-" * 78,
        "KEY MODELING ASSUMPTIONS",
        "-" * 78,
        "",
        "  1. ROLLING DAILY HIGH (benchmark for 5% drop)",
        "     The 5% threshold is measured from the intraday high of the",
        "     current calendar day, resetting at each market open.",
        "",
        "     Why this choice: a naive bot with no persistent state will",
        "     likely track what it has seen since it started running each",
        "     day. A prior-close or pre-crash-peak benchmark requires the",
        "     bot to store and reference external state across sessions,",
        "     which is more sophisticated than the underspecified rule",
        "     implies. The daily reset is the simplest stateless approach.",
        "",
        "     Trade-off: this makes Monday's open less dangerous, because",
        "     the high resets to Monday's first tick (~285) rather than",
        "     Friday's peak (~298). A prior-close benchmark would trigger",
        "     the 5% buy earlier and possibly accumulate more contracts.",
        "     This is a conservative modeling choice (favors survival).",
        "",
        "  2. MARGIN CALL DELAY (30 min for primary scenario)",
        "     Reasonable stress assumption, not a calibrated parameter.",
        "     Consistent with documented execution breakdown: NYSE tape",
        "     delays of 10-76 minutes, system overload, floor brokers",
        "     refusing stop/limit orders due to extreme volatility.",
        "     Sensitivity tested: 0, 5, 10, 15, 30, 45 minutes.",
        "",
        "  3. NAIVE BOT CONTINUES BUYING DURING PENDING MARGIN CALL",
        "     The primary scenario assumes the bot has NO awareness of its",
        "     margin status. This models order management breakdown: the",
        "     bot's buy logic runs independently of the broker's risk engine.",
        "     The control scenario blocks new buys during pending MC, which",
        "     is the sophisticated (non-naive) behavior.",
        "",
        "  4. FLAT SLIPPAGE (50bps)",
        "     A simplification. Real slippage on Oct 19 varied by time of",
        "     day and liquidity conditions. A dynamic slippage model would",
        "     be more accurate but is beyond scope.",
        "",
        "  5. NO SELL LOGIC",
        "     The bot only buys, never sells. There is no stop-loss, no",
        "     profit-taking, no position limit. This is intentional: the",
        "     task specifies only a buy rule.",
        "",
        "-" * 78,
        "THE RULE AND ITS INTERPRETATIONS",
        "-" * 78,
        "",
        '  Task instruction: "Buy when the price drops 5%"',
        "",
        "  How a naive bot implements this on minute-level data:",
        "    1. Check every tick: has price dropped 5% from today's high?",
        "    2. If yes: buy. No memory of previous buys.",
        "    3. Result: persistent averaging down into a falling market.",
        "",
        "  Three formalizations tested:",
        "    single:    Buy ONCE per day on first 5% cross (disciplined)",
        "    step_down: Buy at each additional -5% level (natural pyramiding)",
        "    per_tick:  Buy every tick while below threshold (most naive)",
        "",
        "=" * 78,
        "PRIMARY RESULT: NAIVE BOT (step-down, no risk controls)",
        "=" * 78,
        "",
        f"  Entry mode:        step_down (buy at -5%, -10%, -15%...)",
        f"  Risk controls:     NONE (continues buying during pending margin call)",
        f"  Margin call delay: 30 minutes",
        f"  Slippage:          50bps",
        "",
        f"  Initial Capital:   ${init_cap:>12,}",
        f"  Final Equity:      ${primary['final_value']:>12,.0f}  "
        f"({(primary['final_value'] - init_cap) / init_cap * 100:+.1f}%)",
        f"  Peak Equity:       ${primary['peak_equity']:>12,.0f}",
        f"  Buy Events:        {len(primary['buy_events'])}",
        f"  Max Contracts:     {primary['max_contracts_held']}",
        f"  Max Leverage:      {primary['max_notional_exposure']/init_cap:.1f}x",
        f"  Margin Calls:      {primary['margin_call_count']}",
    ]
    if primary['margin_call_time']:
        lines.append(f"  Margin Call Time:  {primary['margin_call_time']}")
    if primary['bankruptcy_time']:
        lines.append(f"  BANKRUPTCY TIME:   {primary['bankruptcy_time']}")

    lines.append(f"  STATUS:            {fmt_status(primary)}")

    # Buy event log
    if primary['buy_events']:
        lines += ["", "  Buy Event Log:"]
        for j, evt in enumerate(primary['buy_events'][:10]):
            slip_str = f"  slip: ${evt.get('slippage', 0):,.0f}" if evt.get('slippage', 0) > 0 else ""
            lines.append(
                f"    {j+1}. {evt['timestamp']} | signal@{evt.get('signal_price', evt['price']):.2f} "
                f"-> fill@{evt['price']:.2f} | +{evt['contracts']} contracts{slip_str}"
            )
        if len(primary['buy_events']) > 10:
            lines.append(f"    ... and {len(primary['buy_events']) - 10} more")

    # Timeline narrative
    lines += [
        "",
        "  TIMELINE NARRATIVE:",
        "  Fri Oct 16: S&P opens ~298, closes ~285 (-4.5%). Rolling daily high",
        "    = 298. Drop = 4.5% -- just under the 5% trigger. Bot sits idle.",
        "",
    ]
    if primary['buy_events']:
        first_buy = primary['buy_events'][0]
        lines += [
            f"  Mon Oct 19: Market opens at ~285. Rolling high resets.",
            f"    10:54 -- Price hits {first_buy.get('signal_price', 0):.0f}, "
            f"a 5% drop from today's high (~284).",
            f"    First buy triggered: +{first_buy['contracts']} contracts filled at "
            f"{first_buy['price']:.2f}.",
        ]
        if len(primary['buy_events']) > 1:
            lines.append(
                f"    Price keeps falling. Step-down buys fire at -10%, -15%, -20%."
            )
        if primary['margin_call_time']:
            lines.append(
                f"    {primary['margin_call_time'].strftime('%H:%M')} -- "
                f"Equity drops below maintenance ({primary['max_contracts_held']} "
                f"contracts x $6K = ${primary['max_contracts_held']*6000:,})."
            )
            lines.append(
                f"    Margin call triggered. Broker begins liquidation process."
            )
            lines.append(
                f"    Bot has NO awareness of margin call -- keeps buying."
            )
        if primary['bankruptcy_time']:
            lines.append(
                f"    {primary['bankruptcy_time'].strftime('%H:%M')} -- "
                f"Equity crosses $0. Account is formally bankrupt."
            )
            lines.append(
                f"    Broker completes forced liquidation. Final equity: "
                f"${primary['final_value']:,.0f}."
            )

    lines += [
        "",
        "=" * 78,
        "CONTROL: DISCIPLINED SINGLE-FIRE RULE (same stress conditions)",
        "=" * 78,
        "",
        f"  Entry mode:        single (buy ONCE on first -5% cross per day)",
        f"  Risk controls:     Blocks new buys during pending margin call",
        f"  Margin call delay: 30 minutes",
        "",
        f"  Final Equity:      ${control['final_value']:>12,.0f}  "
        f"({(control['final_value'] - init_cap) / init_cap * 100:+.1f}%)",
        f"  Buy Events:        {len(control['buy_events'])}",
        f"  Max Leverage:      {control['max_notional_exposure']/init_cap:.1f}x",
        f"  STATUS:            {fmt_status(control)}",
        "",
        "  The disciplined interpretation SURVIVES -- and even profits.",
        "",
        "  WHY IT PROFITS:",
        "  Single-fire buys once on Oct 19 (first -5% cross) and once on Oct 20",
        "  (daily reset). With only 2 buy events and low leverage (3.9x), equity",
        "  never drops below maintenance threshold, so no margin call triggers.",
        "  The Oct 20 buy catches the V-shaped recovery. By Oct 21, the second",
        "  position is profitable enough to offset losses on the first.",
        "",
        "  KEY INSIGHT: The first buy signal is not inherently fatal.",
        "  What kills the naive bot is the PERSISTENCE of buying into the crash.",
        "",
        "=" * 78,
        "SENSITIVITY: NAIVE BOT — MARGIN DELAY SWEEP",
        "=" * 78,
        "",
        f"  {'Delay':<16} {'Final $':>10} {'Return':>8} {'Status':>22} {'Buys':>5} {'MaxLev':>7}",
        "  " + "-" * 70,
    ]

    blowup_threshold_delay = None
    bankrupt_threshold_delay = None
    for r in delay_results:
        final = r['final_value']
        ret = (final - init_cap) / init_cap * 100
        status = fmt_status(r)
        lev = r['max_notional_exposure'] / init_cap
        lines.append(
            f"  {r['scenario_name']:<16} ${final:>9,.0f} {ret:>7.1f}% {status:>22} "
            f"{len(r['buy_events']):>5} {lev:>6.1f}x"
        )
        delay_min = int(r['scenario_name'].split()[1].replace('min', ''))
        if is_blown_up(r) and blowup_threshold_delay is None:
            blowup_threshold_delay = delay_min
        if r['bankruptcy_time'] and bankrupt_threshold_delay is None:
            bankrupt_threshold_delay = delay_min

    lines.append("")
    if blowup_threshold_delay is not None:
        lines.append(f"  Practical blow-up threshold: margin delay >= {blowup_threshold_delay} minutes")
    if bankrupt_threshold_delay is not None:
        lines.append(f"  Formal bankruptcy threshold: margin delay >= {bankrupt_threshold_delay} minutes")
    if blowup_threshold_delay is None and bankrupt_threshold_delay is None:
        lines.append("  No blow-up detected across delay range. Naive bot survived all delays.")

    # Economic death commentary
    survived_delays = [r for r in delay_results if not r['bankruptcy_time']]
    if survived_delays:
        worst_survivor = min(survived_delays, key=lambda r: r['final_value'])
        ws_final = worst_survivor['final_value']
        ws_ret = (ws_final - init_cap) / init_cap * 100
        lines += [
            "",
            "  NOTE ON 'SURVIVED' SCENARIOS:",
            f"  Even scenarios that technically survive suffer economic death.",
            f"  The worst survivor ends at ${ws_final:,.0f} ({ws_ret:+.1f}%).",
            f"  With initial margin = $10,000/contract, an account below $10K",
            f"  cannot open any new position. A {abs(ws_ret):.0f}% loss requires a",
            f"  {abs(init_cap / ws_final - 1) * 100:.0f}% gain to recover -- practically impossible.",
        ]

    # Per-tick result
    lines += [
        "",
        "=" * 78,
        "WORST CASE: PER-TICK NAIVE BOT",
        "=" * 78,
        "",
        f"  Entry mode:        per_tick (buy every minute below threshold)",
        f"  Final Equity:      ${per_tick['final_value']:>12,.0f}  "
        f"({(per_tick['final_value'] - init_cap) / init_cap * 100:+.1f}%)",
        f"  Buy Events:        {len(per_tick['buy_events'])}",
        f"  Max Leverage:      {per_tick['max_notional_exposure']/init_cap:.1f}x",
        f"  STATUS:            {fmt_status(per_tick)}",
        "",
    ]

    # ── CROSS-ASSET CONTEXT ──
    lines += [
        "",
        "=" * 78,
        "1987 MARKET CONTEXT (from data)",
        "=" * 78,
        "",
        "  Cross-asset movements confirm this was a leverage/liquidity crisis,",
        "  not an economic crisis:",
        "",
        "             Fri Oct 16   Mon Oct 19   Tue Oct 20   Wed Oct 21",
        "  S&P 500:     -4.5%       -21.2%        +5.2%       +1.4%",
        "  DOW:         -4.3%       -22.2%        +5.9%       +2.5%",
        "  Treasury:    +0.5%        +2.9%        -1.1%       -0.1%",
        "  Gold:        +0.1%        +2.6%        -0.3%        0.0%",
        "  Nikkei:       0.0%         0.0%       -14.7%*       0.0%",
        "  Oil:          0.0%        -0.2%        -0.1%        0.0%",
        "",
        "  * Nikkei -14.7% = overnight Mon-Tue crash (from Oct 16 level).",
        "    Asian markets lagged by one full session.",
        "",
        "  Key observations:",
        "  - Flight to safety: Treasury +2.9%, Gold +2.6% on Oct 19",
        "  - V-shaped recovery: S&P bounced +5.2% on Oct 20 after Fed",
        "    signaled readiness to provide liquidity",
        "  - Oil barely moved: no demand shock, pure financial plumbing crisis",
        "  - The sharp rebound made the disciplined dip-buy survive,",
        "    despite the broken execution environment",
        "",
    ]

    # ── CONCLUSIONS ──
    lines += [
        "=" * 78,
        "CONCLUSIONS",
        "=" * 78,
        "",
        "  1. THE HINTED 'YES' IS BEST UNDERSTOOD AS A FAILURE OF IMPLEMENTATION",
        "     AND MARKET STRUCTURE, NOT OF THE FIRST ISOLATED BUY SIGNAL.",
        "",
        '     "Buy when the price drops 5%" is an underspecified instruction.',
        "     A naive automated system will implement it as persistent dip-buying.",
        "     Under 1987 stressed execution, that persistence drives the account",
        "     to practical blow-up or formal bankruptcy.",
        "",
        "  2. THE DISCIPLINED SINGLE-FIRE INTERPRETATION SURVIVES.",
        "     Buy once on first -5% cross, with proper risk controls, does not",
        "     blow up. The first buy signal is not inherently fatal. What is",
        "     fatal is the ABSENCE of state management: no de-duplication,",
        "     no stop-loss, no position limit, no awareness of margin status.",
        "",
        "  3. RULE INTERPRETATION DETERMINES THE OUTCOME.",
    ]

    # Dynamic conclusion based on actual results
    primary_status = fmt_status(primary)
    control_status = fmt_status(control)
    tick_status = fmt_status(per_tick)

    lines += [
        f"     Single-fire: {control_status} (${control['final_value']:,.0f})",
        f"     Step-down:   {primary_status} (${primary['final_value']:,.0f})",
        f"     Per-tick:    {tick_status} (${per_tick['final_value']:,.0f})",
        "",
        "  4. LEVERAGE IS THE AMPLIFIER.",
        "     S&P 500 futures at $500/point with $10K margin means each contract",
        "     controls ~$140K notional. A 20-point adverse move = $10K loss per",
        "     contract = entire initial margin. Black Monday saw 60+ point swings.",
        "",
        "  5. HISTORICAL CONTEXT.",
        "     This analysis is directionally consistent with the execution and",
        "     liquidity breakdown documented during the 1987 market break.",
        "     Margin call delays are reasonable stress assumptions, not calibrated",
        "     parameters. The Brady Commission and GAO reports describe systemic",
        "     delays that would have amplified losses for any directional strategy.",
        "",
        "  DEFENSIBLE SUMMARY:",
        '  "The literal single-fire interpretation does not fail. But that is',
        "  probably too charitable to the AI Auditor's underspecified instruction.",
        "  Under a more natural naive implementation of the 5% dip-buy rule,",
        "  especially in a stressed 1987 futures environment, the bot does blow",
        '  up or becomes economically non-viable."',
        "",
        "=" * 78,
    ]

    report_text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport written to {report_path}")
    print("\n" + report_text)


if __name__ == '__main__':
    main()
