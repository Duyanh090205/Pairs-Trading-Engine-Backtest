"""Compute realized slippage + P&L from live snapshot JSON files.

Run with the path to an exit_<ts> snapshot folder after Friday EOM:
  python submission_data/compute_slippage.py submission_data/exit_<ts>.json

Or against the live API directly:
  python submission_data/compute_slippage.py --live

Outputs per-leg, per-pair, aggregate tables suitable for the submission write-up.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def fetch_live() -> dict:
    """Fetch fresh data from the live dashboard."""
    import urllib.request
    base = "https://week6-paper-dashboard.onrender.com/api"
    out = {}
    for endpoint, key in [
        ("orders?limit=200", "orders"),
        ("positions", "positions"),
        ("trades?limit=50", "trades"),
        ("pnl", "pnl"),
    ]:
        with urllib.request.urlopen(f"{base}/{endpoint}", timeout=10) as r:
            out[key] = json.loads(r.read())
    return out


def adverse_bps(side: str, decision_px: float, fill_px: float) -> float:
    """Adverse slippage in bps (positive = bad for us)."""
    if not decision_px or not fill_px:
        return 0.0
    raw = fill_px - decision_px
    if side == "buy":
        return (raw / decision_px) * 10000     # paid more = adverse
    return (-raw / decision_px) * 10000        # sell: received less = adverse


def analyze_orders(orders: list[dict]) -> None:
    filled = [o for o in orders
              if o.get("status") in ("filled", "OrderStatus.FILLED")
              and o.get("fill_price") and o.get("decision_price")]

    if not filled:
        print("No filled orders with both decision_price and fill_price.")
        return

    print("\n=== Per-leg slippage (entry + exit combined) ===")
    print(f"{'submitted_ts':<24} {'pair':<14} {'ticker':<6} {'side':<5} "
          f"{'qty':>5} {'decision':>10} {'fill':>10} {'adv_bps':>9}")
    print("-" * 96)
    per_pair_legs: dict[str, list] = defaultdict(list)
    total_adv_usd = 0.0
    total_notional = 0.0
    leg_bps: list[float] = []
    for o in sorted(filled, key=lambda x: x["submitted_ts"]):
        dec = float(o["decision_price"])
        fill = float(o["fill_price"])
        qty = float(o["fill_qty"] or o["qty"])
        side = o["side"]
        bps = adverse_bps(side, dec, fill)
        adv_usd = bps / 10000 * qty * dec
        notional = qty * fill
        total_adv_usd += adv_usd
        total_notional += notional
        leg_bps.append(bps)
        per_pair_legs[o["pair_id"]].append(o)
        print(f"{o['submitted_ts'][:23]:<24} {o['pair_id']:<14} "
              f"{o['ticker']:<6} {side:<5} {qty:>5.0f} "
              f"{dec:>10.4f} {fill:>10.4f} {bps:>+9.2f}")

    print("-" * 96)
    print(f"\n=== Aggregate ===")
    print(f"Total filled legs:        {len(filled)}")
    print(f"Total notional traded:    ${total_notional:,.2f}")
    print(f"Total adverse slip $:     ${total_adv_usd:+.2f}")
    print(f"Aggregate slip (bps):     {(total_adv_usd/total_notional*10000) if total_notional else 0:+.2f}")
    print(f"Mean per-leg slip (bps):  {sum(leg_bps)/len(leg_bps):+.2f}")
    print(f"Median per-leg (bps):     {sorted(leg_bps)[len(leg_bps)//2]:+.2f}")
    print(f"Max adverse leg:          {max(leg_bps):+.2f} bps")
    print(f"Min (best) leg:           {min(leg_bps):+.2f} bps")
    print(f"Backtest assumed cost:    +22 bps/trade (target to compare)")


def analyze_positions(positions: list[dict], trades: list[dict]) -> None:
    print("\n=== Open positions ===")
    if not positions:
        print("(none — all flat)")
    for p in positions:
        gross = abs(p["notional_a"]) + abs(p["notional_b"])
        net = (p["notional_a"] if p["direction"] == 1 else -p["notional_a"]) \
            + (-p["notional_b"] if p["direction"] == 1 else p["notional_b"])
        print(f"  {p['pair_id']:<14} dir={p['direction']:+d} "
              f"notA=${p['notional_a']:>7.0f} notB=${p['notional_b']:>7.0f} "
              f"gross=${gross:>7.0f} net=${net:>+7.0f} entry_z={p['entry_z']:+.3f}")

    print("\n=== Closed trades (realized) ===")
    if not trades:
        print("(none yet — submit Friday after EOM flatten)")
        return
    total_pnl = 0.0
    for t in trades:
        pnl = t.get("realized_pnl") or 0
        total_pnl += pnl
        print(f"  {t['pair_id']:<14} entry={t['entry_ts'][:19]} exit={t['exit_ts'][:19]} "
              f"entry_z={t.get('entry_z',0):+.2f} exit_z={t.get('exit_z',0):+.2f} "
              f"reason={t.get('exit_reason','?'):<12} pnl=${pnl:+.2f}")
    print(f"\nTotal realized P&L: ${total_pnl:+.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", help="Snapshot folder OR --live")
    p.add_argument("--live", action="store_true")
    args = p.parse_args()

    if args.live or args.path == "--live":
        data = fetch_live()
        print(f"Using LIVE data from dashboard.\n")
    elif args.path:
        # Find latest exit_*.json files in the folder
        snap_dir = Path(args.path)
        if snap_dir.is_dir():
            ords = sorted(snap_dir.glob("orders_*.json"))[-1]
            pos = sorted(snap_dir.glob("positions_*.json"))[-1]
            data = {
                "orders": json.loads(ords.read_text()),
                "positions": json.loads(pos.read_text()),
                "trades": {"rows": []},
            }
            try:
                tr = sorted(snap_dir.glob("trades_*.json"))[-1]
                data["trades"] = json.loads(tr.read_text())
            except Exception:
                pass
            print(f"Loaded snapshot from {snap_dir}\n")
        else:
            print(f"Path not found: {snap_dir}")
            return 1
    else:
        print("Usage: python compute_slippage.py --live   OR   <snapshot_folder>")
        return 1

    analyze_orders(data["orders"]["rows"])
    analyze_positions(
        data["positions"]["rows"],
        data.get("trades", {}).get("rows", []),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
