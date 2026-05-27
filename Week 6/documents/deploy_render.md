# Render.com deployment — step-by-step

Live paper trading dashboard target: `https://<your-app>.onrender.com/`.
Render reads `render.yaml` for service config; secrets entered in UI.

## 0. One-time preparation

- Push the Week 6 repo to **private** GitHub. `.env` is gitignored — verify it
  is NOT in the push (`git status` should not list it).
- Revoke + regenerate Alpaca keys if any were exposed in chat.

## 1. Render account

1. Sign up at https://render.com (GitHub-OAuth).
2. Dashboard → **New → Blueprint**.
3. Connect the GitHub repo → Render reads `render.yaml`.
4. Click **Apply**.

## 2. Set secrets

In Render UI for the service:
- `ALPACA_API_KEY` — paste from Alpaca paper dashboard
- `ALPACA_SECRET_KEY` — paste secret
- `DISCORD_WEBHOOK_URL` — paste webhook URL (optional, but required for alerts)

Deploy will trigger automatically on save.

## 3. Verify

After "live" status:
- Visit `https://<your-app>.onrender.com/api/health` → `{"ok": true, "db_exists": true}`
- Visit `https://<your-app>.onrender.com/` → dashboard renders
- Visit `https://<your-app>.onrender.com/api/status` → kill-switch + hardstop status

## 4. Upgrade to always-on ($7/month)

Free tier sleeps after 15 min inactivity → the 16:00 ET daily cron would miss.
Upgrade to **Starter** ($7/mo) BEFORE Phase 5 (live run).

Service Settings → Instance Type → Starter.

## 5. Persistent state

`render.yaml` mounts a 1GB disk at `/data` and `STATE_DB_PATH=/data/live_state.db`
so SQLite survives redeploys.

## 6. Deliverable artifacts

For the assignment submit:
1. `dashboard_url.txt` — the `onrender.com` URL
2. `dashboard_screenshots/` — at least 3 screenshots (market open / midday / close)
3. `disconnect_drill.md` — output of `python live/drills/drill_disconnect_5min.py`
4. `restart_drill.md` — output of `python live/drills/drill_restart_recovery.py`
5. `drift_report.md` — generate at end of paper run via `live/monitor/reports.month_end_summary`
6. `live_vs_predicted.md` — honest post-mortem comparing live to backtest predictions

## Anti-patterns (don't do)

- ❌ Don't commit `.env` (gitignore is enforced; verify with `git status` before push)
- ❌ Don't use the free tier for the 1-month run (will miss crons)
- ❌ Don't share the live dashboard URL publicly (paper account but still your account)
