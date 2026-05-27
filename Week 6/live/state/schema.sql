-- Week 6 live paper trading state store
-- Single SQLite file. ACID, recoverable after kill -9.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- Open / historical pair positions
CREATE TABLE IF NOT EXISTS positions (
    pair_id          TEXT NOT NULL,
    side_a           TEXT NOT NULL,            -- long ticker
    side_b           TEXT NOT NULL,            -- short ticker (hedge)
    beta             REAL NOT NULL,            -- hedge ratio at entry
    direction        INTEGER NOT NULL,         -- +1 long-A short-B, -1 reverse
    notional_a       REAL NOT NULL,            -- $ per leg
    notional_b       REAL NOT NULL,
    entry_ts         TEXT NOT NULL,            -- ISO8601 UTC
    entry_z          REAL NOT NULL,
    exit_ts          TEXT,                     -- NULL while open
    exit_z           REAL,
    exit_reason      TEXT,                     -- 'zero_cross' | 'hard_sl' | 'eom' | 'kill_switch'
    realized_pnl     REAL,
    predicted_cost_bps REAL,
    realized_cost_bps  REAL,
    PRIMARY KEY (pair_id, entry_ts)
);

CREATE INDEX IF NOT EXISTS idx_positions_open
    ON positions(pair_id) WHERE exit_ts IS NULL;

-- Order ledger (idempotent submission)
CREATE TABLE IF NOT EXISTS orders (
    client_order_id  TEXT PRIMARY KEY,        -- hash(pair_id, bar_ts, leg, side, qty, order_type, limit_price)
    broker_order_id  TEXT,                    -- assigned by Alpaca
    pair_id          TEXT NOT NULL,
    bar_ts           TEXT NOT NULL,           -- decision bar timestamp (groups legs of one entry/exit)
    ticker           TEXT NOT NULL,
    side             TEXT NOT NULL,           -- 'buy' | 'sell'
    qty              REAL NOT NULL,
    order_type       TEXT NOT NULL,           -- 'market' | 'limit' | 'moo'
    limit_price      REAL,
    status           TEXT NOT NULL,           -- 'submitted' | 'filled' | 'partial' | 'rejected' | 'canceled'
    fill_price       REAL,
    fill_qty         REAL,
    decision_price   REAL,                    -- price at decision time (for slippage)
    submitted_ts     TEXT NOT NULL,           -- when our process INSERTed the row
    filled_ts        TEXT,
    raw_response     TEXT                     -- JSON dump from broker
);

-- idx_orders_pair_bar lives in persist._apply_migrations because bar_ts is an
-- added column on legacy DBs; creating the index from schema.sql would fail
-- before the migration runs.

CREATE INDEX IF NOT EXISTS idx_orders_pair ON orders(pair_id, submitted_ts);
-- idx_orders_broker / idx_orders_pair_bar live in persist._apply_migrations
-- so they don't fail on legacy DBs that pre-date those columns.

-- Daily bars (engine inputs)
CREATE TABLE IF NOT EXISTS bars (
    ticker     TEXT NOT NULL,
    ts_date    TEXT NOT NULL,                 -- YYYY-MM-DD
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    source     TEXT NOT NULL,                 -- 'alpaca_official' | 'built_from_ticks'
    PRIMARY KEY (ticker, ts_date)
);

-- Raw ticks (for bar audit; rolling 7-day retention)
CREATE TABLE IF NOT EXISTS ticks (
    ticker  TEXT NOT NULL,
    ts      TEXT NOT NULL,                    -- ISO8601 UTC ms
    price   REAL NOT NULL,
    size    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_ticker_ts ON ticks(ticker, ts);

-- Audit log (alerts, disconnects, reconnects, fills, errors)
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event      TEXT NOT NULL,                 -- 'disconnect' | 'reconnect' | 'fill' | 'alert' | 'halt' | ...
    level      TEXT NOT NULL,                 -- 'CRITICAL' | 'ERROR' | 'WARN' | 'INFO'
    message    TEXT NOT NULL,
    payload    TEXT                            -- JSON
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);

-- Kill-switch state (singleton row id=1)
CREATE TABLE IF NOT EXISTS kill_switch (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    halted          INTEGER NOT NULL DEFAULT 0,  -- 0/1
    reason          TEXT,
    triggered_ts    TEXT,
    cleared_ts      TEXT
);

INSERT OR IGNORE INTO kill_switch (id, halted) VALUES (1, 0);

-- Regime decision per month (composite filter result)
CREATE TABLE IF NOT EXISTS regime_decisions (
    month       TEXT PRIMARY KEY,             -- YYYY-MM
    stress_z    REAL NOT NULL,
    q67_thresh  REAL NOT NULL,
    halted      INTEGER NOT NULL,
    decided_ts  TEXT NOT NULL
);
