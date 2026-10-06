# TradeLedger — Project Vision

> This document is the **canonical description of what TradeLedger is, what it aims to
> become, and how every part of the application works**. It is intentionally broader and
> deeper than `README.md` (which stays operational: quick start, configuration reference,
> API tables). Here the goal is completeness: every subsystem, every responsibility, every
> design decision that shapes the product — so that a new maintainer (human or LLM) can
> understand the whole machine without reading all ~45 000 lines of source code first.

---

## 1. What TradeLedger Is

TradeLedger is a **self-hosted, AI-powered trading bot for the Italian market**
(Borsa Italiana / Euronext Milan, XMIL). It trades **medium-to-long-term** horizons
(timeframes from 1h up to 5Y — hours, days, weeks, months, years — not seconds), across
three asset classes:

| Asset class | Source of truth | Yahoo symbol shape | Notes |
|---|---|---|---|
| Italian stocks | Borsa Italiana (Euronext Milan) | `TICKER.MI` | Discovered via Wikipedia / FinanceDatabase / static lists |
| UCITS ETFs (Italian-domiciled) | Borsa Italiana | `TICKER.MI` | Discovered via FinanceDatabase keyword matching |
| BTPs (Italian government bonds) | Borsa Italiana MOT bond pages | `ITxxxxxxxxxxx` (ISIN) | Scraped + token-authenticated BI API; YTM, maturity, coupon handled natively |

**Currency: EUR. Timezone: Europe/Rome. Broker reality modelled: Intesa Sanpaolo Investo**
(the fee model, Tobin tax, BTP fee rules, and tick sizes in the code mirror that retail bank).

The intelligence is delegated to an **external LLM** (Ollama, any OpenAI-compatible
endpoint, or g4f free providers). The LLM is not a gimmick bolted onto a classic bot:
**the LLM is the decision-maker** — it selects which symbols to track, proposes trading
strategies with explicit risk parameters, proposes backtest variants, reviews backtest
results, decides to pause or resume the whole bot, and even tunes operational parameters.
Everything around the LLM exists to feed it the best possible context and to constrain,
validate, execute, and audit what it decides.

Two operation modes:

- **`paper`** — a full trading simulator with realistic fees, dynamic slippage, volume
  caps, stop/limit/trailing order types, SQLite persistence, and silent Telegram
  notifications. This is the primary mode and the one under active development.
- **`notify`** — signal-only mode: audible Telegram alerts for BUY/SELL, and manual trade
  logging through the web dashboard (for when you execute by hand at your bank).

There is **no live-trading broker integration by design**. The project deliberately stops
at paper simulation + notification; connecting real money is out of scope for now.

## 2. What the Project Aims to Be

1. **A trustworthy decision-support machine, not a slot machine.** Every LLM idea must
   survive parsing, semantic validation, parameter clamping, risk validation, backtesting,
   a second LLM review, and post-decision filters before it can touch the (simulated)
   portfolio. The bot must be profitable *after* real-world costs — the system prompt
   itself teaches the LLM the exact round-trip fee percentages for €1 000 and €10 000
   trades and forbids take-profits below break-even.

2. **Honest about uncertainty.** The LLM must be able to say **HOLD** with confidence.
   Low-conviction ideas are rejected (`confidence_rejection_threshold`), position sizing is
   confidence-weighted, and the system prompt encodes risk-appetite regimes (normal /
   probing / conservative) driven by market breadth and P&L.

3. **Safe by default — fail closed.** When anything is uncertain (Redis down, market
   clock unknown, LLM provenance unprovable, database unwritable, pause state unreadable),
   the bot degrades to the *safe* side: no new BUYs, HOLD downgrade, forced pause. This
   principle is enforced in dozens of dedicated code paths and tests
   (`test_drawdown_breaker_fail_closed.py`, `test_pause_resume_fail_closed.py`,
   `test_llm_provenance_gate.py`, …).

4. **Cheap to run.** LLM calls are the main cost. The system attacks token usage from
   every angle: response caching, snapshot-hash semantic decision caching (reuse an
   LLM-reviewed decision when nothing material changed), provider prompt caching via
   strictly prefix-stable prompts, prompt compaction (including a TOON-based serializer),
   chunk-split/summarize for oversized payloads, per-role model tiers (weak actuator vs.
   strong mind chosen dynamically by market complexity), token-budget semaphores,
   evaluation intervals scaled to timeframe, and an optional external token-compression
   MITM proxy (`llmtrim`) wired into the Docker deployment.

5. **Observable.** Every LLM call is metered (tokens, latency, cost, provider, model,
   request type) into `llm_metrics`; every decision is stored with outcome tracking in
   `llm_decision_quality`; failures feed an automatic model blacklist; logs stream to the
   dashboard via Redis; the dashboard shows LLM metrics charts, decision quality, model
   failures & blacklist state, and raw logs. Decision-cache hits — which skip the LLM
   call entirely and so leave no `llm_metrics` row — are metered separately into
   `decision_cache_metrics` (§5.1) with estimated prompt tokens saved. A self-analysis
   loop ("analyze wrong decisions") periodically asks the LLM to critique its own
   losing patterns and feeds the result back into future prompts (the "Past Mistakes
   Analysis" block).

6. **A one-person, self-hosted product.** Single Docker Compose stack (bot + PostgreSQL +
   external Redis), a PWA dashboard installable on a phone, a Telegram bot as the
   mobile control surface, JSON structured logs, health endpoints, and supervised tasks
   that restart themselves. Nothing requires a Kubernetes cluster or a SaaS account.

7. **An honest research lab.** The Simulation Lab in the dashboard can run the full
   Step-1a → Step-1b → backtest → Step-2 pipeline for any symbol without executing
   anything, and backtest results (with walk-forward validation for long histories) are
   persisted so the LLM's proposals are judged on evidence.

### Non-goals (for now)

- No live broker/order-routing integration (paper + notify only).
- No non-Italian markets (architecture is market-aware but `TARGET_COUNTRY`/`TICKER_SUFFIX`
  parameterized).
- No high-frequency trading: minimum entry-condition timeouts, 15–30 min+ evaluation
  cycles, and hour-long-plus hold times are deliberate.
- No "fallback LLM" silently making trading decisions: the fallback chain exists, but
  fallback calls are labelled and (for portfolio selection) not promoted to final
  decisions; the user can enforce a strict no-fallback policy in code defaults.

## 3. Design Principles

These principles are visible throughout the codebase and any change should respect them:

1. **LLM decides, code constrains.** The LLM proposes; deterministic validators
   (`llm_parser`, `validator`, `post_decision_manager`, `risk_manager`) dispose.
2. **Fail closed.** Uncertainty ⇒ safe side (no BUY, HOLD, pause). Never fail open on
   risk paths.
3. **Supervise everything.** Every background loop runs under an Erlang-style
   `TaskSupervisor` (restart with backoff, health reporting, notifier alerts).
4. **Decouple via events.** Engine components communicate through an in-process
   `EventBus` (`publish` for notifications, `request` for command/query with error
   propagation), keeping `engine.py` a thin composition root and façade.
5. **Dedicated executors, bounded concurrency.** Separate thread pools for DB writes,
   downloads, quotes, LLM calls, and backtests; semaphores for exchanges, news,
   indicators, downloads, and symbol processing — one noisy subsystem must never starve
   the web server or Telegram bot.
6. **Cache what is expensive, invalidate honestly.** Quotes, candles, indicators,
   sentiment, LLM responses, decisions, correlation matrices, market clock — all cached
   with explicit TTLs and cache-invalidation hooks on settings reload.
7. **Prefix-stable prompts.** Volatile content (past-mistakes analysis) goes at the end
   of the user message, never into the system prompt, to preserve provider prompt caches.
8. **Hot-reloadable configuration.** `settings.reload()` re-reads `.env`, and registered
   callbacks propagate changes (semaphores, sessions, caches) without restart.
9. **Persistence is part of the design.** Positions, queued orders, balances, indicators,
   signals, PnL snapshots, backtests, dividends, LLM metrics, and decision outcomes all
   survive restarts; state is saved periodically and on shutdown signals.
10. **The Italian market is a feature, not a bug.** Fee models (Tobin tax), XMIL market
    calendar with Italian holidays, BTP mechanics (par at maturity, no trailing stops,
    YTM prompts), Banca d'Italia news — domain specificity is the edge.

## 4. Configuration & Entry Points

### 4.1 Static configuration — `src/config/settings.py`

The entire parameter surface of the bot lives in **one ~1 800-line
pydantic-settings `Settings` class**, loaded from `.env` (with `override=True`) and
exposed as the module-level `settings` singleton. Every field is typed and most are
guarded by `@field_validator`s that clamp unsafe values (e.g. re-evaluation intervals
must be ≥ 300 s, slippage percentages ≥ 0, temperatures within 0–2). The settings
groups, in the order they appear:

- **Trading mode & market identity** — `TRADING_MODE` (`paper` / `notify`),
  `TICKER_SUFFIX` (`.MI`), `TARGET_COUNTRY` (`italy`), `ETF_ITALY_KEYWORDS`.
- **Paper trading** — initial balance (€10 000 default), poll intervals, dynamic
  slippage (base/max percent scaled by volume), dividend reinvestment.
- **Cadence & rhythm** — engine loop interval (60 s), per-timeframe evaluation
  intervals, max skip-before-forced-reevaluation, *active windows* around market
  open/close (extra-frequent evaluation during opening/closing minutes), entry-signal
  monitor interval, cooldowns between forced LLM evaluations, minimum entry-condition
  timeout as a multiple of the candle timeframe (anti-HFT guard).
- **Portfolio & symbol selection** — `MAX_SYMBOLS`, `MAX_OPEN_POSITIONS`,
  `MIN_SYMBOLS` (LLM must select at least 3 unless pausing), `ALWAYS_INCLUDE_ETFS`,
  `ADDITIONAL_TICKERS`, `EXCLUDED_SYMBOLS`, FinanceDatabase discovery toggle,
  strict-country filter, composite trend/sentiment score weights, LLM chunk size for
  candidate evaluation, incremental rotating-batch re-evaluation.
- **Risk management** — portfolio cooldown after consecutive losses, max daily loss
  (5 % of initial balance pauses the day), `HARD_MAX_LOSS_PCT` (15 % force-exit) with
  per-timeframe overrides, BTP-specific loss/take-profit caps (3 % — bonds trade in
  narrow ranges), max LLM stop-loss/take-profit reviews before force-sell (with weekly
  and long-term reduced caps), partial take-profit and dust-sweep review caps with a
  dust auto-sell timeout, native stop-fill timeout before manual market-sell fallback,
  max position age as a multiple of the LLM's original `max_hold_time`.
- **Pause/resume governance** — max consecutive "keep paused" LLM decisions (3) before
  force-resume with a reduced risk multiplier (0.5), force-resume blocked when account
  drawdown exceeds 15 %, minimum LLM pause duration (30 min).
- **Market data** — OHLCV timeframes (`5Y` → … multi-timeframe analysis), download /
  quote-refresh / full-asset-backfill intervals, download staggering and per-call candle
  caps, OHLCV retention days (years, for long timeframes), quote staleness thresholds
  scaled by timeframe, news fetch intervals (fast refresh for tracked symbols),
  news-driven and RSS-based symbol discovery.
- **LLM** — provider selection (`ollama` / `openai` / `g4f`), **per-role models**:
  *mind* (deep reasoning: selection, strategy generation), *actuator* (fast
  time-critical reviews), *weak* (summarization), *sentiment* (dedicated news-scoring
  model), and an *Always-Online locally-hosted* last-resort model — each with its own
  provider overrides, temperature (optionally a `min-max` range picked by prompt
  complexity), timeout, thinking/reasoning-effort control and token limits. Fallback
  provider settings default to **disabled** (fail-closed: if the primary fails, trading
  stops rather than silently delegating to weaker models), and `validate_llm_settings()`
  enforces this as a **hard startup invariant** — if `LLM_FALLBACK_ENABLED` is false
  while any fallback provider is still configured, the process refuses to boot (a stale
  or half-migrated `.env` can never silently route decisions to unreviewed models).
  Provider selection is validated per-provider here too (Ollama needs a model, OpenAI
  needs a key + model, g4f self-manages). Also: provider prompt
  caching flags + `cache_control`-capable provider list, snapshot-hash semantic
  decision cache, `LLM_CACHE_VERSION`, LLM circuit-breaker threshold, response cache
  TTLs, per-role timeouts, and the **`llmtrim` egress-proxy** settings
  (`LLM_PROXY_URL` + CA bundle) that route all LLM HTTP traffic through a local
  token-compression MITM proxy.
- **Backtesting** — variants per cycle, parallelism cap, minimum candles, max data age
  before skipping (don't backtest on stale downloads).
- **Infrastructure** — dedicated thread-pool sizes (DB writes, downloads, quotes, LLM,
  backtests), async semaphore limits (exchanges, news, indicators, symbol processing),
  Redis connection, optional PostgreSQL (used instead of SQLite when fully configured),
  data directory, Telegram token/chat id, web host/port.

**Hot reload.** `settings.reload()` re-reads `.env` **in place** (via
`load_dotenv(override=True)`, then copying every field onto the live singleton so all
importers see the change): it skips the connection/infrastructure fields that must
never change at runtime — `DATABASE_BACKEND`/`DATABASE_PATH`,
`REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`/`REDIS_TLS`, `WEB_HOST`/`WEB_PORT`, and the
executor worker counts — regenerates `LLM_CACHE_VERSION` (a UUID) whenever any
LLM/provider/fee/prompt-affecting field changes (auto-detected by the `LLM_`/`OLLAMA_`/
`OPENAI_`/`G4F_` prefixes plus an explicit fee/prompt field set), which invalidates
every semantic LLM cache — note that **temperature-only changes deliberately do not
invalidate the cache** (same prompt, same key) — flags a paper-balance change
(`PAPER_BALANCE_CHANGED`, triggering a soft paper-state restart), fires all registered
reload callbacks (semaphores, sessions, caches re-tune live), and invalidates the
primary-model cache in `llm/cache.py` so market-hours changes take effect immediately.
Configuration is therefore a runtime object, not a restart.

### 4.2 LLM-decided runtime configuration — `src/config/config_service.py`

A small `UnifiedConfigService` bridges the two config layers: parameters the **LLM is
allowed to tune** (e.g. evaluation cadence adjustments) are stored in Redis under a
`trading:` prefix with a 7-day TTL, read through a 60-second in-memory cache, and
degrade to safe defaults when Redis is unavailable. Static settings define the
guardrails; the LLM adjusts behavior inside them at runtime.

### 4.3 Entry point — `src/main.py`

`main.py` is both the process entry (`python -m src.main`) and the observability
bootstrap:

- **Structured logging**: a `JsonFormatter` on the root logger emits one JSON object
  per record (timestamp, level, logger, message, extras, exception). Uvicorn is given
  the same formatter via a customized logging dict; an `UvicornAccessFilter` drops
  `/health` access logs entirely and downgrades other access logs to DEBUG unless
  `LOG_LEVEL=DEBUG`. Noisy libraries (httpx, urllib3, asyncio) are silenced.
- **Dashboard log stream**: `RedisLogHandler` pushes INFO+ records into a bounded
  Redis list (`logs:recent`, 200 entries) through a daemon flusher thread and a
  1 000-entry in-memory queue — logging never blocks the event loop, and a full/down
  Redis can never break the application.
- **Startup validation** (`_validate_startup_settings`): Redis must be reachable
  (critical exit otherwise), LLM settings validated, and the database must pass a
  write/read round-trip test — the "fail closed" principle applied at boot.
- **`main()` coroutine**: sizes the default executor (100 threads), initializes the
  DB pool, syncs the model blacklist from the DB, pre-creates the yfinance cache
  directory, seeds the Telegram chat id from env, tests RSS feeds, constructs the
  `TradingEngine` and injects it into the web app, then **starts the web server
  immediately** (dashboard is reachable while the engine warms up) and wires the
  Telegram bot as the engine's notifier.
- **Supervision at the top level**: the engine runs under a `TaskSupervisor` (restart
  with 10 s delay, max 10 restarts), the Telegram bot under its own; a health monitor
  logs supervisor state every 60 s.
- **Graceful shutdown**: SIGINT/SIGTERM set an event; on shutdown the engine stops,
  Telegram stops, uvicorn exits, and the DB pool closes.

### 4.4 Deployment — `Dockerfile`, `docker-compose.yml`, `nginx.conf`

- **Dockerfile**: `python:3.11-slim`; compiles the **TA-Lib C library from source**
  (technical-analysis core); installs requirements **plus dev requirements and runs
  the full pytest suite at build time** — a failing test fails the image build — then
  uninstalls dev dependencies and deletes the tests. `/app/data` is created and
  chowned to UID 1000; the container health-checks `/health`; the command is
  `python -m src.main`.
- **docker-compose.yml**: three services. `postgres` (18-alpine, port bound to the LAN
  interface only, health-checked, data under `./postgresql/data`); `bot` (built from
  the Dockerfile, `.env` mounted read-only, `./data` volume for SQLite/logs/cache,
  and `LLM_PROXY_URL`/`LLM_PROXY_CA_BUNDLE` pointing at the external **llmtrim** MITM
  proxy so all LLM payloads are compressed in transit); and a `redis-dummy` alpine
  container whose only job is to wait for the **external Redis** (running outside the
  stack on the LAN host) so the bot's `depends_on` ordering is satisfied.
- **nginx.conf**: an optional reverse-proxy sample for exposing the dashboard on a
  domain — WebSocket upgrade for `/ws` with 24 h read/send timeouts, plain proxy for
  everything else.

## 5. Data Layer

The data layer is a deliberate two-tier design: **SQL is the durable truth**
(trades, candles, decisions, metrics — everything that must survive a restart),
**Redis is the hot layer** (quotes, caches, pause state, rate limits, log stream —
everything that is cheap to lose and must be fast). Redis may vanish and the bot
keeps running in degraded no-cache mode; SQL may not.

### 5.1 SQL persistence — `src/database.py` (~3 600 lines)

One module owns all SQL in the system. It supports **two interchangeable backends**
chosen by `DATABASE_BACKEND`:

- **PostgreSQL** (production): psycopg3 with a `psycopg_pool.ConnectionPool`
  (min 10 / max 50 connections, 30 s acquire timeout, 60 s max idle, up to 300 s
  reconnect), connections validated in autocommit (`SELECT 1` + `SET search_path`)
  and `dict_row` results. A `_PgConnectionWrapper` makes `close()` roll back any
  pending transaction *before* returning the connection to the pool — so an exception
  mid-transaction can never leak an "INTRANS" connection back into the pool (and if
  `putconn` itself fails, the connection is closed outright rather than poisoned).
- **SQLite** (single-node default): one connection per thread, opened with
  `PRAGMA journal_mode=WAL` and a 30 s busy timeout, wrapped so that `close()` is a
  **no-op** (connections are reused for the thread's lifetime) with a
  `weakref.finalize` plus a class-level `WeakSet` guaranteeing the real file handle is
  closed when a thread dies; `close_all_sqlite_connections()` force-closes every
  wrapper (used when the DB file is replaced), `vacuum_database()` closes everything
  then runs `VACUUM`.
- Both are reached through the same `get_connection()`, and a `get_connection_ctx()`
  context manager that returns pool connections to PostgreSQL on exit and is a no-op
  on SQLite. A `retry_on_db_lock` decorator (default 2 retries, exponential backoff
  capped at 1 s) absorbs SQLite `database is locked`, PostgreSQL deadlock (`40P01`)
  and serialization failure (`40001`), and transient connection/timeout/OSError —
  i.e. the same decorator covers both backends' contention modes.

All SQL is written once in PostgreSQL style (`%s` placeholders) and adapted to
SQLite (`?`) by `_adapt_sql`; a dummy `psycopg` namespace keeps exception handlers
working on the SQLite backend. This is what makes "one-person self-hosted" possible:
the same code runs on a laptop SQLite file or a LAN PostgreSQL.

**Schema (17 tables, created idempotently; `_migrate_db` adds the ~23 historical
columns to existing installations as one atomic transaction — all migrations are
applied together and rolled back together, so an installation is never half-migrated):**

| Table | Purpose |
|---|---|
| `trading_state` | key/value store for the engine's full state (positions, balances, counters) — persisted periodically and on shutdown |
| `telegram_state` | Telegram chat id and bot bookkeeping |
| `trade_history` | every executed fill: order id, symbol, timeframe, side, amount, price, cost, fees, realized PnL, cost basis, strategy type, exit reason, hold time, buy confidence |
| `signals` | every LLM decision record: action, confidence, full reasoning, strategy parameters (entry condition, SL/TP, trailing, position size fraction, max hold, order type/limit price), plus provider/model provenance |
| `market_data` | OHLCV candles, unique on (symbol, timeframe, timestamp), with **covering indexes** (`INCLUDE(ohlc, volume)`) for index-only scans on the hot read paths |
| `indicators` | latest computed indicator set per symbol/timeframe (JSON) |
| `quotes` | latest quote per symbol (bid/ask/last/volume/change) |
| `latest_close_prices` | close-price fallback so valuation survives quote outages |
| `discovered_symbols` | the **asset registry**: symbol, ISIN (+ manual override), asset type (stock/ETF/BTP), name, BTP maturity & coupon, country |
| `position_pnl` | periodic per-position PnL snapshots (unrealized/realized/percent), with retention cleanup |
| `backtest_results` | backtest variants keyed by (symbol, timeframe, params_hash) with stats JSON and LLM-readable summary |
| `llm_metrics` | one row per LLM call: provider, model, model_type, prompt/completion/total tokens, cache hit, latency, error, request type, fallback flag |
| `llm_decision_quality` | outcome tracking: decision → later outcome price → profitable flag → feeds the self-analysis loop |
| `decision_cache_metrics` | one row per decision-cache outcome: hit (the Step-2 LLM call was skipped) or miss (cold / changed / rebuild-failed), with estimated prompt tokens saved on hits |
| `dividends` | ex-dates and amounts per symbol, with reinvestment marking |
| `portfolio_equity` | peak total equity — the reference for the drawdown circuit breaker |
| `llm_model_blacklist` | persistent model blacklist with reason and expiry |
| `news_articles` | stored news with sentiment label/compound per article |

Around these tables the module exposes the entire data-access API: trade insertion
and performance computation, PnL snapshot history, backtest result storage with
dedup by params hash, state/balance/order persistence, news storage + per-symbol
sentiment aggregation, OHLCV batch upserts and batch reads/summaries, indicator
persistence, quote batch writes, discovered-symbol registry maintenance (including
malformed-symbol cleanup and ISIN/BTP detail lookups), signal pagination, LLM
metrics summaries and time series (dashboard aggregation), dividend tracking with
yield computation, and the decision-quality lifecycle (`insert_llm_decision` →
`update_llm_decision_outcome` → `get_llm_decision_quality_metrics`). BTP yield-to-
maturity is computed in SQL-adjacent Python (`compute_btp_ytm`). Operational hygiene
is a first-class part of the API: a `cleanup_old_*` family (news, market data, PnL
snapshots, backtests, LLM metrics, dividends, decisions, decision-cache metrics)
bounds every append-only table, `reset_paper_trading_data` / `reset_llm_metrics` / `reset_llm_decision_quality`
/ `clear_all_blacklisted_models` give each dashboard "reset" action a scoped SQL
counterpart (paper reset can even keep trade history), and `get_pool_stats` exposes
the PostgreSQL pool to the health endpoint.

`init_db()` creates the schema, then — **only when the database is brand new** —
flushes stale Redis caches so a fresh database never serves cached data from a
previous life. The flush is deliberately surgical: a declared list of ~40 key
prefixes is deleted with `SCAN`+`DELETE` per prefix, never `flushdb()`, so a Redis
instance shared with other services is left intact. It then runs the migrations,
cleans malformed discovered symbols, invalidates the `tradable_assets:{country}`
cache so the tradable universe is rebuilt from clean data, and backfills the latest
close prices.

### 5.2 Redis — `src/utils/redis_client.py` and friends

Redis is a hard dependency (startup exits without it) but a **soft dependency at
runtime**: `get_redis_client()` returns a real client or, when connectivity is lost,
a `DummyRedis` no-op object — reads return safe defaults (`None`, `{}`, `0`, `False`,
`-1`, `set()`) with rate-limited warnings, writes log a data-loss **error** every time
and pretend to succeed, and `__getattr__` catches every command not explicitly stubbed.
`set_redis_available` flips a global flag and logs a critical alert; the whole
codebase branches on `is_redis_available()`. Nothing that touches Redis can crash the
engine.

The real client is a **single shared, lazily-created singleton pool** (max 50
connections, 5 s socket and connect timeouts, 30 s health-check pings,
`retry_on_timeout`, decoded responses) — one pool for the whole process, so heavy
logging and concurrent dashboard calls cannot exhaust file descriptors or Redis's
client limit. Outages **self-heal**: `check_redis_connection()` runs at startup, from
the dashboard health endpoint, and periodically from the background task manager (on
the DB executor, off the event loop); a successful ping flips availability back to
true and the same pooled client resumes serving, so a Redis blip costs cache warmth
for a few minutes, not a restart.

Redis holds the *hot* state:

- **Market-data caches** — `quote:{sym}`, `latest_close_prices:{sym}`,
  `ohlcv:{sym}:{tf}:{limit}` and `ohlcv_range:{sym}:{tf}:{start}:{limit}` (candle
  caches), `news:*` / `news_summary:{sym}` / `sentiment:{sym}`, ISIN and name lookups
  (`stock_name:{sym}`, `isin_not_found:{sym}`), BTP detail caches (`btp_details:`,
  `btp_bonds_list`, `italian_ucits_etfs`), Yahoo-side caches (`yahoo_quote:`,
  `yahoo_fundamentals:`, `yahoo_dividends:`, `yahoo_analyst_ratings:`,
  `yahoo_insider_transactions:`, `yahoo_options_summary:`), and
  `tradable_assets:{country}` (the current tradable universe).
- **The trading pause state machine** — `trading:paused`, `trading:pause_source`,
  `trading:pause_start`, `trading:pause_duration`, `trading:pause_reason`,
  `trading:llm_pause_time`, managed centrally by `src/utils/pause_utils.py`: a single
  writer that **clears the whole key family before setting** (so a pause can never be
  half-applied with stale metadata left behind), with an optional TTL on the family and
  a permanent 7-day TTL on `trading:pause_duration`. When Redis is unreachable, an
  **in-process fallback pause flag** takes over so fail-safe paths (drawdown breaker
  etc.) still block new BUYs — pause is fail-closed in both directions.
- **Risk & circuit-breaker state** — `trading:peak_total_equity` (drawdown reference),
  `trading:has_open_positions` (a 300 s TTL heartbeat refreshed on fills, so the flag
  self-expires if the engine dies), `trading:last_triggered_reeval` (2 h TTL reeval
  throttle), `llm:fail_count:{model}` (1 h sliding window),
  `llm:blacklist_level:{model}` (kept 7 days), `llm:consecutive_failures` (1 h TTL,
  the LLM circuit-breaker counter — incremented by callers whose outer timeout cancels
  a call, so external timeouts are not undercounted). The model blacklist lives in
  Redis for speed and in `llm_model_blacklist` for durability.
- **Derived market context** — `market:breadth:full` (breadth snapshot, 600 s TTL,
  feeding risk regimes and the pause/resume decision), `atr_percentile:{sym}`,
  `volume_trend:ratio:{sym}` and `volume_trend:ema:{sym}` (percentile/trend values
  reused across evaluation cycles). Note what is deliberately *not* here: the market
  clock is an in-process TTL cache in `MarketDataManager`, and LLM-decided regime
  thresholds are `trading:*` keys owned by `UnifiedConfigService` (§4.2).
- **LLM response cache** — `llm:{sha256(cache-key)}` semantic response caching, plus
  `llm:wrong_decision_analysis` (cached self-analysis of past mistakes).
- **Web infrastructure** — dashboard `session:{token}` and `csrf:{token}` (1-day TTL),
  per-IP rate limiting (`rate_limit:{ip}`, `ws_rate_limit:{ip}`), the bounded
  `logs:recent` list (200 entries) that streams logs to the dashboard, `web:messages`
  (100 entries) as the Telegram→dashboard notification stream, and
  `metrics:unexpected_exception:*` for the dashboard's error panel.

### 5.3 What persists where (summary)

- **Restart-survivable truth** → SQL: trades, the engine's live state (symbols,
  positions, queued orders, recent signals, pending entries, entry-signal state,
  eval snapshots, cooldowns, risk multiplier — all as `trading_state` key/value rows),
  candles, indicators, signals, backtests, dividends, LLM metrics, decision outcomes,
  model blacklist, peak equity (single-row `portfolio_equity`, mirrored in Redis for
  fast reads).
- **Fast and ephemeral** → Redis: quotes, candle/news/sentiment/Yahoo caches, pause
  keys, counters, sessions, logs. Losing it costs performance and cache warmth, never
  correctness.
- **The one deliberate exception: pause state is Redis-only.** It is not mirrored to
  SQL, which is a considered trade-off, not an oversight — a restart with Redis intact
  resumes exactly where the pause left off, and because `trading:` is in the flush
  list, a *fresh-database* start starts unpaused by design (a new life has no reason to
  inherit an old halt). While Redis is down, the in-process fallback flag carries the
  halt, so the fail-safe paths never depend on Redis being reachable.
- **LLM-decided parameters** → Redis via `UnifiedConfigService` (§4.2), TTL-bounded
  so stale LLM tuning self-expires.

## 6. Market Data Layer

### 6.1 Quotes — a chain of responsibility — `src/exchanges/market_data.py`

`get_quotes()` sanitizes the symbol list (strips `$` and the `/EUR` suffix), builds a
`QuoteContext` (symbols, result, `missing_symbols`, redis client) and pushes it through
a **chain of `QuoteHandler`s**; each handler fills what it can and passes only the
symbols still missing to the next, so the chain short-circuits as soon as everything is
resolved:

1. `RedisQuoteHandler` — hot cache hit (`quote:{symbol}`).
2. `DatabaseQuoteHandler` — recent rows from the `quotes` table
   (`DB_QUOTE_MAX_AGE_SECONDS`).
3. `BorsaItalianaQuoteHandler` — the exchange-native source: BTPs, and any symbol whose
   Italian ISIN is known.
4. `YFinanceQuoteHandler` — batch download for non-BTP symbols with a verified Italian
   ISIN.
5. `AlphaVantageQuoteHandler` / `IEXQuoteHandler` — optional keyed fallbacks, each capped
   at 10 symbols per call to respect their free-tier quotas.

Every source is guarded by its own circuit breaker, and the whole chain is bypassed
*before it starts* when the two primary sources (yfinance **and** Borsa Italiana) are
both open: `get_quotes()` then serves `get_quotes_cached()` — a strictly **no-network**
path (Redis → `quotes` table → last close prices, rejecting closes older than
`STALE_QUOTE_THRESHOLD_HOURS`) — so a Yahoo outage can never block the trading loop.
Source availability is reported to `HealthMetrics` on every call.

After the chain, `_finalize_and_persist_quotes` computes `change_24h` and `percentage`
from the daily candles stored in the database, fills symbols that still have no price
from the last close, **validates every quote against the stored one** (a newer stored
quote wins; a price deviating by more than `QUOTE_DEVIATION_THRESHOLD` is reverted when
the stored quote is less than an hour old, and accepted as a genuine move when it is
older), guarantees `bid`/`ask` are never null when a price exists, persists to Redis
(`QUOTE_CACHE_TTL`) and the `quotes` table, and enriches BTP quotes with maturity,
coupon, name and computed YTM.

Staleness is deliberately handled *above* this layer, where the decision is made:
`MarketDataManager` attaches a visible ⚠️ warning to the prompt for quotes sourced from
`db_close`/`db_quotes`/`yfinance` that are more than 15 minutes old, and
`_is_quote_too_stale()` blocks **new entries** with a threshold scaled by the symbol's
timeframe — `max(QUOTE_MAX_STALENESS_SECONDS, min(tf_seconds × 0.1, 6 h))`, so a 5Y
position may act on staler data than a 1h one — with an explicit exemption while the
market is closed, which stops every weekend and holiday from looking like a data
failure.

Candles flow through `get_multi_timeframe_bars` (per-timeframe Redis cache: 60 s for
intraday, 300 s otherwise) and `get_bars_range` (same policy anchored at a start
timestamp, with the 730-day cap Yahoo imposes on intraday history). The source policy is
country-first: **BTPs are served exclusively by Borsa Italiana**, a symbol with a known
Italian ISIN is fetched from Borsa Italiana first with yfinance only as fallback, and a
symbol **without** a verified Italian ISIN is never sent to yfinance at all — the bot
prefers no data to data for the wrong country. Long timeframes (6M/1Y/3Y/5Y) are
aggregated from monthly candles; results from all sources are merged with Borsa Italiana
taking precedence per timestamp, then validated. ISINs are resolved on demand when
missing.

### 6.2 Source adapters — `src/exchanges/`

Every external source is wrapped in its own module with a dedicated rate limiter
and circuit breaker, so one angry provider cannot take the whole chain down. The
adapters:

- **Yahoo Finance** (`yahoo_finance.py`, `yf_session.py`): quotes, fundamentals,
  dividends, insider transactions, analyst ratings, and options summaries.
  `yf_session.py` hardens access because Yahoo increasingly blocks non-browser
  clients with 401/429: a **curl_cffi session impersonating Chrome's TLS
  fingerprint** (`impersonate="chrome"`, 15 s default timeout, 401/403/429 raise
  session errors), a **fail-fast sliding-window rate limiter** that *raises*
  instead of sleeping — a blocked worker would hold the shared download thread
  pool hostage, and raising lets the caller fall back to the database
  immediately — and downloads routed through `_yf_download_with_timeout` on a
  2-worker thread pool with a hard 30 s timeout so a hung call cannot stall a
  request. A circuit breaker trips after `YF_MAX_ERRORS` (20) errors in a 300 s
  window and blocks for 300 s; the impersonated session is invalidated early
  (after 5 errors, or on trip) and rebuilt from scratch. A logging filter
  (`YFinance401Filter`) suppresses the resulting 401 noise, and Italian ISINs are
  validated with `_is_valid_italian_isin` before any symbol mapping is trusted.
- **Borsa Italiana** (`borsa_italiana_utils.py`): the token-authenticated BI
  market API. Access tokens are **scraped from the public
  `grafici.borsaitaliana.it` summary-chart pages** (BeautifulSoup with a regex
  fallback, 3 retries), cached per market with `BORSA_TOKEN_CACHE_TTL`, and
  invalidated on 401 so a rotated token is re-acquired transparently. Calls are
  paced by a 1-second minimum-interval limiter and guarded by a circuit breaker
  (20 errors → 300 s block, degraded state logged at 10) over a shared pooled
  `httpx` client. Beyond `get_borsa_italiana_quote` and
  `get_borsa_italiana_candles` (the BTP-native candle source), the module is the
  **source of truth for symbol metadata in strict mode**:
  `_get_isin_and_info_from_borsa_italiana` resolves ISIN/country, while
  `_fetch_btp_details` (maturity/coupon from MOT bond pages) and
  `discover_btp_bonds` (the BTP universe) cover the sovereign-bond side.
- **Keyed fallbacks** (`alphavantage_utils.py`, `iex_utils.py`): Alpha Vantage and
  IEX Cloud — the two optional, API-key-gated quote *and* candle sources behind
  `AlphaVantageQuoteHandler` / `IEXQuoteHandler` (§6.1). Both are dormant without
  their key (`ALPHAVANTAGE_ENABLED`/`IEX_ENABLED` plus key), refuse BTP ISINs
  outright (there is no US-listed proxy for a sovereign bond), strip the `TICKER_SUFFIX`
  because both services speak plain US-style tickers, and route outbound calls
  through the shared proxy rotator. Alpha Vantage maps internal timeframes onto
  `TIME_SERIES_*` functions (1h → 60-min intraday, 1d/1w/1M; anything longer is
  refused) and switches to `outputsize=full` when a start date or more than 100
  bars is requested; IEX maps timeframes to chart ranges (1h → 1d, 1d → 1m,
  1w → 3m, 1M/3M/6M → 1y, 1Y/3Y/5Y → 5y). Both pace themselves with the shared
  `YFinanceRateLimiter` (Alpha Vantage at `ALPHAVANTAGE_RATE_LIMIT_PER_MIN`, IEX at
  the free tier's 100 req/min), run raw rows through `_validate_and_clean_candles`
  before returning, and collapse every failure mode to `None` so the chain simply
  moves on to the next handler. Neither exposes a true order book: Alpha Vantage
  mirrors `last` into both bid and ask, and IEX uses its `iexBidPrice`/`iexAskPrice`
  fields with the same last-price fallback — which is exactly what the §6.1
  guarantee ("bid/ask never null when a price exists") consumes. The safety net
  lives in the handlers in `market_data.py`: a dedicated `CircuitBreaker` per
  source, at most 10 symbols attempted per call, and a cross-check of any fetched
  price against the one already resolved upstream — a deviation beyond
  `QUOTE_DEVIATION_THRESHOLD` is logged and rejected in favour of the earlier source.
- **Asset discovery** (`asset_discovery.py`): `get_tradable_assets()` assembles
  the candidate universe from an **ordered source list** — Wikipedia FTSE MIB +
  FTSE Italia All-Share constituent tables (both language editions),
  user-configured additional tickers, FinanceDatabase keyword-matched Italian
  UCITS ETFs (gated by availability), tickers harvested from news feeds, a static
  CSV, and a hardcoded last-resort list — then verifies country via
  `_fetch_info`, which tries **yfinance → Borsa Italiana search → DuckDuckGo
  text search** (`duckduckgo_utils.py`, capped by `MAX_DDG_LOOKUPS`) before
  giving up. The whole chain degrades gracefully: if live discovery fails, it
  falls back to the Redis cache (`tradable_assets:{country}:{strict}`, 24 h)
  merged with the `discovered_symbols` DB registry, then to DB-only, and finally
  to a notifier alert ("⚠️ Market Data Discovery Failure") while trading idles
  rather than acting on a broken universe. In strict mode, DB writes are
  **deferred until after country filtering** so unverified symbols never enter
  the registry.
- **Candle hygiene** (`candle_utils.py`): validation and cleaning of raw candles
  (non-positive prices, negative volume, OHLC-consistency violations, duplicate
  timestamps keep-last), aggregation of lower timeframes into 6M/1Y/3Y/5Y bars
  (half-year buckets for 6M, calendar years for 1Y, year-thirds/fifths for
  3Y/5Y), merging of Borsa + Yahoo candle sets with **Borsa precedence per
  timestamp**, and data-quality issue detection (>20 % price jumps, >10 % gaps,
  zero-volume candles) that surfaces a bounded, summarized alert instead of
  silently feeding bad data to the LLM.
- **Fees** (`fees.py`): `calculate_transaction_costs()` implements the **Intesa
  Sanpaolo Investo Standard Profile** —
  `max(STOCK_FEE_MIN, gross × STOCK_FEE_PERC) + STOCK_FEE_FIXED` — plus the
  **Tobin tax on BUYs only**, and a separate **BTP fee policy** via `BTPPolicy`
  (different fee shape; sovereign bonds are exempt from Tobin tax). This module
  is the reason the bot knows its real break-even.
- **Proxies** (`proxy_utils.py`): a `DynamicProxyRotator` that scrapes free-proxy
  lists, validates candidates concurrently against a test endpoint, and
  refreshes the pool every 30 minutes; outbound requests pick randomly between
  the static proxy and the dynamic pool when proxying is enabled.

### 6.3 Technical indicators — `src/indicators.py`

All indicators are computed with **TA-Lib** (compiled into the Docker image), with a
parallel set of `_compute_simple_*` pure-Python implementations used as fallbacks
for degenerate inputs — long timeframes with only a handful of candles, where
TA-Lib would return `None`; fewer than two candles yields an empty result by
design. The catalogue: ATR (value + series), RSI, MACD, Bollinger Bands,
EMA 9/21, Stochastic, ADX/+DI/−DI, OBV, MFI, CCI, Williams %R, Ichimoku,
Donchian Channels, Parabolic SAR, Keltner Channels, VWAP, and pivot points.

`compute_all_indicators(candles, config, requested_indicators)` organizes the
catalogue into 16 keyword groups (`_KEY_GROUPS`) and can restrict computation to
**only the groups a prompt actually requests** — a token-saving feature as much
as a performance one. Honest note: the selective path exists and works, but no
current caller passes `requested_indicators`, so prompts today receive the full
catalogue; the capability is wired and waiting for prompt-size optimization.

Persistence is an event-driven pipeline, not inline computation:
`compute_and_store_indicators` (`trading/components/market_data_manager.py`)
subscribes to the indicator EventBus event, skips work when stored indicators are
already current for the latest candle, computes off the request path on a
dedicated download executor with concurrency capped by `INDICATOR_SEMAPHORE_LIMIT`
(4), and persists via `save_indicators(symbol, timeframe, latest_ts, indicators)`
to the `indicators` table.

### 6.4 News & sentiment — `src/news/fetcher.py` (~1 400 lines)

A source-abstracted news pipeline with **credential-based auto-enablement**:
NewsAPI, Twitter/X, Reddit, Facebook, YouTube (keyed) plus Google News,
StockTwits, DuckDuckGo and configurable RSS feeds (free). A source without
credentials is never touched; sources that fail permanently land in a
runtime-maintained disabled set. Resilience is layered per source: RSS fetches
are cached in-process for 5 minutes, a feed failing 3 consecutive times is
benched for an hour (parse errors are exempt — one malformed entry shouldn't
kill a good feed), and the shared news rate limiter reads its limits live from
settings so tuning needs no restart. Highlights:

- **Relevance filtering** (`_is_relevant`) scores articles against symbol,
  company name, and English + Italian finance keywords (title hits weigh
  double) so prompts aren't polluted by name-alike noise.
- **Per-symbol fan-out**: `fetch_news_for_symbol()` queries sources with a
  combined OR query (symbol, ISIN, company name), gathers concurrently with
  per-source exception isolation, and deduplicates by URL and normalized title.
  Sentiment cache keys embed an md5 fingerprint of the enabled-source set, so a
  source being added or lost invalidates stale aggregates naturally.
- **LLM-powered sentiment**: `_batch_analyze_sentiments()` batches 8 articles
  per call (small batches cap blast radius), sorts deterministically and hashes
  the batch (sha256 over titles + source + market state) for cache reuse with a
  24 h TTL, caps each text to 200 characters, and enforces a strict JSON
  contract (`label` ∈ positive/negative/neutral, `compound` ∈ −1.0…1.0) via the
  dedicated sentiment model. When the market is closed the call is
  **fail-closed** — the batch is stored as neutral without spending an LLM call;
  an open-market failure gets exactly one retry under a distinct cache key.
  StockTwits articles arrive pre-scored (Bullish +0.5 / Bearish −0.5) and skip
  the LLM. Aggregated sentiment lands in the `news_articles` table and Redis.
- **News as a discovery channel**: `discover_trending_stocks()` samples gainers,
  ranks them by DB-aggregated sentiment, caches under `news:trending_stocks_raw`
  for an hour, and supports a `cache_only` mode for degraded operation;
  `discover_tickers_from_news()` mines feeds for `.MI`-suffix symbols and feeds
  them into the candidate universe.
- **Calendar awareness**: `detect_upcoming_events()` scans stored articles for
  seven keyword categories (earnings, dividends, ex-dates, AGMs, splits, IPOs,
  guidance) straight from the database — no network, no cost — and
  `get_upcoming_earnings()` adds yfinance earnings dates. (Code observation:
  its circuit check is inverted — `if not _check_yf_circuit(): return None`,
  where `_check_yf_circuit()` returns True when the circuit is *open* — so the
  yfinance path currently runs only while the circuit is tripped; flagged for
  later review.)
- **Banca d'Italia BTP news**: a dedicated scraper pages through ~10 Banca
  d'Italia pages (paced at 0.5 s) for central-bank bond news.
- `test_rss_feeds()` runs at startup (called from `main.py`) so a broken feed
  configuration is known immediately, not discovered at the first missed signal.

## 7. LLM Layer

### 7.1 Provider abstraction — `llm/llm_client.py` + `llm/g4f_client.py`

`LLMProvider` abstract base with three implementations, all returning
`{content, usage}` dicts:

- `OllamaProvider` — local/self-hosted models via Ollama's `/api/chat`; token
  usage read from `prompt_eval_count`/`eval_count` (heuristic estimate when the
  server omits them).
- `OpenAIProvider` — any OpenAI-compatible endpoint (OpenAI, DeepSeek, local
  servers, …); adds `cache_control: ephemeral` markers on the system message and
  first user message when prompt caching is enabled for that provider.
- `G4FProvider` — **gpt4free**: free upstream providers, routed through the
  llmtrim proxy when configured and otherwise through the rotating proxy pool;
  token usage is estimated.

All providers always send `reasoning_effort` ("low" when thinking is disabled,
the computed value when enabled) and honor `max_tokens` (Ollama `num_predict`).

`OllamaProvider` and `OpenAIProvider` funnel through `_execute_llm_request`:
httpx with an explicit timeout split (connect 10 s, write 10 s, pool 5 s,
read = the per-role timeout), 3 retries with exponential backoff (2^attempt) on
429/500/502/503/504 and network errors, `Retry-After` honored (seconds form,
capped at 60 s), non-retryable statuses raised immediately with the response
body, every attempt recorded in health metrics. When `LLM_PROXY_URL` is set, all
LLM traffic — health checks included — routes through the **llmtrim MITM proxy**
(payload compression in transit, `LLM_PROXY_CA_BUNDLE` root CA).

`G4FProvider` deliberately does **not** share that helper: g4f is driven through
its own async client (`ClientFactory.create_async_client`) rather than raw
httpx, so it re-implements the same retry / backoff / health-metrics contract
independently, and proxies via `LLM_PROXY_URL` when configured, otherwise
through the rotating proxy pool. A request issued while an event loop is
already running is executed on a thread pool, and g4f token usage is always
estimated.

The g4f model roster is **discovered at runtime** from g4f's model map (cached
1 h; audio/image/video models plus the generic `default`/`custom`/`video`/`auto`
entries excluded, vision models intentionally kept; a hard-coded roster is used
when discovery itself fails) and tier-categorized by keyword heuristics into
mind/actuator/weak in this precedence: **explicit overrides first** (`o4-mini`
→ mind, `haiku-4-5` → weak), then `mini`/`haiku`/`flash` → actuator, then the
weak, mind and actuator keyword lists in that order, then model-size regexes
with word boundaries (1–14 b weak, 20–104 b actuator, 120–675 b mind); unknown
models default to actuator.

`get_llm_response[_async]` is the backward-compatible text entry point
(5-minute TTL; an empty response is an error, not a silent success);
`check_llm_health()` probes the mind/actuator/weak/aol roles via `/models` /
`/api/tags` with a 10 s timeout for the dashboard — g4f roles are reported as
connected with model `dynamic` **without probing**, because their roster is
discovered on demand.

### 7.2 Execution & caching core — `llm/cache.py` (~1 870 lines)

The single gateway every LLM call in the system passes through:
`get_cached_llm_response[_async]`. The async wrapper runs the blocking call on a
**dedicated 20-worker thread pool** (with a separate 5-worker pool for
split/merge summarization) — deliberately isolated from asyncio's default
executor so a stalled LLM can never starve the web server or the Telegram bot.
The pipeline, in order:

1. **Market gate (fail-closed).** `is_market_session_active()` wraps
   `_should_use_primary_model(allow_open_positions_bypass=False)`: a trading day
   is a weekday that is not an **Italian holiday** (ten fixed dates plus
   computed Easter Monday); the session is active during the configured open
   window or the 60-minute pre-market before it; the result is cached 30 s and
   a calendar failure fails **closed**. Outside the session **no LLM calls
   happen at all** — the gateway serves from Redis only and otherwise raises
   `MarketClosedError`; callers must degrade to non-LLM behaviour (HOLD / skip),
   never retry. The cache-only path probes a candidate key for the
   **OpenAI-configured** model of the role plus a synthetic
   `gate/market_closed` key, so decisions cached during market hours stay
   reusable after the close; ollama/g4f cache keys are not probed on this path,
   which is a known limitation of the gate.
   `force_primary_model` bypasses only the fallback-model downgrade, never this
   gate, and open positions do not bypass it.
2. **Role resolution.** `model_type` (mind / actuator / weak / sentiment)
   selects provider, model list, base URL and API key (per-role settings with
   global fallbacks), per-role temperature (explicit → per-role range parsed
   via `parse_temperature_range` → global; rounded to 2 decimals for the cache
   key), per-role thinking flag and reasoning effort, per-role timeout
   (actuator shorter — the market moves), per-role cache TTL
   (`LLM_MIND_CACHE_TTL` / `LLM_SENTIMENT_CACHE_TTL` / `LLM_CACHE_TTL`),
   per-role input token budget (`_get_max_input_tokens`, provider × role ×
   fallback aware) and the sentiment output cap (`LLM_SENTIMENT_MAX_TOKENS`).
   A sentiment role with no models configured degrades to weak. When the market
   is closed and fallbacks are enabled, the call is **downgraded to a fallback
   model up front** (`random.choice` over the fallback roster) so out-of-session
   work never spends the strong model's quota; if no fallback is configured it
   logs a warning and keeps the primary model rather than failing.
3. **Semantic cache.** The key is a SHA-256 over normalized messages/prompt,
   system prompt, model type, provider, model, temperature, the
   `LLM_CACHE_VERSION` salt and the **fee fingerprint** — an md5 over the
   seven fee settings — so a fee change invalidates decisions priced under the
   old fees. Numbers in the prompt text are rounded to 4 decimals for key
   purposes only (the prompt itself is sent unmodified). When `market_hash` is
   provided the key is **composite** —
   `llm:{version}:{fee_fp}:{provider}:{model}:{role}:market:{market_hash}:sys:{sha256[:16]}:t{temp}`
   — where `market_hash` is `compute_market_hash` over the normalized decision
   snapshot: volatile keys (`timestamp`, `fetched_at`, `published_at`,
   `last_eval`, `last_auto_resume`, `source`, any `*_time`, …) excluded, candle
   timestamps stripped for candle keys, floats rounded to 6 significant figures,
   `None` normalized to `"null"`, nesting capped at depth 10 — "same market
   state" maps to "same key", so an LLM-reviewed decision is reused while
   nothing material changed. Cache hits are recorded as zero-token metrics.
4. **Context-window management.** The budget is 80 % of the role's max input
   tokens; an over-budget prompt goes through `_split_and_merge_prompt`: the
   "Return JSON:" output-format section is carved out and preserved verbatim,
   the rest is chunked (paragraphs → lines → words), chunks are summarized
   **in parallel by the weak model** on the dedicated pool (temperature 0.1,
   thinking disabled, per-chunk timeout `min(LLM_TIMEOUT, 60 s)`, chunk limit
   `max(weak_budget × 0.6 / (1 + 0.5 × depth), 1 000)`) under a global timeout
   (`LLM_SUMMARY_TIMEOUT_MIN` … `chunks × LLM_SUMMARY_TIMEOUT_PER_CHUNK`),
   timed-out or failed chunks fall back to truncation, and recursion (depth
   ≤ 3, shrinking chunk limit) re-merges until the prompt fits. For the
   messages form only the **last user message** is compressed, leaving the
   system prompt and the rest of the conversation intact.
5. **Primary call.** Blacklisted models are filtered out, the remaining list
   is shuffled to spread quota, and each model is tried with 3 retries; an
   empty response counts as a failure.
6. **Fallback chain (labelled, not promoted).** On primary failure →
   `_execute_fallback_call` → `_try_fallback_models` (per-role fallback
   provider/model/URL/key chain, its own split/merge at 80 % of the fallback
   budget, `LLM_FALLBACK_TIMEOUT`) → last resort `_try_aol_model` (the
   Always-Online locally hosted model, dedicated timeout and token budget,
   which bypasses the blacklist when all its models are blacklisted). Fallback
   responses carry `is_fallback=True` in metrics and cache entries and are
   never promoted to final portfolio decisions. **By design the chain is
   disabled**: `LLM_FALLBACK_ENABLED` defaults to `False` and startup
   validation rejects any fallback configuration while disabled — a failed
   primary raises, because the bot prefers no decision over a decision from a
   weaker model.
7. **Bookkeeping.** Success deletes the consecutive-failure counter; every
   call — hit or miss, success or failure — stores a metric row (provider,
   model, role, prompt/completion tokens, latency, `request_type`,
   `is_fallback`); a total failure increments the circuit-breaker counter
   before re-raising.

**Model reliability machinery** lives here too: per-model failure counters
(`llm:fail_count:{model}`, 1-hour window) escalate into **blacklist levels** —
5 failures blacklist a model for `min(1800 × level, 7200)` seconds, the level
persists 7 days, entries are persisted to the `llm_model_blacklist` table,
re-synced to Redis at startup, and released early by a recovery probe (a
blacklist key seen with < 60 s TTL is dropped immediately, giving the model a
fresh trial). A model that answers successfully clears its counters and its
blacklist row. The blacklist **check itself fails open**: if Redis cannot be
read, models are treated as allowed, because a Redis outage must not blackhole
every LLM path. Failure statistics are served to the dashboard with SCAN-based
key iteration (no full keyspace scans). The **LLM circuit breaker**
(`llm:consecutive_failures`, 1-hour TTL, threshold
`LLM_CIRCUIT_BREAKER_THRESHOLD`; callers whose own outer timeout cancels a
call count it too) short-circuits risk-management LLM reviews after repeated
consecutive failures so the engine force-executes deterministically instead of
waiting on a broken LLM. A breaker read failure reports "inactive", letting
decisions proceed.

### 7.3 Prompt engineering

- **`system_prompt.py`** — `build_system_prompt(task_type)` instantiates a
  static template with live settings on every call (fee changes need no
  re-import): trading principles (timeframe hierarchy, confidence-scaled
  sizing, ≥2-indicator confirmation, no chasing breakouts), the
  stop-loss/take-profit contract (ATR multiples 2.0–5.0, TP strictly > SL,
  trailing-stop fields), max-hold-time guidelines per timeframe, risk-appetite
  regimes (Normal / Conservative / Probing keyed on breadth and P&L), the
  pause/resume contract, the two-step decision process, entry-condition types
  (`limit_price`, `rsi_threshold`, `delay`, `indicator_combo`) with minimum
  timeouts, the strict compact-JSON output contract, asset-specific rules
  (BTPs: no trailing stops, wider stops, YTM comparison; leveraged-ETF decay,
  earnings gaps) and the **fee tables computed from current settings** —
  round-trip costs for €1 000 and €10 000 trades, stocks (Tobin tax on buys
  only) and BTPs (zero-fee variant for primary issuance) — with the CRITICAL
  rule that take-profit must exceed break-even. The template *teaches the LLM
  its real costs* — the foundation of the "profitable after fees" goal.
- **Prefix-stability rule.** The **Past Mistakes Analysis** block
  (`get_past_mistakes_block`, Redis `llm:wrong_decision_analysis` — the LLM's
  self-critique of its losing patterns) is appended to the **end of the user
  message**, never the system prompt, because volatile content in the system
  prompt would invalidate provider prompt caches on every call.
- **`prompts.py`** (~870 lines) — `StrategyPromptData` (**103 typed fields**:
  quote, multi-timeframe candles and indicators, position state with
  triggered-review counters for stop-loss / take-profit / partial-TP /
  dust-sweep, portfolio exposure, market breadth, global risk multiplier,
  fundamentals, analyst ratings, insider transactions, options summary,
  dividends, macro context, symbol events) + `build_strategy_prompt`, the
  Step-1 decision prompt assembled in fixed sections — portfolio context,
  exposure summary, **queued orders** (with the instruction not to emit new
  BUY/SELL while an order is queued), cycle budget, global risk multiplier,
  timeframe/regime, macro context, session and evaluation-interval context,
  ATR and its percentile, **fee break-even calculation**, YTM / dividend
  yield / next ex-dividend, **the LLM's own previous decision for this
  symbol**, the position block that states the trade semantics explicitly
  (*BUY = add to the position, SELL = close the entire position*), multi-TF
  OHLCV summaries and indicator table, past trades, aggregate news sentiment
  and trend, market breadth, analyst/insider/options data, fundamentals,
  trade pattern analysis, and the triggered max-hold / stop-loss / take-profit
  / partial-TP / dust-sweep blocks, each phrased as "SELL, or HOLD with a new
  value — HOLD without one means auto-sell" — under a strict JSON response
  contract. Long-term timeframes suppress the short-term-only sections
  (sentiment, breadth, analyst, insider, options, news, events), and the
  minimum hold time quoted in the prompt is the **same value the validator
  enforces**, capped at ~1 year for 6M/1Y and fixed at ~1 year for 3Y/5Y.
  Token hygiene is built into the assembly: tickers reduced to the fields the
  LLM can act on, recent/past trades capped at 20, historical backtests at 5,
  multi-TF raw candles at the last 200 per timeframe, `raw_candles` at 500,
  `historical_ohlcv` at 1 000, non-assigned timeframes carrying a reduced
  indicator set. `build_analysis_prompt` is the compact Step-1a variant
  (direction only, no parameters); `build_strategy_messages` /
  `build_analysis_messages` produce the multi-turn form used for prompt
  caching.
- **`backtest_prompts.py`** — `BacktestPromptData` +
  `build_backtest_variants_prompt` (**Step 1b**: the LLM is shown its own
  Step-1a analysis and asked to translate it into concrete parameters and
  1–`MAX_BACKTEST_VARIANTS` variants, each with a **required**
  `backtest_entry_config`, under the validator's real minimums and the fee
  break-even) and `build_final_decision_prompt` (**Step 2**: preliminary
  decision + every backtest variant's summary and full statistics +
  historical results → the final BUY/SELL/HOLD, reviewed by the strong model).
  The Step-2 prompt encodes the **buy criteria explicitly** (positive
  expectancy — profit factor > 1.0, win rate > 40 %, or materially reduced
  drawdown versus buy-and-hold; a BUY is warranted on better risk-adjusted
  returns even when total return trails buy-and-hold; HOLD only when all
  variants consistently lose; variants with < 5 trades are unreliable and must
  not by themselves veto a BUY) and flags a **timeframe fallback** when a
  variant was backtested on a different timeframe than the assigned one. Its
  static decision rules and JSON schema are placed **before** the volatile
  per-symbol data so the cached prefix covers them.
  `build_backtest_variants_messages` / `build_final_decision_messages` are the
  prompt-caching forms.
- **`stock_selection_prompts.py`** — the two-phase selection prompts:
  `build_stock_selection_prompt` evaluates chunks of `LLM_CHUNK_SIZE`
  candidates (quotes, tickers with per-symbol min trade cost and sentiment,
  multi-timeframe OHLCV summaries, indicators, trend-quality scores,
  correlation matrix, dividend yields, BTP YTM, corporate events, news,
  performance, market regime), and `build_final_selection_prompt` aggregates
  the chunk shortlists (deduplicated, capped at 50 batches) with open
  positions, breadth, tenure and market context. Two properties matter:
  **the selection response is not just a symbol list** — it returns the
  engine's own operating parameters (skip-evaluation thresholds, regime
  thresholds, minimum ATR/hold multipliers, review caps, pause durations,
  portfolio exposure and stop-risk caps, risk-reward and confidence
  thresholds, re-evaluation interval, global risk multiplier), i.e. the LLM
  tunes the machine that executes its decisions; and the `MIN_SYMBOLS` floor
  is stated as **mandatory with an explicit warning that the engine overrides
  a smaller selection**, while `MAX_SYMBOLS` caps it. Timeframe assignment is
  constrained to timeframes that actually have OHLCV data for that symbol,
  selection-level analysis is limited to the long-term frames (5Y/3Y/1Y/6M/3M/1w
  summaries, 5Y/1Y/3M indicators) and news to the top 20 candidates,
  correlations are shown only above |0.3|, and candidate lists are trimmed to
  100 — all to keep the prompt inside the context window.
- **`prompt_utils.py`** — token hygiene: recursive float rounding for cache
  stability, **TOON serialization** (`python-toon`) for compact tabular data,
  whitespace compaction, OHLCV summarization (window stats instead of raw
  candle walls), trade-pattern formatting (best entry conditions / timeframes
  / exit reasons / confidence ranges / best & worst symbols with win rates,
  5 items each), news formatting caps (5 articles, 200-char summaries),
  timeframe parsing including monthly and yearly units, and
  `get_cached_news_summary[_async]` — a one-sentence (≤ 120 chars)
  news-sentiment summary written by the weak model, Redis-cached under
  `news_summary:{symbol}` with 60 s TTL on the empty/failure paths to avoid
  hammering the database, and the full `NEWS_CACHE_TTL_SECONDS` on success.

### 7.4 Summarization — `llm/summarizer.py`

`summarize_text[_async]`: text already under the length cap passes through
unchanged; otherwise the **weak model** summarizes it (24-hour Redis cache,
`request_type="summarization"`), preserving numbers, dates and entity names.
On failure it retries once with the actuator role under
`request_type="summarization_fallback"` and `force_primary_model=True` —
skipped when the caller already demanded the primary model — and if that fails
too the original text is returned: summarization must never lose data, only
compress it. The async wrapper runs on the dedicated LLM thread pool, so a
slow summarizer cannot block the event loop.

## 8. Trading Engine & Strategies

This is the largest subsystem: a thin engine façade, **35 dedicated components**
(`src/trading/components/`), a strategy layer that turns free-text LLM output into
validated, executable, auditable trading decisions, and a paper broker that
simulates the real one.

### 8.1 The `Signal` — the universal decision object — `strategies/base.py`

Every decision in the system, whatever its origin, is a `Signal` dataclass:
`action` (BUY/SELL/HOLD), `confidence`, `reasoning`, and 43 further optional
fields (46 in total — count verified) the LLM may set, grouped as:

- **Risk/exit geometry** — `stop_loss`/`take_profit` percentages,
  `stop_loss_method` with `stop_loss_atr_multiple` / `take_profit_atr_multiple`
  (ATR-based exits), `trailing_stop` plus distance / ATR-multiple / activation,
  `max_hold_time_seconds`, `cooldown_after_loss_seconds`, `risk_level`.
- **Sizing** — `position_size` (fraction of the per-symbol budget) and
  `confidence_sizing_weight` (how much confidence scales that size).
- **Delayed execution** — `entry_condition` (e.g. `{"type": "limit_price",
  "price": …, "timeout_seconds": …}`) and the `timeframe` its indicator conditions
  are evaluated against.
- **Portfolio vote** — `portfolio_risk_adjustment_factor` (0.1–1.0): a per-symbol
  LLM vote, and the engine takes the **min** across symbols, so the LLM can only
  ever *lower* global risk, never raise it.
- **Native order geometry** — `order_type`, `limit_price`, `stop_price`,
  `trail_offset`, and the per-side `stop_loss_order_type/stop_price/limit_price/
  trail_offset` and `take_profit_order_type/limit_price` the exit-order manager
  places as real orders.
- **Backtest requests** — `backtest_period_days` and `backtest_variants` (the
  LLM's own proposed parameter sets to simulate before it is believed).
- **Free-form LLM payload & absolute prices** — `price`/`sl`/`tp` (absolute levels,
  converted to percentages by the validator), `strategy_type` + `strategy_params`
  (the LLM-defined parameter bag), `indicator_config`, `backtest_summary`, and
  `reason` (free-text explanation for logging).
- **LLM identity** — `model_type`, `llm_provider`, `llm_model`, recorded with every
  decision so per-model quality can be measured.
- **Provenance** — `step2_reviewed` (set **only** on genuine Step-2 LLM success
  paths in `BacktestManager.run_step2_llm_call`, never on fallback or downgrade
  paths), `origin` (non-LLM sources, e.g. `risk_manager` for circuit-breaker exits,
  which are risk-reducing and exempt from the provenance gate), and
  `decision_source` (`live` vs `cache`, observability only).

`Signal.from_dict` is fail-safe by construction: a missing, malformed or
out-of-vocabulary `action` becomes **HOLD** — an unparseable LLM reply can never
default into an executable decision. `Strategy` / `LLMStrategy` give the engine one
uniform `generate_signal(market_data)` interface, so a pre-computed LLM decision is
just another strategy, and a future non-LLM strategy is a drop-in replacement.

### 8.2 Parsing & validation — `strategies/llm_parser.py`, `strategies/validator.py`

The LLM's raw text goes through a gauntlet, and **every gate that fails produces a
HOLD**, never a repaired trade.

**Parsing** (`llm_parser.py`, 691 lines): `_extract_first_json` scans brace depth to
find the first complete JSON object inside free text; Pydantic models
(`LLMResponseModel`, `StrategyModel`, `EntryConditionModel`) enforce structure;
`_validate_semantic_quality` rejects economically nonsensical decisions
(take-profits below the round-trip fee, stops above caps) and can downgrade the
action; `_clamp_parameter_ranges` hard-clamps every numeric parameter, with
`stop_loss_pct` clamped by horizon (0.80 for ≥1Y, 0.65 for ≥6M, 0.5 otherwise);
`_score_reasoning_quality` scores the reasoning text 0–1, which is recorded and fed
back into later prompts. Result: an `LLMStrategy` wrapping a clean `Signal`.

**Deterministic risk validation** (`validator.py`, 648 lines) — `validate_signal`
wraps `_validate_signal_impl` and logs every BUY/SELL→HOLD downgrade. The gates run
in a fixed order, each returning a HOLD on failure:

1. **Action gate** — anything not BUY/SELL becomes HOLD; absolute `sl`/`tp` are
   converted to percentages against the live price.
2. **`_validate_backtest_entry_config`** (BUY only) — the EMA/ADX entry filter the
   backtest must use; missing gets a sane default (`ema_period 21`, `above`,
   `min_adx 20`, `logic: and`), malformed is a HOLD.
3. **`_validate_stop_loss`** — ATR-based stops scaled to a **daily equivalent**
   (`(86400 / timeframe_seconds) ** 0.5`), so a 5-minute ATR is never used to size a
   6-month stop; floor 1 %, cap 50 %; timeframe-aware defaults when ATR is
   unavailable (15 % ≥1Y, 10 % ≥1M, 5 % medium-term).
4. **`_apply_required_defaults`** — `trailing_stop` defaults to False;
   `position_size_fraction` is **derived from the risk budget**
   (`max_risk_per_trade_pct / stop_loss_pct`, floor 0.01) and only falls back to
   timeframe heuristics (0.25 / 0.15 / 0.10); `max_hold_time_seconds` defaults to
   10× the timeframe, capped at 5 years.
5. **`_validate_take_profit`** — a **1.5:1 reward:risk floor** (3×ATR or 1.5×SL,
   capped at 500 %), never below break-even.
6. **`_validate_trailing_stop`** — gated by `BTPPolicy.supports_trailing_stop`
   (BTPs never trail).
7. **`_validate_required_params`** — `position_size_fraction` ∈ (0, 1],
   `max_hold_time_seconds` ≥ `min_hold_time_mult × timeframe` (capped ~5 years), so
   a 5-minute timeframe cannot be traded with a 20-minute intended holding period.
8. **`_validate_optional_params`** — ~40 optional parameters type- and
   range-checked (fee and slippage models, direction, `global_risk_multiplier`,
   `confidence_sizing_weight`, gap tolerance and `on_gaps`, backtest balance and
   trade caps, `reasoning_quality_score`, portfolio exposure and stop-risk
   limits …).
9. **`_validate_logical_consistency`** — take-profit must exceed stop-loss (skipped
   only when both are ATR-based), trailing distance must be tighter than the stop,
   and trailing stops are rejected outright for BTP symbols.

Confidence is deliberately **not** a rejection criterion here — it only scales
position size later. `VALID_STRATEGY_TYPES` is the closed vocabulary
(`momentum`, `mean_reversion`, `breakout`, `swing`, `position`).

### 8.3 Backtesting — `strategies/backtester.py` (~1 090 lines)

A full simulation engine used to judge LLM proposals **before** real money is put
behind them. It is deliberately built to be pessimistic in the same ways a real
account is pessimistic.

- **`BacktestConfig`** — ~47 fields: entry geometry (`backtest_entry_config` with
  EMA period/direction and `min_adx` gate), stop loss and take profit as both
  percentages and ATR multiples, trailing stop and *trailing take-profit*, partial
  take-profit levels, breakeven move, max-unrealized-loss force exit, max hold
  time, position sizing and exposure caps, direction (long/short), RSI/MACD
  filters, dividend reinvestment, gap tolerance and `on_gaps` behaviour, and the
  fee/slippage model switches. This object is what the LLM is allowed to propose,
  so its field list is effectively the strategy vocabulary the system permits.
- **Real costs** — `_compute_intesa_fees` applies the actual Intesa fee schedule
  (with the BTP branch) to every simulated fill, and
  `_compute_dynamic_slippage` derives per-fill slippage from the trade's size
  against rolling average volume and ATR, capped by configuration. A strategy that
  trades too often, too big, or in illiquid names is punished here, not in
  production.
- **Data honesty** — `_detect_gaps` finds discontinuities in the candle history
  (tolerance scaled by ATR via `BACKTEST_GAP_TOLERANCE_MULT`) and warns or skips,
  so a "great" result produced on a broken history is labelled as such and shown
  to the reviewing model rather than silently trusted.
- **`backtest_strategy`** replays candles through the entry gate, the position
  lifecycle and every exit rule, producing a trade list with per-trade PnL.
- **`_compute_stats`** aggregates that list into the numbers the LLM and the human
  both read: win rate, profit factor, average win/loss, max drawdown, Sharpe,
  annualized gross **and net** return (annualization is time-weighted, so a
  3-week backtest is not presented as a yearly result), total fees paid, and an
  explicit **buy-and-hold comparison** over the same window — the answer to "would
  doing nothing have been better?"
- **`format_backtest_summary`** renders those stats as a compact, token-cheap,
  LLM-readable block (the same numbers are also stored to SQL through
  `save_backtest_result`, with a params hash so identical proposals are reused
  rather than recomputed).
- **`walk_forward_backtest` / `format_walk_forward_summary`** split long histories
  into rolling train/validate windows and report the distribution of results
  across them. This exists for one reason: to make an overfitted single-window
  strategy visible as what it is.

### 8.4 The engine — `trading/engine.py` (composition root & façade)

`TradingEngine` is intentionally **thin**: it owns no trading logic of its own. Its
job is to be the single place where the machine is assembled and where the two
human surfaces have something to call.

- **Composition root.** `__init__` builds `SharedState`, the `EventBus`, the
  `UnifiedConfigService` and the Redis client, then constructs every component
  with `(engine, event_bus)` and performs the few explicit cross-wirings that
  cannot go through the bus (notably `OrderExecutor` ↔ `BuyExecutor` ↔
  `ExitOrderManager`, and the notifier). Components talk to each other through
  bus `publish` (notification) and `request` (command, with errors propagated),
  never by reaching into each other's internals.
- **Resource isolation.** Six async semaphores (exchange calls, news, indicators,
  backtests, downloads, symbol processing) and three dedicated thread pools
  (`dbwriter`, `downloader`, `quotes`). The point is specific: a slow Yahoo
  download or a burst of DB writes must never occupy the default pool that serves
  the web dashboard and Telegram.
- **Startup hygiene.** On boot it clears time-sensitive Redis keys
  (`_clear_time_sensitive_redis_keys`) and stale local pause keys
  (`clear_trading_pause_keys`) so a container restart cannot inherit a paused or
  half-decided market state; it registers a settings-reload callback
  (`_on_settings_reload`) that rebuilds the semaphores, invalidates the yfinance
  session and asks the market-data manager to drop its clock cache; and it logs a
  subscription summary so a missing handler is visible in the log at startup
  rather than as a silent no-op later.
- **Public API.** The methods web and Telegram actually call: `reset_paper_trading_state`,
  `trigger_symbol_reevaluation(force)`, `force_download_all_assets`,
  `force_download_tracked_symbols`, `get_performance_summary`, `get_pause_status`,
  `sell_all_positions`, `sell_position`, `simulate_backtest`, `simulate_decision`,
  `set_notifier`, plus `_is_market_open` and the risk-multiplier setter. Everything
  else is internal plumbing.
- **Self-monitoring.** `_record_unexpected_exception` writes
  `metrics:unexpected_exception:{context}:{exc_type}` to Redis (24 h TTL) and
  raises an alert on the third occurrence in the window — an exception that keeps
  happening becomes a notification, not a log line nobody reads.
- **`engine_utils.py`** holds the shared helpers the components need: timeframe
  conversions, display-symbol formatting, exclusion rules,
  `normalize_llm_symbol` (mapping the LLM's sloppy `'ENI'` back onto
  `'ENI.MI/EUR'`), and `get_effective_refresh_interval`.

### 8.5 The components — `trading/components/` (35 files, ~21 300 lines)

This is where the engine's behaviour actually lives. Grouped by responsibility
(line counts verified from the tree):

- **State & orchestration**
  - `shared_state.py` (321) — the single mutable-state object: positions, queued
    orders, cycle spend, pending entries, per-symbol evaluation state, trade
    history (capped at 500), daily realized PnL and buy fees. Every field is
    behind a named lock, and the class docstring states a **mandatory lock
    ordering** (positions → queued orders → cycle spent → eval state → pending
    entries) with an `ordered_locks` helper, because this is the one object
    touched by every loop in the system and the natural place for a deadlock.
  - `engine_orchestrator.py` (401) — builds and runs the **26 supervised
    background loops** (each wrapped in `TaskSupervisor`: crash → log → notify →
    restart), the main `run()` loop (engine interval, 120 s open-positions
    heartbeat key, symbol processing skipped while re-evaluation is running,
    periodic dirty-state save), the XMIL market-clock monitor (pause on close,
    **never overriding a pause whose source is `llm`**, session countdown
    notifications, 30-min weekday / 2-h weekend cadence), and the
    **wrong-decisions self-analysis loop** (every 6 h, weak model, market-open
    only, last 20 wrong decisions → `llm:wrong_decision_analysis`, which is fed
    back into the system prompt).
  - `background_task_manager.py` (1 505) — the bodies of those periodic loops:
    data refresh, indicator computation, news fetch, risk checks, dividend
    reinvestment buys, LLM decision-outcome evaluation, Redis health checks.
  - `evaluation_scheduler.py` (134) — decides *when* each symbol is evaluated, and
    enforces the fail-closed gate: when the market session is not active it
    returns **no symbols at all**, so trading and re-evaluation stop together.
    Base per-timeframe intervals are halved (floor 900 s) in the open/close
    windows and under extreme market breadth, shortened to ≤1 800 s on sentiment
    shift, doubled (cap 28 800 s) in a quiet market, then overridden by any
    LLM-set per-symbol `strategy_interval`.
  - `state_initializer.py` (132) / `state_persistence.py` (403) — load on boot,
    periodic + on-shutdown saving of positions, orders, balances, pause state.

- **Decision pipeline (the two-step LLM contract)**
  - `signal_processor.py` (1 865, the second-largest component) — the per-symbol
    pipeline: fetch market data, classify market regime, compute ATR percentile,
    decide `should_skip_llm_eval`, gather all prompt context (multi-timeframe
    indicators, candles, news, dividends, analyst targets, options data, volume
    trend), and build the analysis prompt plus the decision snapshot.
  - `signal_market_data.py` (137) — multi-timeframe indicator computation for prompts.
  - `simulation_manager.py` (557) — Step-1 data preparation and Step-1 LLM calls
    (also the surface behind the dashboard's "simulate" endpoints).
  - `llm_step_manager.py` (428) — Step 1a analysis with retry/correction and the
    explicit `_create_fallback_hold_signal` (`llm_provider="fallback"`), plus Step 1b.
  - `backtest_manager.py` (1 347) — prepares the LLM-proposed variants, runs them
    in parallel under the backtest semaphore, then makes the **Step-2 final
    review** call to the strong model. This is the only place `step2_reviewed`
    is set to `True` (line-verified); every failure path explicitly sets it
    `False`, and `decision_cache.py` re-asserts `True` only when carrying an
    already-reviewed decision over from the cache. The whole step is gated by
    the decision cache: a snapshot-hash hit reuses the previously reviewed
    decision and carries over the execution-critical fields, so a cache hit
    still satisfies the provenance gate.
  - `model_tier_manager.py` (582) — `compute_prompt_complexity` from candidate
    count, volatility percentile, RSI/MACD/Bollinger state and portfolio risk,
    mapping to mind / actuator / weak tier with dynamic threshold adjustment,
    effective temperature and reasoning effort.
  - `decision_cache.py` (196) — deterministic snapshot hash over the exact Step-2
    prompt inputs; store/get/invalidate; only genuine Step-2 successes are stored;
    every outcome is metered into `decision_cache_metrics` (§5.1) with estimated
    prompt tokens saved on hits — a cache hit writes no `llm_metrics` row, so this
    table is the only place the cache's savings are visible.
  - `post_decision_manager.py` (976) — the last gate before execution:
    **`check_llm_provenance`** (BUY/SELL require a real provider+model **and**
    `step2_reviewed=True`; only `origin="risk_manager"` sells are exempt;
    violations are forced to HOLD, counted and alerted), trade filters, sector
    concentration check, triggered-flag handling, entry-condition registration,
    decision logging and quality recording.

- **Positions & risk**
  - `position_manager.py` (1 409) — cost-basis integrity, exposure summary,
    performance summary, per-position risk metrics, equity & drawdown series,
    trade-pattern analysis (fed back into prompts), `update_position_params`,
    `_close_btp_at_par`, and `reconcile_positions` against the broker's view.
  - `risk_manager.py` (2 262, the largest component) — the SL/TP monitoring loop:
    hard and soft stops, news-sentiment exits, trailing stop and trailing
    take-profit maintenance, native stop-order updates, partial take-profits,
    dust sweeps, max-hold/position-age enforcement, manual SL/TP handling,
    breakeven moves, drawdown circuit breaker, daily-loss limit, loss cooldown,
    correlation risk and portfolio VaR/stress checks, and the global risk
    multiplier.

- **Execution**
  - `order_executor_base.py` (121) / `order_executor.py` (803) — order dispatch,
    limit-price distance checks, fee application, fill processing, `execute_signal`,
    `sell_all_positions`, `sell_position`.
  - `buy_executor.py` (1 061) — position sizing from risk, SL/TP parameter
    computation, min-profit and min-order-size checks, buy limit-price computation,
    position create/update, fill recording and notification.
  - `sell_executor.py` (939) — partial sells with level labels and exit reasons,
    fill processing, cleanup callbacks.
  - `exit_order_manager.py` (612) — native stop/limit/trailing exit orders, price
    computation from entry price + signal + ATR, fill-timeout fallback to market sells.
  - `entry_signal_manager.py` (454) — delayed entries: condition checking, timeout
    handling, pending-entry processing.
  - `manual_trade_logger.py` (148) — journaling of trades made by hand in notify mode.

- **Symbol re-evaluation subsystem** (the periodic LLM portfolio review) —
  `symbol_reevaluator.py` (520, `ReevalContext` phase state machine) plus its
  `reeval_*` family: `reeval_data_fetcher.py` (716, candidate gathering, quote
  fallbacks, filtering), `reeval_shortlist_builder.py` (452, LLM shortlist +
  minimum-symbol enforcement + OHLCV summaries), `reeval_llm_runner.py` (531,
  chunked selection calls behind `_TokenBudgetSemaphore`),
  `reeval_response_processor.py` (167, parse/validate selections),
  `reeval_config_manager.py` (208, LLM-decided parameters → Redis),
  `reeval_market_condition_monitor.py` (159, conditions that trigger early
  re-evaluation), `reeval_pause_resume_manager.py` (133, LLM pause/resume
  decisions with global risk multiplier), `reeval_post_selection_manager.py`
  (135, backfill + stale-state pruning as supervised background tasks),
  `reeval_notifier.py` (146, completion notifications).

- **Market data & pause state**
  - `market_data_manager.py` (1 077) — `ClockInfo`/`AssetInfo`, XMIL session
    phases and staleness, asset universes (stocks, ETFs, BTPs) with TTL caches,
    VIX, batched async quotes, OHLCV backfill and gap filling, indicator
    computation and storage, force-download paths.
  - `pause_resume_manager.py` (400) — the LLM pause/resume decision, evaluated
    fail-closed (see §5.2/§9): a pause whose source is `llm` is not auto-resumed
    by the clock monitor.

### 8.6 The paper trader — `trading/paper_trader.py` (~830 lines)

The broker stand-in. It exists so the system can be run end-to-end, forever, with
real decisions and real consequences, at zero financial risk — and so that the
consequences are **realistic enough to be worth learning from**.

- **Order types** — `market`, `limit`, `stop`, `stop_limit` and `trailing_stop`,
  buy and sell, each with time-in-force and fill timeout, mirroring what the real
  interface exposes. Orders and balances are persisted to SQL, so a restart does
  not create phantom positions.
- **Dynamic slippage** — `_get_dynamic_slippage` derives the fill penalty from the
  order's size against the last 21 daily candles' volume **plus an ATR term**
  (`slippage += atr_pct × 0.05`), capped by
  `PAPER_SLIPPAGE_MAX_PCT` (with a short cache so repeated fills are consistent).
- **Volume reality** — `_get_max_fillable_volume` limits a fill to a fraction of
  recent volume (`PARTIAL_FILL_VOLUME_CAP_PCT`): you cannot buy more than the
  market traded, and oversized orders fill **partially**. `_compute_market_impact_pct`
  adds a self-inflicted price impact, `0.05 * ratio**0.5`, where ratio is order size
  vs available volume. Illiquid names and large positions therefore behave the way
  they do in real markets.
- **Fill polling** — an adaptive poller: `_poll_interval_base * 1.5 ** min(idle_polls, 4)`,
  so a busy order book is polled promptly and a quiet one is polled cheaply.
  Cancellation, order lookup and complete trade history are provided.
- **Fees** — delegated to `src/exchanges/fees.py`, the same schedule used by the
  backtester, so simulated costs and backtested costs agree.

This is the "broker reality" the whole system is calibrated against: the LLM is
judged, the validator is tuned, and the risk manager is evaluated against a
counterparty that charges, slips, partially fills and refuses to move the market
for free.

---

## 9. Interfaces — The Two Human Surfaces

The engine has no UI of its own. It is reached through exactly two surfaces, both thin,
both deliberately redundant with each other (everything the dashboard can do, the phone
can do, and vice versa), and both speaking to the engine through the same façade methods
and `EventBus` requests the engine exposes — never by reaching into components.

### 9.1 Web dashboard — `src/web/app.py` (FastAPI, ~1 100 lines)

A single-file FastAPI application that is at once the operator console, the read-only
telemetry surface, and the Docker health endpoint.

**Security model (single-user, self-hosted).**

- `verify_auth` — a `session_token` cookie checked against Redis (`session:{token}`,
  24 h TTL). If `WEB_USERNAME`/`WEB_PASSWORD` are unset, auth is disabled entirely
  (trusted-LAN deployment).
- `verify_csrf` — double-submit CSRF: the `X-CSRF-Token` header must equal the
  `csrf_token` cookie, which must equal `csrf:{session_token}` in Redis, all compared
  with `secrets.compare_digest`. Every mutating route requires it.
- `login` / `logout` on a **public router**; everything else lives on the
  auth-guarded `http_router` (`/api/v1`). Login compares credentials with
  `secrets.compare_digest`, mints a `token_urlsafe(32)` session id in Redis
  (`session:{token}`, 24 h) and a **second** token bound to that session
  (`csrf:{token}`). The session cookie is `httponly`; the CSRF cookie is
  deliberately **not** httponly, because the dashboard's JavaScript has to read it
  and echo it back in a header.
- **Rate limiting middleware**: sliding-window counters in Redis sorted sets
  (`rate_limit:{ip}`, `rate_limit:global` at 10× the per-IP budget,
  `ws_rate_limit:{ip}` for WebSocket handshakes), limits read fresh from `settings`
  on every request so `settings.reload()` takes effect instantly. Each request is
  recorded as a `uuid4` member scored by timestamp, so two requests in the same
  millisecond cannot overwrite each other's count. Over budget ⇒ `429`. On Redis
  failure the limiter **fails open** — the dashboard must stay usable when the cache
  layer is down, and this is one of the few places where fail-open is intentional.
- `/health` is registered directly on the app, unauthenticated, and returns 503
  until the engine is attached — it is the Docker `HEALTHCHECK` target and cannot be
  shadowed by the authed router.
- `/api/v1/health` is a different, **authenticated** endpoint: the deep health
  check. It pings Redis and all three LLM roles (mind / actuator / weak) through
  `check_llm_health` and answers `ok` or `degraded` with each role's status, so
  "the container is up" and "the system can actually think" are two separate
  questions with two separate endpoints.

**Read endpoints** — `/status` (symbols + ISINs, positions with live P&L, balances in
native and base currency via `{CUR}EUR=X` FX quotes, pause state, market open/phase,
queued orders, Redis availability), `/trades` (open trades enriched with SL/TP,
trailing-stop config, order ids, max-hold time, live P&L), `/profit`, `/performance`,
`/risk`, `/market-status`, `/news` (per-symbol LLM news summaries), `/messages`
(the notification mirror), `/logs` (Redis log stream), `/history` (closed trades),
`/signals` (paginated LLM decision records, page size clamped to ≤ 100),
`/discovered-symbols` (asset registry with ISIN, asset type, country and candle
count — also the autocomplete source), `/manual-trades` (the journal of trades the
human logged by hand), `/config` (effective trading mode, base currency, max
symbols, web port, and the **resolved provider/model for each LLM role** —
`_resolve_llm_role_settings` resolves per-role overrides, falling back to the
global provider and, for `g4f`, to the live `_get_g4f_models` list),
`/ohlcv/{symbol}` (sanitized candles, limit clamped to ≤ 1 000 — non-finite floats
become `null` for JSON compliance), `/ticker/{symbol}` and `/tickers?symbols=…`
(both degrade to `null` fields rather than 500 when quotes fail), `/llm-metrics`,
`/llm-metrics/timeseries` (hour/day/week/month or explicit date range, model filter),
`/llm-decision-quality`, `/decision-cache` (hit rate and estimated token savings of
the snapshot-hash decision cache — a cache hit skips the Step-2 call and so writes
no `llm_metrics` row; these counters are the only place its savings are visible).

**Write endpoints** (all CSRF-guarded; anything that would block is handed to a
background task, so an HTTP request never waits on the engine):

- `/pause` — `set_trading_pause(redis, "manual", …)`: the pause is recorded as
  **human**-sourced, which is precisely what keeps the engine's auto-resume logic from
  overriding a deliberate decision.
- `/resume` — refused with 400 while the market is closed, then `clear_trading_pause_keys`:
  the same key-namespace clear the bot uses, so pause and resume stay symmetric
  operations on one set of keys.
- `/sell?symbol=` (or all positions) — refused while the market is closed; invalidates
  the shared WebSocket payload cache first so the open tabs repaint immediately, then
  `asyncio.create_task(engine.sell_position(…))`.
- `/manual-trade` — journals a trade the human executed at the real broker, validated
  hard: side ∈ {buy, sell}, quantity > 0, money_spent > 0, the symbol must exist in the
  discovered-asset registry, and the implied price (`money_spent / quantity`) must fall
  within **10 %** of the live quote. A typo is rejected, not silently recorded; and if
  the quote cannot be fetched the request fails **503** — an unverifiable price is not a
  valid price.
- `/update-isin`, `/clear-isin` — manual ISIN override/removal, the escape hatch when
  discovery gets it wrong.
- `/reload` — `settings.reload()`; every live-value reader picks the new values up on
  its next read.
- `/force-reeval`, `/force-download`, `/force-backfill` — symbol re-evaluation,
  tracked-symbol OHLCV download, full-universe backfill.
- `/restart` — `await engine.stop()` then `sys.exit(0)`: a clean shutdown that lets
  Docker's restart policy bring the container back.
- `/llm-metrics/reset` — wipes the SQL side (`reset_llm_metrics`,
  `clear_all_blacklisted_models`, `reset_llm_decision_quality`) **and** the Redis side
  (`llm:metrics:*`, `llm:blacklist:*`, `llm:fail_count:*`, `llm:blacklist_level:*`):
  a reset that left the circuit breakers armed would be a reset in name only.
- `/simulate/backtest/{symbol}`, `/simulate/decision/{symbol}` — the Simulation Lab:
  run the backtester, or a full Step-1 + Step-2 decision, for one symbol without
  touching live state.

**Market-aware guards.** `/resume`, `/sell` and `/manual-trade` are refused with 400
when the XMIL market is closed — the bot will not act on, or record, trades against a
market that is not trading. This is the fail-closed principle applied to the human
interface, not just to the engine.

**The WebSocket feed** — `/api/v1/ws` pushes the live dashboard payload:

- The handshake is rate-limited by `ws_rate_limit:{ip}`; over budget the socket is
  closed with code 1008 before any work is done.
- Session cookie verified at handshake **and re-verified on every tick**, so an expired
  session closes an already-open socket (policy-violation close code 1008) — a socket is
  not a long-lived credential.
- A **shared payload cache** (`_ws_payload_cache`, 5 s TTL, invalidated by mutating
  actions) means N open tabs cost one set of quote fetches, not N; the first iteration
  of every connection always builds fresh, so a newly opened tab never renders stale
  data.
- The payload is assembled in parallel — `asyncio.gather` over display-name lookups and
  over positions, one batched quote fetch for all open positions,
  `get_isin_map_from_db` for non-BTP symbols — and carries positions with live price,
  unrealized P&L (absolute and percent) and position value; queued orders with the bulky
  embedded `signal` object stripped before transmission; pause state from
  `engine.get_pause_status()`; market phase; discovered symbols; native balances;
  `redis_available`; and **base-currency balances** converted through
  `{CUR}{BASE}=X` FX quotes, where a currency with no rate is reported as `null` rather
  than silently assumed to be 1:1.
- **Diff-send**: the serialized payload is compared with the last one sent and only
  transmitted when it changed, at a 1 s cadence — a quiet portfolio produces no traffic.
- While the engine is booting, clients get `{"status": "initializing"}` instead of an
  error, because the web server starts before the engine is ready by design.

**The dashboard itself** — `src/web/static/` is a dependency-light PWA: a single
`index.html` (~3 070 lines of vanilla JS + CSS), Chart.js 4.4.0 from CDN as the only
script, `manifest.json`, `sw.js`, and an SVG icon.

- **Installable PWA**: `sw.js` is network-first for navigations and caches only the
  manifest/icon (API and WebSocket are always network), an in-app install button uses
  `beforeinstallprompt`, and Chrome-Android PNG icons are **generated on the fly** by
  the `/icon-{size}.png` route (PIL draws the rounded-square logo; a transparent PNG
  is served if PIL is missing) — no binary assets in the repo.
- **Cards**: profit summary, risk metrics, messages, balances, manual-trade form (with
  discovered-symbol autocomplete), latest LLM signals (paginated, click-through),
  positions, queued orders, performance, closed trades, market status, news, manual
  ISIN management, Simulation Lab (search, run-all, per-symbol result), LLM metrics
  (summary, period charts, date-range chart, per-model table, recent calls, model
  failures & blacklist, decision quality), and a live log viewer.
- **Controls**: pause, resume, refresh, force backfill, reload settings, restart app,
  and a client-side update-interval selector (3 s … 60 s).
- **Chart modal**: any symbol is clickable → candle/OHLC chart with timeframe selector
  from 5Y down to intraday, refreshed on interval.
- **UX details**: login overlay, dark/light theme toggle persisted in `localStorage`,
  CSRF token kept in `localStorage` and sent as a header, mode badge, status/trading
  status dots, modals for full signal text and full LLM call text.

### 9.2 Telegram bot — `src/telegram/bot.py` (~1 550 lines)

The mobile control surface and the engine's **notifier** (it is injected as the
engine's notifier in `main.py`, so every engine notification flows through it).

- python-telegram-bot v20 `Application` with `concurrent_updates(True)`, tuned
  long-polling (`poll_interval=5.0`, `poll_timeout=30.0`) to avoid asyncio socket
  churn, and a global `error_handler` that classifies timeouts/network/API errors.
- **Single-operator authorization**: only the configured `TELEGRAM_CHAT_ID` is served;
  every handler starts with `_is_authorized`. With no chat id configured the bot
  answers nobody (and says so in logs).
- **Commands**: `/start` (stores the chat id, shows the menu), `/menu`, `/pause`,
  `/resume`, `/status`, `/trades`, `/profit`, `/performance`, `/news` (free-text
  symbol search), `/news_status`, `/risk`, `/market`, `/sell`, `/backfill`,
  `/signals`, `/reset`, plus a plain-text handler for non-command messages and a global
  error handler.
- **A button keyboard** mirrors the commands for phone use (Status, Trades, Profit,
  Performance, Risk, Signals, News, Re-eval, Pause, Resume, Sell All, Backfill);
  `handle_button` routes button text to the same command implementations, and any
  unrecognized text just re-shows the keyboard.
- **Long replies** are split into ≤ 4 000-character chunks that avoid breaking HTML
  tags mid-tag, with the reply markup attached only to the final chunk and each chunk
  sent under its own `TimedOut`/`NetworkError`/`TelegramError` guard — a partial
  delivery is logged, not raised.
- Command replies are built the same way the dashboard builds its payload: batched
  quote fetches, parallel name/ISIN lookups, per-field timeouts, and graceful
  degradation when a fetch fails.
- **Every dependency call is timeout-wrapped with a user-visible fallback**: 5 s for the
  Redis reads and writes (chat id, pause set/clear, message mirror, audit-log write),
  10 s for the market-open check, 30 s for the paper-state reset, 15 s per outgoing
  chunk. A handler that would have hung now answers "timed out" in one line, because a
  control surface that stops replying is indistinguishable from a dead engine.
- `/resume` refuses while the market is closed, then clears the pause keys **and** calls
  `engine.trigger_symbol_reevaluation()` — resuming is treated as "the universe may have
  changed while we were stopped", not as a flag flip.

**The notification pipeline** (`send_notification`) is a subsystem in its own right:

1. Capture the active traceback for `ERROR` notifications before any `try` can clear it.
2. Resolve the chat id with a 5 s timeout; no chat id ⇒ notification dropped with a
   warning, never an exception into the engine.
3. **Verbosity filter** (`NOTIFICATION_VERBOSITY`): `all`, `none`, `errors_only`,
   `trades_only`; `PAUSE`/`RESUME` always pass, and under `errors_only`/`trades_only`
   a message with no recognizable `action` passes as well — a message the filter
   cannot classify is not silently discarded. An *unknown* verbosity value fails
   closed (nothing is sent), and `none` means none.
4. **Rate limiting**: 15 notifications/minute sliding window. Over the limit,
   notifications are *queued* (not discarded) in a bounded deque of 50 and drained by
   the background queue task after each item it processes — a burst is delayed, never
   lost, and a burst that never ends drops the oldest, not the newest.
5. **Provenance line**: notifications carry the model that produced the decision,
   rendered with a role emoji (🧠 mind, 🤖 actuator, ⚡ weak, 📰 sentiment). When the
   caller supplies provider/model it is used verbatim; when it does not, the bot
   resolves that role's configured provider itself (ollama model, the live g4f model
   list, the openai model) and joins the names with `", "` — the operator sees *which
   brain* spoke even when the caller forgot to say.
6. **Dashboard mirror**: every sent message is also `lpush`'d to `web:messages`
   (trimmed to 100) *before* the Telegram send, so the dashboard shows the same
   stream even if Telegram is unreachable.
7. **Silent by default**: only `BUY`, `SELL` and `ERROR` ring the phone; everything
   else arrives silently — a phone that buzzes for every LLM thought is a phone that
   gets muted.
8. **Delivery** happens on a background queue task (never blocking the caller), with
   3 retries for critical notifications — `BUY`, `SELL`, `ERROR`, the same set that
   rings the phone — and 1 for the rest, chunked sends, and a 15 s per-chunk timeout;
   exhaustion is logged at `critical` level.
9. **Structured audit log**: gated by `NOTIFICATION_LOG_ENABLED`; every summary is
   appended as a compact JSON line to `data/notifications.jsonl` with a UTC timestamp
   injected when the caller supplied none (symbols collapsed to names, sentiment to a
   single compound number rounded to 2 dp, backtests to a one-line string such as
   `1D: 12 trades, 58% win (7W/5L)`), rotated at 512 KB with 10 backups under a
   class-level lock.

**Lifecycle.** `start()` is supervisor-safe: it detects an already-running updater
(restart idempotency), initializes and starts polling, starts the `Application` if it is
not already running, announces the trading mode, launches the notification-queue task,
and then parks on an `asyncio.Event().wait()` so the supervisor sees a live task instead
of a "completed" one — the detail that keeps restart logic from dead-locking on
"Updater is already running". `stop()` unwinds in the safe order: cancel the queue task,
`updater.stop()`, `app.stop()`, `app.shutdown()`. In `main.py` the bot is injected as the
engine's notifier and runs under a `TaskSupervisor` with `max_restarts=10` and a 10 s
restart delay.

---

## 10. Utilities & Cross-Cutting Infrastructure

Everything that is not a layer but a *rule* lives in `src/utils/` (8 modules, ~680 lines)
plus the two modules in `src/config/`. These are the pieces the rest of the codebase leans
on constantly, and they encode the project's operating assumptions: **fail closed around
money, degrade gracefully around data, put a timeout on everything, and never let
infrastructure break the trading loop.**

### 10.1 Event bus — `src/utils/event_bus.py` (70 lines)

The single seam between the engine's 35 components (§8.5). A flat
`dict[str, list[Callable]]` registry with four methods and no priorities, wildcards or
patterns:

- `subscribe(event_name, callback)` — append-only, registration order preserved.
- `publish(event_name, *args, **kwargs)` — **notification**. Awaits every coroutine
  subscriber in order and **swallows handler exceptions** (logged with traceback),
  re-raising only `CancelledError`. A component that crashes while *reacting* to an event
  must not abort the other subscribers and must not abort the loop.
- `request(event_name, *args, **kwargs)` — **command / query**. Returns the *first*
  subscriber's result and **propagates exceptions**. The docstring states the reason
  explicitly: a handler failure must not surface as `None`, because the caller would read
  `None` as a legitimate empty answer — an empty quote set, a missing market clock.
- `log_subscription_summary()` — dumps the whole registry at startup with `qualname`,
  `async`/`sync` and handler index, so the wiring of the machine is visible in the log
  before a single trade happens.

The publish/request split is the project's answer to a common async bug: using one
primitive for both "tell everyone" and "ask someone", with opposite error semantics.

### 10.2 Task supervisor — `src/utils/task_supervisor.py` (108 lines)

The Erlang supervisor pattern, applied to the process's two long-lived tasks
(`main.py`): `TradingEngine.run` and the Telegram bot's `start()`, both configured with
`max_restarts=10` and `restart_delay=10.0`.

- **Normal completion resets the restart counter to 0** — a loop that runs for a week and
  dies once does not spend a restart from a budget meant for crash loops.
- **Exponential backoff**: `min(restart_delay * 2**(restarts-1), max_backoff_delay)`,
  capped at 300 s by default.
- **Budget exhaustion is escalated, not swallowed**: `logger.critical`, `is_healthy =
  False`, a `WARNING` notification pushed through the notifier, a `cooling_off_period`
  (300 s) sleep, counter reset, `is_healthy = True`, and the task starts again. The
  process never gives up on its own trading loop; it slows down and tells the operator.
- `CancelledError` cancels the child and breaks out — shutdown is not a restart.
- `record_task_time(name, duration)` on every normal completion feeds §10.3.
- `get_health()` → `{name, running, is_healthy, restart_count, max_restarts,
  last_failure_time, last_exception}`; `main.py`'s `monitor_supervisor_health` polls it
  every 60 s and logs warnings for unhealthy or restarted tasks.

### 10.3 Health metrics — `src/utils/health_metrics.py` (55 lines)

A thread-safe singleton (`__new__` + double-checked lock) with four recorders:
`record_llm_call(model, success)`, `record_data_source(source, available)`,
`record_loop_latency(name, duration)`, `record_task_time(name, duration)`. Latency and
task lists are **rolling 100-sample windows** (oldest popped), so the averages reflect
*recent* behaviour rather than lifetime history.

| Recorder | Writers |
|---|---|
| `record_llm_call` | `llm/llm_client.py`, `llm/g4f_client.py`, and `post_decision_manager.py` — the latter records the synthetic `"provenance_gate"` model on every decision the provenance gate blocks (§8.5) |
| `record_data_source` | `exchanges/market_data.py` (`yfinance`, `borsa_italiana`, driven by their circuit breakers) and `exchanges/borsa_italiana_utils.py` |
| `record_loop_latency` | `background_task_manager.py` (symbol re-evaluation) |
| `record_task_time` | `task_supervisor.py` |

**Honest note:** `get_metrics()` currently has **no consumer in the repository**. The
metrics are recorded and averaged in-process, but nothing reads them — the dashboard's
health endpoint uses Redis plus live probes instead (§9.1) — and `trading/engine.py`
imports the module without using it. This is a real, small gap: an in-memory observability
buffer waiting for a reader.

### 10.4 The trading gate — `src/utils/pause_utils.py` (113 lines)

Pause state is a **key family**, not a key: `trading:paused`, `pause_source`,
`pause_start`, `pause_duration`, `pause_reason`, `llm_pause_time`.

- `set_trading_pause(...)` **deletes the whole family first**, then writes — stale fields
  from a previous pause can never leak into the new one.
- `clear_trading_pause_keys(...)` deletes the family. It is the single way to resume, used
  by `/resume` (§9.1), the Telegram `/resume` (§9.2) and the auto-resume logic (§8.5).
- Durations are always `setex`'d with a 7-day TTL; the remaining fields take an optional
  TTL.
- `pause_source` is what makes the human/LLM asymmetry work (§5.2): a pause written by a
  human is not overridable by the automatic resume path.

Plus an **in-process fail-safe that needs no Redis at all**:

```python
set_local_pause(reason)   # returns True only on the inactive -> active transition
clear_local_pause()
is_locally_paused()
get_local_pause_reason()
```

Guarded by its own lock, logged at `critical` ("new BUY decisions blocked until Redis
connectivity is restored"), and **one-shot by design**: returning `True` only on the
transition notifies the operator exactly once when the bot enters fail-safe, not once per
cycle. This is the fail-closed principle applied in both directions — when Redis dies the
bot stops buying, and when Redis dies the *pause mechanism itself* keeps working.

### 10.5 Symbol identity & BTP policy — `symbol_utils.py` (22) + `btp_policy.py` (166)

`symbol_utils` is the smallest module in the project and the one everything else trusts:
`^IT[A-Z0-9]{10}$` identifies a BTP ISIN (after stripping the `/QUOTE` suffix), and
`is_italian_isin` is the domestic-asset test.

`BTPPolicy` is the single place where **bond-specific trading rules** live, so no
component has to know what a BTP is:

- `is_btp(symbol)`; `supports_trailing_stop()` → **False for BTPs** (the broker does not
  support trailing stops on these instruments, so the engine must never try to place one).
- `get_max_take_profit_pct` / `get_max_stop_loss_pct` come from settings and return `None`
  for equities — stocks have no bond-shaped risk envelope.
- `get_hard_max_loss_pct` is a **maturity ladder** (`1h`, `1d`, `1w`, `1M`, `3M`,
  `6M/1Y/3Y/5Y`): how much loss is tolerated depends on how long the bond has to run.
- `compute_fees` → **zero** for primary issuance (BTPs bought at auction pay no brokerage),
  otherwise `max(gross * BTP_FEE_PERC, BTP_MIN_FEE)`.
- `get_slippage_pct` → a fixed 0.1 % for BTPs: bond books are thin, so slippage is modelled
  as a constant rather than as a function of volume.
- `compute_btp_metrics(symbol, price)` is real fixed-income math, not a heuristic: it pulls
  coupon and maturity from the DB (§5.1), YTM from `compute_btp_ytm`, then runs a
  semi-annual cash-flow present-value loop to produce **Macaulay duration**, **modified
  duration** `D/(1+y/2)` and **convexity**, so the LLM is given

  ```
  Δprice ≈ −D_mod · Δy + ½ · C · Δy²
  ```

  instead of being asked to guess how a bond behaves. Missing inputs return `None`, never a
  fabricated number.

### 10.6 Macro context — `src/utils/macro_data.py` (75 lines)

Gives the LLM the world the trade lives in: EUR/USD, the US 10-year yield, Brent, gold and
the FTSE MIB (`EURUSD=X`, `^TNX`, `BZ=F`, `GC=F`, `FTSEMIB.MI`), cached in Redis under
`macro:economic_context` with a 1-hour TTL.

Two details that matter:

- The request is **gated by the yfinance circuit breaker** and uses the shared
  Chrome-impersonating session (§6.2) — macro data never adds load to a source that is
  already failing.
- `^TNX` is **divided by 10**, because Yahoo quotes the US 10-year as a percentage index
  (5.00 = 5 %); without that division the prompt would receive a yield 10× too large.

Every per-ticker failure is logged at `debug` and the function returns `{}`: prompts are
built with or without macro context, never blocked by it.

### 10.7 Redis resilience — `src/utils/redis_client.py` (168 lines)

Detailed in §5.2; the design intent is worth restating because it is the most-copied
pattern in the codebase.

- The real client is a **lazily created, lock-guarded singleton** (`max_connections=50`,
  `socket_timeout=5`, `socket_connect_timeout=5`, `health_check_interval=30`,
  `retry_on_timeout=True`, `decode_responses=True`) — every component calls
  `get_redis_client()` and receives the same pool.
- When Redis is unavailable, callers get a **`DummyRedis`**: reads return safe defaults
  (`None`/`0`/`False`), writes log and return safe defaults (`True`/`1`) so callers
  proceed. Reads warn once and then every 50 calls; **writes log at `error` every time**,
  because a dropped write is real data loss. `__getattr__` stubs any command not explicitly
  listed, so a new Redis call cannot crash in degraded mode.
- `check_redis_connection()` flips the global availability flag; the transition to
  unavailable is logged at `critical`. One asymmetry worth knowing: the probe client is
  built *without* `health_check_interval`/`retry_on_timeout`, so the probe is strict while
  the production pool is forgiving.

### 10.8 Hot-reload machinery — `src/config/settings.py` (tail)

§4.1 describes the feature; the mechanism is the part a maintainer needs.

1. `reload()` = `load_dotenv(override=True)` → construct a **fresh** `Settings()` →
   `validate_llm_settings()`. **If validation fails, reload returns** — a broken `.env`
   can never half-apply to a running engine.
2. Fields are copied onto the live singleton with `setattr`, skipping **`unsafe_fields`**
   (`DATABASE_BACKEND`, `DATABASE_PATH`, `REDIS_HOST/PORT/DB/TLS`, `WEB_HOST`, `WEB_PORT`,
   the three executor worker counts). Connection topology genuinely needs a restart;
   pretending otherwise would leave the process holding a pool built from the old host.
3. LLM fields are detected **by prefix** (`LLM_`, `OLLAMA_`, `OPENAI_`, `G4F_`) minus the
   four temperature fields. Temperature is deliberately excluded from cache invalidation:
   it changes the *call*, not the *prompt*, so a temperature tweak must not invalidate
   every cached response.
4. If any LLM field changed, `LLM_CACHE_VERSION` is regenerated (`uuid4`). The cache key in
   `llm/cache.py` embeds it (§7.2), so a provider/model change instantly invalidates every
   cached LLM response **without deleting a single key**.
5. A `PAPER_INITIAL_BALANCE` change sets `PAPER_BALANCE_CHANGED`, consumed and reset by the
   orchestrator (§8.5) to trigger a paper reset.
6. Registered reload callbacks are invoked with exceptions swallowed, then
   `_invalidate_primary_model_cache()` runs so market-hours changes take effect immediately
   rather than after the 30 s clock TTL (§8.5).

The only registered consumer today is the engine (`engine.py:166`), which refreshes its own
cached settings. `/reload` (§9.1) is the human trigger.

### 10.9 LLM-decided runtime config — `src/config/config_service.py` (61 lines)

§4.2 covers the purpose. Mechanically it is a 60-second in-memory cache over Redis keys
prefixed `trading:`, with `asyncio.to_thread` around every Redis call (the engine is async;
the Redis client is not), bytes-decoding, default-on-failure, `set_llm_config` writing with
a 7-day TTL, and both write paths invalidating the local cache. It is the only sanctioned
way for the LLM to change its own operating parameters at runtime — which is why it is a
*service* with a cache and an invalidation contract rather than a bare `redis.get`.

### 10.10 Cross-cutting conventions

Patterns repeated across every module — the project's house style:

| Convention | Where |
|---|---|
| **Every network call has a timeout** | 5 s Redis sockets, 15 s HTTP, 15 s yfinance, 45 s quote batches, 5 s Telegram Redis probes, 10 s market checks, 30 s paper resets — no unbounded awaits |
| **Fail closed for money, fail open for data** | LLM activity gate, provenance gate, pause logic and the validator refuse unsafe decisions (§7.2, §8.2, §8.5); quotes, news, sentiment and macro data degrade silently to `None`/`{}` |
| **Deliberate exceptions to fail-open** | the web rate limiter fails **open** when Redis is lost (§9.1) — locking the operator out of the dashboard during an outage is worse than losing rate limiting |
| **Circuit breakers per external source** | yfinance (§6.2), Borsa Italiana (§6.2), market-data sources (§6.1), the LLM primary chain (§7.2) |
| **Blocking work never runs on the event loop** | `asyncio.to_thread` for Redis and SQLite, three dedicated thread pools (§8.4), a 100-thread default executor (§4.3) |
| **Concurrency is designed, not discovered** | `SharedState` publishes its lock ordering in its own docstring (§8.5); Redis rate windows use sorted sets with `uuid4` members so concurrent requests cannot collide (§9.1) |
| **Cache everything with a TTL, invalidate explicitly on change** | 30 s market clock, 60 s config service, 1 h macro context, 5 s WebSocket payload, cache-version inside LLM keys |
| **A logging handler must never break the application** | `RedisLogHandler` bounds its queue, drops on overflow, never raises (§4.3) |
| **Degraded mode is a supported mode** | `DummyRedis`, `is_redis_available`, `redis_available` surfaced on the dashboard, the in-process fail-safe pause |
| **Tell the operator, at the right level** | `critical` for lost Redis, exhausted supervisors and fail-safe pause; `warning` for unhealthy supervised tasks; `debug` for one missing macro ticker |

---

## 11. Testing & Quality Gates

The project carries a **~3 600-line pytest suite — 28 test files, 276 test functions** —
whose shape mirrors its values: the safety spine (fail-closed paths, the provenance gate)
gets dedicated test files, while pure domain math (fees, YTM, candle hygiene, cache-key
normalization) gets exhaustive edge-case treatment. The suite is not an afterthought —
**it runs inside the Docker image build** (§4.4): a red test means no deployable image.

### 11.1 The harness — `pytest.ini`, `tests/conftest.py`, `requirements-dev.txt`

- **`pytest.ini`** — `testpaths = tests`, standard discovery (`test_*.py` / `Test*` /
  `test_*`), and **`asyncio_mode = auto`**: every `async def test_` runs on a real event
  loop with no decorators or markers, which is why engine-component tests read like
  synchronous ones.
- **`requirements-dev.txt`** — `pytest>=7`, `pytest-asyncio>=0.21`, `pytest-cov>=4`
  (coverage is installed but no threshold is enforced).
- **`tests/conftest.py`** — one **autouse** fixture, `mock_external_services`, that is
  the suite's hermeticity contract: it patches
  `src.utils.redis_client.get_redis_client`, `src.utils.redis_client.is_redis_available`
  (→ `True`), and `src.database.get_connection` with `MagicMock`s. **No test ever
  touches a real Redis or a real database**; infrastructure is mocked at the lowest seam
  so everything above it can be exercised purely in-process. A test that genuinely needs
  SQL semantics must opt out explicitly — which none currently does.

### 11.2 The six test families

**Family 1 — Fail-closed invariants (the crown jewels).**

- `test_drawdown_breaker_fail_closed.py` — the drawdown circuit breaker under hostile
  conditions: Redis unavailable ⇒ breaker trips to pause; breaker exceptions ⇒ pause;
  healthy Redis + no drawdown ⇒ no pause; the BUY executor is blocked by the in-process
  local pause (§10.4); and the fail-safe notification fires **exactly once** on
  activation, not once per cycle.
- `test_pause_resume_fail_closed.py` — an LLM failure while paused **stays paused**
  (never resumes on error), while a genuine LLM resume decision still works.
- `test_llm_provenance_gate.py` — the most elaborate harness in the suite: it builds a
  real `PostDecisionManager` via `__new__` with a mocked engine/event bus and minimal
  `DecisionContext`s, then pins every rule of the gate (§8.5): a reviewed BUY carries
  real provenance; unreviewed SELL→HOLD conversions are blocked (both the direct and the
  `process_entry` path); risk-manager circuit-breaker SELLs are exempt; fallback-provider
  and default-hold-model provenance is rejected; a fallback HOLD passes through *without
  promotion*; a BUY without Step-2 review is blocked; a Step-2 failure path is never
  marked reviewed; and the risk-manager helper tags its origin correctly.

**Family 2 — The money path: paper trader, fees, execution.**

- `test_paper_trader.py` — market BUY/SELL, stop-loss trigger, **trailing-stop
  trigger**, order cancellation, volume-capped buys, and dynamic slippage applied.
- `test_backtester.py` (largest file) — backtest lifecycle (empty candles → error,
  basic run, stop-loss hit, take-profit hit, max-hold exit, missing entry config) plus
  Intesa fee computation (BUY/SELL/small-trade minimum/BTP) and dynamic slippage with
  and without volume.
- `test_fees.py` + `test_btp_policy.py` — the exact fee shapes: stock buy costs, BTP buy
  costs, and **zero fees for BTP primary issuance**.
- `test_buy_executor.py` — confidence-weighted position sizing.
- `test_reinvestment_llm_path.py` — dividend reinvestment is **routed through LLM
  review**; an unreviewed reinvestment BUY never executes; a Step-2 exception fails safe
  with no execution; a just-closed position is not resurrected; the reinvestment signal
  carries dividend metadata; and the BUY executor uses the fixed reinvestment value.

**Family 3 — The decision pipeline.**

- `test_llm_parser.py` — JSON parsing (raw, markdown-fenced, arrays), invalid action
  defaulting to HOLD, first-JSON extraction from noisy text, reasoning-quality scoring,
  semantic rejection of a bad stop-loss.
- `test_signal_pipeline.py` / `test_signal_from_dict.py` / `test_validator.py` — the
  parse→validate chain end to end; `Signal.from_dict` defaults and invalid-action
  mapping (with logged warning); valid-BUY validation.

**Family 4 — LLM infrastructure.**

- `test_decision_cache.py` (26 tests) — snapshot-hash determinism and sensitivity;
  store/get/invalidate of cached decisions; **provenance preserved through the cache**;
  Redis errors never propagate; cache hit skips the LLM call, miss calls it, disabled
  cache calls it, Redis-unavailable still reaches the LLM; the execute-signal hook
  invalidates the cache; and the metric instrumentation: every outcome (hit with saved
  tokens, cold/changed/rebuild-failed misses) is recorded, a failed metric write never
  breaks the decision path, and the SQL record/summary/cleanup helpers are covered.
- `test_cache.py` — token estimation; `compute_market_hash` ignores volatile fields;
  text normalization for cache keys (float rounding, scientific notation, integers,
  empty input); **fee fingerprint consistency** (a fee change must change the key).
- `test_llm_integration.py` — cache miss→set, cache hit, fallback on primary failure,
  and all-providers-fail.
- `test_prompt_prefix_stability.py` — the §3.7 principle as a test: the system prompt
  is byte-stable across volatile Redis content; past-mistakes content stays out of the
  prefix; final-decision message lists keep a stable prefix while preserving the
  information the prompt needs.
- `test_prompt_utils.py` — `round_floats` shapes and `timeframe_to_seconds` across
  minutes → years plus invalid input.

**Family 5 — Market data & domain math.**

- `test_candle_utils.py` — validation/cleaning (non-positive open/close, negative
  volume, high/low violations, duplicate timestamps keep-last, empty and short inputs,
  sorted output) and 6M/1Y aggregation.
- `test_indicators.py` — ATR/RSI/EMA sanity checks.
- `test_database.py` — symbol normalization variants and `compute_btp_ytm` edge cases
  (None inputs, zero/negative price, past maturity, **at / below / above par**, invalid
  maturity format).
- `test_symbol_utils.py` — the BTP ISIN regex against suffix/quote-suffix variants and
  regular-stock rejection.
- `test_settings.py` — `parse_temperature_range` against every shape it can meet: none,
  empty, single value (incl. zero and max), range, spaced range, reversed range,
  out-of-bounds high/low, and malformed formats.

**Family 6 — Concurrency & the web surface.**

- `test_concurrency.py` — `SharedState` under concurrent position updates, cycle-budget
  spending, queued-order writes and trade-list appends; `EventBus` under concurrent
  subscribe and publish (§10.1).
- `test_exit_order_manager.py` — exit-price computation (stop variant and fallback).
- `test_position_manager.py` — portfolio exposure summary computation.
- `test_api.py` — a FastAPI `TestClient` over the real app with **auth and CSRF
  bypassed via dependency overrides** and a fully mocked engine (plus mocked Redis
  views): `/health`, `/status`, `/config`, pause/resume, and force re-evaluation.

### 11.3 What the suite chooses to test — and honest gaps

The suite optimizes for **invariant density, not line coverage**: every fail-closed rule
named in §2.3 has a test that tries to break it, and every pure function a human could
get subtly wrong (YTM near par, fee minimums, temperature parsing) is tested at its
edges. Known gaps, stated honestly:

- **No coverage threshold** — `pytest-cov` is installed but unconfigured, so coverage
  drifts invisibly.
- **The API test's engine is a `MagicMock`** — endpoint wiring and auth/CSRF bypassing
  are verified, but not real handler behaviour against a real engine.
- **No migration test** — `_migrate_db`'s atomic multi-column migration (§5.1) is
  exercised only in production; conftest's global SQL mock means no test round-trips a
  real schema.
- **Untested seams**: news fetching, Borsa Italiana scraping, asset discovery,
  re-evaluation pipeline, Telegram bot, and WebSocket streaming all lack dedicated
  tests — they are covered only indirectly, if at all.


## 12. The Whole Machine — One Pass Through It

The preceding sections describe every subsystem in isolation. This closing chapter is
the synthesis: the same machine, seen end-to-end, in the order the code actually
executes it — plus the one-paragraph answer to "so what *is* this project?".

### 12.1 Boot — `src/main.py`

`main()` builds the runtime in a deliberate order, and every step is a gate:

1. The default asyncio executor is set to 100 workers — the safety margin under the
   dedicated pools and semaphores of §8.4.
2. `init_db()` creates/verifies the schema (§5.1); on a brand-new database it also
   flushes the stale Redis key prefixes so no cache survives its data.
3. `_sync_blacklist_from_db()` restores the persistent LLM model blacklist (§7.2)
   before anything can call a model.
4. `_validate_startup_settings()` is the fail-closed boot gate: Redis unreachable ⇒
   exit; `validate_llm_settings()` (§4.1) rejects a half-migrated fallback
   configuration ⇒ exit; a throwaway `save_trading_state`/`load_trading_state`
   round-trip proves the database is writable ⇒ otherwise exit. A process that
   cannot cache, decide, or persist refuses to start.
5. Housekeeping: pre-create the yfinance cache directory (race avoidance), seed the
   Telegram chat id from env, RSS connectivity test.
6. `TradingEngine()` is constructed — the composition root of §8.4 wires all 35
   components — and handed to the web app via `set_engine` before the server starts,
   so `/health` can answer 503 (engine attached but not ready) rather than crash.
7. Uvicorn starts **first** as an asyncio task: the dashboard is reachable (serving
   `{"status":"initializing"}` on the WebSocket, §9.1) before any trading happens.
8. The Telegram bot object is created and injected as the engine's notifier (§9.2).
9. `engine.run` starts under a `TaskSupervisor` (max 10 restarts, 10 s delay) as a
   background task; the Telegram bot's polling gets its own supervisor; a health
   monitor loop warns every 60 s about any supervisor that has restarted.
10. SIGINT/SIGTERM handlers set a shutdown event; the final await chain stops things
    in reverse order — engine, engine supervisor, Telegram bot (+ its own cleanup),
    uvicorn (`should_exit`), then `close_pool()`.

### 12.2 Life of an evaluation cycle

One pass of the engine's main loop (60 s interval, §4.1), as the §4–§10 subsystems
compose into a single decision:

1. **Who is due?** `evaluation_scheduler` (§8.5) picks symbols whose per-timeframe
   interval has elapsed, shaped by market phase (halved cadence in open/close
   windows, doubled in a quiet market, sentiment shifts pull it in, LLM-set
   `strategy_interval` overrides), and answers **"nobody"** when the XMIL session is
   inactive — the fail-closed gate that stops trading *and* re-evaluation with the
   market.
2. **Gather the truth.** For each due symbol, `signal_processor` assembles the
   decision context: quotes through the chain of responsibility (§6.1 — Redis →
   database → Borsa Italiana → yfinance → keyed fallbacks, each circuit-broken),
   multi-timeframe candles merged with Borsa precedence and validated (§6.2), TA-Lib
   indicators (§6.3), news summaries and sentiment (§6.4), dividends, analyst and
   options data, macro context (§10.6). Staleness is made *visible* (⚠️ in the
   prompt) and entry-blocking thresholds scale with the timeframe.
3. **How hard is this question?** `model_tier_manager` computes prompt complexity
   from volatility, indicator state, candidate count and portfolio risk, and picks
   the tier: strong *mind*, fast *actuator*, or *weak* summarizer — with the
   effective temperature and reasoning effort to match (§4.1).
4. **Ask, cheaply.** Every token is fought for before the call: the decision cache
   (§8.5) short-circuits identical snapshots; prefix-stable prompts (§3.7) keep
   provider prompt caches warm; compaction and chunk/summarize bound the payload
   (§7.2–7.3); token-budget semaphores prevent a burst from stampeding the provider.
5. **Decide — twice.** Step 1a (`llm_step_manager`) produces analysis and a candidate
   signal, with retry/correction and an explicit fallback-HOLD on failure; proposed
   parameters go to `backtest_manager`, which runs the variants in parallel (§8.3)
   and then makes the **Step-2 review call to the strong model** — the only place in
   the system where `step2_reviewed=True` is ever set.
6. **Gate.** `post_decision_manager` runs the provenance gate (§8.5): a BUY/SELL
   without real provider/model provenance and Step-2 review is forced to HOLD and
   alerted. Then trade filters, sector concentration, and risk validation (§8.2).
7. **Execute.** `buy_executor` sizes the position by confidence and risk, applies
   the Intesa fee model (§6.2), computes limit prices and SL/TP, and dispatches
   through `order_executor` into the `paper_trader` (§8.6) — volume caps, dynamic
   slippage, native order semantics. Sells flow from `risk_manager`'s SL/TP
   monitoring loop through `sell_executor` and `exit_order_manager`.
8. **Remember everything.** Fills land in `trade_history`, decisions in `signals`,
   metrics in `llm_metrics`, outcomes in `llm_decision_quality` (§5.1); state is
   snapshotted periodically and on shutdown (§8.5). Every notification rides the
   `EventBus` to the Telegram pipeline (§9.2) and the dashboard mirror.
9. **Learn.** In the background, the loops of `background_task_manager` refresh
   data, compute indicators, fetch news, track dividends, score decision outcomes;
   every 6 h the self-analysis loop asks the weak model to critique the last 20
   wrong decisions and feeds the result back into future prompts (§8.5) — the only
   feedback loop in the system that closes.

### 12.3 What TradeLedger ultimately is

A self-hosted trading research machine for the Italian market in which **the LLM is
the strategist and the code is the constitution**. Everything above the LLM exists to
give it the best possible picture of the market; everything below it exists to make
sure that only reviewed, provenanced, risk-bounded decisions can move money — and
that even a simulated portfolio is treated with the accounting, fees, and audit trail
of a real one. It is deliberately narrow (Italian stocks, UCITS ETFs, BTPs), honestly
simulated (paper + notify, never live), and engineered to fail closed: on any
uncertainty — in its data, its cache, its models, or itself — it does less, not more.
That is the vision this codebase implements, line by line.
