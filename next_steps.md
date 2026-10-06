# Next Steps

Handoff notes for the next TradeLedger session. Completed work is committed to `main`,
synced into `project_vision.md` (§2.5, §5.1, §8, §9.1, §11) and recorded in Chroma
project memory (`aiderdesk_tradeledger_memory`, append-only).

## Pre-existing issues found during verification (documented, not fixed)

The SQLite backend is dead code in this deployment (production runs PostgreSQL) and is
broken at HEAD, independent of the decision-cache work:

1. `_SqliteConnectionWrapper.execute(sql, None)` forwards `params=None` straight to
   `sqlite3` → `ProgrammingError: parameters are of unsupported type` for any
   parameterless call.
2. `_get_init_statements()` includes Postgres-only `CREATE INDEX ... INCLUDE (...)`
   statements (src/database.py, ~lines 445/448) → `init_db()` fails on SQLite with
   `OperationalError: near "INCLUDE": syntax error`.

Fix shape when it ever matters: translate `params=None → ()` in the wrapper and move the
INCLUDE indexes into a Postgres-only init branch. Priority: low.

Environment note: the production DB host (192.168.1.13:5432) refuses connections from
this machine, so schema-level verification must use a throwaway PostgreSQL container
(recipe below). Also note `src.config.settings` runs `load_dotenv(override=True)` at
import, which clobbers pre-set `DB_*` env vars — tests/verify scripts must `setattr` on
the settings object before importing `src.database` (the pool is built at that import).

## Candidate next improvements (priority order per mandate)

1. **Cache-TTL policy from measured data.** `decision_cache_metrics` now measures hit
   rate and churn per symbol; the natural follow-up is adapting `LLM_DECISION_CACHE_TTL_SECONDS`
   per symbol (shorten for churny snapshots, lengthen for stable ones). This changes
   decision-path behaviour — requires a backtest/simulation comparison before touching
   the default.
2. **Slippage report (mandate priority 3).** Paper fills apply dynamic slippage but
   there is no report joining `trade_history` fills to quote timestamps to quantify
   execution quality per order. A read-only report + dashboard chart is low-risk and
   makes execution quality measurable.
3. **Drawdown circuit breaker tuning study.** Stress-check parameters have never been
   reviewed against recorded drawdown data; candidate for a simulation study (mandate
   priority 4), not a blind parameter change.

## Verification environment recipe (repeatable)

```bash
docker run -d --name tl-verify-pg -e POSTGRES_DB=tl -e POSTGRES_PASSWORD=test \
  -p 55432:5432 postgres:15-alpine
# in the repo venv, before importing src.database:
#   settings.DB_HOST='127.0.0.1'; settings.DB_PORT=55432; settings.DB_NAME='tl';
#   settings.DB_USER='postgres'; settings.DB_PASSWORD='test'
docker rm -f tl-verify-pg   # when done
```
