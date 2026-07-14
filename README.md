# TradeLedger

An advanced, AI-powered trading bot focused on **medium to long-term investment horizons**. It uses an external LLM (Ollama or OpenAI) to dynamically select Italian stocks, ETFs, and government bonds (BTPs), generate trading strategies, and manage risk. It uses **yfinance** and custom scrapers for market data and supports two operation modes: **Paper Trader** (simulated) and **Notifier** (signal-only with manual trade tracking via web UI).

## Key Features

- **LLM-driven decisions**: It uses external LLMs (Ollama or OpenAI-compatible) to dynamically select Italian stocks, ETFs, and government bonds (BTPs), generate trading strategies, and manage risk. Supports per-role model configuration (mind, actuator, weak) with automatic fallback to secondary providers. It dynamically adjusts LLM model tier, prompt complexity, and temperature based on market volatility and context.
- **Multi-Asset Support**: Trades Italian stocks (Borsa Italiana / Euronext Milan), UCITS ETFs, and BTPs (Italian government bonds). Data for BTPs is fetched from Borsa Italiana.
- **Advanced Technical Analysis**: Computes a wide array of indicators (RSI, MACD, Bollinger Bands, Ichimoku, Parabolic SAR, Keltner Channels, VWAP, ADX, OBV, MFI, CCI, Williams %R, Donchian Channels, Stochastic, ATR, EMA) to feed context to the LLM.
- **Market Regime Classification**: Automatically classifies market conditions (trending, ranging, volatile) to inform strategy decisions.
- **Backtesting Engine**: Simulates strategies on historical OHLCV data, supporting trailing stops, partial take-profit levels, and breakeven logic to validate LLM signals before execution.
- **News & Sentiment Analysis**: Fetches and analyzes financial news from multiple sources (NewsAPI, Twitter, Reddit, RSS, Google News, StockTwits, YouTube), caching sentiment scores to provide additional context to the LLM.
- **Risk Management**: Automatic stop-loss, take-profit, and trailing stops for every position. Includes hard maximum loss limits, daily loss circuit breakers, portfolio drawdown protection, and pause/resume logic based on consecutive losses and market conditions. Supports periodic portfolio rebalancing.
- **Two operation modes**:
  - `paper`: Custom paper trading simulator with configurable fees, SQLite persistence, and silent Telegram notifications.
  - `notify`: Signal-only mode that sends audible Telegram alerts for BUY/SELL signals and allows manual trade logging via the web dashboard.
- **Market hours**: Uses `pandas_market_calendars` to determine when Euronext Milan (XMIL) is open, with configurable pre-market and post-market active periods.
- **Telegram bot**: Receive trade notifications (audible for signals, silent for routine updates) and control the bot with commands.
- **Web dashboard**: Real-time view of positions, balances, trades, and profit. Includes manual trade entry form in notify mode, OHLCV charting, LLM metrics dashboard, and simulation lab for backtesting and decision testing. The dashboard is a Progressive Web App (PWA) with offline support, installable on mobile devices, and includes CSRF protection, session-based authentication, and rate limiting.
- **Database-backed indicators**: Technical indicators are pre-computed in background jobs after each candle download and stored in a dedicated database table. This decouples indicator computation from symbol re-evaluation, making the re-evaluation loop faster and non-blocking.
- **Redis caching**: LLM responses and market data are cached to reduce API calls. Logs are also pushed to Redis for the web dashboard.
- **Dockerized**: Ready to run with Docker Compose (includes Redis).

## How It Works

The bot operates through a series of asynchronous background loops managed by the `TradingEngine`:

1. **Data Collection**: Periodically downloads OHLCV data for all tradable assets and fills historical gaps. After each symbol's candles are downloaded, technical indicators (RSI, MACD, Bollinger Bands, ADX, ATR, etc.) are computed in a background job and stored in the database. News articles are fetched and analyzed for sentiment.
2. **Symbol Reevaluation**: The LLM evaluates the current portfolio and available assets, selecting the optimal mix of symbols to track based on performance, market breadth, sentiment, and pre-computed indicators fetched from the database.
3. **Signal Generation**: For each tracked symbol, the bot fetches pre-computed indicators from the database, classifies the market regime, and builds a complex prompt. The LLM generates a preliminary trading decision (BUY/SELL/HOLD) with specific parameters (stop-loss, take-profit, hold time).
4. **Backtesting & Validation**: The preliminary decision is backtested against historical data. The `Signal` is validated to ensure risk-reward ratios and stop-loss distances are sane.
5. **Final Decision**: A final LLM call is made with the backtest results and preliminary decision to confirm the trade.
6. **Execution**: If confirmed, the bot executes the trade (via paper trader or notification) and places exit orders. It continuously monitors positions for risk management, partial take-profits, and trailing stops.

## Supported Assets

- **Stocks**: Italian stocks listed on Borsa Italiana (suffix `.MI`). Discovered via Wikipedia, FinanceDatabase, and static lists.
- **ETFs**: Italian UCITS ETFs.
- **BTPs**: Italian government bonds (BTPs) discovered from Borsa Italiana. Candle data is fetched from Borsa Italiana.

## Technical Indicators

The bot uses the following indicators to build LLM prompts:
- **Trend**: EMA, MACD, ADX, Ichimoku Cloud, Parabolic SAR, VWAP, Pivot Points.
- **Momentum**: RSI, Stochastic, MFI, CCI, Williams %R, OBV.
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian Channels.

## News & Sentiment

The bot can fetch and analyze news from the following sources (if API keys are provided):
- NewsAPI
- Twitter (via Tweepy)
- Reddit (via PRAW)
- RSS Feeds (configurable list)
- Google News RSS
- StockTwits
- YouTube Data API v3
- Facebook Graph API
- Banca d'Italia BTP News

Sentiment is analyzed using an LLM, and results are cached in the database to provide context to the LLM during strategy generation.

## Libraries & Technologies

- **Language**: Python 3.11+
- **Web Framework**: FastAPI, Uvicorn, WebSockets
- **Database**: SQLite (via `sqlite3`) or PostgreSQL (via `psycopg`)
- **Caching/Queue**: Redis
- **Market Data**: `yfinance`, `pandas_market_calendars`, `financedatabase`, `beautifulsoup4` (for scraping Borsa Italiana and Banca d'Italia)
- **LLM**: `httpx` (for OpenAI-compatible API and Ollama API)
- **Telegram**: `python-telegram-bot`
- **Data Validation**: `pydantic`, `pydantic-settings`
- **Data Processing**: `pandas`, `numpy`, `TA-Lib`

## Configuration

Copy `.env.example` to `.env` and fill in your settings. Here are the key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TRADING_MODE` | `paper` (simulated) or `notify` (signal-only) | `paper` |
| `TICKER_SUFFIX` | Yahoo Finance suffix for Italian stocks | `.MI` |
| `TARGET_COUNTRY` | Country filter for stock discovery | `italy` |
| `BASE_CURRENCY` | Quote currency | `EUR` |
| `PAPER_INITIAL_BALANCE` | Initial balance for paper trading | `10000.0` |
| `MAX_SYMBOLS` | Maximum number of stocks to trade simultaneously | `10` |
| `LLM_PROVIDER` | `ollama` or `openai` | `ollama` |
| `LLM_TIMEOUT` | LLM request timeout in seconds | `60` |
| `LLM_TEMPERATURE` | Base LLM temperature (0.0–2.0) | `0.1` |
| `LLM_MIND_TEMPERATURE` | Temperature range for mind model (e.g. "0.2-0.5") | |
| `LLM_ACTUATOR_TEMPERATURE` | Temperature range for actuator model (e.g. "0.0-0.1") | |
| `LLM_WEAK_PROVIDER` | Provider for weak model (empty = use global) | |
| `LLM_WEAK_TEMPERATURE` | Temperature range for weak model (e.g. "0.1") | |
| `LLM_MIND_MODEL` | Model name for mind role (e.g., `gpt-4o`) | |
| `LLM_ACTUATOR_MODEL` | Model name for actuator role (e.g., `gpt-4o-mini`) | |
| `LLM_WEAK_MODEL` | Model name for weak role | |
| `LLM_FALLBACK_ENABLED` | Enable automatic fallback to secondary provider | `true` |
| `LLM_MIND_PROVIDER` | Provider for mind model (empty = use global) | |
| `LLM_ACTUATOR_PROVIDER` | Provider for actuator model (empty = use global) | |
| `MAX_DAILY_LOSS_PCT` | Maximum daily loss as a fraction of initial balance (0.05 = 5%) | `0.05` |
| `HARD_MAX_LOSS_PCT` | Hard maximum unrealized loss percentage that forces exit (0.15 = 15%) | `0.15` |
| `BTP_MAX_TAKE_PROFIT_PCT` | Maximum take-profit percentage for BTPs (0.03 = 3%) | `0.03` |
| `BTP_MAX_STOP_LOSS_PCT` | Maximum stop-loss percentage for BTPs (0.03 = 3%) | `0.03` |
| `BTP_HARD_MAX_LOSS_PCT` | Hard maximum unrealized loss percentage for BTPs (0.05 = 5%) | `0.05` |
| `PORTFOLIO_REBALANCE_ENABLED` | Enable periodic portfolio rebalancing | `true` |
| `PORTFOLIO_REBALANCE_INTERVAL_SECONDS` | Interval for portfolio rebalance | `7776000` |
| `MAX_POSITION_AGE_MULTIPLIER` | Maximum position age as a multiple of the original max_hold_time_seconds | `2.0` |
| `PORTFOLIO_COOLDOWN_MAX_CONSEC_LOSSES` | Maximum consecutive losses before portfolio cooldown | `5` |
| `PORTFOLIO_COOLDOWN_SECONDS` | Portfolio cooldown duration in seconds | `3600` |
| `REINVEST_DIVIDENDS` | Enable dividend reinvestment | `false` |
| `NEWS_ENABLED` | Enable news fetching and sentiment | `false` |
| `NEWS_API_KEY` | NewsAPI.org API key | |
| `TWITTER_BEARER_TOKEN` | Twitter API v2 bearer token | |
| `REDDIT_CLIENT_ID` | Reddit API client ID | |
| `REDDIT_CLIENT_SECRET` | Reddit API client secret | |
| `RSS_FEEDS` | List of RSS feed URLs | `[]` |
| `YOUTUBE_API_KEY` | YouTube Data API v3 key | |
| `YAHOO_FINANCE_ENABLED` | Enable Yahoo Finance fallback for quotes | `true` |
| `ALPHAVANTAGE_ENABLED` | Enable Alpha Vantage for quotes/candles | `false` |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage API key | |
| `IEX_ENABLED` | Enable IEX Cloud for quotes/candles | `false` |
| `IEX_API_KEY` | IEX Cloud API key | |
| `BANCA_D_ITALIA_BTP_NEWS_ENABLED` | Enable Banca d'Italia BTP news scraping | `false` |
| `FACEBOOK_PAGE_ACCESS_TOKEN` | Facebook Graph API token | |
| `FACEBOOK_PAGE_ID` | Facebook Page ID | |
| `WEB_USERNAME` | Web dashboard username (optional) | |
| `WEB_PASSWORD` | Web dashboard password (optional) | |
| `WEB_RATE_LIMIT_REQUESTS` | Web rate limit requests per window | `300` |
| `WEB_RATE_LIMIT_WINDOW` | Web rate limit window in seconds | `60` |
| `BTP_FEE_PERC` | Fee percentage for BTP trades | `0.0024` |
| `BTP_MIN_FEE` | Minimum fee for BTP trades | `3.50` |
| `BTP_IS_PRIMARY_ISSUANCE` | Whether BTPs are primary issuance (zero fees) | `false` |
| `STOCK_FEE_PERC` | Fee percentage for stock trades | `0.0024` |
| `STOCK_FEE_MIN` | Minimum fee for stock trades | `3.50` |
| `STOCK_FEE_FIXED` | Fixed fee for stock trades | `2.50` |
| `TOBIN_TAX_RATE` | Tobin tax rate for stock buys | `0.0012` |
| `REDIS_HOST` | Redis host | `redis` |
| `REDIS_PORT` | Redis port | `6379` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (optional) | |
| `TELEGRAM_CHAT_ID` | Default chat ID (set via /start) | |
| `WEB_HOST` | Web server host | `0.0.0.0` |
| `WEB_PORT` | Web server port | `8083` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `NOTIFICATION_VERBOSITY` | Notification verbosity: all, errors_only, trades_only, none | `all` |
| `OHLCV_TIMEFRAMES` | List of timeframes to fetch | `["5Y", "3Y", "1Y", "6M", "3M", "1M", "1w", "1d", "1h"]` |
| `OHLCV_RETENTION_DAYS` | Days of historical OHLCV data to keep | `3650` |
| `MARKET_DATA_REFRESH_SECONDS` | Interval for tracked symbol OHLCV download | `900` |
| `FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS` | Interval for all-asset OHLCV download | `21600` |
| `DB_HOST` | PostgreSQL host (empty = use SQLite) | |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | PostgreSQL database name | `trade_ledger` |
| `DB_USER` | PostgreSQL user | `trade_ledger` |
| `DB_PASSWORD` | PostgreSQL password | |

## Quick Start (Docker)

1. Ensure Docker and Docker Compose are installed.
2. Clone the repository and navigate to its directory.
3. Copy `.env.example` to `.env` and edit as needed. To use PostgreSQL instead of SQLite, fill in the `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD` variables.
4. Run:
   ```bash
   docker-compose up -d
   ```
   This starts Redis, PostgreSQL (optional), and the bot.
5. Access the web dashboard at `http://localhost:8083`.
6. If Telegram is configured, send `/start` to your bot to begin receiving notifications.

## Running Locally

1. Install Python 3.11+, Redis, and (optionally) PostgreSQL.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **Note:** TA-Lib C library must be installed separately. See [TA-Lib installation instructions](https://github.com/TA-Lib/ta-lib-python#installation).
3. Copy `.env.example` to `.env` and configure. To use PostgreSQL, set the `DB_*` variables; otherwise SQLite is used by default.
4. Start Redis (e.g., `redis-server`).
5. Run the bot:
   ```bash
   python -m src.main
   ```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Register chat for notifications |
| `/menu` | Show the interactive keyboard menu |
| `/pause` | Pause trading |
| `/resume` | Resume trading |
| `/status` | Show current symbols, positions, and balances |
| `/trades` | Show open trades and queued orders |
| `/profit` | Show profit/loss summary |
| `/performance` | Show performance by symbol |
| `/risk` | Show risk metrics |
| `/market` | Show market status |
| `/news` | Show news summaries for tracked symbols |
| `/news_status` | Show news article counts for tracked symbols |
| `/sell` | Sell all positions or a specific one by ID |
| `/reset` | Reset paper trading state |
| `/backfill` | Force backfill of all discovered symbols |
| `/signals` | Show latest LLM signals |

## Web API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check (includes Redis status) |
| `GET /api/status` | Current symbols, positions, balances, paused state |
| `GET /api/trades` | Open trades |
| `GET /api/profit` | Profit/loss summary |
| `GET /api/performance` | Performance by symbol |
| `GET /api/risk` | Risk metrics |
| `GET /api/history` | Closed trades |
| `GET /api/config` | Current configuration |
| `POST /api/pause` | Pause trading |
| `POST /api/resume` | Resume trading |
| `POST /api/sell` | Sell all or specific position |
| `POST /api/manual-trade` | Log a manual trade (notify mode only) |
| `GET /api/manual-trades` | List manual trades |
| `GET /api/ohlcv/{symbol}` | OHLCV candles for charting |
| `POST /api/force-reeval` | Force immediate symbol re-evaluation |
| `POST /api/force-download` | Force download of all asset OHLCV data |
| `POST /api/restart` | Restart the application |
| `POST /api/simulate/backtest/{symbol}` | Simulate a backtest for a symbol |
| `POST /api/simulate/decision/{symbol}` | Simulate an LLM decision for a symbol |
| `WS /ws` | Real-time dashboard data |
| `POST /api/force-backfill` | Force backfill of all discovered symbols |
| `POST /api/reload` | Reload settings from `.env` |
| `GET /api/discovered-symbols` | Return all discovered symbols for autocomplete |
| `GET /api/ticker/{symbol}` | Return quote for a single symbol |
| `GET /api/tickers` | Return quotes for a comma-separated list of symbols |
| `GET /api/llm-metrics` | Return aggregated LLM metrics for the dashboard |
| `POST /api/llm-metrics/reset` | Wipe all LLM metrics |
| `GET /api/llm-metrics/timeseries` | Return aggregated LLM metrics for charting |
| `GET /api/llm-decision-quality` | Return LLM decision quality metrics |
| `GET /api/logs` | Return the most recent log entries from Redis |
| `GET /api/messages` | Return messages stored for the web interface |
| `GET /api/market-status` | Return market status |
| `GET /api/news` | Return news summaries for tracked symbols |
| `POST /api/config/update-interval` | Update the WebSocket payload cache TTL |
| `POST /api/login` | Authenticate and set session cookie |
| `POST /api/logout` | Clear session cookie |

## Project Structure

```
.
├── src/
│   ├── config/          # Settings and validation
│   ├── exchanges/       # yfinance market data, fees, BTP scrapers, Borsa Italiana API
│   ├── llm/             # LLM client, caching, prompts
│   ├── strategies/      # Signal, strategy, validation, backtester
│   ├── trading/         # Engine, paper trader (no live trader — paper only)
│   ├── telegram/        # Telegram bot
│   ├── news/            # News fetcher and sentiment analysis
│   ├── utils/           # Redis client, retry
│   └── web/             # FastAPI app and dashboard
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## License

MIT License – see `LICENSE` file.
