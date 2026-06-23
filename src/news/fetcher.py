import logging
import re
import threading
from typing import List, Dict, Any, Optional
import hashlib
import httpx
import json
import time
import feedparser
from urllib.parse import quote

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config.settings import settings
from src.database import get_aggregate_sentiment_from_db
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

_sentiment_analyzer = SentimentIntensityAnalyzer()

# Event keyword categories for detecting upcoming corporate events from news
_EVENT_KEYWORDS = {
    "earnings": [
        "earnings", "quarterly results", "q1 results", "q2 results",
        "q3 results", "q4 results", "revenue report", "eps",
        "earnings report", "earnings call", "earnings release",
        "fiscal quarter", "financial results", "earnings date",
        "reports earnings", "earnings announcement",
    ],
    "fda": [
        "fda", "clinical trial", "drug approval", "phase 1",
        "phase 2", "phase 3", "nda", "biologics license",
        "regulatory approval", "filing accepted", "pdufa",
    ],
    "ma": [
        "merger", "acquisition", "buyout", "takeover",
        "acquire", "merge", "tender offer",
    ],
    "dividend": [
        "dividend", "ex-dividend", "dividend declaration",
        "special dividend", "dividend payment", "dividend date",
    ],
    "split": [
        "stock split", "reverse split", "forward split",
        "split announcement",
    ],
    "guidance": [
        "guidance", "outlook", "forecast revision",
        "raise guidance", "lower guidance", "preliminary results",
        "preannounce", "pre-announcement",
    ],
    "other": [
        "ipo", "analyst day", "investor day", "shareholder meeting",
        "annual meeting", "proxy vote", "restructuring", "layoff",
        "ceo change", "executive departure", "management change",
        "product launch", "recall",
    ],
}

# Cache for RSS feed content: {url: (timestamp, feed_content)}
_rss_cache = {}
_rss_cache_lock = threading.Lock()
_rss_cache_last_cleanup = 0.0


def _cleanup_rss_cache():
    """Remove RSS cache entries older than 10 minutes."""
    global _rss_cache_last_cleanup
    now = time.time()
    # Throttle: only run cleanup once per 60 seconds
    if now - _rss_cache_last_cleanup < 60:
        return
    with _rss_cache_lock:
        expired = [
            url for url, (ts, _) in _rss_cache.items()
            if now - ts > 600  # 10 minutes
        ]
        for url in expired:
            del _rss_cache[url]
    _rss_cache_last_cleanup = now


class RateLimiter:
    """Thread-safe per-source rate limiter."""
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_request: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, source: str):
        """Block until the required interval has passed since the last request for this source."""
        if not settings.NEWS_RATE_LIMIT_ENABLED:
            return
        with self._lock:
            now = time.time()
            last = self._last_request.get(source, 0.0)
            wait_time = self.min_interval - (now - last)
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()  # re-read after sleep
            self._last_request[source] = now


# Global rate limiter instance, initialized lazily to avoid import order issues.
_rate_limiter: Optional[RateLimiter] = None

# Sources that have returned a permanent error and should be skipped for the rest of the run.
_permanently_disabled_sources: set = set()


def _get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(settings.NEWS_RATE_LIMIT_PER_SOURCE_SECONDS)
    return _rate_limiter


def _get_enabled_sources() -> List[str]:
    """Return a list of source names that are enabled based on configured credentials."""
    sources = []
    if settings.NEWS_API_KEY and "newsapi" not in _permanently_disabled_sources:
        sources.append("newsapi")
    if settings.TWITTER_BEARER_TOKEN and "twitter" not in _permanently_disabled_sources:
        sources.append("twitter")
    if settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET and "reddit" not in _permanently_disabled_sources:
        sources.append("reddit")
    if settings.FACEBOOK_PAGE_ACCESS_TOKEN and settings.FACEBOOK_PAGE_ID and "facebook" not in _permanently_disabled_sources:
        sources.append("facebook")
    if settings.YOUTUBE_API_KEY and "youtube" not in _permanently_disabled_sources:
        sources.append("youtube")
    # Google News is free
    if "googlenews" not in _permanently_disabled_sources:
        sources.append("googlenews")
    # StockTwits is free/public – no API key required
    if "stocktwits" not in _permanently_disabled_sources:
        sources.append("stocktwits")
    if settings.RSS_FEEDS and "rss" not in _permanently_disabled_sources:
        sources.append("rss")
    logger.debug(f"News sources auto-enabled: {sources}")
    return sources


def _analyze_sentiment(text: str) -> Dict[str, Any]:
    """Return sentiment label and compound score for a text."""
    scores = _sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "compound": round(compound, 4)}


def _is_relevant(symbol: str, title: str, summary: str) -> bool:
    """Return True if the article is likely relevant to the trading symbol."""
    text = f"{title} {summary}".lower()
    sym_lower = symbol.split("/")[0].lower()

    # If searching by company name or BTP name (contains a space),
    # check if the first word is in the text.
    if " " in sym_lower:
        first_word = sym_lower.split()[0]
        if first_word in text:
            # Give it a high score so it passes the threshold
            score = 3
            stock_keywords = [
                "stock", "equity", "etf", "market", "trading", "bullish", "bearish",
                "price", "volume", "breakout", "support", "resistance",
                "earnings", "revenue", "dividend", "sector", "index",
                "fed", "interest rate", "inflation", "gdp", "jobs report",
                "analyst", "upgrade", "downgrade", "ipo", "merger", "acquisition",
                "bond", "btp", "treasury", "yield", "maturity",
            ]
            for kw in stock_keywords:
                if kw in text:
                    score += 1
            return score >= 3
        return False

    # Must mention the symbol at least once
    if sym_lower not in text:
        return False
    # Stock/ETF‑specific keywords that indicate relevance
    stock_keywords = [
        "stock", "equity", "etf", "market", "trading", "bullish", "bearish",
        "price", "volume", "breakout", "support", "resistance",
        "earnings", "revenue", "dividend", "sector", "index",
        "fed", "interest rate", "inflation", "gdp", "jobs report",
        "analyst", "upgrade", "downgrade", "ipo", "merger", "acquisition",
        "bond", "btp", "treasury", "yield", "maturity",
    ]
    # Score: +2 for symbol in title, +1 for each stock keyword found
    score = 0
    if sym_lower in title.lower():
        score += 2
    for kw in stock_keywords:
        if kw in text:
            score += 1
    # Require at least 3 points (symbol in title + one keyword, or three keywords)
    return score >= 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_for_symbol(symbol: str, name: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Fetch news articles for a trading symbol from all enabled sources.
    Returns a list of dicts with keys:
        title, source, url, published_at, summary
    Results are cached in Redis for NEWS_CACHE_TTL_SECONDS.
    """
    if not settings.NEWS_ENABLED:
        return []

    # Use base symbol (e.g., "AAPL") for caching, not the full pair
    base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
    if base_symbol.endswith(settings.TICKER_SUFFIX):
        base_symbol = base_symbol[:-len(settings.TICKER_SUFFIX)]

    search_terms = [base_symbol]
    if name:
        search_terms.append(name)

    start_time = time.time()
    logger.debug(f"Fetching news for {symbol} (base symbol: {base_symbol}, name: {name})...")

    redis_client = get_redis_client()
    cache_key = f"news:{base_symbol}:{name if name else ''}:{_source_fingerprint()}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            articles = json.loads(cached)
            logger.debug(f"News for {base_symbol} served from cache ({len(articles)} articles)")
            return articles
        except Exception:
            pass

    articles: List[Dict[str, str]] = []

    enabled = _get_enabled_sources()
    logger.debug(f"Enabled news sources for {symbol}: {enabled}")
    for source in enabled:
        for term in search_terms:
            source_start = time.time()
            if source == "newsapi":
                articles.extend(_fetch_newsapi(term))
            elif source == "twitter":
                if term == base_symbol:
                    articles.extend(_fetch_twitter(term, use_cashtag=True))
                else:
                    articles.extend(_fetch_twitter(term, use_cashtag=False))
            elif source == "reddit":
                articles.extend(_fetch_reddit(term))
            elif source == "facebook":
                articles.extend(_fetch_facebook(term))
            elif source == "youtube":
                articles.extend(_fetch_youtube(term))
            elif source == "googlenews":
                articles.extend(_fetch_googlenews(term))
            elif source == "stocktwits":
                # StockTwits only supports ticker symbols, not company names
                if term == base_symbol:
                    articles.extend(_fetch_stocktwits(term))
            elif source == "rss":
                articles.extend(_fetch_rss(term))
            source_time = time.time() - source_start
            if source_time > 2.0:
                logger.debug(f"Slow news source '{source}' for {term}: {source_time:.2f}s")

    # Deduplicate by URL
    seen = set()
    unique = []
    for a in articles:
        url = a.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(a)

    # Limit per symbol
    unique = unique[:settings.NEWS_MAX_ARTICLES_PER_SYMBOL]

    # Cache
    try:
        redis_client.setex(cache_key, settings.NEWS_CACHE_TTL_SECONDS, json.dumps(unique))
    except Exception as e:
        logger.warning(f"Failed to cache news for {base_symbol}: {e}")

    total_time = time.time() - start_time
    logger.info(f"News for {symbol}: {len(unique)} articles from {len(enabled)} sources in {total_time:.2f}s")
    if total_time > 5.0:
        logger.debug(f"News fetch for {symbol} took {total_time:.2f}s – consider reducing sources or increasing cache TTL")

    return unique


def discover_trending_stocks(
    base_currency: str,
    existing_pairs: List[str],
    max_symbols: int = 5,
    min_sentiment: float = 0.3,
    min_articles: int = 3,
) -> List[str]:
    """
    Discover trending stocks not already in existing_pairs by looking at
    top daily gainers among tradable assets and filtering by positive news sentiment.
    """
    if not settings.NEWS_ENABLED or not settings.NEWS_SYMBOL_DISCOVERY_ENABLED:
        return []

    from src.exchanges.market_data import get_quotes

    # Use a subset of existing_pairs to avoid excessive API calls (max 200)
    sample = existing_pairs[:200]
    if not sample:
        return []

    try:
        quotes = get_quotes(sample)
    except Exception as e:
        logger.warning(f"Failed to fetch quotes for stock discovery: {e}")
        return []

    # Build list of (symbol, change_24h)
    gainers = []
    for sym in sample:
        q = quotes.get(sym)
        if q and q.get("change_24h") is not None:
            gainers.append((sym, q["change_24h"]))
    # Sort by change descending (biggest gainers first)
    gainers.sort(key=lambda x: x[1], reverse=True)

    existing_symbols = {pair.split("/")[0].lower() for pair in existing_pairs}
    candidates = []
    for sym, change in gainers:
        if sym in existing_pairs:
            continue
        base = sym.split("/")[0] if "/" in sym else sym
        if base.lower() in existing_symbols:
            continue
        # Check news sentiment
        agg = get_aggregate_sentiment_from_db(base, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
        if agg and agg["total_articles"] >= min_articles and agg["avg_compound"] >= min_sentiment:
            candidates.append((sym, agg["avg_compound"]))
        if len(candidates) >= max_symbols:
            break

    # Sort by sentiment descending and take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    discovered = [pair for pair, _ in candidates[:max_symbols]]
    if discovered:
        logger.info(f"News-driven stock discovery found: {discovered}")
    return discovered


def detect_upcoming_events(symbol: str) -> Optional[Dict[str, Any]]:
    """Scan recent news articles for event-related keywords.

    Uses articles already stored in the database — no additional API calls.
    Returns a dict with event information, or None if no events detected.
    """
    from src.database import get_news_for_symbol

    if not settings.NEWS_ENABLED:
        return None

    base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
    articles = get_news_for_symbol(base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
    if not articles:
        return None

    detected_types = set()
    detected_keywords = []

    for article in articles:
        title = (article.get("title", "") or "").lower()
        summary = (article.get("summary", "") or "").lower()
        text = f"{title} {summary}"

        for event_type, keywords in _EVENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    detected_types.add(event_type)
                    if kw not in detected_keywords:
                        detected_keywords.append(kw)

    if not detected_types:
        return None

    return {
        "has_event": True,
        "event_types": sorted(detected_types),
        "keywords": detected_keywords[:10],
    }


def _source_fingerprint() -> str:
    """Create a short fingerprint of the current source configuration for cache key."""
    raw = f"{_get_enabled_sources()}:{settings.NEWS_MAX_ARTICLES_PER_SYMBOL}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------

def _fetch_newsapi(symbol: str) -> List[Dict[str, str]]:
    if not settings.NEWS_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("newsapi")
        logger.debug(f"Fetching NewsAPI for {symbol}...")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{symbol.split('/')[0]} stock",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": settings.NEWS_MAX_ARTICLES_PER_SYMBOL,
            "apiKey": settings.NEWS_API_KEY,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        
        # Check HTTP status
        if response.status_code != 200:
            logger.warning(
                f"NewsAPI returned HTTP {response.status_code} for {symbol}: "
                f"{response.text[:200]}"
            )
            return []
        
        # Safely parse JSON
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(
                f"NewsAPI JSON decode failed for {symbol}: {e}. "
                f"Response text: {response.text[:200]}"
            )
            return []
        
        articles = []
        for art in data.get("articles", []):
            title = art.get("title", "")
            description = art.get("description", "") or ""
            text = f"{title} {description}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, description):
                continue
            articles.append({
                "title": title,
                "source": art.get("source", {}).get("name", "NewsAPI"),
                "url": art.get("url", ""),
                "published_at": art.get("publishedAt", ""),
                "summary": description[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"NewsAPI returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Twitter (X) via API v2
# ---------------------------------------------------------------------------

def _fetch_twitter(symbol: str, use_cashtag: bool = True) -> List[Dict[str, str]]:
    if not settings.TWITTER_BEARER_TOKEN:
        return []
    try:
        import tweepy
    except ImportError:
        logger.warning("tweepy not installed. Install with: pip install tweepy")
        return []
    try:
        _get_rate_limiter().wait("twitter")
        logger.debug(f"Fetching Twitter for {symbol}...")
        client = tweepy.Client(bearer_token=settings.TWITTER_BEARER_TOKEN, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        base = symbol.split('/')[0]
        if use_cashtag:
            query = f"${base} stock -is:retweet lang:en"
        else:
            query = f"{base} stock -is:retweet lang:en"
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(settings.NEWS_MAX_ARTICLES_PER_SYMBOL, 10),
            tweet_fields=["created_at", "text"],
        )
        articles = []
        if tweets.data:
            for tweet in tweets.data:
                sentiment = _analyze_sentiment(tweet.text)
                if not _is_relevant(symbol, tweet.text[:100], tweet.text):
                    continue
                articles.append({
                    "title": tweet.text[:100],
                    "source": "Twitter",
                    "url": f"https://twitter.com/i/web/status/{tweet.id}",
                    "published_at": str(tweet.created_at) if tweet.created_at else "",
                    "summary": tweet.text,
                    "sentiment": sentiment,
                })
        logger.debug(f"Twitter returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Twitter fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

def _fetch_reddit(symbol: str) -> List[Dict[str, str]]:
    if not settings.REDDIT_CLIENT_ID or not settings.REDDIT_CLIENT_SECRET:
        return []
    try:
        import praw
    except ImportError:
        logger.warning("praw not installed. Install with: pip install praw")
        return []
    try:
        _get_rate_limiter().wait("reddit")
        logger.debug(f"Fetching Reddit for {symbol}...")
        reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT,
            timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS,
        )
        submissions = reddit.subreddit("all").search(
            f"{symbol.split('/')[0]} stock",
            sort="relevance",
            time_filter="week",
            limit=settings.NEWS_MAX_ARTICLES_PER_SYMBOL,
        )
        articles = []
        for sub in submissions:
            text = f"{sub.title} {sub.selftext[:300] if sub.selftext else ''}"
            sentiment = _analyze_sentiment(text)
            reddit_summary = sub.selftext[:300] if sub.selftext else sub.title
            if not _is_relevant(symbol, sub.title, reddit_summary):
                continue
            articles.append({
                "title": sub.title,
                "source": f"Reddit r/{sub.subreddit.display_name}",
                "url": f"https://reddit.com{sub.permalink}",
                "published_at": str(sub.created_utc),
                "summary": reddit_summary,
                "sentiment": sentiment,
            })
        logger.debug(f"Reddit returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Reddit fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Facebook (Graph API)
# ---------------------------------------------------------------------------

def _fetch_facebook(symbol: str) -> List[Dict[str, str]]:
    if not settings.FACEBOOK_PAGE_ACCESS_TOKEN or not settings.FACEBOOK_PAGE_ID:
        return []
    try:
        _get_rate_limiter().wait("facebook")
        logger.debug(f"Fetching Facebook for {symbol}...")
        url = f"https://graph.facebook.com/v19.0/{settings.FACEBOOK_PAGE_ID}/posts"
        params = {
            "fields": "message,created_time,permalink_url",
            "limit": settings.FACEBOOK_POST_LIMIT,
            "access_token": settings.FACEBOOK_PAGE_ACCESS_TOKEN,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for post in data.get("data", []):
            message = post.get("message", "")
            if not message:
                continue
            # Simple relevance check: symbol appears in the post
            sym_lower = symbol.split('/')[0].lower()
            message_lower = message.lower()
            if " " in sym_lower:
                if sym_lower.split()[0] not in message_lower:
                    continue
            else:
                if sym_lower not in message_lower:
                    continue
            sentiment = _analyze_sentiment(message)
            articles.append({
                "title": message[:100],
                "source": "Facebook",
                "url": post.get("permalink_url", ""),
                "published_at": post.get("created_time", ""),
                "summary": message[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"Facebook returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Facebook fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# YouTube Data API v3
# ---------------------------------------------------------------------------

def _fetch_youtube(symbol: str) -> List[Dict[str, str]]:
    if not settings.YOUTUBE_API_KEY:
        return []
    try:
        _get_rate_limiter().wait("youtube")
        logger.debug(f"Fetching YouTube for {symbol}...")
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": f"{symbol.split('/')[0]} stock",
            "type": "video",
            "maxResults": settings.YOUTUBE_MAX_RESULTS,
            "order": "date",
            "relevanceLanguage": "en",
            "key": settings.YOUTUBE_API_KEY,
        }
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        articles = []
        for item in data.get("items", []):
            snippet = item["snippet"]
            title = snippet.get("title", "")
            description = snippet.get("description", "")
            text = f"{title} {description}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, description[:300]):
                continue
            articles.append({
                "title": title,
                "source": "YouTube",
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "published_at": snippet.get("publishedAt", ""),
                "summary": description[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"YouTube returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"YouTube fetch failed for {symbol}: {e}")
        return []




# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------

def _fetch_googlenews(symbol: str) -> List[Dict[str, str]]:
    """Fetch news from Google News RSS feed."""
    try:
        _get_rate_limiter().wait("googlenews")
        logger.debug(f"Fetching Google News for {symbol}...")
        base = symbol.split("/")[0]
        encoded_base = quote(base)
        url = f"https://news.google.com/rss/search?q={encoded_base}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:settings.GOOGLE_NEWS_MAX_ARTICLES]:
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            text = f"{title} {summary}"
            sentiment = _analyze_sentiment(text)
            if not _is_relevant(symbol, title, summary[:300]):
                continue
            articles.append({
                "title": title,
                "source": entry.get("source", {}).get("title", "Google News"),
                "url": entry.get("link", ""),
                "published_at": entry.get("published", ""),
                "summary": summary[:300],
                "sentiment": sentiment,
            })
        logger.debug(f"Google News returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"Google News fetch failed for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# StockTwits API
# ---------------------------------------------------------------------------

def _fetch_stocktwits(symbol: str) -> List[Dict[str, str]]:
    """Fetch recent twits from StockTwits public API (no key required)."""
    try:
        _get_rate_limiter().wait("stocktwits")
        logger.debug(f"Fetching StockTwits for {symbol}...")
        base = symbol.split("/")[0]
        # Public endpoint uses the raw ticker (e.g., AAPL), no .X suffix
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{base}.json"
        params = {"limit": settings.STOCKTWITS_MAX_POSTS}
        response = httpx.get(url, params=params, timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS)

        if response.status_code != 200:
            if response.status_code == 404:
                logger.debug(f"StockTwits: symbol {base} not found (404)")
            else:
                logger.warning(
                    f"StockTwits returned HTTP {response.status_code} for {symbol}: "
                    f"{response.text[:200]}"
                )
            return []

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"StockTwits JSON decode failed for {symbol}: {e}")
            return []

        articles = []
        for msg in data.get("messages", []):
            try:
                body = msg.get("body", "")
                title = body[:100]

                # Safe access: user may be None
                user = msg.get("user") or {}
                username = user.get("username", "")
                msg_id = msg.get("id", "")

                # Safe access: entities may be None
                entities = msg.get("entities") or {}
                sentiment_obj = entities.get("sentiment") or {}
                sentiment_label = sentiment_obj.get("basic", "")

                if sentiment_label == "Bullish":
                    label = "positive"
                    compound = 0.5
                elif sentiment_label == "Bearish":
                    label = "negative"
                    compound = -0.5
                else:
                    sentiment = _analyze_sentiment(body)
                    label = sentiment["label"]
                    compound = sentiment["compound"]

                if not _is_relevant(symbol, title, body[:300]):
                    continue

                articles.append({
                    "title": title,
                    "source": "StockTwits",
                    "url": f"https://stocktwits.com/{username}/message/{msg_id}",
                    "published_at": msg.get("created_at", ""),
                    "summary": body[:300],
                    "sentiment": {"label": label, "compound": compound},
                })
            except Exception as e:
                logger.warning(
                    f"StockTwits: failed to process message id={msg.get('id', '?')} "
                    f"for {symbol}: {e}. Raw message: {json.dumps(msg, default=str)[:500]}"
                )
                continue
        logger.debug(f"StockTwits returned {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.warning(f"StockTwits fetch failed for {symbol}: {e}")
        return []




# ---------------------------------------------------------------------------
# RSS Feeds
# ---------------------------------------------------------------------------

def _fetch_rss(symbol: str) -> List[Dict[str, str]]:
    """Fetch news from configured RSS feeds, filtering for symbol mentions."""
    _cleanup_rss_cache()
    articles = []
    for feed_url in settings.RSS_FEEDS:
        try:
            # Check cache first
            with _rss_cache_lock:
                cached = _rss_cache.get(feed_url)
                if cached and (time.time() - cached[0]) < 300:  # 5-minute TTL
                    feed_content = cached[1]
                else:
                    feed_content = None

            if feed_content is None:
                _get_rate_limiter().wait(feed_url)
                logger.debug(f"Fetching RSS feed: {feed_url}")
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0; +https://github.com/your-repo)"
                }
                # Retry on 429 with exponential backoff
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        resp = httpx.get(
                            feed_url,
                            headers=headers,
                            timeout=settings.NEWS_HTTP_TIMEOUT_SECONDS,
                            follow_redirects=True,
                        )
                        resp.raise_for_status()
                        feed_content = resp.text
                        break
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 429 and attempt < max_retries - 1:
                            wait = 2 ** attempt
                            logger.warning(
                                f"RSS feed {feed_url} rate limited, retrying in {wait}s..."
                            )
                            time.sleep(wait)
                        else:
                            raise
                # Cache the successful response
                with _rss_cache_lock:
                    _rss_cache[feed_url] = (time.time(), feed_content)

            feed = feedparser.parse(feed_content)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                combined = f"{title} {summary}".lower()
                sym_lower = symbol.split("/")[0].lower()
                if " " in sym_lower:
                    # For company names, check if the first word is in the text
                    if sym_lower.split()[0] not in combined:
                        continue
                else:
                    if sym_lower not in combined:
                        continue
                text = f"{title} {summary}"
                sentiment = _analyze_sentiment(text)
                if not _is_relevant(symbol, title, summary[:300]):
                    continue
                articles.append({
                    "title": title,
                    "source": feed.feed.get("title", "RSS"),
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", ""),
                    "summary": summary[:300],
                    "sentiment": sentiment,
                })
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"RSS feed not found (404): {feed_url}")
            elif e.response.status_code == 403:
                logger.warning(f"RSS feed access forbidden (403): {feed_url}")
            else:
                logger.warning(f"RSS fetch failed for {feed_url}: {e}")
        except Exception as e:
            logger.warning(f"RSS fetch failed for {feed_url}: {e}")
    logger.debug(f"RSS total articles for {symbol}: {len(articles)}")
    return articles


def discover_tickers_from_news(existing_pairs: Optional[List[str]] = None) -> List[str]:
    """Scan configured RSS feeds for potential stock tickers.

    Looks for words ending with the configured TICKER_SUFFIX (e.g., ``.MI``)
    in feed entry titles and summaries. Returns a list of unique base symbols
    (suffix stripped) that are NOT already in existing_pairs (if provided).
    Handles exceptions gracefully and returns an empty list on failure.
    """
    import re

    suffix = settings.TICKER_SUFFIX
    if not suffix:
        return []

    if not settings.NEWS_TICKER_DISCOVERY_ENABLED:
        return []

    # Escape the suffix for regex (e.g., ".MI" -> "\.MI")
    pattern = re.compile(rf"\b([A-Z0-9]+{re.escape(suffix)})\b")
    discovered: set = set()

    existing_set = set()
    if existing_pairs:
        for pair in existing_pairs:
            base = pair.split("/")[0] if "/" in pair else pair
            existing_set.add(base.upper())

    _cleanup_rss_cache()
    limiter = _get_rate_limiter()

    for feed_url in settings.RSS_FEEDS:
        try:
            # Use the existing RSS cache to avoid redundant HTTP requests
            with _rss_cache_lock:
                cached = _rss_cache.get(feed_url)
                if cached and (time.time() - cached[0]) < 300:  # 5-minute TTL
                    feed_content = cached[1]
                else:
                    feed_content = None

            if feed_content is None:
                limiter.wait(feed_url)
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0; +https://github.com/your-repo)"
                }
                resp = httpx.get(
                    feed_url,
                    headers=headers,
                    timeout=15.0,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                feed_content = resp.text
                with _rss_cache_lock:
                    _rss_cache[feed_url] = (time.time(), feed_content)

            feed = feedparser.parse(feed_content)
            for entry in feed.entries:
                title = entry.get("title", "") or ""
                summary = entry.get("summary", "") or entry.get("description", "") or ""
                text = f"{title} {summary}"
                for match in pattern.findall(text):
                    # Strip the suffix to get the base symbol
                    base = match[: -len(suffix)]
                    if base and base.upper() not in existing_set:
                        discovered.add(base)
                        if len(discovered) >= settings.NEWS_TICKER_DISCOVERY_MAX_SYMBOLS:
                            break
                if len(discovered) >= settings.NEWS_TICKER_DISCOVERY_MAX_SYMBOLS:
                    break
        except Exception as e:
            logger.debug(f"Ticker discovery from RSS feed {feed_url} failed: {e}")
        if len(discovered) >= settings.NEWS_TICKER_DISCOVERY_MAX_SYMBOLS:
            break

    return list(discovered)[:settings.NEWS_TICKER_DISCOVERY_MAX_SYMBOLS]


def test_rss_feeds():
    """Check each configured RSS feed and log whether it is reachable."""
    logger.debug(f"Testing {len(settings.RSS_FEEDS)} RSS feeds...")
    for url in settings.RSS_FEEDS:
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0; +https://github.com/your-repo)"},
                timeout=10,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                logger.debug(f"RSS OK: {url}")
            else:
                logger.warning(f"RSS {url} returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"RSS {url} failed: {e}")
