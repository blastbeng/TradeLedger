"""Model tier selection and prompt complexity computation."""
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings


class ModelTierManager:
    """Handles model tier selection and prompt complexity scoring."""

    def __init__(self, engine):
        self.engine = engine

    def choose_model_tier(
        self,
        atr: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        macd_hist: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_middle: Optional[float] = None,
        bb_lower: Optional[float] = None,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        stochastic_k: Optional[float] = None,
        adx: Optional[float] = None,
        plus_di: Optional[float] = None,
        minus_di: Optional[float] = None,
        mfi: Optional[float] = None,
        cci: Optional[float] = None,
        williams_r: Optional[float] = None,
        ichimoku: Optional[Dict[str, Any]] = None,
        market_regime: str = "",
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_val: Optional[float] = None,
        volume_trend: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        is_critical: bool = False,
        trading_paused: bool = False,
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        consecutive_losses: int = 0,
        current_price: Optional[float] = None,
    ) -> str:
        """Return "mind" or "actuator" based on market complexity."""
        if is_critical:
            return "mind"

        score = 0.0
        max_score = 0.0

        # === Critical factors (weight 2.0) ===
        if rsi is not None and macd_hist is not None:
            max_score += 2.0
            if (rsi < settings.MODEL_TIER_RSI_EXTREME and macd_hist < 0) or (rsi > (100 - settings.MODEL_TIER_RSI_EXTREME) and macd_hist > 0):
                score += 2.0

        if all(v is not None for v in (ema_9, ema_21, adx, plus_di, minus_di)):
            max_score += 2.0
            if (ema_9 > ema_21) != (plus_di > minus_di) and adx > settings.MODEL_TIER_ADX_STRONG:
                score += 2.0

        if drawdown_pct is not None:
            max_score += 2.0
            if drawdown_pct > settings.MODEL_TIER_DRAWDOWN_PCT:
                score += 2.0

        if symbol_event is not None:
            max_score += 2.0
            if symbol_event.get("has_event"):
                score += 2.0

        max_score += 2.0
        if consecutive_losses >= settings.MODEL_TIER_CONSECUTIVE_LOSSES:
            score += 2.0

        # === Significant factors (weight 1.5) ===
        if atr_percentile is not None:
            max_score += 1.5
            if atr_percentile > settings.MODEL_TIER_ATR_PERCENTILE_HIGH or atr_percentile < settings.MODEL_TIER_ATR_PERCENTILE_LOW:
                score += 1.5

        if market_regime:
            max_score += 1.5
            if any(kw in market_regime for kw in ("high volatility", "squeeze", "expansion", "ranging")):
                score += 1.5

        if sentiment_trend_val is not None:
            max_score += 1.5
            if abs(sentiment_trend_val) > settings.MODEL_TIER_SENTIMENT_TREND_MAG:
                score += 1.5

        if all(v is not None for v in (bb_upper, bb_lower, bb_middle)) and bb_middle > 0:
            max_score += 1.5
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < settings.MODEL_TIER_BB_WIDTH_SQUEEZE or bb_width > settings.MODEL_TIER_BB_WIDTH_EXPANSION:
                score += 1.5

        if portfolio_exposure_pct is not None:
            max_score += 1.5
            if portfolio_exposure_pct > settings.MODEL_TIER_PORTFOLIO_EXPOSURE_HIGH:
                score += 1.5

        if portfolio_stop_risk_pct is not None:
            max_score += 1.5
            if portfolio_stop_risk_pct > settings.MODEL_TIER_PORTFOLIO_STOP_RISK_HIGH:
                score += 1.5

        if unrealized_pnl is not None:
            max_score += 1.5
            if unrealized_pnl < 0:
                score += 1.5

        if market_breadth is not None:
            max_score += 1.5
            pos_pct = market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                score += 1.5

        if full_market_breadth is not None:
            max_score += 1.5
            pos_pct = full_market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                score += 1.5

        # === Standard factors (weight 1.0) ===
        if macd is not None and macd_signal is not None and macd != 0:
            max_score += 1.0
            if abs(macd - macd_signal) < settings.MODEL_TIER_MACD_HIST_CHANGE * abs(macd):
                score += 1.0

        if stochastic_k is not None:
            max_score += 1.0
            if stochastic_k < settings.MODEL_TIER_STOCH_EXTREME or stochastic_k > (100 - settings.MODEL_TIER_STOCH_EXTREME):
                score += 1.0

        if mfi is not None:
            max_score += 1.0
            if mfi < settings.MODEL_TIER_MFI_EXTREME or mfi > (100 - settings.MODEL_TIER_MFI_EXTREME):
                score += 1.0

        if cci is not None:
            max_score += 1.0
            if cci < -settings.MODEL_TIER_CCI_EXTREME or cci > settings.MODEL_TIER_CCI_EXTREME:
                score += 1.0

        if williams_r is not None:
            max_score += 1.0
            if williams_r < -(100 - settings.MODEL_TIER_WILLIAMS_R_EXTREME) or williams_r > -settings.MODEL_TIER_WILLIAMS_R_EXTREME:
                score += 1.0

        if ichimoku is not None and current_price is not None:
            cloud_top = ichimoku.get("cloud_top")
            cloud_bottom = ichimoku.get("cloud_bottom")
            if cloud_top is not None and cloud_bottom is not None:
                max_score += 1.0
                if cloud_bottom <= current_price <= cloud_top:
                    score += 1.0

        if volume_trend is not None:
            max_score += 1.0
            if volume_trend > settings.MODEL_TIER_VOLUME_TREND_HIGH:
                score += 1.0

        if fundamentals is not None:
            pe = fundamentals.get("pe_ratio")
            if pe is not None:
                max_score += 1.0
                if pe > settings.MODEL_TIER_PE_HIGH or pe < 0:
                    score += 1.0
            margins = fundamentals.get("profit_margins")
            if margins is not None:
                max_score += 1.0
                if margins < 0:
                    score += 1.0

        if max_score == 0:
            return "actuator"
        normalized_score = score / max_score
        return "mind" if normalized_score >= settings.LLM_MIND_MODEL_THRESHOLD * 1.5 else "actuator"

    def compute_prompt_complexity(
        self,
        num_candidates: int = 0,
        volatility_percentile: Optional[float] = None,
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        macd_hist: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_middle: Optional[float] = None,
        bb_lower: Optional[float] = None,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        stochastic_k: Optional[float] = None,
        adx: Optional[float] = None,
        plus_di: Optional[float] = None,
        minus_di: Optional[float] = None,
        mfi: Optional[float] = None,
        cci: Optional[float] = None,
        williams_r: Optional[float] = None,
        ichimoku: Optional[Dict[str, Any]] = None,
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_magnitude: Optional[float] = None,
        volume_trend: Optional[float] = None,
        market_regime: str = "",
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        is_critical: bool = False,
        trading_paused: bool = False,
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        consecutive_losses: int = 0,
        current_price: Optional[float] = None,
        fear_greed: Optional[Dict[str, Any]] = None,
        conflicting_signals: bool = False,
    ) -> float:
        """Return a complexity score between 0.0 (simple) and 1.0 (very complex)."""
        tech_score = 0.0
        if rsi is not None and (rsi < settings.MODEL_TIER_RSI_EXTREME or rsi > (100 - settings.MODEL_TIER_RSI_EXTREME)):
            tech_score = max(tech_score, 0.15)
        if stochastic_k is not None and (stochastic_k < settings.MODEL_TIER_STOCH_EXTREME or stochastic_k > (100 - settings.MODEL_TIER_STOCH_EXTREME)):
            tech_score = max(tech_score, 0.12)
        if mfi is not None and (mfi < settings.MODEL_TIER_MFI_EXTREME or mfi > (100 - settings.MODEL_TIER_MFI_EXTREME)):
            tech_score = max(tech_score, 0.12)
        if cci is not None and (cci < -settings.MODEL_TIER_CCI_EXTREME or cci > settings.MODEL_TIER_CCI_EXTREME):
            tech_score = max(tech_score, 0.12)
        if williams_r is not None and (williams_r < -(100 - settings.MODEL_TIER_WILLIAMS_R_EXTREME) or williams_r > -settings.MODEL_TIER_WILLIAMS_R_EXTREME):
            tech_score = max(tech_score, 0.12)
        if macd is not None and macd_signal is not None and macd != 0:
            if abs(macd - macd_signal) < settings.MODEL_TIER_MACD_HIST_CHANGE * abs(macd):
                tech_score = max(tech_score, 0.10)
        if bb_upper is not None and bb_lower is not None and bb_middle is not None and bb_middle > 0:
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < settings.MODEL_TIER_BB_WIDTH_SQUEEZE or bb_width > settings.MODEL_TIER_BB_WIDTH_EXPANSION:
                tech_score = max(tech_score, 0.15)
        if ichimoku is not None and current_price is not None:
            cloud_top = ichimoku.get("cloud_top")
            cloud_bottom = ichimoku.get("cloud_bottom")
            if cloud_top is not None and cloud_bottom is not None:
                if cloud_bottom <= current_price <= cloud_top:
                    tech_score = max(tech_score, 0.12)

        conflict_score = 0.0
        if rsi is not None and macd_hist is not None:
            if (rsi < settings.MODEL_TIER_RSI_EXTREME and macd_hist < 0) or (rsi > (100 - settings.MODEL_TIER_RSI_EXTREME) and macd_hist > 0):
                conflict_score = max(conflict_score, 0.20)
        if ema_9 is not None and ema_21 is not None and adx is not None and plus_di is not None and minus_di is not None:
            ema_bullish = ema_9 > ema_21
            di_bullish = plus_di > minus_di
            if ema_bullish != di_bullish and adx > settings.MODEL_TIER_ADX_STRONG:
                conflict_score = max(conflict_score, 0.15)
        if conflicting_signals:
            conflict_score = max(conflict_score, 0.05)

        market_score = 0.0
        if volatility_percentile is not None and (volatility_percentile > settings.MODEL_TIER_ATR_PERCENTILE_HIGH or volatility_percentile < settings.MODEL_TIER_ATR_PERCENTILE_LOW):
            market_score = max(market_score, 0.15)
        if market_regime and any(kw in market_regime for kw in ("high volatility", "squeeze", "expansion")):
            market_score = max(market_score, 0.12)
        if market_breadth:
            pos_pct = market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                market_score = max(market_score, 0.12)
        if full_market_breadth:
            pos_pct = full_market_breadth.get("positive_pct", 50)
            if pos_pct > settings.MODEL_TIER_MARKET_BREADTH_EXTREME or pos_pct < (100 - settings.MODEL_TIER_MARKET_BREADTH_EXTREME):
                market_score = max(market_score, 0.10)
        if sentiment_trend_magnitude is not None and sentiment_trend_magnitude > settings.MODEL_TIER_SENTIMENT_TREND_MAG:
            market_score = max(market_score, 0.12)
        if volume_trend is not None and volume_trend > settings.MODEL_TIER_VOLUME_TREND_HIGH:
            market_score = max(market_score, 0.10)

        portfolio_score = 0.0
        if portfolio_exposure_pct is not None and portfolio_exposure_pct > settings.MODEL_TIER_PORTFOLIO_EXPOSURE_HIGH:
            portfolio_score = max(portfolio_score, 0.15)
        if portfolio_stop_risk_pct is not None and portfolio_stop_risk_pct > settings.MODEL_TIER_PORTFOLIO_STOP_RISK_HIGH:
            portfolio_score = max(portfolio_score, 0.15)
        if drawdown_pct is not None and drawdown_pct > settings.MODEL_TIER_DRAWDOWN_PCT:
            portfolio_score = max(portfolio_score, 0.15)
        if unrealized_pnl is not None and unrealized_pnl < 0:
            portfolio_score = max(portfolio_score, 0.05)
        if consecutive_losses >= settings.MODEL_TIER_CONSECUTIVE_LOSSES:
            portfolio_score = max(portfolio_score, 0.12)

        critical_score = 0.0
        if is_critical:
            critical_score = max(critical_score, 0.15)
        if symbol_event is not None and symbol_event.get("has_event"):
            critical_score = max(critical_score, 0.10)
        if fundamentals is not None:
            pe = fundamentals.get("pe_ratio")
            if pe is not None and (pe > settings.MODEL_TIER_PE_HIGH or pe < 0):
                critical_score = max(critical_score, 0.08)
            margins = fundamentals.get("profit_margins")
            if margins is not None and margins < 0:
                critical_score = max(critical_score, 0.08)
        if trading_paused:
            critical_score = max(critical_score, 0.05)

        candidate_score = 0.0
        if num_candidates > 20:
            candidate_score = 0.03
        elif num_candidates > 10:
            candidate_score = 0.02

        legacy_score = 0.0
        if fear_greed:
            fg = fear_greed.get("value", 50)
            if fg <= 25 or fg >= 75:
                legacy_score = 0.02

        total = tech_score + conflict_score + market_score + portfolio_score + critical_score + candidate_score + legacy_score
        return min(1.0, total)

    def _get_effective_temperature(
        self,
        model_type: str,
        complexity: float,
        is_critical: bool = False,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
    ) -> float:
        """Return the temperature to use for a given model_type and complexity score (0-1).
        
        Higher complexity leads to lower temperature for more deterministic and focused decisions.
        Critical decisions always use the lowest temperature.
        """
        from src.config.settings import Settings
        raw = settings.LLM_MIND_TEMPERATURE if model_type == "mind" else settings.LLM_ACTUATOR_TEMPERATURE
        parsed = Settings.parse_temperature_range(raw)
        if parsed is None:
            return settings.LLM_TEMPERATURE
        lo, hi = parsed
        if lo == hi:
            return lo
        
        if is_critical:
            return lo

        # Position risk: large exposure, near stop-loss, or deep drawdown
        # warrants the lowest temperature for deterministic decisions.
        if (
            portfolio_exposure_pct is not None
            and portfolio_exposure_pct > settings.MODEL_TIER_PORTFOLIO_EXPOSURE_HIGH
        ) or (
            portfolio_stop_risk_pct is not None
            and portfolio_stop_risk_pct > settings.MODEL_TIER_PORTFOLIO_STOP_RISK_HIGH
        ) or (
            drawdown_pct is not None
            and drawdown_pct > settings.MODEL_TIER_DRAWDOWN_PCT
        ):
            return lo
        
        # Invert the scale: higher complexity -> lower temperature (more deterministic)
        return hi - (hi - lo) * complexity

    def _compute_reasoning_effort(
        self,
        model_type: str,
        complexity: float,
        is_critical: bool = False,
    ) -> str:
        """Compute reasoning effort level from thinking_enabled and prompt complexity.

        When thinking is disabled for the model type, always returns 'low'.
        When thinking is enabled, maps the complexity score to 'low', 'medium', or 'high'.
        Critical decisions always get 'high' when thinking is enabled.
        """
        if model_type == "mind":
            thinking_enabled = settings.LLM_MIND_THINKING_ENABLED
        elif model_type == "weak":
            thinking_enabled = settings.LLM_WEAK_THINKING_ENABLED
        else:
            thinking_enabled = settings.LLM_ACTUATOR_THINKING_ENABLED

        if not thinking_enabled:
            return "low"

        if is_critical:
            return "high"

        if complexity < 0.35:
            return "low"
        elif complexity < 0.65:
            return "medium"
        else:
            return "high"

    def compute_model_tier_and_temperature(
        self,
        atr: Optional[float],
        atr_percentile: Optional[float],
        rsi: Optional[float],
        macd: Optional[float],
        macd_signal: Optional[float],
        macd_hist: Optional[float],
        bb_upper: Optional[float],
        bb_middle: Optional[float],
        bb_lower: Optional[float],
        ema_9: Optional[float],
        ema_21: Optional[float],
        stochastic_k: Optional[float],
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
        mfi: Optional[float],
        cci: Optional[float],
        williams_r: Optional[float],
        ichimoku: Optional[Dict[str, Any]],
        market_regime: str,
        market_breadth: Optional[Dict[str, Any]],
        full_market_breadth: Optional[Dict[str, Any]],
        sentiment_trend_val: Optional[float],
        volume_trend_val: Optional[float],
        unrealized_pnl: Optional[float],
        drawdown_pct: Optional[float],
        portfolio_exposure_pct: float,
        portfolio_stop_risk_pct: float,
        is_critical: bool,
        trading_paused: bool,
        symbol_event: Optional[Dict[str, Any]],
        fundamentals: Optional[Dict[str, Any]],
        consecutive_losses: int,
        current_price: float,
        timeframe: Optional[str] = None,
        num_candidates: int = 0,
    ) -> Tuple[str, float, str]:
        """Compute the strategy model type and effective temperature."""
        _conflicting = False
        if rsi is not None and macd_hist is not None:
            if (rsi < settings.MODEL_TIER_RSI_EXTREME and macd_hist < 0) or (rsi > (100 - settings.MODEL_TIER_RSI_EXTREME) and macd_hist > 0):
                _conflicting = True

        strategy_complexity = self.compute_prompt_complexity(
            num_candidates=num_candidates,
            volatility_percentile=atr_percentile,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            ema_9=ema_9,
            ema_21=ema_21,
            stochastic_k=stochastic_k,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            mfi=mfi,
            cci=cci,
            williams_r=williams_r,
            ichimoku=ichimoku,
            market_breadth=market_breadth,
            full_market_breadth=full_market_breadth,
            sentiment_trend_magnitude=abs(sentiment_trend_val) if sentiment_trend_val is not None else None,
            volume_trend=volume_trend_val,
            market_regime=market_regime,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=drawdown_pct,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=is_critical,
            trading_paused=trading_paused,
            symbol_event=symbol_event,
            fundamentals=fundamentals,
            consecutive_losses=consecutive_losses,
            current_price=current_price,
            conflicting_signals=_conflicting,
        )

        if is_critical:
            strategy_model_type = "mind"
        else:
            threshold = settings.LLM_MIND_MODEL_THRESHOLD * 1.5
            # Long-term positions benefit from the "mind" model, so use a lower threshold
            if timeframe and timeframe in ("1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y"):
                threshold = settings.LLM_MIND_MODEL_THRESHOLD * 1.2
            strategy_model_type = "mind" if strategy_complexity >= threshold else "actuator"

        effective_temp = self._get_effective_temperature(
            strategy_model_type,
            strategy_complexity,
            is_critical,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            drawdown_pct=drawdown_pct,
        )

        reasoning_effort = self._compute_reasoning_effort(
            strategy_model_type, strategy_complexity, is_critical,
        )

        return strategy_model_type, effective_temp, reasoning_effort
