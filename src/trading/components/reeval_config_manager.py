"""Handles storage of LLM-decided configuration parameters for re-evaluation."""
import logging
from typing import Any, Dict

from src.config.settings import settings

logger = logging.getLogger(__name__)


class ReevalConfigManager:
    """Manages LLM-decided parameters extracted from the re-evaluation response."""

    def __init__(self, engine):
        self.engine = engine

    async def store_llm_decided_parameters(self, parsed: Dict[str, Any]) -> None:
        """Store LLM-decided parameters from the stock selection response to Redis.

        Handles: max_positions_per_sector, portfolio risk thresholds, confidence
        rejection, limit price distance, min viable trade amount, eval skip
        thresholds, regime thresholds, stop/hold multipliers, review limits,
        pause settings, and re-evaluation interval.
        """
        engine = self.engine

        # Parse max_positions_per_sector from LLM
        max_positions_per_sector = parsed.get("max_positions_per_sector")
        if max_positions_per_sector is not None and isinstance(max_positions_per_sector, int) and max_positions_per_sector > 0:
            await engine.config_service.set_llm_config("max_positions_per_sector", max_positions_per_sector)
            logger.info(f"LLM set max positions per sector to {max_positions_per_sector}")
        else:
            await engine.config_service.clear_llm_config("max_positions_per_sector")

        # Parse LLM-decided portfolio risk thresholds
        max_port_exp = parsed.get("max_portfolio_exposure_pct")
        if max_port_exp is not None and isinstance(max_port_exp, (int, float)) and 0.0 <= float(max_port_exp) <= 0.95:
            await engine.config_service.set_llm_config("max_portfolio_exposure_pct", float(max_port_exp))
        else:
            await engine.config_service.clear_llm_config("max_portfolio_exposure_pct")

        max_port_risk = parsed.get("max_portfolio_stop_risk_pct")
        if max_port_risk is not None and isinstance(max_port_risk, (int, float)) and 0.0 <= float(max_port_risk) <= 1.0:
            await engine.config_service.set_llm_config("max_portfolio_stop_risk_pct", float(max_port_risk))
        else:
            await engine.config_service.clear_llm_config("max_portfolio_stop_risk_pct")

        min_rr = parsed.get("min_risk_reward_ratio")
        if min_rr is not None and isinstance(min_rr, (int, float)) and min_rr >= 1.0:
            if min_rr > 2.0:
                logger.warning(f"min_risk_reward_ratio {min_rr} is too high, capping at 2.0")
                min_rr = 2.0
            await engine.config_service.set_llm_config("min_risk_reward_ratio", float(min_rr))
        else:
            await engine.config_service.clear_llm_config("min_risk_reward_ratio")

        conf_rejection = parsed.get("confidence_rejection_threshold")
        if conf_rejection is not None and isinstance(conf_rejection, (int, float)) and 0.0 <= float(conf_rejection) <= 1.0:
            await engine.config_service.set_llm_config("confidence_rejection_threshold", float(conf_rejection))
            logger.info(f"LLM set confidence rejection threshold to {float(conf_rejection):.2f}")
        else:
            await engine.config_service.clear_llm_config("confidence_rejection_threshold")

        # Parse LLM-controlled limit price max distance
        limit_price_max_dist = parsed.get("limit_price_max_distance_pct")
        if limit_price_max_dist is not None and isinstance(limit_price_max_dist, (int, float)) and 0.0 <= float(limit_price_max_dist) <= 1.0:
            await engine.config_service.set_llm_config("limit_price_max_distance_pct", float(limit_price_max_dist))
        else:
            await engine.config_service.clear_llm_config("limit_price_max_distance_pct")

        # Parse LLM-controlled minimum viable trade amount
        min_viable = parsed.get("min_viable_trade_amount")
        if min_viable is not None and isinstance(min_viable, (int, float)) and min_viable > 0:
            await engine.config_service.set_llm_config("min_viable_trade_amount", float(min_viable))
            logger.info(f"LLM set min viable trade amount to {float(min_viable):.2f}")
        else:
            await engine.config_service.clear_llm_config("min_viable_trade_amount")

        # Parse LLM evaluation skip thresholds
        skip_price_mult = parsed.get("skip_eval_price_change_atr_mult")
        if skip_price_mult is not None and isinstance(skip_price_mult, (int, float)) and skip_price_mult > 0:
            await engine.config_service.set_llm_config("skip_eval_price_change_atr_mult", float(skip_price_mult))
        else:
            await engine.config_service.clear_llm_config("skip_eval_price_change_atr_mult")

        skip_rsi = parsed.get("skip_eval_rsi_change")
        if skip_rsi is not None and isinstance(skip_rsi, (int, float)) and skip_rsi > 0:
            await engine.config_service.set_llm_config("skip_eval_rsi_change", float(skip_rsi))
        else:
            await engine.config_service.clear_llm_config("skip_eval_rsi_change")

        skip_rsi_oversold = parsed.get("skip_eval_rsi_oversold")
        if skip_rsi_oversold is not None and isinstance(skip_rsi_oversold, (int, float)) and skip_rsi_oversold > 0:
            await engine.config_service.set_llm_config("skip_eval_rsi_oversold", float(skip_rsi_oversold))
        else:
            await engine.config_service.clear_llm_config("skip_eval_rsi_oversold")

        skip_rsi_overbought = parsed.get("skip_eval_rsi_overbought")
        if skip_rsi_overbought is not None and isinstance(skip_rsi_overbought, (int, float)) and skip_rsi_overbought > 0:
            await engine.config_service.set_llm_config("skip_eval_rsi_overbought", float(skip_rsi_overbought))
        else:
            await engine.config_service.clear_llm_config("skip_eval_rsi_overbought")

        skip_macd = parsed.get("skip_eval_macd_hist_change")
        if skip_macd is not None and isinstance(skip_macd, (int, float)) and skip_macd > 0:
            await engine.config_service.set_llm_config("skip_eval_macd_hist_change", float(skip_macd))
        else:
            await engine.config_service.clear_llm_config("skip_eval_macd_hist_change")

        # Parse LLM-driven market regime thresholds
        regime_adx_strong = parsed.get("regime_adx_strong")
        if regime_adx_strong is not None and isinstance(regime_adx_strong, (int, float)) and regime_adx_strong > 0:
            await engine.config_service.set_llm_config("regime_adx_strong", float(regime_adx_strong))
        else:
            await engine.config_service.clear_llm_config("regime_adx_strong")

        regime_adx_moderate = parsed.get("regime_adx_moderate")
        if regime_adx_moderate is not None and isinstance(regime_adx_moderate, (int, float)) and regime_adx_moderate > 0:
            await engine.config_service.set_llm_config("regime_adx_moderate", float(regime_adx_moderate))
        else:
            await engine.config_service.clear_llm_config("regime_adx_moderate")

        regime_vol_high = parsed.get("regime_volatility_high_pct")
        if regime_vol_high is not None and isinstance(regime_vol_high, (int, float)) and regime_vol_high > 0:
            await engine.config_service.set_llm_config("regime_volatility_high_pct", float(regime_vol_high))
        else:
            await engine.config_service.clear_llm_config("regime_volatility_high_pct")

        regime_vol_low = parsed.get("regime_volatility_low_pct")
        if regime_vol_low is not None and isinstance(regime_vol_low, (int, float)) and regime_vol_low > 0:
            await engine.config_service.set_llm_config("regime_volatility_low_pct", float(regime_vol_low))
        else:
            await engine.config_service.clear_llm_config("regime_volatility_low_pct")

        regime_bb_squeeze = parsed.get("regime_bb_squeeze_width")
        if regime_bb_squeeze is not None and isinstance(regime_bb_squeeze, (int, float)) and regime_bb_squeeze > 0:
            await engine.config_service.set_llm_config("regime_bb_squeeze_width", float(regime_bb_squeeze))
        else:
            await engine.config_service.clear_llm_config("regime_bb_squeeze_width")

        regime_bb_expansion = parsed.get("regime_bb_expansion_width")
        if regime_bb_expansion is not None and isinstance(regime_bb_expansion, (int, float)) and regime_bb_expansion > 0:
            await engine.config_service.set_llm_config("regime_bb_expansion_width", float(regime_bb_expansion))
        else:
            await engine.config_service.clear_llm_config("regime_bb_expansion_width")

        min_stop_atr_mult = parsed.get("min_stop_loss_atr_mult")
        if min_stop_atr_mult is not None and isinstance(min_stop_atr_mult, (int, float)) and min_stop_atr_mult > 0:
            await engine.config_service.set_llm_config("min_stop_loss_atr_mult", float(min_stop_atr_mult))
        else:
            await engine.config_service.clear_llm_config("min_stop_loss_atr_mult")

        min_hold_time_mult = parsed.get("min_max_hold_time_mult")
        if min_hold_time_mult is not None and isinstance(min_hold_time_mult, (int, float)) and min_hold_time_mult > 0:
            await engine.config_service.set_llm_config("min_max_hold_time_mult", float(min_hold_time_mult))
        else:
            await engine.config_service.clear_llm_config("min_max_hold_time_mult")

        max_sl_reviews = parsed.get("max_stop_loss_reviews")
        if max_sl_reviews is not None and isinstance(max_sl_reviews, int) and 1 <= max_sl_reviews <= 20:
            await engine.config_service.set_llm_config("max_stop_loss_reviews", max_sl_reviews)
        else:
            await engine.config_service.clear_llm_config("max_stop_loss_reviews")

        max_tp_reviews = parsed.get("max_take_profit_reviews")
        if max_tp_reviews is not None and isinstance(max_tp_reviews, int) and 1 <= max_tp_reviews <= 20:
            await engine.config_service.set_llm_config("max_take_profit_reviews", max_tp_reviews)
        else:
            await engine.config_service.clear_llm_config("max_take_profit_reviews")

        min_llm_pause = parsed.get("min_llm_pause_duration_seconds")
        if min_llm_pause is not None and isinstance(min_llm_pause, int) and 300 <= min_llm_pause <= 14400:
            await engine.config_service.set_llm_config("min_llm_pause_duration", min_llm_pause)
        else:
            await engine.config_service.clear_llm_config("min_llm_pause_duration")

        pause_max_keep = parsed.get("pause_max_consecutive_keep")
        if pause_max_keep is not None and isinstance(pause_max_keep, int) and 1 <= pause_max_keep <= 10:
            await engine.config_service.set_llm_config("pause_max_consecutive_keep", pause_max_keep)
        else:
            await engine.config_service.clear_llm_config("pause_max_consecutive_keep")

        pause_force_mult = parsed.get("pause_force_resume_risk_multiplier")
        if pause_force_mult is not None and isinstance(pause_force_mult, (int, float)) and 0.0 <= float(pause_force_mult) <= 1.0:
            await engine.config_service.set_llm_config("pause_force_resume_risk_multiplier", float(pause_force_mult))
        else:
            await engine.config_service.clear_llm_config("pause_force_resume_risk_multiplier")

        max_partial_tp = parsed.get("max_partial_tp_reviews")
        if max_partial_tp is not None and isinstance(max_partial_tp, int) and 1 <= max_partial_tp <= 20:
            await engine.config_service.set_llm_config("max_partial_tp_reviews", max_partial_tp)
        else:
            await engine.config_service.clear_llm_config("max_partial_tp_reviews")

        max_dust_sweep = parsed.get("max_dust_sweep_reviews")
        if max_dust_sweep is not None and isinstance(max_dust_sweep, int) and 1 <= max_dust_sweep <= 20:
            await engine.config_service.set_llm_config("max_dust_sweep_reviews", max_dust_sweep)
        else:
            await engine.config_service.clear_llm_config("max_dust_sweep_reviews")

        # Optional: LLM can set the global symbol re-evaluation interval
        new_interval = parsed.get("stock_revaluation_interval_seconds")
        if new_interval is not None:
            if isinstance(new_interval, (int, float)) and new_interval >= 3600:
                clamped = max(new_interval, settings.MIN_SYMBOL_REEVALUATION_INTERVAL)
                engine._symbol_reevaluation_interval = clamped
                logger.info(f"LLM set symbol re-evaluation interval to {clamped}s (requested {new_interval}s)")
            else:
                logger.warning(f"Invalid stock_revaluation_interval_seconds: {new_interval} (must be >= 3600)")
