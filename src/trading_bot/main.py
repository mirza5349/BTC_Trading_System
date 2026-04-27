"""Application main entrypoint.

Initializes settings, logging, and provides the top-level run function
that will orchestrate the trading bot pipeline in future steps.
"""

from __future__ import annotations

from trading_bot import __app_name__, __version__
from trading_bot.logging_config import get_logger, setup_logging
from trading_bot.settings import load_settings

logger = get_logger(__name__)


def run() -> None:
    """Initialize and start the trading bot application.

    Currently performs startup diagnostics only.
    Real pipeline orchestration will be implemented in later steps.
    """
    settings = load_settings()
    setup_logging(log_level=settings.log_level.value)

    logger.info(
        "application_starting",
        app=__app_name__,
        version=__version__,
        environment=settings.app_env.value,
        asset=settings.asset.symbol,
        paper_only=settings.risk.paper_trading_only,
    )

    logger.info(
        "configuration_summary",
        timeframe=settings.asset.timeframe,
        model=settings.model.algorithm,
        prediction_horizon=f"{settings.model.prediction_horizon_minutes}m",
        market_source=settings.data_sources.market,
        news_source=settings.data_sources.news,
    )

    logger.info("startup_complete", status="ready")


if __name__ == "__main__":
    run()
