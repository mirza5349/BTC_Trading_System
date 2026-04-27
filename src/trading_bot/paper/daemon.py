"""Background entrypoint for the scheduled offline paper-trading loop."""

from __future__ import annotations

import argparse

from trading_bot.paper.loop import PaperLoopSelection, ScheduledPaperTradingService
from trading_bot.settings import load_settings


def main() -> None:
    """Run the configured paper loop until interrupted."""
    parser = argparse.ArgumentParser(description="Scheduled offline paper-trading loop")
    parser.add_argument("--loop-id", default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--target-key", default=None)
    parser.add_argument("--model-run-id", default=None)
    parser.add_argument("--optimization-id", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    settings = load_settings()
    service = ScheduledPaperTradingService.from_settings(settings)
    default_selection = service.default_selection()
    selection = PaperLoopSelection(
        loop_id=args.loop_id or default_selection.loop_id,
        target_name=args.target_name or default_selection.target_name,
        target_key=args.target_key or default_selection.target_key,
        model_run_id=args.model_run_id or default_selection.model_run_id,
        optimization_id=args.optimization_id or default_selection.optimization_id,
        symbol=args.symbol or default_selection.symbol,
        timeframe=args.timeframe or default_selection.timeframe,
        feature_version=args.feature_version or default_selection.feature_version,
        provider=args.provider or default_selection.provider,
    )
    service.run_forever(selection=selection)


if __name__ == "__main__":
    main()
