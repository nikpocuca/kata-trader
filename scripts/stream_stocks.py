"""
stream_stocks.py

Use the engine to stream stock quote results to a stream directory. 
"""

from src.kata_alpaca_engine.ingestion_engine import IngestionEngine
from src.script_utilities import WorkloadResult, create_workload_parser, workload_logger, check_market_close

def main() -> WorkloadResult:
    """
    main function for streaming incoming trades.

    """

    workload_parser = create_workload_parser("stream_stocks")
    workload_parser.add_argument('--symbol', required=True)
    args = workload_parser.parse_args()

    try:
        # declare engine and wait to receive stock_info info.
        _ = IngestionEngine(archive_mode=False, 
                            secret_path=args.config,
                            stocks=(args.symbol))

        while not check_market_close(): 
            workload_logger.info("Market is still open")

    except Exception as error:
        workload_logger.info(f"workload failed {error.__repr__()}")
        return WorkloadResult.FAILURE

    workload_logger.info("workload success")
    return WorkloadResult.SUCCESS


if __name__ == "__main__":
    main()
