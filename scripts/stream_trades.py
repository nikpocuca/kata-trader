"""
stream_trades.py


Use the engine to stream results to a stream directory. 
"""

# import sys
# sys.path.append('..')
from src.kata_alpaca_engine import KataAlpacaEngine
from src.script_utilities import WorkloadResult, create_workload_parser, workload_logger


def main() -> WorkloadResult:
    """
    main function for streaming incoming trades.

    """

    workload_parser = create_workload_parser("stream_trades")
    args = workload_parser.parse_args()

    try:

        # declare engine and wait to receive trade info.
        _ = KataAlpacaEngine(archive_mode=False, secret_path=args.config)

    except Exception as error:
        workload_logger.info(f"workload failed {error.__repr__()}")
        return WorkloadResult.FAILURE

    workload_logger.info("workload success")
    return WorkloadResult.SUCCESS


if __name__ == "__main__":
    main()
