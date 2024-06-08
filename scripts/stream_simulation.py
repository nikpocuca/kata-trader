"""
stream_simulation.py

Simulations to stream stock quotes for testing purposes, pushes results in a stream directory. 
"""

from src.kata_alpaca_engine.ingestion_engine import IngestionEngine
from src.kata_alpaca_engine.simulation_engine import SimulationEngine

from src.script_utilities import WorkloadResult, create_parser, workload_logger

def main() -> WorkloadResult:
    """
    main function for streaming trades.

    """

    workload_parser = create_parser("simulation_workload")
    workload_parser.add_argument('--file', required=True, type=str)
    workload_parser.add_argument('--demo', required=False, action='store_true')
    args = workload_parser.parse_args()

    try:
        engine = SimulationEngine(args.file)
        some_list = []

        engine.attach(some_list)
        engine.start_simulation_in_background()

        if args.demo:
            breakpoint() 

    except Exception as error:
        workload_logger.info(f"workload failed {error.__repr__()}")
        return WorkloadResult.FAILURE

    workload_logger.info("workload success")
    return WorkloadResult.SUCCESS


if __name__ == "__main__":
    main()
