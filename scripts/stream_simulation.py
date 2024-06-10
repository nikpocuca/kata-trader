"""
stream_simulation.py

Simulations to stream stock quotes for testing purposes, pushes results in a stream directory. 
"""

from src.kata_cp import KataChangePointEngine
from src.kata_alpaca_engine.simulation_engine import SimulationEngine
import time

import traceback

from src.script_utilities import WorkloadResult, create_parser, workload_logger

def main() -> WorkloadResult:
    """
    main function for streaming trades.

    """

    workload_parser = create_parser("simulation_workload")
    workload_parser.add_argument('--file', required=True, type=str)
    workload_parser.add_argument('--demo', required=False, action='store_true')
    workload_parser.add_argument('--seconds_wait',type=int, default=5)
    args = workload_parser.parse_args()

    try:
        simulation_engine = SimulationEngine(args.file, flat_sim_time=0.001)
        
        
        kata_model_engine = KataChangePointEngine(output_dir_path='output',
                                                  name='test',
                                                  parsing_function=parse_weights)

        simulation_engine.attach(kata_model_engine.data_list_json)
        simulation_engine.start_simulation_in_background()

        if args.demo:
            workload_logger.info("Running in demo mode")
            kata_model_engine.run_model()
        else:
            workload_logger.info("Running in standard mode")
            kata_model_engine.start_changepoint_model_in_background()

        while not simulation_engine.end_of_simulation:
            time.sleep(args.seconds_wait)
            continue

    except Exception as error:
        trace_back_error = traceback.format_exc()
        workload_logger.info(f"workload failed {error.__repr__()} \n {trace_back_error}")
        return WorkloadResult.FAILURE

    workload_logger.info("workload success")
    return WorkloadResult.SUCCESS



def parse_asks(json_dict: dict):
    """
    simple return ask price 
    """ 
    return json_dict['ask_price']

def parse_weights(json_dict: dict):
    return json_dict['weighted_price']

    

if __name__ == "__main__":
    main()
