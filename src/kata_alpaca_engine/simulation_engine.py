"""

simulation_engine.py 

contains SimulationEngine class which outputs 
data at a certain speed to mimic real life data streams. 

"""

import os
from typing import List
from collections.abc import Iterator

from .ingestion_engine import IngestionEngine
from enum import Enum 
from .logging_utilities import create_logger, Logger
import asyncio
import threading
from asyncio.unix_events import _UnixSelectorEventLoop as EventLoop
from threading import Thread
import time 


"""
type defined seconds for easy reading
"""
seconds = int 


class SimulationMode(Enum): 
    """
    Different mode settings for simulation engine, 

    - RealTime implies that the start timestamp is time_0, and 
    attaches datapoints in a sequence based on the difference between the next
    consecutive timestamp. 
    
    - FlatTime implies a flat sequence based on a pre-specified time differential.
    I.e. every 1 seconds, 2 seconds, etc. Uses a standard sleep call to attach data

    """
    RealTime = 0 
    FlatTime = 1

class SimulationEngine:
    """
    Simulates data streams in a real-time fashion for purposes of testing. 
    
    """
    loaded_file_path: str
    json_data_list: List[dict]
    symbol_list = []
    logger: Logger
    sim_time: List[seconds]
    sim_time_iter: Iterator 
    simulation_thread: Thread

    # attached data information 
    attach_data_list: List[float]
    attached: bool
    attached_iter: Iterator

    def __init__(self, 
                 file_path: os.PathLike, 
                 mode: SimulationMode = SimulationMode.FlatTime, 
                 flat_sim_time: seconds = 1) -> None:
        """
        file_path: file path of stream file to be used for simulation 
        mode: The simulating time differential to attach data onto a list

        """
        if os.path.exists(file_path): 
            self.loaded_file_path = file_path
            self.json_data_list = IngestionEngine.parse_stream_data(self.loaded_file_path) 

            match mode: 
                case SimulationMode.FlatTime:
                    
                    simulation_logger = create_logger(f'simulation_logger_{SimulationMode.FlatTime._name_}')
                    self.logger = simulation_logger
                    self.sim_time = [flat_sim_time]* len(self.json_data_list)
                    self.sim_time_iter = iter(self.sim_time)

                case SimulationMode.RealTime: 
                    raise NotImplementedError(f"Real time mode is not available yet")


    def attach(self, data: List[float]):
        """
        attaches the list as a pass by reference,
        must be run before the simulation begins otherwise 
        data will have nothing to attach to. 
        """
        
        self.logger.info(f"Attaching data at id {id(data)}")

        self.attach_data_list = data
        self.attached = True 
        self.attached_iter = iter(self.json_data_list)

    def detach(self):
        """
        detaches attached list from its source data, 
        only removes the reference from this class but does not delete the original data. 
        """

        self.logger.info(f"Dettaching data at id {id(self.attach_data_list)}")

        del self.attach_data_list 
        self.attached = False 
        del self.attached_iter
    
    async def simulate(self):
        """
        begins the asyncronous call to simulate data and starts to attach the target list.    
        """
        while True:
            try:
                
                # sleep the simulated time delta. 
                sim_time = next(self.sim_time_iter)
                time.sleep(sim_time)

                # attach the point when done sleeping
                data_point_json: dict = next(self.attached_iter)
                self.attach_data_list.append(data_point_json)
                
            except StopIteration:
                self.logger("End of iterator reached, simulation is done")
                break        

    def start_simulation_in_background(self) -> None:
        """
        starts a stream in the background
        """

        # set the loop
        self.stock_stream_loop = asyncio.get_event_loop()
        self.logger.info("simulation starting.")

        self.simulation_thread = threading.Thread(
            target=self.background_job, args=(self.stock_stream_loop,)
        )

        self.simulation_thread.daemon = True # if main thread dies, then this dies as well. 
        self.simulation_thread.start()

    def background_job(self, loop: EventLoop) -> None:
        """
        run the actual job in the background.
        """
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.simulate())

