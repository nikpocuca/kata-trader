"""
change_point.py version 0.1.0 

Contains class on changepoint and methods related to attaining changepoint information. 

"""

import ruptures as rpt
import numpy as np 


""" 
Type definitions on data size and dimensionality

"""


class KataChangePoint: 
    """
    encapsulates changepoint detection and calculation. 

    wraps around the ruptures package 
    """


    data: np.array 
    penalty: float 
    
