"""

This module provides tests (pytest) for methods in ts_analytics module (implementation of basic time series analytic methods).

2015.01, Tomasz Wiktorski, University of Stavanger, Norway
"""

import ts_analytics as tsa
import pytest
import mlpy

def test_random_walk_length():
    rw1 = tsa.random_walk(27)
    assert len(rw1) == 27

def test_random_walk_scale():
    """
    This test is not fully precise, it might fail randomly but rarely. It checks if scale parameter works in general, but does not check correctness of the particular scale value.
    """
    rw1 = tsa.random_walk(100, scale=1)
    rw2 = tsa.random_walk(100, scale=11)
    rw1 = [abs(val) for val in rw1]
    rw2 = [abs(val) for val in rw2]
    assert sum(rw1)*3 < sum(rw2)
    
def test_random_walk_real():
    rw1 = tsa.random_walk(27)
    assert type(rw1[1]) is float
    
def test_random_walk_values_only():
    rw1 = tsa.random_walk(27, values_only=False)
    assert rw1[0] == [0,0]
    
def test_dtw_short():
    """
    This test calculates DTW using DTW 1.0 by Rouanet (modified to remove normalization), dtw_std from mlpy and compares it with DTW in ts_analytics. It uses two ten element lists as using by Rouanet in http://nbviewer.ipython.org/github/pierre-rouanet/dtw/blob/master/simple%20example.ipynb.
    """
    x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    
    mlpy_dist = mlpy.dtw_std(x, y)
    tsa_dist = tsa.dtw(x, y)
    
    assert tsa_dist == mlpy_dist
