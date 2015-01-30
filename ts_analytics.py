"""
This file contains implementation of basic time series analytic methods.

The implementation is based upon many theoretical and prectical sources, including:
"Timeseries Classification: KNN & DTW" by Mark Regan (http://nbviewer.ipython.org/github/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/blob/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb)
"Learning DTW Global Constraint for Time Series Classification" Niennattrakul and Ratanamahatana
"DTW (Dynamic Time Warping) python module" by Pierre Rouanet (https://github.com/pierre-rouanet/dtw; https://github.com/pierre-rouanet/dtw/blob/master/dtw.py)

2014.12-2015.01, Tomasz Wiktorski, University of Stavanger, Norway
"""

import random
import math
from numpy import zeros, argmin, inf

def random_walk(length=10, scale=1, real=True, values_only=True):
    """
    This function generates a random walk either of integer or real values.
    
    Parameters:
    length - length of resulting random walk, default 10
    scale - scale value for each step of the walk, default 1
    real - specifies if the values in the random walk are integer or real
    values_only - if True simple list is returned with values only, if False list of lists is returned with each value indexed by an integer
    
    Example usage:
    random_walk()
    [0,
     -0.616911095864209,
     -1.202496981836522,
     -0.40840394893314147,
     0.3405551590671432,
     0.7761595862027906,
     1.4461579752084943,
     1.5912390003180445,
     2.367535494551449,
     3.0357579591128516]
     
    random_walk(real=False, values_only=False)
    [[0, 0],
     [1, -1],
     [2, -2],
     [3, -3],
     [4, -2],
     [5, -3],
     [6, -2],
     [7, -3],
     [8, -2],
     [9, -1]] 
    """
    curr_val=0
    if values_only:
        answer=[0]
    else:
        answer=[[0,0]]
    
    
    for t in range(length-1):
        val=random.uniform(-1,1)
        if not real:
            val=int(math.copysign(1,val))
        curr_val+=val*scale
        if values_only:
            answer.append(curr_val)
        else:
            answer.append([t+1,curr_val])
        
    return answer



def dtw(list1, list2, dist=lambda x,y: abs(x-y), dist_only=True):
    """
    This function returns DTW distance between list1 and list2 and a set of supporting parameters. Lists can be of different lengths. You can define a distance measure as a parameter, by default it is the absolute value.
    
    Parameters:
    list1, list2 - time series values
    dist - distance measure as function taking two scalar values, currently absolute value of difference that corresponds in this case with Euclidean distance and L2-norm, squared difference is also common in literature
    dist_only - if True returns only DTW distance as output
    
    Outputs (in order):
    DTW distance between the lists, unscaled
    warp path, list of tuples representing indexes from each input list respectively
    length of shorter list, can be used for scaling
    length of longer list, can be used for scaling
    length of warping path, can be used for scaling
    
    Example usage:
    
    In: x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    In: y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    In: dtw(x,y)
    Out: 
    (4.0,
     [(0, 0),
      (1, 0),
      (2, 1),
      (3, 2),
      (4, 3),
      (4, 4),
      (4, 5),
      (4, 6),
      (5, 7),
      (6, 8),
      (7, 8),
      (8, 8),
      (9, 9)],
     10,
     10,
     13)
    """ 
    list1l=len(list1)
    list2l=len(list2)
    
    dtwm=zeros((list1l+1,list2l+1)) #rows by columns, +1 for boundary conditions
    dtwm[0, 1:] = inf #fill first column with infinity
    dtwm[1:, 0] = inf #fill first raw with infinity
    
    #calculate DTW matrix
    for i in range(list1l):
        for j in range(list2l):
           dtwm[i+1,j+1]=dist(list1[i],list2[j])+min(dtwm[i, j], dtwm[i, j+1], dtwm[i+1, j])
        
    #trace the warping path
    i,j=(list1l -1,list2l -1)  
    wp=[(i,j)]
    
    dtwm=dtwm[1:, 1:]
    
    while (i>0 and j>0):
        dir=argmin((dtwm[i-1, j-1], dtwm[i-1, j], dtwm[i, j-1]))
        
        if dir==0:
            i-=1
            j-=1
        elif dir==1:
            i-=1
        elif dir==2:
            j-=1
        
        wp.insert(0,(i,j))
      
    wp.insert(0, (0,0))
  
    if dist_only:
        return dtwm[-1,-1]
    else:
        return dtwm[-1,-1], wp, min(list1l, list2l), max(list1l,list2l), len(wp)
  
  