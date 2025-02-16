#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_tools.py
This script contains all checking tools.

"""
## Importing Moduels
import sys
import os

### Function To See If Testing Should be Turned on
def check_testing():
    '''
    The purpose of this function is to turn on testing codes
    INPUTS:
        void
    OUTPUTS:
        True or False depending if you are on the server
    '''
    ## Checking Path If In Server
    # if sys.prefix != '/Users/jiexi/anaconda'
    # and sys.prefix != r'C:\Users\jiexi\anaconda3'
    # and sys.prefix != r'C:\Users\jiexi\anaconda3\envs\tf':
    
    if sys.platform == "linux":
        testing = False
    else:
        print("*** Testing Mode Is On ***")
        testing = True
    
    return testing


### Function to Check Multiple Paths
def check_multiple_paths( *paths ):
    '''
    Function that checks multiple paths
    INPUTS:
        *paths: any number of paths
    OUTPUTS:
        correct_path: [list]
            list of corrected paths
    '''
    correct_path = []
    ## Looping Through
    for each_path in paths:
        ## Correcting
        correct_path.append(get_paths(each_path))
    
    ## Converting to Tuple
    correct_path = tuple(correct_path)
    
    return correct_path

if __name__ == '__main__':
    
    path = r"C:\\Users\shi.1909\OneDrive - The Ohio State University\GitHub\pt111_energy_project\md_simulations\\" + r"lammps_data"
    
    print (get_paths(path))
    # print (check_testing())
