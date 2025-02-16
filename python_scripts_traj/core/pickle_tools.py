# -*- coding: utf-8 -*-
"""
pickle_tools.py contains all pickling tools that will need to load and store pickles.

https://docs.python.org/3/library/pickle.html
https://python3-cookbook.readthedocs.io/zh_CN/latest/c05/p21_serializing_python_objects.html

Functions:
    pickle_results:             function to pickle results
    load_pickle_first_result:   loads pickle but outputs the first result
    load_pickle_results:        loads pickle and gives general result

"""
## Importing Functions
import pickle
import sys
import os
import pandas as pd

### Function To Store Results
def pickle_results(results,
                   pickle_path,
                   protocol = 4,        # Protocol version 4 was added in Python 3.4. It adds support for very large objects,
                                        # Protocol version 5 was added in Python 3.8. It adds support for out-of-band data and speedup for in-band data.
                   verbose = False):
    '''
    This function stores the results for pickle.
    INPUTS:
        results: [list]
            list of results
        pickle_path: [str]
            path to pickle location
        verbose: [bool]
            True if want verbosely print
    OUTPUTS:
        no output text, just store pickle
    '''
    ## Checking If Results is a List, if not, Turn it into a List
    if type(results) != list:
        results = [results]
        
    ## Verbose
    if verbose is True:
        print("Storing pickle at: %s"%(pickle_path) )
        
    ## Storing Pickles
    with open( os.path.join( pickle_path ), 'wb') as file:
        pickle.dump(results, file, protocol = protocol)
    return


### Function to Load Pickle Given a Pickle Directory
def load_pickle_results(pickle_path, verbose = False):
    '''
    This function loads pickle file.
    INPUTS:
        Pickle_path: [str]
            path to the pickle file
        verbose: [bool, default = False]
            True if want to verbosely tell where the pickle is from
    OUTPUTS:
        results from the pickle file
    '''
    
    # Printing
    if verbose == True:
        print("LOADING PICKLE FROM: %s"%(pickle_path))
    
    ## Loading The Data
    with open(pickle_path,'rb') as f:
        # multi_traj_results = pickle.load(f)
        try:
            results = pickle.load(f, encoding = 'latin1')
        except ImportError:
            results = pd.read_pickle( pickle_path )
    return results


### Function To Load Pickle And Outputs The First Result
def load_pickle_first_result(Pickle_path, verbose = False):
    '''
    This function loads pickle file and outputs the first result.
    INPUTS:
        Pickle_path: [str]
            path to the pickle file
        verbose: [bool, default = False]
            True if you want to verbosely tell you where the pickle is from
    OUTPUTS:
        results from your pickle
    '''
    # Printing
    if verbose == True:
        print("LOADING PICKLE FROM: %s"%(Pickle_path) )
    
    ## Loading The Data
    with open(Pickle_path,'rb') as f:
        # multi_traj_results = pickle.load(f)
        try:
            results = pickle.load(f, encoding = 'latin1')
        except OSError:     ## Loading Normally
            results = pickle.load(f)
    return results[0]


### Function to Save Pickle (and Reloading if Necessary)
def save_and_load_pickle(function,
                         inputs,
                         pickle_path,
                         rewrite = False,
                         verbose = True):
    '''
    Save and load pickle whenever we have a function we want to run and we want to store the outputs.
    To lower the amount of redundant calculations by storing the outputs into a pickle file, then reload if necessary.
    
    INPUTS:
        function: [func]
            function we want to run
        inputs: [dict]
            dictionary of inputs
        pickle_path: [str]
            path to store the pickle in
        rewrite: [bool]
            True if we want to rewrite the pickle file
        verbose: [bool]
            True if we want to print the details
    OUTPUTS:
        results: [list]
            list of the results
    '''
    ## Running the Function
    if os.path.isfile(pickle_path) == False or rewrite is True:
        ## Printing
        if verbose is True:
            print("Since either pickle path does not exist or rewrite is true, we are running the calculation!")
            print("Pickle path: %s"%(pickle_path) )
        ## Performing The Task
        results = function(**inputs)
        
        if verbose is True:
            print("Saving the pickle file in %s"%(pickle_path) )
        
        ## Storing It
        pickle_results(results = results,
                       pickle_path = pickle_path)

    ## Loading The File (If pickle path exists and No Rewrite)
    else:
        ## Loading The Pickle
        results = load_pickle_results(pickle_path = pickle_path)
        ## Printing
        if verbose is True:
            print("Loading the pickle file from %s"%(pickle_path) )
    
    return results

