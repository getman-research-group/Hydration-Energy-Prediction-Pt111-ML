# -*- coding: utf-8 -*-
"""
initialize.py contains all importing and exporting functions for analysis tools.

FUNCTIONS:
    ### Trajectory Functions
    load_traj_from_dir: Load md trajectory given gro + lammpstrj
    print_traj_general_info: Prints general information of the trajectory
    
    ### Time Functions
    convert2HoursMinSec: Converts seconds, minute, hours, etc. to more user friendly times
    getDate: Simply gets the date for today
    
    ### Directory Functions
    check_dir: Creates a directory if it does not exist
    
    ### Dictionary Functions
    make_dict_avg_std: Makes average and standard deviation as a dictionary

"""
### Importing Modules
import time
import sys
import os

### Function To Keep Track Of Time
def convert2HoursMinSec( seconds ):
    '''
    Take the total seconds and converts it to hours, minutes, and seconds
    For Radial Distribution Function Script
    INPUTS:
        seconds: Total seconds
    OUTPUTS:
        h: hours
        m: minutes
        s: seconds
    '''
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

### Function to Get Today's Date
def getDate():
    '''
    Get the date for figures, etc.
    INPUTS: None
    OUTPUTS:
        Date: Date as a year/month/date
    '''
    Date = time.strftime("%y%m%d") # Date for the figure name
    return Date

def getDateTime():
    '''
    Get date + hour, minute, seconds
    INPUTS:
        NONE
    OUTPUTS:
        Date_time: date and time as YEAR, MONTH, DAY, HOUR, MINUTE, SECONDS (e.g. '2017-12-15 08:27:38')
    '''
    Date_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return Date_time

### Function to Convert Time
def convert_hms_to_Seconds(hms):
    h = hms[0]
    m = hms[1]
    s = hms[2]
    total_time_seconds = h * 60 * 60 + m * 60 + s
    return total_time_seconds

### Function to Create Directories
def check_dir(directory):
    '''
    This function checks if the directory exists. If not, it will create one.
    INPUTS:
        directory: directory path
    OUTPUTS:
        none, a directory will be created
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

### Function to Make Dictionary of Average and Standard Deviations
def make_dict_avg_std(average,std):
    '''
    The purpose of this script is simply to take your mean (average) and standard deviation to create a dictionary.
    INPUTS:
        average: average value(s)
        std: standard deviation
    OUTPUTS:
        dict_object: dictionary containing {'avg': value, 'std': value}
    '''
    return {'avg':average, 'std':std}
