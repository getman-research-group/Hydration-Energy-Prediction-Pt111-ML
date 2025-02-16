# -*- coding: utf-8 -*-
"""
path.py
    This script contains all path functions.

Functions:
    get_base_paths: function to locate the base_path
    get_paths: function to find paths based on the operating system

┍━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━┑
│ System              │ Value               │
┝━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━┥
│ Linux               │ linux or linux2 (*) │
│ Windows             │ win32               │
│ Windows/Cygwin      │ cygwin              │
│ Windows/MSYS2       │ msys                │
│ Mac OS X            │ darwin              │
┕━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━┙

"""

## Importing Modules
import sys
import os
import getpass


## Importing Nomenclature
try:
    from core.nomenclature import read_combined_name, extract_representation_inputs, extract_sampling_inputs
except ImportError:
    from nomenclature import read_combined_name, extract_representation_inputs, extract_sampling_inputs


def get_base_paths():
    # Get the username and home directory
    username = getpass.getuser()
    print('username: ', username)
    homedir = os.path.expanduser('~')
    print('homedir: ', homedir)

    # Set the base path based on the username and home directory
    if username == 'shi.1909' and homedir == r'C:\Users\shi.1909':    # Windows
        base_path = r"C:\Users\shi.1909\OneDrive - The Ohio State University\GitHub\pt111_energy_project"
    elif username == 'jiexin' and homedir == '/Users/jiexin':          # Mac OS
        base_path = "/Users/jiexin/Library/CloudStorage/OneDrive-TheOhioStateUniversity/GitHub/pt111_energy_project"
    elif username == 'jiexins' and homedir == '/users/PAS2536/jiexins':         # Linux
        base_path = "/fs/ess/PAS2536/jiexins/pt111_energy_project"
    else:
        raise ValueError("Unsupported user or home directory")
    
    return base_path




def get_paths(path):
    
    # Set the base path based on the operating system
    if sys.platform.startswith('linux'):
        base_path = "/fs/ess/PAS2536/jiexins/pt111_energy_project"
    elif sys.platform.startswith('darwin'):
        base_path = "/Users/jiexin/Library/CloudStorage/OneDrive-TheOhioStateUniversity/GitHub/pt111_energy_project"
    elif sys.platform.startswith('win'):
        base_path = r"C:\Users\shi.1909\OneDrive - The Ohio State University\GitHub\pt111_energy_project"
    else:
        raise ValueError("Unsupported operating system")

    # Define the paths
    path_dic = {
        'database_path'          : os.path.join(base_path, 'database'),
        'label_data_path'        : os.path.join(base_path, 'database', 'label_data'),
        'md_descriptors_90'      : os.path.join(base_path, 'database', 'label_data', 'E_int_90_traj.csv'),
        'md_descriptors_450'     : os.path.join(base_path, 'database', 'label_data', 'E_int_450_config.csv'),
        
        'simulation_path'        : os.path.join(base_path, 'md_simulations'),
        'lammps_data_path'       : os.path.join(base_path, 'md_simulations', 'lammps_data'),

        'combined_database_path' : os.path.join(base_path, 'combined_data_set'),
        'csv_file_path'          : os.path.join(base_path, 'output_csv'),
        'output_path'            : os.path.join(base_path, 'output_cnn'),
        'output_3dcnn_path'      : os.path.join(base_path, 'output_3d_cnn'),
        'output_figure_path'     : os.path.join(base_path, 'output_figures'),
    }
    
    return path_dic.get(path, "Path not found")


if __name__ == '__main__':
    print (get_paths('path_image_dir'))
    print (sys.platform)
    