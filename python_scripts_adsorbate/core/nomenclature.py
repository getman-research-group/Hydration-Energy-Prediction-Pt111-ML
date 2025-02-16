# -*- coding: utf-8 -*-
"""
nomenclature.py contains all nomenclature information

Functions:
    convert_to_single_name: converts to a single name
    extract_instance_names: extracts the instance name
    extract_representation_inputs: extract representation inputs as a dictionary
    
    ## For Training
    get_combined_name: gets combined names
    read_combined_name: reverses the get combined name function

"""

### Imporing Important Modules
import sys


### Function To Convert Adsorbates etc. to Name
def convert_to_single_name(adsorbate,
                           config,
                           ):
    ''' This function converts multiple arguments to a single name '''
    
    ## Defining Order
    order = [adsorbate, str(config)]
    
    ## Filtering Out All None Flags
    order = filter(None, order)
    
    ## Getting String Name
    str_name = '_'.join(order)
    
    return str_name

### Function To Extract Names
def extract_instance_names(name):
    '''
    Extracts instance names from the given name.

    Args:
        name (str): Name of the instance, e.g. 'A01_1'

    Returns:
        dict: Name dictionary containing the extracted names.

    Example:
        >>> extract_instance_names('A01_1')
        {'adsorbate': 'A01', 'config': '1'}
    '''
    ## Splitting
    split_name = name.split('_')
    
    ## Defining Name Dictionary
    name_dict = {
            'adsorbate':    split_name[0],  # adsorbate name
            'config':       split_name[1],  # configuration
            }
    return name_dict

### Function That Takes Representation Inputs Based On Type
def extract_representation_inputs(representation_type, representation_inputs):
    '''
    This function is to extract representation inputs based on type.
    For example, 'split_avg_nonorm_perc' has three inputs:
        - num splits
        - percentage
        - total frames
    We would like to extract the inputs correctly.
    
    INPUTS:
        representation_type: [str]
            representation type that we are interested in
        representation_inputs: [list]
            list of representation inputs
    
    OUTPUTS:
        representation_inputs_dict: [dict]
            representation inputs as a dictionary
        
    '''
    ## Fixing Representation Inputs
    if representation_type != 'split_avg_nonorm_perc':
        representation_inputs_dict = {
                                    'num_splits': int(representation_inputs[0])
                                    }
        if representation_type == 'split_avg_nonorm_sampling_times':
            representation_inputs_dict = {
                                        'initial_frame': int(representation_inputs[0]),
                                        'last_frame': int(representation_inputs[1]),
                                        'num_splits': int(representation_inputs[2]),
                                        'perc': float(representation_inputs[3]),
                                        }
            
    else:   # if representation_type == 'split_avg_nonorm_perc':
        representation_inputs_dict = {
                                    'num_splits': int(representation_inputs[0]),
                                    'perc': float(representation_inputs[1]),
                                    'total_frames': int(representation_inputs[2]),
                                    }
    return representation_inputs_dict

### Function To Extract Sampling Inputs Based On Type
def extract_sampling_inputs( sampling_type,
                             sampling_inputs,):
    '''
    Extract the sampling inputs into a format that is understandable.
    The sampling information is output into the training algorithm.
    
    Available sampling types:
        train_perc:
            stratified sampling (by default), allowing us to split training and test sets
        spec_train_tests_split:
            way to optimize the number of trianing and testing splits.
            We assume that the training and test sets are selected from the end of the trajectory,
            where the last N_test is the test set and N_train is the training set.
    
    INPUTS:
        sampling_type: [str]
            sampling type that you are trying to use
        sampling_inputs: [list]
            sampling inputs
    
    OUTPUTS:
        sampling_dict: [dict]
            dictionary for sampling
    '''
    ## Storing The Name
    sampling_dict = {
            'name': sampling_type,
            }
    
    ## Defining Available Sampling Dicts
    available_sampling_dict = [ 'train_perc', 'spec_train_tests_split', ]
    
    ## Defining Learning Type
    if sampling_type == 'train_perc':
        sampling_dict['split_percentage'] =  float(sampling_inputs[0])
    elif sampling_type == 'spec_train_tests_split':
        sampling_dict['num_training'] = int(sampling_inputs[0])
        sampling_dict['num_testing'] = int(sampling_inputs[1])
    else:
        print("Error! sampling_type is not correctly defined. Please check the 'extract_sampling_inputs' function to ensure the sampling dictionary is specified!")
        print("Available sampling types are:")
        print("%s"%(', '.join( available_sampling_dict ) ) )
        sys.exit(1)
    return sampling_dict
    

### Function To Decide The Combined Name
def get_combined_name(representation_type,
                      representation_inputs,
                      adsorbate_list,
                      solvent_list,
                      surface_data,
                      data_type = "20_20_20",   # default data type
                      want_adsorbate_list = False
                      ):
    '''
    The purpose of this function is to combine all the names into a single framework that we can store files in.
    
    INPUTS:
        representation_type: [str]
            string of representation types
        representation_inputs: [dict]
            dictionary for the representation input
        adsorbate_list: [list]
            list of adsorbates you are interested in
        solvent_list: [list]
            list of solvent data
        surface_data: [list]
            list of catalytic surface data
    
    OUTPUTS:
        unique_name: [str]
            unique name characterizing all of this
    '''
    
    ## Soring Adsorbate Names
    adsorbate_list.sort()
    ## Sorting Solvent Names
    solvent_list.sort()
    ## Sorting Catalytic Surface Information
    surface_data.sort()
    ## Sort Representation Inputs as a List
    try:
        representation_inputs_list = [ str(representation_inputs[each_key]) for each_key in sorted(representation_inputs) ]
    except TypeError:
        representation_inputs_list = [ str(representation_inputs[each_key]) for each_key in sorted(representation_inputs) ]
    
    if want_adsorbate_list == True:
        unique_name =   data_type + '-' + \
                        representation_type + '-' + \
                        '_'.join(representation_inputs_list) + '-' + \
                        '_'.join(adsorbate_list) + '-' + \
                        '_'.join(solvent_list) + '-' + \
                        '_'.join(surface_data)
    
    if want_adsorbate_list == False:
        unique_name =   data_type + '-' + \
                        representation_type + '-' + \
                        '_'.join(representation_inputs_list) + '-' + \
                        str(len(adsorbate_list)) + 'ads-' + \
                        '_'.join(solvent_list) + '-' + \
                        '_'.join(surface_data)
    
    return unique_name

### Function to Read Combined Name
def read_combined_name(unique_name, reading_type = "post_training"):
    '''
    This function is to go from combined name back to representative inputs.
    INPUTS:
        unique_name: [str], e.g:
            
            20_20_20_100ns_updated-
            split_avg_nonorm_sampling_times-
            10_0.1_0_10000-
            spec_train_tests_split-
            1_2-
            solvent_net-
            500-
            CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF
            
            unique name characterizing all of this
            
        reading_type: [str, default = 'post_training']
            type to read, e.g
                post_training:
                    post training examples
                instances:
                    combined training instances
                
    OUTPUTS:
        combined_name_info: [dict]
            dictionary with the combined names revived
    '''
    ## Defining Empty
    combined_name_info = {}
    
    ## Splitting
    split_name = unique_name.split('-')
    
    ## Extraction
    if reading_type == 'post_training':
        combined_name_info['data_type'] = split_name[0]
        combined_name_info['representation_type'] = split_name[1]
        combined_name_info['representation_inputs'] = split_name[2]
        # combined_name_info['adsorbate_list'] = split_name[3].split('_')
        
        combined_name_info['solvent_list'] = split_name[4]
        combined_name_info['surface_data'] = split_name[5]
        
        combined_name_info['cnn_type'] = split_name[6]
        combined_name_info['epochs'] = split_name[7]
        

        combined_name_info['sampling_type'] = split_name[8]
        combined_name_info['sampling_inputs'] = split_name[9]

        
    elif reading_type == 'instances':
        combined_name_info['data_type'] = split_name[0]                         # 10_10_10_5ns_2_water_adsorbate
        combined_name_info['representation_type'] = split_name[1]               # split_avg_nonorm
        combined_name_info['representation_inputs'] = split_name[2].split('_')  # 5
        # combined_name_info['adsorbate_list'] = split_name[3].split('_')
        combined_name_info['solvent_list'] = split_name[4].split('_')           #
        combined_name_info['surface_data'] = split_name[5].split('_')
    
    else:
        print("Error, no reading type found for: %s"%(reading_type  ) )
        print("Check read_combined_name code in core > nomenclature")
        sys.exit()
    
    return combined_name_info

if __name__ == '__main__':
    pickle_name = "10_10_10_5ns_2_water_adsorbate-split_avg_nonorm-5-HOH-PTS"
    print(read_combined_name(pickle_name, reading_type = "instances"))
