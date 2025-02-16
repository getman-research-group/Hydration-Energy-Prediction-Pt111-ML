# -*- coding: utf-8 -*-
"""
database_scripts.py contains all dataframe code

FUNCTIONS:
    find_index_matching_columns_for_df: function that matches column indices for multiple dfs

"""
import pandas as pd
import numpy as np

### Function Rename DF
def rename_df_column_entries(df,
                             col_name = 'adsorbate',
                             change_col_list = [ '001', '002' ]
                             ):
    '''
    This function is to rename df column entries.
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe
        col_name: [str]
            column name
        change_col_list: [list]
            list of columns we want to change
    OUTPUTS:
        updated df (changed in place)
    '''
    ## Changing Column Names (If Necessary)
    df.loc[df.adsorbate == change_col_list[0], col_name] = change_col_list[-1]
    return df

### Function To Match And Create An Index List
def find_index_matching_columns_for_df( dfs,
                                        cols_list,
                                        index_to_list_col_name = 'index_to_csv'
                                        ):
    '''
    This function is to find the index matching between columns.
    
    Inputs:
        dfs: [list]
            list of dfs. 1st one is the reference. 2nd is the one we are looking at.
        cols_list: [list]
            list of matching instances.
            For example, suppose df 1 has 'adsorbate','solvent','surface' and df 2 has 'adsorbate','solvent','surface'. You will match with a list of list:
                col_list = [['adsorbate','solvent','surface'],
                            ['adsorbate','solvent','surface']]
        index_to_list_col_name: [str, default = 'index_to_csv']
    
    Outputs:
        dfs: [list]
            same dfs, except df 1 has a new column based on index_to_list_col_name name, which can be used to reference df 2. Check out dfs[0]['index_to_csv']
    '''
    ## Defining DFs
    instances_df = dfs[0]
    csv_file = dfs[1]
    
    ## Definign Column Lists
    cols_instances = cols_list[0]
    cols_csv_file = cols_list[1]
    
    ## Adding Empty Column of Nans
    instances_df["index_to_csv"] = np.nan
    column_index = instances_df.columns.get_loc("index_to_csv")
    
    ## Finding Locating Labels
    locating_labels_csv_file = np.array([ csv_file[each_label] for each_label in cols_csv_file]).T.astype('str')
    
    ## Creating Index List
    index_list = []
    
    ## Looping Through Each Instance
    for index, row in instances_df.iterrows():
        ## Finding Results
        current_labels = np.array([ row[each_col] for each_col in cols_instances ]).astype('str')
        ## Finding Index
        try:
            index_to_csv = int(np.argwhere( (locating_labels_csv_file == current_labels).all(axis=1) )[0][0])
        except IndexError:
            print("Error found in label:", current_labels)
        ## Appending Index
        instances_df.iloc[index, column_index] = index_to_csv
        index_list.append(index)
        
    ## Converting To Int
    instances_df["index_to_csv"] = instances_df["index_to_csv"].astype('int')
    
    return dfs
