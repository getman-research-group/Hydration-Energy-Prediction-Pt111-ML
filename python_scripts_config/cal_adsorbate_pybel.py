"""
calculate_logP_pybel.py

Calculate Partition Coefficient logP value for adsorbate
It can load xyz, pdb or sdf files and do calculation.

"""
import csv
import os
import numpy as np
import pandas as pd
from functools import reduce

from core.global_vars import ADSORBATE_TO_NAME_DICT, PATH_MAIN_PROJECT
from core.path import get_paths

from openbabel import pybel
# import pybel

# http://openbabel.org/wiki/Main_Page
# https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_PybelAPI.html




class Open_Babel_Descriptors():

    def __init__(   self,
                    adsorbate_list,
                    save_data,
                    csv_file,
                    ):

        ## Storing Initial Information
        self.adsorbate_list = adsorbate_list
        self.save_data = save_data
        self.csv_file = csv_file
        
        self.dict_pdb = {}
        self.dict_xyz = {}
        self.dict_sdf = {}
        
        self.descriptor_dataframes = []
        
        # Create a molecule as an example
        mol = pybel.readstring("smi", "CCO")  # 使用乙醇分子作为示例

        # Get all available descriptors
        descriptors = mol.calcdesc()
        descriptor_names = descriptors.keys()

        # Print all available descriptors
        print("Available descriptors:")
        for desc in descriptor_names:
            print(desc)
    
        # Initialize file dictionaries
        for adsorbate in self.adsorbate_list:
            pdb_file = os.path.join(get_paths('lammps_data_path'), adsorbate, "prod2.pdb")
            xyz_file = os.path.join(get_paths('simulation_path'), 'xyz_file_adsorbate_only', adsorbate + '.xyz')
            sdf_file = os.path.join(get_paths('lammps_data_path'), adsorbate, "prod2.sdf")
            self.dict_pdb[adsorbate] = pdb_file
            self.dict_xyz[adsorbate] = xyz_file
            self.dict_sdf[adsorbate] = sdf_file
        
        
        # Compute descriptors
        self.calculate_descriptor(self.calculate_logP_xyz)
        self.calculate_descriptor(self.calculate_logP_sdf)
        self.calculate_descriptor(self.calculate_adsorbate_rotatable_bonds)
        self.calculate_descriptor(self.calculate_molar_refractivity)
        self.calculate_descriptor(self.calculate_hba1)
        # self.calculate_descriptor(self.calculate_hba2) # duplicated with hbonds_donor_acceptor
        # self.calculate_descriptor(self.calculate_hbd)  # duplicated with hbonds_donor_acceptor
        
        # Combine all dataframes
        self.combine_dataframes()

        # Save dataframe to csv file
        if self.save_data:
            self.df.to_csv(self.csv_file, header=True, index=False, encoding='utf-8')



    def calculate_descriptor(self, calculation_func):
        descriptor_data = []
        descriptor_name, file_dict, calc_func = calculation_func()

        for adsorbate, file_path in file_dict.items():
            value = calc_func(file_path, adsorbate)
            
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * 5,
                'config': list(range(5)),
                f'adsorbate_{descriptor_name}': [value] * 5
            })
            descriptor_data.append(df)

        result_df = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(result_df)

    
    def calculate_logP_xyz(self):
        print("\n--- Calculating logP for XYZ files")
        def logP_calculator(file_path, adsorbate):
            mol = next(pybel.readfile("xyz", file_path))
            logP = mol.calcdesc(["logP"])["logP"]
            print(f'    LogP for {adsorbate}: {logP:.5f}')
            return logP

        return 'logP_xyz', self.dict_xyz, logP_calculator

    
    def calculate_logP_sdf(self):
        print("\n--- Calculating logP for SDF files")
        def logP_calculator(file_path, adsorbate):
            mol = next(pybel.readfile("sdf", file_path))
            logP = mol.calcdesc(["logP"])["logP"]
            print(f'    LogP for {adsorbate}: {logP:.5f}')
            return logP

        return 'logP_sdf', self.dict_sdf, logP_calculator



    def calculate_adsorbate_rotatable_bonds(self):
        print("\n--- Calculating Number of Rotatable Bonds for SDF files")
        def rotors_calculator(file_path, adsorbate):
            mol = next(pybel.readfile("sdf", file_path))
            rotors = mol.calcdesc(["rotors"])["rotors"]
            print(f'    Number of Rotatable Bonds for {adsorbate}: {rotors:.1f}')
            return rotors

        return 'rotors', self.dict_sdf, rotors_calculator

    # https://openbabel.org/docs/Descriptors/descriptors.html
    def calculate_molar_refractivity(self):
        print("\n--- Calculating Molar Refractivity for SDF files")
        def molar_refractivity_calculator(file_path, adsorbate):
            mol = next(pybel.readfile("sdf", file_path))
            mr = mol.calcdesc(["MR"])["MR"]
            print(f'    Molar Refractivity for {adsorbate}: {mr:.5f}')
            return mr

        return 'molar_refractivity', self.dict_sdf, molar_refractivity_calculator

    # https://openbabel.org/docs/Descriptors/descriptors.html
    def calculate_hba1(self):
        print("\n--- Calculating HBA1 for SDF files")
        def hba1_calculator(file_path, adsorbate):
            mol = next(pybel.readfile("sdf", file_path))
            hba1 = mol.calcdesc(["HBA1"])["HBA1"]
            print(f'    HBA1 for {adsorbate}: {hba1:.5f}')
            return hba1

        return 'hba1', self.dict_sdf, hba1_calculator

    # https://openbabel.org/docs/Descriptors/descriptors.html
    def calculate_hba2(self):
        print("\n--- Calculating HBA2 for SDF files")
        def hba2_calculator(file_path, adsorbate):
            mol = next(pybel.readfile("sdf", file_path))
            hba2 = mol.calcdesc(["HBA2"])["HBA2"]
            print(f'    HBA2 for {adsorbate}: {hba2:.5f}')
            return hba2

        return 'hba2', self.dict_sdf, hba2_calculator


    def calculate_hbd(self):
        print("\n--- Calculating HBD for SDF files")
        def hbd_calculator(file_path, adsorbate):
            mol = next(pybel.readfile("sdf", file_path))
            hbd = mol.calcdesc(["HBD"])["HBD"]
            print(f'    HBD for {adsorbate}: {hbd:.5f}')
            return hbd

        return 'hbd', self.dict_sdf, hbd_calculator


    def combine_dataframes(self):
        if not self.descriptor_dataframes:
            raise ValueError("No descriptor dataframes to combine")
        self.df = reduce(lambda left, right: pd.merge(left, right, on=['adsorbate', 'config'], how='outer'), self.descriptor_dataframes)
        return self.df
        
    
    
    
    

if __name__ == "__main__":
    
    csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_450_adsorbate_pybel_new.csv")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    # 'A01',
                    # 'A02',
                    # 'A40',
                    # 'A60',
                    '254',
                    '264',
                      ]

    # adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    openbabeldescriptors = Open_Babel_Descriptors(
                                        adsorbate_list = adsorbate_list,
                                        save_data = True,   # True or False
                                        csv_file = csv_file,
                                        )
    
