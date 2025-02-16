"""
https://www.kaggle.com/code/tosmin/cancer-prediction-hyperparameter-tuning-molecule#Calculate-molecular-descriptor
"""
import os
import h5py
import ase.io
from dscribe.descriptors import CoulombMatrix

import sys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths


class Coulomb_Matrix:

    def __init__(   self,
                    file_dir,
                    flatten,
                    sort_type,
                    adsorbate_list = ['A01'],
                    save_data = False,
                    csv_file_path = os.path.join(get_paths("database_path"), "label_data"),
                    ):

        ## Storing Initial Information
        self.adsorbate_list = adsorbate_list
        
        ## If use flatten Coulomb Matrix
        self.flatten = flatten
        
        ## Defines the method for handling permutational invariance
        self.sort_type = sort_type
        
        ## If save data into csv file
        self.save_data = save_data
        
        ## Directory to load xyz files
        self.file_dir = file_dir
        
        ## File to store descriptors Value
        self.csv_file_path = csv_file_path
        
        ## Dictionary to store XYZ file paths for each adsorbate
        self.xyz_dict = {}
        
        ## Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
            
            ## Get file_path
            file_path = os.path.join(self.file_dir, adsorbate + '.xyz')
            
            ## Store the file path to the xyz_dict
            self.xyz_dict[adsorbate] = file_path

        ## Dictionary to store atoms objects for each adsorbate
        self.atoms_dict = {}
        for adsorbate, file_path in self.xyz_dict.items():
            atoms = ase.io.read(file_path,
                                format = 'xyz',
                                )
            print('    atoms: ', atoms)
            self.atoms_dict[adsorbate] = atoms
        
        ## Determine the maximum number of atoms among all molecules
        self.max_number_of_atoms = 0
        for atoms in self.atoms_dict.values():
            number_of_atoms = atoms.get_global_number_of_atoms()
            self.max_number_of_atoms = max(self.max_number_of_atoms, number_of_atoms)
        
        print('--- max_number_of_atoms: ', self.max_number_of_atoms)
    
        ## Compute Coulomb Matrices
        self.compute_coulomb_matrices()
        
        ## Save DF to CSV
        if self.save_data:
            # Save Coulomb Matrices to h5py file
            self.save_unflattened_coulomb_matrices_as_h5py()
            
            # Save Coulomb Matrices to CSV file
            self.save_flattened_coulomb_matrices_as_csv()
    
    
    ## compute_coulomb_matrices
    def compute_coulomb_matrices(self):
        
        # Create Coulomb Matrix object
        if self.sort_type == 'sort_all':
            cm = CoulombMatrix(n_atoms_max = self.max_number_of_atoms,
                               permutation = 'none',
                               )
        else:
            cm = CoulombMatrix(n_atoms_max = self.max_number_of_atoms,
                               permutation = self.sort_type,
                               )
        
        # dict for storing Coulomb Matrix
        self.cm_dict = {}

        # Compute Coulomb matrix for each molecule
        for adsorbate, atoms in self.atoms_dict.items():
            
            ## Create matrix for atoms
            coulomb_matrix_flat = cm.create(atoms)
            
            ## if sort_type is not 'eigenspectrum', coulomb_matrix_flat.shape = (self.max_number_of_atoms ** 2,)
            if self.sort_type in ['none', 'sorted_l2', 'sort_all']:
                
                if self.flatten == True:
                    # Check if the flattened matrix shape matches the expected dimensions
                    if coulomb_matrix_flat.shape != (self.max_number_of_atoms ** 2, ):
                        print(f"--- Error: Flattened Coulomb Matrix for {adsorbate} has unexpected shape: {coulomb_matrix_flat.shape}")
                        print(f"--- STOP Getting Coulomb matrices...")
                        break  # Stop the loop
                    print('    Coulomb Matrix for %s: %s' % (adsorbate, coulomb_matrix_flat.shape))
                    self.cm_dict[adsorbate] = coulomb_matrix_flat
                    
                    if self.sort_type == 'sort_all':
                        for adsorbate, coulomb_matrix in self.cm_dict.items():
                            # sort matrix
                            sorted_matrix = np.sort(coulomb_matrix)[::-1]
                            self.cm_dict[adsorbate] = sorted_matrix
                    
                elif self.flatten == False:
                    # Unflatten the matrix
                    coulomb_matrix_2d = cm.unflatten(coulomb_matrix_flat)
                    # Check if the unflattened matrix shape matches the expected dimensions
                    if coulomb_matrix_2d.shape != (self.max_number_of_atoms, self.max_number_of_atoms):
                        print(f"--- Error: Unflattened Coulomb Matrix for {adsorbate} has unexpected shape: {coulomb_matrix_2d.shape}")
                        print(f"--- STOP Getting Coulomb matrices...")
                        break  # Stop the loop
                    print('    Coulomb Matrix for %s: %s' % (adsorbate, coulomb_matrix_2d.shape))
                    self.cm_dict[adsorbate] = coulomb_matrix_2d

            ## if sort_type 'eigenspectrum', coulomb_matrix_flat.shape = (self.max_number_of_atoms,)
            elif self.sort_type == 'eigenspectrum':
                # In eigenspectrum case, no need for flattening or unflattening
                if coulomb_matrix_flat.shape != (self.max_number_of_atoms, ):
                    print(f"--- Error: Eigenspectrum for {adsorbate} has unexpected shape: {coulomb_matrix_flat.shape}")
                    print(f"--- STOP Getting Coulomb matrices...")
                    break  # Stop the loop
                print('    Coulomb Matrix for %s: %s' % (adsorbate, coulomb_matrix_flat.shape))
                self.cm_dict[adsorbate] = coulomb_matrix_flat
        
        
        ## Turn Coulomb Matrics into DataFrame
        data = []
        for adsorbate, cm in self.cm_dict.items():
            # Flatten Coulomb matrix if not already flattened
            if len(cm.shape) > 1:
                cm = cm.flatten()
            data.append([adsorbate] + cm.tolist())
        column_names = ["adsorbate"] + [f"cm_{i}" for i in range(len(data[0]) - 1)]
        self.df_cm = pd.DataFrame(data,
                                  columns = column_names)
    
    
    def save_unflattened_coulomb_matrices_as_h5py(self):
        if self.flatten:
            print(f"--- Coulomb matrices not saved to h5py file because they are flattened")
        else:
            filename = f"Coulomb_Matrix-unflattened-{self.sort_type}.h5"
            file_path = os.path.join(self.csv_file_path, 'adsorbate_fingerprints', filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with h5py.File(file_path, 'w') as f:
                for name, matrix in self.cm_dict.items():
                    f.create_dataset(name, data = matrix)
            
            print(f"--- Coulomb matrices saved to {filename}")
    
    
    def save_flattened_coulomb_matrices_as_csv(self):
        
        if self.flatten:
            filename = f"Coulomb_Matrix-flattened-{self.sort_type}.csv"
            
            self.df_cm.to_csv(os.path.join(self.csv_file_path,
                                            'adsorbate_fingerprints',
                                            filename,
                                            ),
                                index = False,
                                )
            print(f"--- Coulomb matrices saved to {filename}")
        else:
            print(f"--- Sine matrices not saved to CSV file because they are not flattened")
        
            
    

if __name__ == "__main__":
    
    ## Defining Input Location
    xyz_file_path = os.path.join(get_paths("simulation_path"), "xyz_file_adsorbate_only")
    csv_file_path = get_paths("database_path")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    'A01',
                    'A02',
                    'A27',
                    'A44',
                    'A83',
                      ]
    #    'A01':  '1-CH2OH-CHOH-CH2OH',
    #    'A02':  '2-CH2OH-COH-CH2OH',
    
    adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    # flatten = False
    flatten = True

    save_data = True
        

    for sort_type in [
                      'none',
                      'sorted_l2',
                      'eigenspectrum',
                      'sort_all',
                      ]:
    
        cm = Coulomb_Matrix(file_dir = xyz_file_path,
                            adsorbate_list = adsorbate_list,
                            flatten = flatten,
                            sort_type = sort_type,
                            save_data = save_data,
                            csv_file_path = csv_file_path,
                            )
    