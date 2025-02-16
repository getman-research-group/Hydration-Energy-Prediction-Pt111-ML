"""
https://www.kaggle.com/code/tosmin/cancer-prediction-hyperparameter-tuning-molecule#Calculate-molecular-descriptor
"""
import os
import h5py
import ase.io
from ase import Atoms
from dscribe.descriptors import SineMatrix
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths


class Sine_Matrix:

    def __init__(   self,
                    file_dir,
                    flatten,
                    sort_type,
                    adsorbate_list = ['A01'],
                    save_data = False,
                    csv_file_path = os.path.join(get_paths("database_path"), "label_data"),
                    ):

        # Storing Initial Information
        self.adsorbate_list = adsorbate_list
        
        # If use flatten Sine Matrix
        self.flatten = flatten
        
        # Defines the method for handling permutational invariance
        self.sort_type = sort_type
        
        # If save data into csv file
        self.save_data = save_data
        
        # Directory to load xyz files
        self.file_dir = file_dir
        
        # File to store descriptors Value
        self.csv_file_path = csv_file_path
        
        # Dictionary to store XYZ file paths for each adsorbate
        self.xyz_dict = {}
        
        # Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
            
            # Get file_path
            file_path = os.path.join(self.file_dir, adsorbate + '.xyz')
            
            # Store the file path to the xyz_dict
            self.xyz_dict[adsorbate] = file_path
        
        # Define the unit cell for all molecules
        unit_cell = np.array([
                            [8.415900, 0.000000, 0.000000],
                            [4.207950, 7.288380, 0.000000],
                            [0.000000, 0.000000, 24.000001]
                        ])
        
        # Dictionary to store atoms objects for each adsorbate
        self.atoms_dict = {}
        for adsorbate, file_path in self.xyz_dict.items():
            atoms = ase.io.read(file_path, format = 'xyz',)
            
            # Set the unit cell
            atoms.set_cell(unit_cell)
            
            # Set the periodic boundary conditions
            atoms.set_pbc([True, True, False])
            
            print('    atoms: ', atoms)
            self.atoms_dict[adsorbate] = atoms
        
        # Determine the maximum number of atoms among all molecules
        self.max_number_of_atoms = 0
        for atoms in self.atoms_dict.values():
            number_of_atoms = atoms.get_global_number_of_atoms()
            self.max_number_of_atoms = max(self.max_number_of_atoms, number_of_atoms)
        
        print('\n--- max_number_of_atoms: ', self.max_number_of_atoms)
    
        # Compute Sine Matrices
        self.compute_Sine_matrices()
        
        # Save DF to CSV
        if self.save_data:
            
            # Save Sine Matrices to h5py file
            self.save_unflattened_Sine_matrices_as_h5py()
            
            # Save Sine Matrices to CSV file
            self.save_flattened_Sine_matrices_as_csv()
    
    # compute_Sine_matrices
    def compute_Sine_matrices(self):
        
        # Create Sine Matrix object
        if self.sort_type == 'sort_all':
            sm = SineMatrix(n_atoms_max = self.max_number_of_atoms,
                            permutation = 'none',
                            )
        else:
            sm = SineMatrix(n_atoms_max = self.max_number_of_atoms,
                            permutation = self.sort_type,
                            )
        
        # dict for storing Sine Matrix
        self.sm_dict = {}

        # Compute Sine Matrix for each molecule
        for adsorbate, atoms in self.atoms_dict.items():
            
            # Create matrix for atoms
            sine_matrix_flat = sm.create(atoms)
            
            # if sort_type is not 'eigenspectrum', sine_matrix_flat.shape = (self.max_number_of_atoms ** 2,)
            if self.sort_type in ['none', 'sorted_l2', 'sort_all']:
                
                if self.flatten == True:
                    # Check if the flattened matrix shape matches the expected dimensions
                    if sine_matrix_flat.shape != (self.max_number_of_atoms ** 2, ):
                        print(f"--- Error: Flattened Sine Matrix for {adsorbate} has unexpected shape: {sine_matrix_flat.shape}")
                        print(f"--- STOP Getting Sine matrices...")
                        break  # Stop the loop
                    print('    Sine Matrix for %s: %s' % (adsorbate, sine_matrix_flat.shape))
                    self.sm_dict[adsorbate] = sine_matrix_flat
                    
                    if self.sort_type == 'sort_all':
                        for adsorbate, sine_matrix in self.sm_dict.items():
                            # sort matrix
                            sorted_matrix = np.sort(sine_matrix)[::-1]
                            self.sm_dict[adsorbate] = sorted_matrix
                    
                elif self.flatten == False:
                    # Unflatten the matrix
                    sine_matrix_2d = sm.unflatten(sine_matrix_flat)
                    # Check if the unflattened matrix shape matches the expected dimensions
                    if sine_matrix_2d.shape != (self.max_number_of_atoms, self.max_number_of_atoms):
                        print(f"--- Error: Unflattened Sine Matrix for {adsorbate} has unexpected shape: {sine_matrix_2d.shape}")
                        print(f"--- STOP Getting Sine matrices...")
                        break  # Stop the loop
                    print('    Sine Matrix for %s: %s' % (adsorbate, sine_matrix_2d.shape))
                    self.sm_dict[adsorbate] = sine_matrix_2d

            # if sort_type 'eigenspectrum', sine_matrix_flat.shape = (self.max_number_of_atoms,)
            elif self.sort_type == 'eigenspectrum':
                # In eigenspectrum case, no need for flattening or unflattening
                if sine_matrix_flat.shape != (self.max_number_of_atoms, ):
                    print(f"--- Error: Eigenspectrum for {adsorbate} has unexpected shape: {sine_matrix_flat.shape}")
                    print(f"--- STOP Getting Sine matrices...")
                    break  # Stop the loop
                print('    Sine Matrix for %s: %s' % (adsorbate, sine_matrix_flat.shape))
                self.sm_dict[adsorbate] = sine_matrix_flat
        
        # Turn Sine Matrics into DataFrame
        data = []
        for adsorbate, sm in self.sm_dict.items():
            # Flatten Sine Matrix if not already flattened
            if len(sm.shape) > 1:
                sm = sm.flatten()
            data.append([adsorbate] + sm.tolist())
        column_names = ["adsorbate"] + [f"sm_{i}" for i in range(len(data[0]) - 1)]
        self.df_sm = pd.DataFrame(data,
                                  columns = column_names)
    
    
    def save_unflattened_Sine_matrices_as_h5py(self):
        if self.flatten:
            print(f"--- Sine matrices not saved to h5py file because they are flattened")
        else:
            filename = f"Sine_Matrix-unflattened-{self.sort_type}.h5"
            file_path = os.path.join(self.csv_file_path, 'adsorbate_fingerprints', filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with h5py.File(file_path, 'w') as f:
                for name, matrix in self.sm_dict.items():
                    f.create_dataset(name, data = matrix)
            
            print(f"--- Sine matrices saved to {os.path.join(self.csv_file_path, filename)}")
    
    
    def save_flattened_Sine_matrices_as_csv(self):
        
        if self.flatten:
            filename = f"Sine_Matrix-flattened-{self.sort_type}.csv"
            
            self.df_sm.to_csv(os.path.join(self.csv_file_path, 'adsorbate_fingerprints', filename),
                                index = False,
                                )
            print(f"--- Sine matrices saved to {os.path.join(self.csv_file_path, filename)}")
        else:
            print(f"--- Sine matrices not saved to CSV file because they are not flattened")

        
            
    

if __name__ == "__main__":
    
    # Defining Input Location
    xyz_file_path = os.path.join(get_paths("simulation_path"), "xyz_file_adsorbate_only")
    csv_file_path = get_paths("database_path")
    
    adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    # flatten = False
    flatten = True
    
    # save_data = False
    save_data = True
    
    for sort_type in [
                    'none',
                    'sorted_l2',
                    'eigenspectrum',
                    'sort_all',
                    ]:
        
        sm = Sine_Matrix(file_dir = xyz_file_path,
                         adsorbate_list = adsorbate_list,
                         flatten = flatten,
                         sort_type = sort_type,
                         save_data = save_data,
                         csv_file_path = csv_file_path,
                         )
    