"""
https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.structure.html#matminer.featurizers.structure.bonding.BagofBonds
https://github.com/hackingmaterials/matminer/blob/3b76466c1ce74c25956d905096f912e9c853e540/matminer/featurizers/structure/matrix.py
https://github.com/hackingmaterials/matminer/blob/3b76466c1ce74c25956d905096f912e9c853e540/matminer/featurizers/structure/bonding.py
"""
import os
import sys
import csv
import ase.io
from pymatgen.core import Lattice, Structure, Molecule
from matminer.featurizers.structure.bonding import BagofBonds
from matminer.featurizers.structure.matrix import CoulombMatrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths


class Bag_of_Bonds:

    def __init__(   self,
                    file_dir,
                    adsorbate_list = ['A01'],
                    save_data = False,
                    csv_file_path = os.path.join(get_paths("database_path"), "label_data"),
                    ):

        ## Storing Initial Information
        self.adsorbate_list = adsorbate_list
        
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

        ## Compute Bag of Bonds
        self.compute_bag_of_bonds()

    
    def compute_bag_of_bonds(self):

        # An iterable of pymatgen Structure objects which will be used to
        # determine the allowed bond types and bag lengths.
        self.mol_list = []
        
        # Maximum number of atoms in the dataset
        self.max_atoms = 0
        
        # Loop through the adsorbates and load the structures
        for adsorbate, file_path in self.xyz_dict.items():
            
            # Load the structure from the xyz file
            mol = Molecule.from_file(file_path)
            print(f"--- Loading Molecule object from {adsorbate}.xyz file")
            
            # Append the molecule to the mol_list
            self.mol_list.append(mol)
            
            # Check number of atoms of the current molecule
            current_num_atoms = mol.num_sites
            print(f"    {adsorbate} has {current_num_atoms} atoms")
            
            # Update the max_atoms if the current molecule has more atoms
            if current_num_atoms > self.max_atoms:
                self.max_atoms = current_num_atoms
        
        
        # Create a CoulombMatrix featurizer
        cm = CoulombMatrix(flatten = False)
        
        # Create a BagofBonds featurizer
        bob = BagofBonds(coulomb_matrix = cm)
        
        # Define the bags using a list of structures.
        bob.fit(self.mol_list)
        
        # Create a list of Bag of Bonds for each molecule
        self.bag_of_bonds_list = [bob.featurize(mol) for mol in self.mol_list]
        
        # Create a dictionary to store the Bag of Bonds
        self.bag_of_bonds_dict = {}
        
        # Loop through the adsorbates and store the Bag of Bonds in the dictionary
        for adsorbate, mol in zip(self.xyz_dict.keys(), self.mol_list):
            self.bag_of_bonds_dict[adsorbate] = bob.featurize(mol)
        
        # Check the length of each feature set in the dictionary
        expected_length = self.max_atoms ** 2
        all_correct = True  # Flag to track if all feature lengths are correct

        for features in self.bag_of_bonds_dict.values():
            if len(features) != expected_length:
                all_correct = False
                break
        
        print(f"\n--- Maximum atom count: {self.max_atoms} atoms")
        
        if all_correct:
            print(f"--- All features have the correct length of {expected_length}.")
        else:
            print(f"--- Error: Not all features have the expected length of {expected_length}.")
        
        # Create a DataFrame to store the Bag of Bonds
        data = []
        for adsorbate, features in self.bag_of_bonds_dict.items():
            data.append([adsorbate] + features)
        
        # Get the column names
        column_names = ["adsorbate"] + [f"bob_{i}" for i in range(len(data[0]) - 1)]
        
        # Create a DataFrame
        self.df_bob = pd.DataFrame(data, columns = column_names)
        
        # Save DF to CSV
        if self.save_data:
            self.save_Bag_of_Bonds_as_csv()
    
    
    
    def save_Bag_of_Bonds_as_csv(self):
        
        filename = f"Bag_of_Bonds-flattened-none_A44.csv"
        
        # Save the Bag of Bonds to a CSV file
        self.df_bob.to_csv(os.path.join(self.csv_file_path,
                                        'adsorbate_fingerprints',
                                        filename,
                                        ),
                           index = False,
                           )

        print(f"--- Bag of Bonds saved to {os.path.join(self.csv_file_path, filename)}")

        
            
    

if __name__ == "__main__":
    
    ## Defining Input Location
    xyz_file_path = os.path.join(get_paths("simulation_path"), "xyz_file_adsorbate_only")
    csv_file_path = get_paths("database_path")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    'A01',
                    # 'A02',
                    # 'A27',
                    # 'A44',
                    # 'A83',
                      ]
    #    'A01':  '1-CH2OH-CHOH-CH2OH',
    #    'A02':  '2-CH2OH-COH-CH2OH',
    
    adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    save_data = True
    
    bagofbonds = Bag_of_Bonds(file_dir = xyz_file_path,
                              adsorbate_list = adsorbate_list,
                              save_data = save_data,
                              csv_file_path = csv_file_path,
                              )
    