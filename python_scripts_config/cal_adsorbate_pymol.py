"""
cal_adsorbate_descriptors.py
calculate the molecular descriptors for machine learning model.
It will read MD configurations from MD Analysis,
and finally write descriptors information into CSV files.

"""
import os
import sys
import csv
from functools import reduce

sys.path.append('/Applications/PyMOL.app/Contents/MacOS/python')

## imports the main module, which contains the PyMOL interpreter
import __main__
## Tells PyMOL to run in "quiet" mode with no GUI, and to exit once it has finished running the script
__main__.pymol_argv = ['pymol','-qc']



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mdtraj as md
import MDAnalysis as mda

from morfeus import read_xyz, read_geometry
from morfeus import SASA, Dispersion, BuriedVolume

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from collections import Counter

from core.global_vars import ADSORBATE_TO_NAME_DICT, PATH_MAIN_PROJECT
from core.path import get_paths

from read_md_config_mdanalysis import configMDAnalysis
from read_md_config_mdtraj import configMDTraj
from read_md_config_ase import universeAse

## Import pymol
import pymol
from pymol import cmd
pymol.finish_launching()

class calculateDescriptorsPyMol:

    def __init__(   self,
                    adsorbate_list,
                    verbose,
                    save_data,
                    csv_file,
                    ):

        ## Storing Initial Information
        self.adsorbate_list = adsorbate_list
        self.verbose = verbose
        self.save_data = save_data
        
        ## File to store descriptors Value
        self.csv_file = csv_file

        self.dict_mdanalysis_config = {}
        self.dict_mdtraj_traj = {}
        self.dict_ase_universe = {}
        self.dict_pdb = {}
        self.dict_xyz = {}
        self.dict_xyz_slab = {}
        
        self.descriptor_dataframes = []
        
        ## Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
            
            # mdanalysis universe object
            config_mda = configMDAnalysis(adsorbate = adsorbate)
            self.dict_mdanalysis_config[adsorbate] = config_mda
            
            # mdtraj trajectory
            config_mdtraj = configMDTraj(adsorbate = adsorbate)
            self.dict_mdtraj_traj[adsorbate] = config_mdtraj
            
            # ase atoms object
            config_ase = universeAse(adsorbate = adsorbate)
            self.dict_ase_universe[adsorbate] = config_ase
            
            # pdb file
            pdb_file = os.path.join(get_paths('lammps_data_path'), adsorbate, "prod2.pdb")
            self.dict_pdb[adsorbate] = pdb_file
            
            # xyz file adsorbate only
            xyz_file = os.path.join(get_paths('simulation_path'), 'xyz_file_adsorbate_only', adsorbate+'.xyz')
            self.dict_xyz[adsorbate] = xyz_file

            # xyz file adsorbate with Pt slab
            xyz_with_slab = os.path.join(get_paths('simulation_path'), 'xyz_file', adsorbate+'.xyz')
            self.dict_xyz_slab[adsorbate] = xyz_with_slab

        
        # # Calculate Solvent Accessible Surface Area for adsorbate
        # self.calculate_adsorbate_sasa_biopython()
        # self.calculate_adsorbate_sasa_morfeus()
        
        # Calculate vdW surface area and vdW polar surface area
        self.calculate_adsorbate_vdw_sa_pymol()
        self.calculate_adsorbate_vdw_sa_polar_pymol()
        
        # Calculate solvent accessible surface area and solvent accessible polar surface area
        self.calculate_adsorbate_sasa_pymol()
        
        # Calculate hydrophobic and polar solvent accessible surface area
        self.calculate_adsorbate_sasa_polar_pymol()
        self.calculate_adsorbate_sasa_hydrophobic_pymol()
        
        # Calculate solvent accessible oxygen area
        self.calculate_adsorbate_sa_oxy_area_pymol()
        
        # Calculate adsorbate hydroxyl fraction
        self.calculate_adsorbate_hydroxyl_fraction_pymol()
        
        # Calculate adsorbate negative charge surface area and ratio
        self.calculate_adsorbate_neg_sasa_pymol()
        self.calculate_adsorbate_neg_sasa_fraction_pymol()

        
        ## Combine All dataframes
        self.combine_dataframes()

        
        if self.save_data:
            self.df.to_csv(self.csv_file, index=False)
            print(f"Data saved to {self.csv_file}")
    
    
    def calculate_single_descriptor(self, calculation_func, descriptor_name, config_dict):
        if self.verbose:
            print(f"\n--- Calculating {descriptor_name.replace('_', ' ')} for Adsorbates {self.adsorbate_list}")

        descriptor_data = []
        
        for adsorbate, config in config_dict.items():
            value = calculation_func(config)
            print(f"    {adsorbate} {descriptor_name.replace('_', ' ')}: {value:.3f}")
            
            if config_dict in [self.dict_pdb, self.dict_xyz, self.dict_xyz_slab]:
                adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            else:
                adsorbate_name = config.adsorbate_name
                
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * 5,
                'adsorbate_name': [adsorbate_name] * 5,
                'config': list(range(5)),
                descriptor_name: [value] * 5
            })
            
            descriptor_data.append(df)
        
        result_df = pd.concat(descriptor_data)
        
        self.descriptor_dataframes.append(result_df)

        return result_df
    
    

    def combine_dataframes(self):
        if self.verbose:
            print("\n--- Combining DataFrames for All Descriptors")

        if not self.descriptor_dataframes:
            raise ValueError("No descriptor dataframes to combine")

        self.df = reduce(lambda left, right: pd.merge(left, right, on=['adsorbate', 'adsorbate_name', 'config'], how='outer'), self.descriptor_dataframes)
        return self.df
       
    
    def calculate_adsorbate_sasa_biopython(self):
        def sasa_calculator_biopython(config):
            pdb_file = config
            
            # https://biopython.org/docs/dev/api/Bio.PDB.PDBParser.html
            pdbparser = PDBParser(QUIET=True)
            # https://biopython.org/docs/dev/api/Bio.PDB.Structure.html
            structure = pdbparser.get_structure(id=config, file=pdb_file)
            # https://biopython.org/docs/dev/api/Bio.PDB.Residue.html
            residue = list(structure.get_residues())[1]  # adsorbate
            
            sr = ShrakeRupley(n_points=960)
            sr.compute(entity=residue, level="R")
            
            asa_biopython = round(residue.sasa, 2)
            return asa_biopython

        return self.calculate_single_descriptor(sasa_calculator_biopython, 'adsorbate_sasa_biopython', self.dict_pdb)
    
    
    def calculate_adsorbate_sasa_morfeus(self):
        def sasa_calculator_morfeus(config):
            xyz_file = config
            elements, coordinates = read_xyz(xyz_file)
            sasa = SASA(elements, coordinates)
            return sasa.area
        return self.calculate_single_descriptor(sasa_calculator_morfeus, 'adsorbate_SASA_morfeus', self.dict_xyz)


    def calculate_adsorbate_vdw_sa_pymol(self):
        def vdw_sa_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')
            # Set dot density
            cmd.set('dot_density', 4)
            # 0 = calculate molecular surface area {default}
            cmd.set('dot_solvent', 0)
            # select adsorbate atoms
            cmd.select('adsorbate_atoms', 'elem H or elem O or elem C')
            # calculate the solvent accessible surface area of the selected atoms
            vdwsa = cmd.get_area('adsorbate_atoms', load_b=1)
            return vdwsa
        return self.calculate_single_descriptor(vdw_sa_calculator_pymol, 'adsorbate_vdw_sa_pymol', self.dict_xyz_slab)
    
    
    def calculate_adsorbate_vdw_sa_polar_pymol(self):
        def vdw_sa_polar_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')
            # Set dot density
            cmd.set('dot_density', 4)
            # 0 = calculate molecular surface area {default}
            cmd.set('dot_solvent', 0)
            # Select the oxygen atoms and their neighboring hydrogen atoms
            cmd.select('oxy', 'elem O')
            cmd.select('hyr', 'elem H within 1.2 of elem O')
            # Select the polar atoms
            cmd.select('polar_atoms', 'oxy or hyr')
            polar_sasa = cmd.get_area('polar_atoms', load_b=1)
            return polar_sasa
        return self.calculate_single_descriptor(vdw_sa_polar_calculator_pymol, 'adsorbate_vdw_sa_polar_pymol', self.dict_xyz_slab)
    
    
    
    def calculate_adsorbate_sasa_pymol(self):
        def sasa_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')
            # Set dot density and solvent radius
            cmd.set('dot_density', 4)
            cmd.set('solvent_radius', 1.4)
            # 1 = calculate solvent accessible surface area (SASA)
            cmd.set('dot_solvent', 1)
            # select adsorbate atoms
            cmd.select('adsorbate_atoms', 'elem H or elem O or elem C')
            # calculate the solvent accessible surface area of the selected atoms
            sasa = cmd.get_area('adsorbate_atoms', load_b=1)
            return sasa
        return self.calculate_single_descriptor(sasa_calculator_pymol, 'adsorbate_sasa_pymol', self.dict_xyz_slab)
    
    
    def calculate_adsorbate_sasa_polar_pymol(self):
        def sasa_polar_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')
            # Set dot density and solvent radius
            cmd.set('dot_density', 4)
            cmd.set('solvent_radius', 1.4)
            # 1 = calculate solvent accessible surface area (SASA)
            cmd.set('dot_solvent', 1)
            # Select the oxygen atoms and their neighboring hydrogen atoms
            cmd.select('oxy', 'elem O')
            cmd.select('hyr', 'elem H and neighbor elem O')
            # Select the polar atoms
            cmd.select('polar_atoms', 'oxy or hyr')
            sasa_polar = cmd.get_area('polar_atoms', load_b=1)
            return sasa_polar
        return self.calculate_single_descriptor(sasa_polar_calculator_pymol, 'adsorbate_sasa_polar_pymol', self.dict_xyz_slab)
   
    
    def calculate_adsorbate_sasa_hydrophobic_pymol(self):
        def hydrophobic_sasa_calculator(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')

            # Read charges from the xyz file
            atom_charges = self.read_charges_from_xyz(xyz_file)
            # Set charges in PyMOL
            for atom, charge in atom_charges.items():
                cmd.alter(f"adsorbate and id {atom}", f"partial_charge={charge}")

            # Set dot density and solvent radius
            cmd.set('dot_density', 4)
            cmd.set('solvent_radius', 1.4)
            # 1 = calculate solvent accessible surface area (SASA)
            cmd.set('dot_solvent', 1)
            # Select hydrophobic atoms (|qi| < 0.2)
            cmd.select('hydrophobic_atoms_positive', 'partial_charge < 0.2 and (elem C or elem H or elem O)')
            cmd.select('hydrophobic_atoms_negative', 'partial_charge > -0.2 and (elem C or elem H or elem O)')
            cmd.select('hydrophobic_atoms', 'hydrophobic_atoms_positive and hydrophobic_atoms_negative')
            # Calculate SASA of hydrophobic atoms
            hydrophobic_sasa = cmd.get_area('hydrophobic_atoms', load_b=1)
            return hydrophobic_sasa
        
        return self.calculate_single_descriptor(hydrophobic_sasa_calculator, 'adsorbate_sasa_hydrophobic_pymol', self.dict_xyz_slab)
    
        
    def calculate_adsorbate_sa_oxy_area_pymol(self):
        def sa_oxy_area_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')
            # Set dot density and solvent radius
            cmd.set('dot_density', 4)
            cmd.set('solvent_radius', 1.4)
            # 1 = calculate solvent accessible surface area (SASA)
            cmd.set('dot_solvent', 1)
            # Select the oxygen atoms and their neighboring hydrogen atoms
            cmd.select('oxy', 'elem O')
            # Calculate the solvent accessible surface area (SASA) of the oxygen atoms
            oxygen_sasa = cmd.get_area('oxy', load_b=1)
            return oxygen_sasa
        return self.calculate_single_descriptor(sa_oxy_area_calculator_pymol, 'adsorbate_sa_oxy_area_pymol', self.dict_xyz_slab)

    
    def calculate_adsorbate_hydroxyl_fraction_pymol(self):
        def ahf_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')
            # Set dot density and solvent radius
            cmd.set('dot_density', 4)
            cmd.set('solvent_radius', 1.4)
            # Select adsorbate atoms (H, O, C)
            cmd.select('adsorbate_atoms', 'elem H or elem O or elem C')
            # Calculate total SASA for adsorbate
            total_sasa = cmd.get_area('adsorbate_atoms', load_b=1)
            # Select hydroxyl groups (O and H directly bonded)
            cmd.select('hydroxyl_oxygens', 'elem O and neighbor elem H')
            cmd.select('hydroxyl_hydrogens', 'elem H and neighbor elem O')
            cmd.select('hydroxyl_atoms', 'hydroxyl_oxygens or hydroxyl_hydrogens')
            hydroxyl_sasa = cmd.get_area('hydroxyl_atoms', load_b=1)
            # Calculate AHF
            ahf = hydroxyl_sasa / total_sasa
            return ahf
        return self.calculate_single_descriptor(ahf_calculator_pymol, 'adsorbate_hydroxyl_fraction_pymol', self.dict_xyz_slab)


    def calculate_adsorbate_neg_sasa_pymol(self):
        def neg_sasa_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')

            # Read charges from the xyz file
            atom_charges = self.read_charges_from_xyz(xyz_file)
            # Set charges in PyMOL
            for atom, charge in atom_charges.items():
                cmd.alter(f"adsorbate and id {atom}", f"partial_charge={charge}")

            # Set dot density and solvent radius
            cmd.set('dot_density', 4)
            cmd.set('solvent_radius', 1.4)
            # Calculate the total solvent accessible surface area (SASA)
            cmd.set('dot_solvent', 1)
            cmd.select('adsorbate_atoms', 'elem H or elem O or elem C')
            total_sasa = cmd.get_area('adsorbate_atoms', load_b=1)
            
            # Select negatively charged atoms
            cmd.select('neg_atoms', 'partial_charge < 0 and (elem C or elem H or elem O)')
            neg_sasa = cmd.get_area('neg_atoms', load_b=1)

            return neg_sasa
        
        return self.calculate_single_descriptor(neg_sasa_calculator_pymol, 'adsorbate_neg_sasa_pymol', self.dict_xyz_slab)
    
    
    def calculate_adsorbate_neg_sasa_fraction_pymol(self):
        def neg_sasa_fraction_calculator_pymol(config):
            xyz_file = config
            # Reset PyMOL's state
            cmd.reinitialize()
            # Load the xyz file
            cmd.load(xyz_file, 'adsorbate')

            # Read charges from the xyz file
            atom_charges = self.read_charges_from_xyz(xyz_file)
            # Set charges in PyMOL
            for atom, charge in atom_charges.items():
                cmd.alter(f"adsorbate and id {atom}", f"partial_charge={charge}")

            # Set dot density and solvent radius
            cmd.set('dot_density', 4)
            cmd.set('solvent_radius', 1.4)
            # Calculate the total solvent accessible surface area (SASA)
            cmd.set('dot_solvent', 1)
            cmd.select('adsorbate_atoms', 'elem H or elem O or elem C')
            total_sasa = cmd.get_area('adsorbate_atoms', load_b=1)
            
            # Select negatively charged atoms
            cmd.select('neg_atoms', 'partial_charge < 0 and (elem C or elem H or elem O)')
            neg_sasa = cmd.get_area('neg_atoms', load_b=1)
            
            # Calculate NCPPSA
            ncppsa = neg_sasa / total_sasa if total_sasa != 0 else 0
            return ncppsa
        
        return self.calculate_single_descriptor(neg_sasa_fraction_calculator_pymol, 'adsorbate_neg_sasa_fraction_pymol', self.dict_xyz_slab)
 
    

    
    
    
    
    
    def read_charges_from_xyz(self, xyz_file):
        charges = {}
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[2:]):  # Skip the first two lines
                parts = line.split()
                if len(parts) < 5:  # Stop reading if the line does not have at least 5 elements
                    break
                atom_id = i + 1  # PyMOL atom IDs start from 1
                charge = float(parts[4])  # Assuming the charge is in the fifth column
                charges[atom_id] = charge
        return charges



    

if __name__ == "__main__":
    
    csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_450_adsorbate_pymol_new.csv")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    # 'A01',
                    # 'A02',
                    # 'A03',
                    # 'A04',
                    # 'A05',
                    # 'A06',
                    # 'A30',
                    # 'A27',
                    # 'A44',
                    # 'A83',
                    '254',
                    '264',
                    
                      ]
    #    'A01':  '1-CH2OH-CHOH-CH2OH',
    #    'A02':  '2-CH2OH-COH-CH2OH',
    
    # adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    descriptors_pymol = calculateDescriptorsPyMol(adsorbate_list = adsorbate_list,
                                                  verbose = True,
                                                  save_data = True,  # True, False
                                                  csv_file = csv_file,
                                                  )
