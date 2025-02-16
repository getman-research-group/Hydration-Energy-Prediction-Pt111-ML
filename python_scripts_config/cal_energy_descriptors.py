"""
cal_adsorbate_descriptors.py
calculate the molecular descriptors for machine learning model.
It will read MD configurations from MD Analysis,
and finally write descriptors information into CSV files.

"""

import os
import numpy as np
import pandas as pd
from functools import reduce
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from MDAnalysis.lib.distances import distance_array, minimize_vectors
from MDAnalysis import transformations
from MDAnalysis.analysis.rdf import InterRDF
from ase.geometry import get_distances

from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.global_vars_lammps import ATOM_TYPE_TO_ELEMENT, ATOM_TYPE_TO_LJ_EPSILON, ATOM_TYPE_TO_LJ_SIGMA
from core.path import get_paths

from read_md_config_mdanalysis import configMDAnalysis
from read_md_config_mdtraj import configMDTraj
from read_md_config_ase import universeAse

# https://docs.mdanalysis.org/2.7.0/documentation_pages/analysis/wbridge_analysis.html



    
class calculateEnergyDescriptors:

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
        
        # Dictionaries to store MDAnalysis, MDTraj, ASE objects
        self.dict_mdanalysis_config = {}
        self.dict_mdtraj_traj = {}
        self.dict_ase_universe = {}
        
        self.descriptor_dataframes = []
        
        # Frame indices of interest
        self.frame_indices = [230, 290, 350, 410, 470]
        
        ## Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
            
            # mdanalysis universe object
            config_mda = configMDAnalysis(adsorbate = adsorbate)
            self.dict_mdanalysis_config[adsorbate] = config_mda
        
        # Calculate van der Waals energy descriptor
        self.calculate_van_der_Waals_energy(cutoff = 12)

        # Calculate electrostatic energy descriptor
        self.calculate_electrostatic_energy(cutoff = 12)
        
        # get combined vdw and electrostatic energy
        self.calculate_vdw_elec_combined()


        # # Calculate van der Waals energy descriptor using X_CSCORE
        # # https://link.springer.com/article/10.1023/A:1016357811882
        # self.calculate_van_der_Waals_energy_X_CSCORE()
        
        # # https://doi.org/10.1021/acs.jcim.7b00017
        # self.calculate_hydrophobic_ChemScore()
        
        
        # # Calculate AutoDock energy descriptors
        # # https://pubs.acs.org/doi/10.1021/ci300604z
        # self.calculate_gaussian_energy_AutoDock(o = 1.5, w = 0.3)
        # self.calculate_repulsion_energy_AutoDock(o = 0.4)
        # self.calculate_van_der_Waals_energy_AutoDock()
        # self.calculate_hydrophobic_AutoDock(b = 1.5)
        
        
        
        
        # Combine all dataframes
        self.combine_dataframes()











    
    def calculate_van_der_Waals_energy(self, cutoff = 10):
        print("\n--- Calculating van der Waals energy descriptor")
        descriptor_data = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions
                
                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
                water_atoms = config_mda.universe.select_atoms('resname HOH')
                
                distances = distance_array(adsorbate_atoms.positions, water_atoms.positions, box=box)
                within_radius_indices = np.any(distances <= cutoff, axis=0)
                close_water_atoms = water_atoms[within_radius_indices]
                close_water_residues = close_water_atoms.residues
                
                vdw_energy = 0.0

                # loop through all water molecules that are within the cutoff distance
                for residue in close_water_residues:
                    water = residue.atoms
                    # Calculate vdW energy for all atoms in the water molecule
                    for atom1 in adsorbate_atoms:
                        for atom2 in water:
                            distance = distance_array(atom1.position, atom2.position, box=box)[0][0]
                            atom1_type = int(atom1.type)
                            atom2_type = int(atom2.type)
                            sigma_i = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom1_type]
                            sigma_j = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom2_type]
                            epsilon_i = ATOM_TYPE_TO_LJ_EPSILON[adsorbate][atom1_type]
                            epsilon_j = ATOM_TYPE_TO_LJ_EPSILON[adsorbate][atom2_type]
                            
                            sigma_ij = (sigma_i + sigma_j) / 2
                            epsilon_ij = np.sqrt(epsilon_i * epsilon_j)
                            
                            vdw_energy_kcal = 4 * epsilon_ij * ((sigma_ij / distance) ** 12 - (sigma_ij / distance) ** 6)
                            vdw_energy += vdw_energy_kcal
                            
                vdw_energy_ev = vdw_energy * 0.0433641  # 1 kcal/mol = 0.0433641 eV
                values.append(vdw_energy_ev)
            
            formatted_values = [f"{value:.2f}" for value in values]
            print(f"    {adsorbate} vdw_energy: {formatted_values}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'vdw_energy': values,
            })
            descriptor_data.append(df)

        self.df_vdw_energy = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_vdw_energy)



    def calculate_electrostatic_energy(self, cutoff=10):
        print("\n--- Calculating electrostatic energy descriptor")
        descriptor_data = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions
                
                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
                water_atoms = config_mda.universe.select_atoms('resname HOH')
                
                distances = distance_array(adsorbate_atoms.positions, water_atoms.positions, box=box)
                within_radius_indices = np.any(distances <= cutoff, axis=0)
                close_water_atoms = water_atoms[within_radius_indices]
                close_water_residues = close_water_atoms.residues
                
                elec_energy = 0.0

                for residue in close_water_residues:
                    water = residue.atoms
                    # Calculate electrostatic energy for all atoms in the water molecule
                    for atom1 in adsorbate_atoms:
                        for atom2 in water:
                            distance = distance_array(atom1.position, atom2.position, box=box)[0][0]
                            charge1 = atom1.charge
                            charge2 = atom2.charge
                            
                            elec_energy_kcal = 332.0637 * (charge1 * charge2) / distance
                            elec_energy += elec_energy_kcal

                elec_energy_ev = elec_energy * 0.0433641  # 1 kcal/mol = 0.0433641 eV
                values.append(elec_energy_ev)
            
            formatted_values = [f"{value:.2f}" for value in values]
            print(f"    {adsorbate} elec_energy: {formatted_values}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'elec_energy': values,
            })
            descriptor_data.append(df)

        self.df_elec_energy = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_elec_energy)

    
    def calculate_vdw_elec_combined(self):
        print("\n--- Calculating combined van der Waals and electrostatic energy descriptor")

        # Ensure both dataframes exist
        if not hasattr(self, 'df_vdw_energy') or not hasattr(self, 'df_elec_energy'):
            raise AttributeError("vdw_energy or elec_energy dataframes do not exist. Please run the respective calculations first.")

        # Merge the two dataframes on common columns
        df_combined = pd.merge(self.df_vdw_energy, self.df_elec_energy, on=['adsorbate', 'adsorbate_name', 'config'])

        # Calculate the combined energy
        df_combined['vdw_elec_energy'] = df_combined['vdw_energy'] + df_combined['elec_energy']

        # Keep only the necessary columns
        df_vdw_elec = df_combined[['adsorbate', 'adsorbate_name', 'config', 'vdw_elec_energy']]

        self.df_vdw_elec = df_vdw_elec
        self.descriptor_dataframes.append(self.df_vdw_elec)
        # Optionally, save to a file
        # df_vdw_elec.to_csv('vdw_elec_combined.csv', index=False)

        formatted_values = [f"{value:.2f}" for value in df_vdw_elec['vdw_elec_energy']]
        print(f"Combined vdw_elec_energy: {formatted_values}")
















    # https://link.springer.com/article/10.1023/A:1016357811882
    def calculate_van_der_Waals_energy_X_CSCORE(self):
        print("\n--- Calculating van der Waals energy descriptor using X_CSCORE")
        descriptor_data = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions

                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS and not name H*')
                water_atoms = config_mda.universe.select_atoms('resname HOH and not name H*')

                vdw_energy = 0.0  # kcal/mol

                for atom1 in adsorbate_atoms:
                    for atom2 in water_atoms:
                        distance = distance_array(atom1.position, atom2.position, box=box)[0][0]
                        atom1_type = atom1.name[0]  # Using the first letter of the atom name
                        atom2_type = atom2.name[0]  # Using the first letter of the atom name
                        
                        # Dictionary for atomic radii
                        ATOM_TYPE_TO_RADIUS = {'C': 1.9, 'O': 1.7}
                        r_i = ATOM_TYPE_TO_RADIUS[atom1_type]
                        r_j = ATOM_TYPE_TO_RADIUS[atom2_type]
                        d_o = r_i + r_j

                        vdw_energy_term = (d_o / distance) ** 8 - 2 * (d_o / distance) ** 4
                        vdw_energy += vdw_energy_term

                vdw_energy_ev = vdw_energy * 0.0433641  # 1 kcal/mol = 0.0433641 eV
                values.append(vdw_energy_ev)
                
            formatted_values = [f"{value:.2f}" for value in values]
            print(f"    {adsorbate} vdw_energy_X_CSCORE: {formatted_values}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'vdw_energy_X_CSCORE': values,
            })
            descriptor_data.append(df)

        self.df_vdw_energy_Autodock = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_vdw_energy_Autodock)

    
    
    
    


    # https://doi.org/10.1021/acs.jcim.7b00017
    def calculate_hydrophobic_ChemScore(self):
        print("\n--- Calculating hydrophobic contact descriptor using ChemScore")
        descriptor_data = []
        
        def hydrophobic_function(d, d0):
            if d <= d0 + 0.5:
                return 1.0
            elif d0 + 0.5 < d <= d0 + 2.0:
                return (1 / 1.5) * (d0 + 2.0 - d)
            else:
                return 0.0

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions

                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS and not name H*')
                water_atoms = config_mda.universe.select_atoms('resname HOH and not name H*')

                hydrophobic_contact = 0.0

                for atom1 in adsorbate_atoms:
                    for atom2 in water_atoms:
                        distance = distance_array(atom1.position, atom2.position, box=box)[0][0]
                        atom1_type = atom1.name[0]  # Using the first letter of the atom name
                        atom2_type = atom2.name[0]  # Using the first letter of the atom name
                        ATOM_TYPE_TO_RADIUS = {'C': 1.9, 'O': 1.7}
                        radius_i = ATOM_TYPE_TO_RADIUS[atom1_type]
                        radius_j = ATOM_TYPE_TO_RADIUS[atom2_type]

                        d0 = radius_i + radius_j

                        hydrophobic_contact += hydrophobic_function(distance, d0)

                values.append(hydrophobic_contact)

            formatted_values = [f"{value:.2f}" for value in values]
            print(f"    {adsorbate} hydrophobic_contact: {formatted_values}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'hydrophobic_ChemScore': values,
            })
            descriptor_data.append(df)

        self.df_hydrophobic_contact = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_hydrophobic_contact)








    # https://pubs.acs.org/doi/10.1021/ci300604z
    def calculate_gaussian_energy_AutoDock(self,o=1.5, w=0.3):
        print("\n--- Calculating Gaussian energy descriptor")
        descriptor_data = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions

                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS and not name H*')
                water_atoms = config_mda.universe.select_atoms('resname HOH and not name H*')

                gauss_energy = 0.0  # Initialize Gaussian energy

                # Calculate Gaussian energy for each atom pair and accumulate
                for atom1 in adsorbate_atoms:
                    for atom2 in water_atoms:
                        distance = distance_array(atom1.position, atom2.position, box=box)[0][0]
                        
                        atom1_type = int(atom1.type)
                        atom2_type = int(atom2.type)
                        sigma_i = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom1_type]
                        sigma_j = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom2_type]
                        d_opt = sigma_i + sigma_j
                        
                        atom1_name = atom1.name[0]
                        atom2_name = atom2.name[0]
                        ATOM_TYPE_TO_RADIUS = {'C': 1.7, 'O': 1.52}
                        r_i = ATOM_TYPE_TO_RADIUS[atom1_name]
                        r_j = ATOM_TYPE_TO_RADIUS[atom2_name]
                        d_opt = r_i + r_j
                        
                        d_diff = distance - d_opt

                        # Calculate Gaussian energy
                        gauss_energy_term = np.exp(-((d_diff - o) ** 2) / w ** 2)
                        gauss_energy += gauss_energy_term

                # Convert Gaussian energy from arbitrary units to eV (assuming initial unit is kcal/mol)
                gauss_energy_ev = gauss_energy * 0.0433641  # 1 kcal/mol = 0.0433641 eV
                values.append(gauss_energy_ev)
            
            formatted_values = [f"{value:.2f}" for value in values]
            print(f"    {adsorbate} gauss_energy_AutoDock: {formatted_values}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'gauss_energy_AutoDock': values,
            })
            descriptor_data.append(df)

        self.df_gauss_energy = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_gauss_energy)
    
    
    # https://pubs.acs.org/doi/10.1021/ci300604z
    def calculate_repulsion_energy_AutoDock(self, o=0.4):
        print("\n--- Calculating repulsion energy descriptor")
        descriptor_data = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions

                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS and not name H*')
                water_atoms = config_mda.universe.select_atoms('resname HOH and not name H*')

                repulsion_energy = 0.0  # Initialize repulsion energy

                # Calculate repulsion energy for each atom pair and accumulate
                for atom1 in adsorbate_atoms:
                    for atom2 in water_atoms:
                        distance = distance_array(atom1.position, atom2.position, box=box)[0][0]
                        
                        atom1_type = int(atom1.type)
                        atom2_type = int(atom2.type)
                        sigma_i = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom1_type]
                        sigma_j = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom2_type]
                        d_opt = sigma_i + sigma_j
                        
                        atom1_name = atom1.name[0]
                        atom2_name = atom2.name[0]
                        ATOM_TYPE_TO_RADIUS = {'C': 1.7, 'O': 1.52}
                        r_i = ATOM_TYPE_TO_RADIUS[atom1_name]
                        r_j = ATOM_TYPE_TO_RADIUS[atom2_name]
                        d_opt = r_i + r_j
                        
                        d_diff = distance - d_opt

                        # Calculate repulsion energy
                        if d_diff < o:
                            repulsion_energy_term = (d_diff - o) ** 2
                        else:
                            repulsion_energy_term = 0
                        
                        repulsion_energy += repulsion_energy_term

                # Convert repulsion energy from arbitrary units to eV (assuming initial unit is kcal/mol)
                repulsion_energy_ev = repulsion_energy * 0.0433641  # 1 kcal/mol = 0.0433641 eV
                values.append(repulsion_energy_ev)
                
            formatted_values = [f"{value:.2f}" for value in values]
            print(f"    {adsorbate} repulsion_energy_AutoDock: {formatted_values}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'repulsion_energy_AutoDock': values,
            })
            descriptor_data.append(df)

        self.df_repulsion_energy = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_repulsion_energy)

    
    # https://pubs.acs.org/doi/10.1021/ci300604z
    def calculate_van_der_Waals_energy_AutoDock(self):
        print("\n--- Calculating van der Waals energy descriptor")
        descriptor_data = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions

                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS and not name H*')
                water_atoms = config_mda.universe.select_atoms('resname HOH and not name H*')

                vdw_energy = 0.0  # kcal/mol

                for atom1 in adsorbate_atoms:
                    for atom2 in water_atoms:
                        distance = distance_array(atom1.position, atom2.position, box=box)[0][0]
                        
                        atom1_name = atom1.name[0]
                        atom2_name = atom2.name[0]
                        ATOM_TYPE_TO_RADIUS = {'C': 1.9, 'O': 1.7}
                        r_i = ATOM_TYPE_TO_RADIUS[atom1_name]
                        r_j = ATOM_TYPE_TO_RADIUS[atom2_name]
                        d_o = r_i + r_j

                        vdw_energy_term = (d_o / distance) ** 8 - 2 * (d_o / distance) ** 4
                        vdw_energy += vdw_energy_term

                vdw_energy_ev = vdw_energy * 0.0433641  # 1 kcal/mol = 0.0433641 eV
                values.append(vdw_energy_ev)
                
            formatted_values = [f"{value:.2f}" for value in values]
            print(f"    {adsorbate} vdw_energy_AutoDock: {formatted_values}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'vdw_energy_AutoDock': values,
            })
            descriptor_data.append(df)

        self.df_vdw_energy = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_vdw_energy)
    
    
    # https://pubs.acs.org/doi/10.1021/ci300604z
    def calculate_hydrophobic_AutoDock(self, b=1.5, charge_threshold=0.2):

        def hydrophobic(d_diff, not_hydrophobic_a1, not_hydrophobic_a2, b=b):
            if not_hydrophobic_a1 or not_hydrophobic_a2:
                return 0
            elif d_diff < 0.5:
                return 1
            elif d_diff >= b:
                return 0
            else:
                return (d_diff - b) / (0.5 - b)

        def non_hydrophobic(d_diff, is_hydrophobic_a1, is_hydrophobic_a2):
            if is_hydrophobic_a1 or is_hydrophobic_a2:
                return 0
            elif d_diff < 0.5:
                return 1
            elif d_diff >= 1.5:
                return 0
            else:
                return 1.5 - d_diff

        def is_hydrophobic(atom, charge_threshold):
            # print('abs(atom.charge): ', abs(atom.charge))
            return abs(atom.charge) < charge_threshold

        print("\n--- Calculating hydrophobic and non_hydrophobic descriptors")
        descriptor_data = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            hydrophobic_values = []
            non_hydrophobic_values = []

            for frame_index in self.frame_indices:
                config_mda.universe.trajectory[frame_index]
                box = config_mda.universe.trajectory.ts.dimensions

                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS and not name H*')
                water_atoms = config_mda.universe.select_atoms('resname HOH and not name H*')

                hydrophobic_value = 0.0
                non_hydrophobic_value = 0.0

                for atom1 in adsorbate_atoms:
                    for atom2 in water_atoms:
                        distance = distance_array(atom1.position, atom2.position, box=box)[0][0]

                        atom1_type = int(atom1.type)
                        atom2_type = int(atom2.type)
                        sigma_i = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom1_type]
                        sigma_j = ATOM_TYPE_TO_LJ_SIGMA[adsorbate][atom2_type]
                        d_opt = sigma_i + sigma_j
                        
                        atom1_name = atom1.name[0]
                        atom2_name = atom2.name[0]
                        ATOM_TYPE_TO_RADIUS = {'C': 1.9, 'O': 1.7}
                        r_i = ATOM_TYPE_TO_RADIUS[atom1_name]
                        r_j = ATOM_TYPE_TO_RADIUS[atom2_name]
                        d_opt = r_i + r_j
                        
                        d_diff = distance - d_opt

                        not_hydrophobic_a1 = not is_hydrophobic(atom1, charge_threshold)
                        not_hydrophobic_a2 = not is_hydrophobic(atom2, charge_threshold)
                        is_hydrophobic_a1 = is_hydrophobic(atom1, charge_threshold)
                        is_hydrophobic_a2 = is_hydrophobic(atom2, charge_threshold)

                        hydrophobic_value += hydrophobic(d_diff, not_hydrophobic_a1, not_hydrophobic_a2, b)
                        non_hydrophobic_value += non_hydrophobic(d_diff, is_hydrophobic_a1, is_hydrophobic_a2)

                hydrophobic_values.append(hydrophobic_value)
                non_hydrophobic_values.append(non_hydrophobic_value)

            formatted_hydrophobic_values = [f"{value:.2f}" for value in hydrophobic_values]
            formatted_non_hydrophobic_values = [f"{value:.2f}" for value in non_hydrophobic_values]
            print(f"    {adsorbate} hydrophobic: {formatted_hydrophobic_values}")
            print(f"    {adsorbate} non_hydrophobic: {formatted_non_hydrophobic_values}")

            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate] * len(self.frame_indices),
                'adsorbate_name': [adsorbate_name] * len(self.frame_indices),
                'config': list(range(len(self.frame_indices))),
                'hydrophobic_AutoDock': hydrophobic_values,
                'non_hydrophobic_AutoDock': non_hydrophobic_values
            })
            descriptor_data.append(df)

        self.df_hydrophobic = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_hydrophobic)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def combine_dataframes(self):
        if self.verbose:
            print("\n--- Combining DataFrames for All Descriptors")

        if not self.descriptor_dataframes:
            raise ValueError("No descriptor dataframes to combine")

        self.df = reduce(lambda left, right: pd.merge(left, right, on=['adsorbate', 'adsorbate_name', 'config'], how='outer'), self.descriptor_dataframes)
        
        if self.save_data:
            self.df.to_csv(self.csv_file, index=False)
        if self.verbose:
            print(self.df)
        return self.df



if __name__ == "__main__":
    
    csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_450_energy_descriptors1.csv")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    'A01',
                    'A14',
                    # 'A22',
                    # 'A44',
                    # 'A56',
                      ]

    adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    energy_descriptors = calculateEnergyDescriptors(adsorbate_list = adsorbate_list,
                                                  verbose = True,
                                                  save_data = True,  # True, False
                                                  csv_file = csv_file,
                                                  )




