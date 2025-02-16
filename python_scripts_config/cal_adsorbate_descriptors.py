"""
cal_adsorbate_descriptors.py
calculate the molecular descriptors for machine learning model.
It will read MD configurations from MD Analysis,
and finally write descriptors information into CSV files.

"""

import os
from functools import reduce

import numpy as np
import pandas as pd

from morfeus import read_xyz, read_geometry
from morfeus import Dispersion, XTB

from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths
from core.global_vars_lammps import ATOM_TYPE_TO_LJ_EPSILON, ATOM_TYPE_TO_LJ_SIGMA

from read_md_config_mdanalysis import configMDAnalysis
from read_md_config_mdtraj import configMDTraj
from read_md_config_ase import universeAse

from MDAnalysis.lib.distances import distance_array

class calculateAdsorbateDescriptors:

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
            
            # xyz file
            xyz_file = os.path.join(get_paths('simulation_path'), 'xyz_file_adsorbate_only', adsorbate+'.xyz')
            self.dict_xyz[adsorbate] = xyz_file
            
            
            

        
        '''
        Adsorbate Environment Descriptors
        '''
        # Calculate molecular Weight for Adsorbate
        self.calculate_adsorbate_weight()
        
        # Calculate Number of Atoms in Adsorbate
        self.calculate_adsorbate_C_count()
        self.calculate_adsorbate_H_count()
        self.calculate_adsorbate_O_count()
        self.calculate_adsorbate_non_H_count()
        self.calculate_adsorbate_hydroxyl_count()
        
        # Calculate the distance of oxygen atom in adsorbate to the pt surface
        self.calculate_adsorbate_oxygen_distance()
        
        # mdanalysis descriptors
        self.calculate_adsorbate_radius_of_gyration()
        self.calculate_adsorbate_gyration_moments()
        self.calculate_adsorbate_shape_parameter()
        self.calculate_adsorbate_dipole_mdanalysis()
        self.calculate_adsorbate_quadrupole_mdanalysis()
        
        # MORFEUS Descriptors
        self.calculate_adsorbate_dispersion_p()
        self.calculate_adsorbate_ionization_potential()
        self.calculate_adsorbate_ionization_potential_corrected()
        self.calculate_adsorbate_electron_affinity()
        self.calculate_adsorbate_homo()
        self.calculate_adsorbate_lumo()
        self.calculate_adsorbate_dipole_morfeus()
        self.calculate_adsorbate_electrophilicity()
        self.calculate_adsorbate_nucleophilicity()
        
        # Charge descriptors
        self.calculate_adsorbate_total_charge()
        self.calculate_adsorbate_max_charge()
        self.calculate_adsorbate_min_charge()
        self.calculate_adsorbate_sum_negative_charges()
        self.calculate_adsorbate_sum_positive_charges()
        self.calculate_adsorbate_sum_abs_qmin_qmax()
        self.calculate_adsorbate_sum_abs_charges()
        self.calculate_adsorbate_sum_oxygen_charges()
        
        
        # Get adsorbate Lennard Jones Parameters
        self.calculate_adsorbate_LJ_sigma()
        self.calculate_adsorbate_LJ_epsilon()
        
        
        ## Combine All dataframes
        self.combine_dataframes()
    
        if self.save_data == True:
            # and len(self.df) > 449:
            self.df.to_csv(self.csv_file, header=True, index=False, encoding='utf-8')
            print(f"--- Descriptors data saved to {self.csv_file}")
    
    
    
    def calculate_single_descriptor(self, calculation_func, descriptor_name, config_dict):
        if self.verbose:
            print(f"\n--- Calculating {descriptor_name.replace('_', ' ')} for Adsorbates {self.adsorbate_list}")

        descriptor_data = []
        
        for adsorbate, config in config_dict.items():
            value = calculation_func(config)
            print(f"    {descriptor_name.replace('_', ' ')} of adsorbate {adsorbate}: {value:.3f}")
            
            if config_dict in [self.dict_xyz, self.dict_pdb]:
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
    
    
    def calculate_adsorbate_weight(self):
        def weight_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            return atom_group.atoms.total_mass()
        return self.calculate_single_descriptor(weight_calculator, 'adsorbate_MW', self.dict_mdanalysis_config)

    
    def calculate_adsorbate_C_count(self):
        def C_count_calculator(config):
            atoms = config.atoms_list[0]
            mol_ids = atoms.get_array('mol-id')
            adsorbate_atoms = [atom for atom, mol_id in zip(atoms, mol_ids) if atom.number == 6 and mol_id == 2]
            return len(adsorbate_atoms)
        return self.calculate_single_descriptor(C_count_calculator, 'adsorbate_C_count', self.dict_ase_universe)

    
    def calculate_adsorbate_H_count(self):
        def H_count_calculator(config):
            atoms = config.atoms_list[0]
            mol_ids = atoms.get_array('mol-id')
            adsorbate_atoms = [atom for atom, mol_id in zip(atoms, mol_ids) if atom.number == 1 and mol_id == 2]
            return len(adsorbate_atoms)
        return self.calculate_single_descriptor(H_count_calculator, 'adsorbate_H_count', self.dict_ase_universe)

    
    def calculate_adsorbate_O_count(self):
        def O_count_calculator(config):
            atoms = config.atoms_list[0]
            mol_ids = atoms.get_array('mol-id')
            adsorbate_atoms = [atom for atom, mol_id in zip(atoms, mol_ids) if atom.number == 8 and mol_id == 2]
            return len(adsorbate_atoms)
        return self.calculate_single_descriptor(O_count_calculator, 'adsorbate_O_count', self.dict_ase_universe)

    
    def calculate_adsorbate_non_H_count(self):
        def non_H_count_calculator(config):
            atoms = config.atoms_list[0]
            mol_ids = atoms.get_array('mol-id')
            adsorbate_atoms = [atom for atom, mol_id in zip(atoms, mol_ids) if atom.number != 1 and mol_id == 2]
            return len(adsorbate_atoms)
        return self.calculate_single_descriptor(non_H_count_calculator, 'adsorbate_non_H_count', self.dict_ase_universe)


    def calculate_adsorbate_hydroxyl_count(self):
        def hydroxyl_count_calculator(config):
            hydroxyl_count = 0
            # Select all oxygen and hydrogen atoms in ADS
            oxygen_atoms = config.universe.select_atoms('resname ADS and name O*')
            print('oxygen_atoms: ', oxygen_atoms)
            hydrogen_atoms = config.universe.select_atoms('resname ADS and name H*')
            # Define a threshold distance to identify O-H bonds
            threshold_distance = 1.1  # Å
            if len(oxygen_atoms) == 0 or len(hydrogen_atoms) == 0:
                return hydroxyl_count
            # Calculate the distance matrix
            distances = distance_array(oxygen_atoms.positions, hydrogen_atoms.positions, box=config.universe.dimensions)
            # Check if any distance is less than the threshold distance
            for distance_row in distances:
                if np.any(distance_row < threshold_distance):
                    hydroxyl_count += 1
            return hydroxyl_count
        return self.calculate_single_descriptor(hydroxyl_count_calculator, 'adsorbate_hydroxyl_count', self.dict_mdanalysis_config)



    def calculate_adsorbate_oxygen_distance(self):
        def oxygen_distance_calculator(config):
            atom_group_oxygen = config.universe.select_atoms('resname ADS and name O*')
            oxygen_distances = atom_group_oxygen.positions[:, 2]
            if len(oxygen_distances) == 0:
                return 0
            else:
                return np.max(oxygen_distances) - 4.581048
        return self.calculate_single_descriptor(oxygen_distance_calculator, 'adsorbate_oxygen_distance', self.dict_mdanalysis_config)


    def calculate_adsorbate_radius_of_gyration(self):
        def radius_of_gyration_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            if len(atom_group) == 0:
                return 0
            # https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup.radius_of_gyration
            return atom_group.radius_of_gyration(unwrap = True)
        return self.calculate_single_descriptor(radius_of_gyration_calculator, 'adsorbate_radius_of_gyration', self.dict_mdanalysis_config)


    def calculate_adsorbate_gyration_moments(self):
        def gyration_moments_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            # https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup.gyration_moments
            gyration_moments = atom_group.gyration_moments()
            gyration = np.linalg.norm(gyration_moments)
            return gyration
        return self.calculate_single_descriptor(gyration_moments_calculator, 'adsorbate_gyration_moments', self.dict_mdanalysis_config)


    def calculate_adsorbate_shape_parameter(self):
        def shape_parameter_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            # https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup.shape_parameter
            shape_parameter = atom_group.shape_parameter()
            return shape_parameter
        return self.calculate_single_descriptor(shape_parameter_calculator, 'adsorbate_shape_parameter', self.dict_mdanalysis_config)
    
    
    def calculate_adsorbate_dipole_mdanalysis(self):
        def dipole_mdanalysis_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            # https://docs.mdanalysis.org/1.0.0/documentation_pages/transformations/wrap.html
            # https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup.dipole_moment
            dipole_magnitude = atom_group.dipole_moment()
            return dipole_magnitude

        return self.calculate_single_descriptor(dipole_mdanalysis_calculator, 'adsorbate_dipole_mdanalysis', self.dict_mdanalysis_config)
    
    
    def calculate_adsorbate_quadrupole_mdanalysis(self):
        def quadrupole_mdanalysis_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            # https://docs.mdanalysis.org/1.0.0/documentation_pages/transformations/wrap.html
            # https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup.quadrupole_moment
            quadrupole_moment = atom_group.quadrupole_moment()
            return quadrupole_moment

        return self.calculate_single_descriptor(quadrupole_mdanalysis_calculator, 'adsorbate_quadrupole_mdanalysis', self.dict_mdanalysis_config)
    
        
    # https://digital-chemistry-laboratory.github.io/morfeus/dispersion.html
    def calculate_adsorbate_dispersion_p(self):
        def dispersion_calculator(config):
            elements, coordinates = read_geometry(config)
            disp = Dispersion(elements, coordinates)
            self.disp = disp
            p_int = disp.p_int
            return p_int
        return self.calculate_single_descriptor(dispersion_calculator, 'adsorbate_dispersion_p', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_ionization_potential(self):
        def ip_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            return xtb.get_ip(corrected=False)
        return self.calculate_single_descriptor(ip_calculator, 'adsorbate_ionization_potential', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_ionization_potential_corrected(self):
        def ip_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            return xtb.get_ip(corrected=True)
        return self.calculate_single_descriptor(ip_calculator, 'adsorbate_ionization_potential_corr', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_electron_affinity(self):
        def ea_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            return xtb.get_ea()
        return self.calculate_single_descriptor(ea_calculator, 'adsorbate_electron_affinity', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_homo(self):
        def homo_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            return xtb.get_homo()
        return self.calculate_single_descriptor(homo_calculator, 'adsorbate_homo', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_lumo(self):
        def lumo_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            return xtb.get_lumo()
        return self.calculate_single_descriptor(lumo_calculator, 'adsorbate_lumo', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_dipole_morfeus(self):
        def dipole_morfeus_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            dipole = xtb.get_dipole()
            return np.linalg.norm(dipole)
        return self.calculate_single_descriptor(dipole_morfeus_calculator, 'adsorbate_dipole_morfeus', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_electrophilicity(self):
        def electrophilicity_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            return xtb.get_global_descriptor("electrophilicity", corrected=True)
        return self.calculate_single_descriptor(electrophilicity_calculator, 'adsorbate_electrophilicity', self.dict_xyz)
    
    # https://digital-chemistry-laboratory.github.io/morfeus/xtb.html
    def calculate_adsorbate_nucleophilicity(self):
        def nucleophilicity_calculator(config):
            elements, coordinates = read_xyz(config)
            xtb = XTB(elements, coordinates)
            return xtb.get_global_descriptor("nucleophilicity", corrected=True)
        return self.calculate_single_descriptor(nucleophilicity_calculator, 'adsorbate_nucleophilicity', self.dict_xyz)


    
    
    def calculate_adsorbate_total_charge(self):
        def total_charge_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            return atom_group.total_charge()

        return self.calculate_single_descriptor(total_charge_calculator, 'adsorbate_charge_total', self.dict_mdanalysis_config)

    def calculate_adsorbate_max_charge(self):
        def max_charge_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            return np.max(atom_group.charges)

        return self.calculate_single_descriptor(max_charge_calculator, 'adsorbate_charge_max', self.dict_mdanalysis_config)

    def calculate_adsorbate_min_charge(self):
        def min_charge_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            return np.min(atom_group.charges)

        return self.calculate_single_descriptor(min_charge_calculator, 'adsorbate_charge_min', self.dict_mdanalysis_config)

    def calculate_adsorbate_sum_negative_charges(self):
        def sum_negative_charges_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            negative_charges = atom_group.charges[atom_group.charges < 0]
            return np.sum(negative_charges)

        return self.calculate_single_descriptor(sum_negative_charges_calculator, 'adsorbate_sum_negative_charges', self.dict_mdanalysis_config)

    def calculate_adsorbate_sum_positive_charges(self):
        def sum_positive_charges_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            positive_charges = atom_group.charges[atom_group.charges > 0]
            return np.sum(positive_charges)

        return self.calculate_single_descriptor(sum_positive_charges_calculator, 'adsorbate_sum_positive_charges', self.dict_mdanalysis_config)

    def calculate_adsorbate_sum_abs_qmin_qmax(self):
        def sum_abs_qmin_qmax_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            q_min = np.min(atom_group.charges)
            q_max = np.max(atom_group.charges)
            return np.abs(q_min) + np.abs(q_max)

        return self.calculate_single_descriptor(sum_abs_qmin_qmax_calculator, 'adsorbate_sum_abs_qmin_qmax', self.dict_mdanalysis_config)
    
    def calculate_adsorbate_sum_abs_charges(self):
        def sum_abs_charges_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS')
            abs_charges = np.abs(atom_group.charges)
            return np.sum(abs_charges)

        return self.calculate_single_descriptor(sum_abs_charges_calculator, 'adsorbate_sum_abs_charges', self.dict_mdanalysis_config)
    
    def calculate_adsorbate_sum_oxygen_charges(self):
        def sum_oxygen_charges_calculator(config):
            atom_group = config.universe.select_atoms('resname ADS and name O*')
            oxygen_charges = atom_group.charges
            if np.any(oxygen_charges > 0):
                raise ValueError("Some oxygen atoms have positive charges")
            return np.sum(oxygen_charges)

        return self.calculate_single_descriptor(sum_oxygen_charges_calculator, 'adsorbate_sum_oxygen_charges', self.dict_mdanalysis_config)
    
            
    
    def calculate_adsorbate_LJ_sigma(self):
        def avg_LJ_sigma_calculator(config):
            atoms = config.atoms_list[0]
            mol_ids = atoms.get_array('mol-id')
            atom_types = atoms.get_array('type')
            adsorbate_atom_types = [atom_type for atom_type, mol_id in zip(atom_types, mol_ids) if mol_id == 2]
            sigmas = [ATOM_TYPE_TO_LJ_SIGMA[config.adsorbate].get(atom_type) for atom_type in adsorbate_atom_types]
            avg_sigma = np.mean(sigmas)
            return avg_sigma
        return self.calculate_single_descriptor(avg_LJ_sigma_calculator, 'adsorbate_LJ_sigma', self.dict_ase_universe)

    def calculate_adsorbate_LJ_epsilon(self):
        def avg_LJ_epsilon_calculator(config):
            atoms = config.atoms_list[0]
            mol_ids = atoms.get_array('mol-id')
            atom_types = atoms.get_array('type')
            adsorbate_atom_types = [atom_type for atom_type, mol_id in zip(atom_types, mol_ids) if mol_id == 2]
            epsilons = [ATOM_TYPE_TO_LJ_EPSILON[config.adsorbate].get(atom_type) for atom_type in adsorbate_atom_types]
            avg_epsilon = np.mean(epsilons)
            return avg_epsilon
        return self.calculate_single_descriptor(avg_LJ_epsilon_calculator, 'adsorbate_LJ_epsilon', self.dict_ase_universe)





    def combine_dataframes(self):
        if self.verbose:
            print("\n--- Combining DataFrames for All Descriptors ---")

        if not self.descriptor_dataframes:
            raise ValueError("No descriptor dataframes to combine")

        self.df = reduce(lambda left, right: pd.merge(left, right, on = ['adsorbate', 'adsorbate_name', 'config'], how='outer'), self.descriptor_dataframes)
        return self.df
       
    
            
    

if __name__ == "__main__":
    
    csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_450_adsorbate_descriptors_new.csv")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    # 'A01',
                    # 'A14',
                    # 'A22',
                    # 'A44',
                    # 'A56',
                    '254',
                    '264',
                      ]

    # adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    descriptors = calculateAdsorbateDescriptors(adsorbate_list = adsorbate_list,
                                        verbose = True,
                                        save_data = True,  # True, False
                                        csv_file = csv_file,
                                        )
