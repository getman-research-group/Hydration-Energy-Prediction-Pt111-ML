"""
    Identify hydrogen bonds based on cutoffs for the Donor-Acceptor distance and angle.

"""
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from read_md_config_mdanalysis import configMDAnalysis
from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.global_vars_lammps import ATOM_TYPE_TO_ELEMENT, ATOM_TYPE_TO_LJ_EPSILON, ATOM_TYPE_TO_LJ_SIGMA
from core.path import get_paths

from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis import transformations
from MDAnalysis.lib.mdamath import triclinic_box, triclinic_vectors
from MDAnalysis.lib.distances import distance_array, calc_angles

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No hydrogen bonds were found given angle of")



# Define the hydrogen bond score function
# https://pubs.acs.org/doi/full/10.1021/ci050422y
def fermi_function(x, a, b):
    fermi = 1 / (1 + np.exp(-a * x + b))
    return fermi

def hydrogen_bond_score_SLICK(r, alpha, r_ideal=1.85, alpha_ideal=180, a_r=0.25, b_r=0.65, a_alpha=30, b_alpha=80):

    delta_r = abs(r - r_ideal)
    delta_alpha = abs(alpha - alpha_ideal)
    
    f_r = fermi_function(delta_r, a_r, b_r)
    f_alpha = fermi_function(delta_alpha, a_alpha, b_alpha)
    
    score = f_r * f_alpha
    return score


# https://pubs.acs.org/doi/10.1021/acs.jcim.1c01537
def hydrogen_bond_score_AA_score(hb_OO_distance):
    term = (hb_OO_distance / 2.6) ** 6
    score = (1 / (1 + term)) / 0.58
    return score

# https://pubs.acs.org/doi/10.1021/ci300493w
def hydrogen_bond_ID_score(d_AD, theta):
    r0 = 2.8
    theta0 = 135
    term1 = (r0 / d_AD) ** 12
    term2 = (r0 / d_AD) ** 6
    cos_term = np.cos(np.radians(theta - theta0)) ** 2
    score = (term1 - 2 * term2) * cos_term
    return score

# https://academic.oup.com/bioinformatics/article/30/12/1674/2748148
# https://pubs.acs.org/doi/abs/10.1021/jm00145a002
def hydrogen_bond_score_Goodford(r_ij, hb_angle, C = 3855, D = 738):
    term1 = C / (r_ij ** 6)
    term2 = D / (r_ij ** 4)
    cos_term = np.cos(np.radians(hb_angle)) ** 4
    score = (term1 - term2) * cos_term
    return score

# https://doi.org/10.1002/jcc.20634
def hydrogen_bond_score_Goodford_2(r_ij, C = 3855, D = 738):
    term1 = C / (r_ij ** 12)
    term2 = D / (r_ij ** 10)
    score = term1 - term2
    return score

# https://link.springer.com/article/10.1023/A:1016357811882
# https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22058
def hydrogen_bond_score_SFCscore(hb_OO_distance, theta1, theta2):
    d0 = 1.7 * 2
    dist_score = distance_score(hb_OO_distance, d0)
    angle_score1 = angle_score(theta1)
    angle_score2 = angle_score(theta2)
    total_score = dist_score * angle_score1 * angle_score2
    return total_score

def distance_score(d, d0):
    if d <= d0 - 0.7:
        return 1.0
    elif d0 - 0.7 < d <= d0:
        return (1 / 0.7) * (d0 - d)
    else:
        return 0.0

def angle_score(theta):
    if theta <= 60:
        return 0.0
    elif 60 < theta < 120:
        return (theta - 60) / 60
    else:
        return 1.0









class findHbondsDetails:
    
    def __init__(self,
                 adsorbate_list,
                 save_csv,
                 ):
        
        self.adsorbate_list = adsorbate_list
        self.save_csv = save_csv
        
        self.dict_mdanalysis_config = {}
        
        ## Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
                
            ## Load the configuration
            config_mda = configMDAnalysis(adsorbate)
            
            ## Store Configurations for adsorbates
            self.dict_mdanalysis_config[adsorbate] = config_mda

        #  Find the water molecules that are hydrogen bonded to the adsorbate
        self.df_water_hbonds = self.find_hbonds_numbers()

        # Load the MDAnalysis universe and calculate hydrogen bonds
        self.load_mda_universe()
        
        # Summarize the hydrogen bond information
        self.summarize_hbonds()
        
        if self.save_csv and len(self.df_final) > 449:
            csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_450_hbonds_descriptors.csv")
            self.df_final.to_csv(csv_file, index=False)
        
    
    
    def find_hbonds_numbers(self):
        file_dir = get_paths('simulation_path')
        df_water_hbonds = []

        for adsorbate in self.adsorbate_list:
            print(f'--- Finding hydrogen bonded water molecules for adsorbate {adsorbate} ---')
            input_dir_path = os.path.join(file_dir, 'vasp_eint', adsorbate, '3-vasp-eint')
            adsorbate_data = []
            config_hbond_counts = []

            for config in range(5):
                poscar_path = os.path.join(input_dir_path, str(config), 'POSCAR')
                relaxed_atoms = []
                box_vectors = None

                with open(poscar_path) as file:
                    lines = file.readlines()
                    # read the box vectors
                    box_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])
                    for i, line in enumerate(lines[8:]):
                        if line.strip().endswith('T'):
                            coords = list(map(float, line.split()[:3]))
                            coords_cartesian = np.dot(coords, box_vectors)
                            relaxed_atoms.append((config, coords, coords_cartesian))

                num_hbonds = len(relaxed_atoms) // 3
                config_hbond_counts.append((adsorbate, config, num_hbonds))
                adsorbate_data.extend(relaxed_atoms)

            df = pd.DataFrame(adsorbate_data, columns=['config', 'coords', 'coords_cartesian'])
            df['adsorbate'] = adsorbate
            df_water_hbonds.append(df)

            # print the number of hydrogen bonded water molecules for each adsorbate and configuration
            for ads, conf, count in config_hbond_counts:
                print(f'    adsorbate: {ads}, Config: {conf}, Number of hydrogen bonded water molecules: {count}')

        df_water_hbonds = pd.concat(df_water_hbonds, ignore_index=True)
        
        return df_water_hbonds


    def load_mda_universe(self):
        frame_indices = [230, 290, 350, 410, 470]
        frame_to_config = {230: 0, 290: 1, 350: 2, 410: 3, 470: 4}
        dataframes = []

        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            for frame_index in frame_indices:
                config = frame_to_config[frame_index]

                # get the water molecules for the current adsorbate and configuration
                df_poscar_adsorbate = self.df_water_hbonds[self.df_water_hbonds['adsorbate'] == adsorbate]
                df_poscar_config = df_poscar_adsorbate[df_poscar_adsorbate['config'] == config]
                
                number_of_hbonds = len(df_poscar_config)//3
                # calculate hydrogen bonds
                print(f'\n--- adsorbate {adsorbate}, config {config}, number of water molecules: {number_of_hbonds} ---')
                hb_info = self.calculate_hbonds(adsorbate, config_mda, df_poscar_config, frame_index)
                
                # Store the hydrogen bond information
                dataframes.append(hb_info)

        self.df = pd.concat(dataframes, ignore_index=True)












    def calculate_hbonds(self, adsorbate, config_mda, df_poscar_config, frame_index):
        hb_data = []
        
        # Locate the specific frame
        config_mda.universe.trajectory[frame_index]

        # Get water molecule atoms from MDAnalysis universe
        water_atoms = config_mda.universe.select_atoms('resname HOH')
        water_coords = water_atoms.positions
        water_ids = water_atoms.indices  # Get all the water atom indices, eg., 36-107
        
        # Set the threshold for matching the relaxed water coordinates
        threshold = 0.0001

        # Set to track processed water molecules
        processed_water_molecules = set()

        for index, row in df_poscar_config.iterrows():
            
            relaxed_coords = np.array(row['coords_cartesian'])
            
            # Find the matching water atom for relaxed_coords
            for i, water_coord in enumerate(water_coords):
                if np.allclose(relaxed_coords, water_coord, atol = threshold):
                    water_atom_id = water_ids[i]
                    water_molecule = config_mda.universe.atoms[water_atom_id].residue
                    if water_molecule.resid in processed_water_molecules:
                        continue
                    processed_water_molecules.add(water_molecule.resid)
                    break
            else:
                # If no matching atom is found, skip
                continue
            # print('water_molecule: ', water_molecule)
            
            # Find the closest adsorbate atom to the water molecule
            adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
            water_atoms = water_molecule.atoms
            # print('water_atoms: ', water_atoms)
            
            distances_all_atoms = distance_array(adsorbate_atoms.positions,
                                                 water_atoms.positions,
                                                 box = config_mda.universe.dimensions)
            
            # Sort distances and iterate to find valid hydrogen bonds
            sorted_distance_indices = np.dstack(np.unravel_index(np.argsort(distances_all_atoms.ravel()), distances_all_atoms.shape))[0]
            
            valid_hbond_found = False

            for closest_adsorbate_atom_idx, closest_water_atom_idx in sorted_distance_indices:
                closest_adsorbate_atom = adsorbate_atoms[closest_adsorbate_atom_idx]
                closest_water_atom = water_atoms[closest_water_atom_idx]
                
                # Check for water as donor
                if closest_water_atom.name.startswith('H') and closest_adsorbate_atom.name.startswith('O'):
                    hb_distance = distances_all_atoms[closest_adsorbate_atom_idx][closest_water_atom_idx]
                    
                    donor_atom = closest_water_atom.bonded_atoms[0]
                    
                    # Calculate the angle between the hydrogen bond
                    hb_angle_rad = calc_angles(
                        donor_atom.position,                    # water O, hbond donor
                        closest_water_atom.position,            # water H, hbond hydrogen
                        closest_adsorbate_atom.position,        # adsorbate O, hbond acceptor
                        box=config_mda.universe.dimensions
                    )
                    hb_angle = np.degrees(hb_angle_rad)
                    
                    # Calculate O...O distance using distance_array
                    hb_OO_distance = distance_array(closest_adsorbate_atom.position,    # Acceptor
                                                    donor_atom.position,                # Donor
                                                    box = config_mda.universe.dimensions)[0][0]
                    
                    # Get the charges of the atoms involved in the hydrogen bond
                    donor_charge = donor_atom.charge                    # water O, hbond donor
                    hydrogen_charge = closest_water_atom.charge         # water H, hbond hydrogen
                    acceptor_charge = closest_adsorbate_atom.charge     # adsorbate O, hbond acceptor
                    
                    
                    # Calculate the hydrogen bond score using SLICK
                    hb_score_slick = hydrogen_bond_score_SLICK(hb_OO_distance, hb_angle)
                    
                    # Calculate the hydrogen bond score using AA score
                    hb_score_aa_score = hydrogen_bond_score_AA_score(hb_OO_distance)
                    
                    # Calculate the hydrogen bond score using ID score
                    distances_to_atoms = distance_array(closest_adsorbate_atom.position,
                                                        adsorbate_atoms.positions,
                                                        box = config_mda.universe.dimensions)
                    within_cutoff_indices = [i for i, distance in enumerate(distances_to_atoms[0]) if 0.1 < distance <= 1.5]
                    max_theta = 0
                    acceptor_root_atom_position = None
                    for i in within_cutoff_indices:
                        atom = adsorbate_atoms[i]
                        current_theta = calc_angles(donor_atom.position,                    # Donor, water O
                                                    closest_adsorbate_atom.position,        # Acceptor
                                                    atom.position,                          # Acceptor root
                                                    box = config_mda.universe.dimensions)
                        if current_theta > max_theta:
                            max_theta = current_theta
                            acceptor_root_atom_position = atom.position
                    if acceptor_root_atom_position is None:
                        theta_2 = 180.0
                    else:
                        theta_2_rad = calc_angles(donor_atom.position,                  # Donor, water O
                                              closest_adsorbate_atom.position,      # Acceptor
                                              acceptor_root_atom_position,          # Acceptor connected atom
                                              box = config_mda.universe.dimensions)
                        theta_2 = np.degrees(theta_2_rad)
                    hb_score_id_score = hydrogen_bond_ID_score(hb_OO_distance, theta_2)

                    
                    # Calculate the hydrogen bond score using SFCscore
                    hydrogen_positions = [atom.position for atom in donor_atom.bonded_atoms if not np.allclose(atom.position, closest_water_atom.position)]
                    donor_root_position = hydrogen_positions[0]
                    theta_1_rad = calc_angles(
                                        donor_root_position,                # donor root
                                        donor_atom.position,                # donor
                                        closest_adsorbate_atom.position,    # acceptor
                                        box=config_mda.universe.dimensions
                                    )
                    theta_1 = np.degrees(theta_1_rad)
                    hb_score_SFCscore = hydrogen_bond_score_SFCscore(hb_OO_distance, theta_1, theta_2)
                    
                    # Calculate the hydrogen bond score using Goodford's paper's method
                    hb_score_Goodford = hydrogen_bond_score_Goodford(hb_OO_distance, hb_angle)
                    
                    
                    valid_hbond_found = True
                    break

                # Check for adsorbate as donor
                elif closest_water_atom.name.startswith('O') and closest_adsorbate_atom.name.startswith('H'):
                    hb_distance = distances_all_atoms[closest_adsorbate_atom_idx][closest_water_atom_idx]
                    
                    # Select oxygen atoms in the adsorbate
                    adsorbate_oxygen_atoms = adsorbate_atoms.select_atoms('name O*')
                    
                    # Find the closest oxygen atom to this hydrogen in the adsorbate_atoms
                    distances_to_oxygens = distance_array(closest_adsorbate_atom.position,
                                                          adsorbate_oxygen_atoms.positions,
                                                          box=config_mda.universe.dimensions)
                    closest_adsorbate_oxygen_idx = np.argmin(distances_to_oxygens)
                    closest_adsorbate_oxygen_atom = adsorbate_oxygen_atoms[closest_adsorbate_oxygen_idx]
                    donor_atom = closest_adsorbate_oxygen_atom
                    
                    hb_angle_rad = calc_angles(
                        donor_atom.position,                # Donor, adsorbate O
                        closest_adsorbate_atom.position,    # Hydrogen, adsorbate H
                        closest_water_atom.position,        # Acceptor, water O
                        box = config_mda.universe.dimensions
                    )
                    hb_angle = np.degrees(hb_angle_rad)
                    
                    # Calculate O...O distance using distance_array
                    hb_OO_distance = distance_array(donor_atom.position,                        # Donor
                                                    closest_water_atom.position,                # Acceptor
                                                    box=config_mda.universe.dimensions)[0][0]
                    
                    # Get the charges of the atoms involved in the hydrogen bond
                    donor_charge = donor_atom.charge                    # Donor, adsorbate O
                    hydrogen_charge = closest_adsorbate_atom.charge     # Hydrogen, adsorbate H
                    acceptor_charge = closest_water_atom.charge         # Acceptor, water O
                    
                    # Calculate the hydrogen bond score using SLICK
                    hb_score_slick = hydrogen_bond_score_SLICK(hb_OO_distance, hb_angle)
                    
                    # Calculate the hydrogen bond score using AA score
                    hb_score_aa_score = hydrogen_bond_score_AA_score(hb_OO_distance)
                    
                    
                    # Calculate the hydrogen bond score using ID score
                    theta_2_rad_0 = calc_angles(
                        donor_atom.position,                            # Donor, adsorbate O
                        closest_water_atom.position,                    # Acceptor, water O
                        closest_water_atom.bonded_atoms.positions[0],   # Acceptor root 0, water H
                        box=config_mda.universe.dimensions,)
                    theta_2_0 = np.degrees(theta_2_rad_0)
                    theta_2_rad_1 = calc_angles(
                        donor_atom.position,                            # Donor, adsorbate O
                        closest_water_atom.position,                    # Acceptor, water O
                        closest_water_atom.bonded_atoms.positions[1],   # Acceptor root 1, water H
                        box=config_mda.universe.dimensions,)
                    theta_2_1 = np.degrees(theta_2_rad_1)
                    theta_2 = max(theta_2_0, theta_2_1)
                    hb_score_id_score = hydrogen_bond_ID_score(hb_OO_distance, theta_2)
                    
                    # Calculate the hydrogen bond score using SFCscore
                    cutoff_distance = 1.5
                    nearby_atoms = [atom for atom in adsorbate_atoms if atom.index != closest_adsorbate_atom.index and np.linalg.norm(donor_atom.position - atom.position) <= cutoff_distance]
                    donor_root_positions = [atom.position for atom in nearby_atoms]
                    theta_1 = 0
                    for donor_root_position in donor_root_positions:
                        current_theta = calc_angles(
                            donor_root_position,                # donor root
                            donor_atom.position,                # donor
                            closest_water_atom.position,        # acceptor
                            box=config_mda.universe.dimensions
                        )
                        if current_theta > theta_1:
                            theta_1 = current_theta
                    hb_score_SFCscore = hydrogen_bond_score_SFCscore(hb_OO_distance, theta_1, theta_2)

                    # Calculate the hydrogen bond score using Goodford's paper's method
                    hb_score_Goodford = hydrogen_bond_score_Goodford(hb_OO_distance, hb_angle)
                    
                    valid_hbond_found = True
                    break

            if not valid_hbond_found:
                print(f'No valid hydrogen bond found for adsorbate {row["adsorbate"]} config {row["config"]}')
                continue

            hb_data.append({
                'adsorbate': row['adsorbate'],
                'config': row['config'],
                'hb_distance': hb_distance,
                'hb_OO_distance': hb_OO_distance,
                'hb_angle': hb_angle,
                'donor_charge': donor_charge,
                'hydrogen_charge': hydrogen_charge,
                'acceptor_charge': acceptor_charge,
                'hb_score_slick': hb_score_slick,
                'hb_score_aa_score': hb_score_aa_score,
                'hb_score_id_score': hb_score_id_score,
                'hb_score_SFCscore': hb_score_SFCscore,
                'hb_score_Goodford': hb_score_Goodford,
            })

        return pd.DataFrame(hb_data)



    def summarize_hbonds(self):
        # 从 self.df 生成 self.df_final
        summary_data = []
        adsorbate_configs = self.df.groupby(['adsorbate', 'config'])
        
        for (adsorbate, config), group in adsorbate_configs:
            hbonds_count = len(group)
            hb_score_slick_sum = group['hb_score_slick'].sum()
            hb_score_aa_score_sum = group['hb_score_aa_score'].sum()
            hb_score_id_score_sum = group['hb_score_id_score'].sum()
            hb_score_sfc_sum = group['hb_score_SFCscore'].sum()
            hb_score_goodford_sum = group['hb_score_Goodford'].sum()
            
            summary_data.append({
                'adsorbate': adsorbate,
                'config': config,
                'hbonds': hbonds_count,
                'hb_score_slick': hb_score_slick_sum,
                'hb_score_aa_score': hb_score_aa_score_sum,
                'hb_score_id_score': hb_score_id_score_sum,
                'hb_score_SFCscore': hb_score_sfc_sum,
                'hb_score_Goodford': hb_score_goodford_sum
            })
        
        # 添加缺少的 adsorbate 和 config 信息
        for adsorbate in self.adsorbate_list:
            for config in range(5):
                if not any((item['adsorbate'] == adsorbate and item['config'] == config) for item in summary_data):
                    summary_data.append({
                        'adsorbate': adsorbate,
                        'config': config,
                        'hbonds': 0,
                        'hb_score_slick': 0.0,
                        'hb_score_aa_score': 0.0,
                        'hb_score_id_score': 0.0,
                        'hb_score_SFCscore': 0.0,
                        'hb_score_Goodford': 0.0
                    })
        
        # 转换为 DataFrame
        self.df_final = pd.DataFrame(summary_data)
        self.df_final.sort_values(by=['adsorbate', 'config'], inplace=True)
        self.df_final.reset_index(drop=True, inplace=True)




if __name__ == "__main__":

    ## Defining List of Adsorbates
    adsorbate_list = [
                        # 'A01',      # [1, 0, 0, 1, 1],
                        'A02',      # [0, 1, 1, 0, 0],
                        # 'A03',      # [0, 3, 2, 0, 0],
                        # 'A04',      # [3, 2, 2, 3, 1],
                        # 'A05',      # [0, 0, 1, 1, 1],
                        # 'A06',      # [0, 1, 2, 1, 2],
                        # 'A07',      # [2, 1, 3, 3, 1],
                        # 'A08',      # [2, 3, 1, 1, 0],
                        # 'A09',      # [2, 1, 1, 1, 1],
                        # 'A10',      # [1, 2, 1, 1, 1],
                      ]

    adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    find_hbonds_details = findHbondsDetails(adsorbate_list = adsorbate_list,
                                            save_csv = True,)
    
