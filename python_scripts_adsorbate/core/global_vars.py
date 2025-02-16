# -*- coding: utf-8 -*-
"""
global_vars.py
This code contains all global variables.

"""
## Importing Functions
import os
try:
    from path import get_base_paths
except:
    from core.path import get_base_paths
    

ATOMIC_RADII = {'H'   : 0.120, 'He'  : 0.140, 'Li'  : 0.076, 'Be' : 0.059,
                'B'   : 0.192, 'C'   : 0.170, 'N'   : 0.155, 'O'  : 0.152,
                'F'   : 0.147, 'Ne'  : 0.154, 'Na'  : 0.102, 'Mg' : 0.086,
                'Al'  : 0.184, 'Si'  : 0.210, 'P'   : 0.180, 'S'  : 0.180,
                'Cl'  : 0.181, 'Ar'  : 0.188, 'K'   : 0.138, 'Ca' : 0.114,
                'Sc'  : 0.211, 'Ti'  : 0.200, 'V'   : 0.200, 'Cr' : 0.200,
                'Mn'  : 0.200, 'Fe'  : 0.200, 'Co'  : 0.200, 'Ni' : 0.163,
                'Cu'  : 0.140, 'Zn'  : 0.139, 'Ga'  : 0.187, 'Ge' : 0.211,
                'As'  : 0.185, 'Se'  : 0.190, 'Br'  : 0.185, 'Kr' : 0.202,
                'Rb'  : 0.303, 'Sr'  : 0.249, 'Y'   : 0.200, 'Zr' : 0.200,
                'Nb'  : 0.200, 'Mo'  : 0.200, 'Tc'  : 0.200, 'Ru' : 0.200,
                'Rh'  : 0.200, 'Pd'  : 0.163, 'Ag'  : 0.172, 'Cd' : 0.158,
                'In'  : 0.193, 'Sn'  : 0.217, 'Sb'  : 0.206, 'Te' : 0.206,
                'I'   : 0.198, 'Xe'  : 0.216, 'Cs'  : 0.167, 'Ba' : 0.149,
                'La'  : 0.200, 'Ce'  : 0.200, 'Pr'  : 0.200, 'Nd' : 0.200,
                'Pm'  : 0.200, 'Sm'  : 0.200, 'Eu'  : 0.200, 'Gd' : 0.200,
                'Tb'  : 0.200, 'Dy'  : 0.200, 'Ho'  : 0.200, 'Er' : 0.200,
                'Tm'  : 0.200, 'Yb'  : 0.200, 'Lu'  : 0.200, 'Hf' : 0.200,
                'Ta'  : 0.200, 'W'   : 0.200, 'Re'  : 0.200, 'Os' : 0.200,
                'Ir'  : 0.200, 'Pt'  : 0.175, 'Au'  : 0.166, 'Hg' : 0.155,
                'Tl'  : 0.196, 'Pb'  : 0.202, 'Bi'  : 0.207, 'Po' : 0.197,
                'At'  : 0.202, 'Rn'  : 0.220, 'Fr'  : 0.348, 'Ra' : 0.283,
                'Ac'  : 0.200, 'Th'  : 0.200, 'Pa'  : 0.200, 'U'  : 0.186,
                'Np'  : 0.200, 'Pu'  : 0.200, 'Am'  : 0.200, 'Cm' : 0.200,
                'Bk'  : 0.200, 'Cf'  : 0.200, 'Es'  : 0.200, 'Fm' : 0.200,
                'Md'  : 0.200, 'No'  : 0.200, 'Lr'  : 0.200, 'Rf' : 0.200,
                'Db'  : 0.200, 'Sg'  : 0.200, 'Bh'  : 0.200, 'Hs' : 0.200,
                'Mt'  : 0.200, 'Ds'  : 0.200, 'Rg'  : 0.200, 'Cn' : 0.200,
                'Uut' : 0.200, 'Fl'  : 0.200, 'Uup' : 0.200, 'Lv' : 0.200,
                'Uus' : 0.200, 'Uuo' : 0.200,
                }

## Defining Adsorbates, Create a list contains all the possible adsorbates.
ADSORBATE_TO_NAME_DICT =  {
                            'A01':  '1-CH2OH-CHOH-CH2OH',
                            'A02':  '2-CH2OH-COH-CH2OH',
                            'A03':  '3-CHOH-COH-CH2OH',
                            'A04':  '4-CHOH-COH-CHOH',
                            'A05':  '5-COH-COH-CHOH',
                            'A06':  '6-COH-COH-COH',
                            'A07':  '7-CO-COH-CO',
                            'A08':  '7-CO-COH-COH',
                            'A09':  '8-CHOH-CH2OH',
                            'A10':  '8-CO-CHOH-CH2OH',
                            'A11':  '9-CO-COH-CH2OH',
                            'A12':  '9-COH-CH2OH',
                            'A13':  '10-CHOH-CHOH',
                            'A14':  '10-CO-CHOH-CHOH',
                            'A15':  '11-CO-COH-CHOH',
                            'A16':  '11-COH-CHOH',
                            'A17':  '12-CO-CHOH-COH',
                            'A18':  '13-CO-COH-COH',
                            'A19':  '13-COH-COH',
                            'A20':  '14-C-CHOH-CH2OH',
                            'A21':  '14-COH-CHOH-CH2OH',
                            'A22':  '15-CHOH-C-CHOH',
                            'A23':  '16-C-COH-CHOH',
                            'A24':  '17-C-CO-CH2OH',
                            'A25':  '17-COH-CO-CH2OH',
                            'A26':  '18-C-COH-COH',
                            'A27':  '0111-OH',
                            'A28':  '0211-H2O',
                            'A29':  '1101-CH',
                            'A30':  '1111-CHO',
                            'A31':  '1121-HCOO',
                            'A32':  '1122-COOH',
                            'A33':  '1201-CH2',
                            'A34':  '1211-CH2O',
                            'A35':  '1212-CHOH',
                            'A36':  '1221-HCOOH',
                            'A37':  '1301-CH3',
                            'A38':  '1401-CH4',
                            'A39':  '1411-CH3OH',
                            'A40':  '2001-C2',
                            'A41':  '2011-CCO',
                            'A42':  '2021-OCCO',
                            'A43':  '2101-C2H',
                            'A44':  '2111-CHCO',
                            'A45':  '2112-CCHO',
                            'A46':  '2113-CCOH',
                            'A47':  '2121-OCCHO',
                            'A48':  '2122-OCCOH',
                            'A49':  '2201-CH2C',
                            'A50':  '2202-CHCH',
                            'A51':  '2211-CH2CO',
                            'A52':  '2212-CHCHO',
                            'A53':  '2213-CCH2O',
                            'A54':  '2214-CHCOH',
                            'A55':  '2215-CCHOH',
                            'A56':  '2221-OCHCHO',
                            'A57':  '2222-OCCH2O',
                            'A58':  '2224-HOCHCO',
                            'A59':  '2225-HOCCOH',
                            'A60':  '2301-CH3C',
                            'A61':  '2302-CH2CH',
                            'A62':  '2311-CH3CO',
                            'A63':  '2312-CH2CHO',
                            'A64':  '2313-CHCH2O',
                            'A65':  '2314-CH2COH',
                            'A66':  '2315-CHCHOH',
                            'A67':  '2316-CCH2OH',
                            'A68':  '2321-OCH2CHO',
                            'A69':  '2322-HOCCH2O',
                            'A70':  '2323-HOCHCHO',
                            'A71':  '2324-OHCH2CO',
                            'A72':  '2401-CH3CH',
                            'A73':  '2402-CH2CH2',
                            'A74':  '2411-CH3CHO',
                            'A75':  '2412-CH2CH2O',
                            'A76':  '2413-CH3COH',
                            'A77':  '2414-CH2CHOH',
                            'A78':  '2415-CHCH2OH',
                            'A79':  '2421-OCH2CH2O',
                            'A80':  '2422-HOCHCH2O',
                            'A81':  '2423-HOCH2CHO',
                            'A82':  '2424-HOCH2COH',
                            'A83':  '2425-HOCHCHOH',
                            'A84':  '2501-CH3CH2',
                            'A85':  '2511-CH3CH2O',
                            'A86':  '2512-CH3CHOH',
                            'A87':  '2513-CH2CH2OH',
                            'A88':  '2522-HOCH2CHOH',
                            'A89':  '2601-CH3CH3',
                            'A90':  '2611-CH3CH2OH',
                            }

## Defining Preferred Adsorbates
ORDER_OF_ADSORBATES = [
                        'A01','A02','A03','A04','A05','A06','A07','A08','A09','A10',
                        'A11','A12','A13','A14','A15','A16','A17','A18','A19','A20',
                        'A21','A22','A23','A24','A25','A26','A27','A28','A29','A30',
                        'A31','A32','A33','A34','A35','A36','A37','A38','A39','A40',
                        'A41','A42','A43','A44','A45','A46','A47','A48','A49','A50',
                        'A51','A52','A53','A54','A55','A56','A57','A58','A59','A60',
                        'A61','A62','A63','A64','A65','A66','A67','A68','A69','A70',
                        'A71','A72','A73','A74','A75','A76','A77','A78','A79','A80',
                        'A81','A82','A83','A84','A85','A86','A87','A88','A89','A90',
                        ]


## Dictionary for Default Runs For CNN Networks
CNN_DICT = {
        'validation_split': 0.2,
        'batch_size':       18,         # Higher batches results in faster training lower batches can converge faster
        'metrics':          ['mean_squared_error'],
        'shuffle':          True,       # True if we want to shuffle the training data
        }


SAMPLING_DICT = {'name': 'train_perc',
                'split_training': 0.8,
                }


## Defining Path to Main Data
PATH_MAIN_PROJECT = get_base_paths()


## Extraction Of Molecular Descriptors Numbers
DESCRIPTORS_CONFIG = [
                # 'hbonds',
                'hbonds_ads_donor',
                'hbonds_water_donor',
                'hbonds_mdanalysis',
               
                'adsorbate_weight',
                'adsorbate_oxy_distance',
                'adsorbate_oxy_num',
                
                # 'adsorbate_SASA_mdtraj',
                # 'adsorbate_SASA_biopython',
                # 'adsorbate_SASA_morfeus',
                'adsorbate_SAS_volume',
                
                'adsorbate_SASA_pymol',
                'adsorbate_PSA_pymol',
                
                # 'adsorbate_vdw_area',
                # 'adsorbate_vdw_volume',
                
                'adsorbate_dispersion_P',
                
                'adsorbate_logP_sdf',
                'adsorbate_rotors',
                
                'adsorbate_total_charge',
                # 'adsorbate_dipole_x',
                # 'adsorbate_dipole_y',
                # 'adsorbate_dipole_z',
                'adsorbate_dipole_magnitude',
                'adsorbate_dipole_angle',
                
                # 'adsorbate_LJ_epsilon',
                # 'adsorbate_LJ_sigma',
                'adsorbate_LJ_epsilon_weighted',
                'adsorbate_LJ_sigma_weighted',
                
                'water_oxy_to_slab',
                'water_oxy_to_ads',
                'water_dipole_angle',
                'water_dipole_mag',
                'water_ad_dipole_angle',
                
            ]


DESCRIPTORS_TRAJ = [
                # 'hbonds',
                # 'hbonds_ads_donor',
                # 'hbonds_water_donor',
                # 'hbonds_mdanalysis',
               
                'adsorbate_weight',
                # 'adsorbate_oxy_distance',
                # 'adsorbate_oxy_num',
                
                # 'adsorbate_SASA_mdtraj',
                # 'adsorbate_SASA_biopython',
                # 'adsorbate_SASA_morfeus',
                # 'adsorbate_SAS_volume',
                
                # 'adsorbate_SASA_pymol',
                # 'adsorbate_PSA_pymol',
                
                # 'adsorbate_vdw_area',
                # 'adsorbate_vdw_volume',
                
                # 'adsorbate_dispersion_P',
                
                # 'adsorbate_logP_sdf',
                # 'adsorbate_rotors',
                
                # 'adsorbate_total_charge',
                # 'adsorbate_dipole_x',
                # 'adsorbate_dipole_y',
                # 'adsorbate_dipole_z',
                # 'adsorbate_dipole_magnitude',
                # 'adsorbate_dipole_angle',
                
                # 'adsorbate_LJ_epsilon',
                # 'adsorbate_LJ_sigma',
                # 'adsorbate_LJ_epsilon_weighted',
                # 'adsorbate_LJ_sigma_weighted',
                
                # 'water_oxy_to_slab',
                # 'water_oxy_to_ads',
                # 'water_dipole_angle',
                # 'water_dipole_mag',
                # 'water_ad_dipole_angle',
                
            ]


if __name__ == "__main__":
    print(PATH_MAIN_PROJECT)
    print(list(ADSORBATE_TO_NAME_DICT))