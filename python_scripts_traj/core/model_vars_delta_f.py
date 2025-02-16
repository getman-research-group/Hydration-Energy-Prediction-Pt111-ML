## Import Machine Learning Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor



MODEL_DICT = {
    'Multi_Linear':     LinearRegression(),
    'SVR':              SVR(),
    'LASSO':            Lasso(),
    'Ridge':            Ridge(),
    'SGD':              SGDRegressor(),
    'Bayesian_Ridge':   BayesianRidge(),
    'K_Neighbors':      KNeighborsRegressor(),
    'Decision_Tree':    DecisionTreeRegressor(),
    'Random_Forest':    RandomForestRegressor(),
    'Gradient_Boost':   GradientBoostingRegressor(),
    'XGBoost':          XGBRegressor(),
    'MLP':              MLPRegressor(),
        }


# Extraction Of Molecular Descriptors
DESCRIPTORS_TRAJ = {
    
    'E_int_90_adsorbate_3d_d_moments.csv':[
                                            'adsorbate_Ctd-Mean',
                                            'adsorbate_Ctd-Sigma',
                                            'adsorbate_Ctd-Skewness',
                                            
                                            'adsorbate_Cst-Mean',
                                            'adsorbate_Cst-Sigma',
                                            'adsorbate_Cst-Skewness',
                                            
                                            'adsorbate_Fct-Mean',
                                            'adsorbate_Fct-Sigma',
                                            'adsorbate_Fct-Skewness',
                                            
                                            'adsorbate_Ftf-Mean',
                                            'adsorbate_Ftf-Sigma',
                                            'adsorbate_Ftf-Skewness',
                                            ],
    
    
    'E_int_90_adsorbate_3d_whim_vdw.csv':[
                                            'adsorbate_3d_whim_vdw_Wlambda1',
                                            'adsorbate_3d_whim_vdw_Wlambda2',
                                            'adsorbate_3d_whim_vdw_Wlambda3',

                                            'adsorbate_3d_whim_vdw_WT',
                                            'adsorbate_3d_whim_vdw_WA',
                                            'adsorbate_3d_whim_vdw_WV',
                                            ],
    
    
    'E_int_90_adsorbate_chemaxon.csv': [
                                        'adsorbate_logP_chemaxon',
                                        'adsorbate_HLB',
                                        'adsorbate_logS_chemaxon',
                                        'adsorbate_polarizability_chemaxon',
                                        ],
    
    
    'E_int_90_adsorbate_CPSA.csv': [
                                    'adsorbate_MSA',
                                    'adsorbate_ASA',
                                    'adsorbate_PNSA1',
                                    'adsorbate_PPSA1',
                                    'adsorbate_PNSA2',
                                    'adsorbate_PPSA2',
                                    'adsorbate_PNSA3',
                                    'adsorbate_PPSA3',
                                    'adsorbate_DPSA1',
                                    # 'adsorbate_DPSA2',
                                    # 'adsorbate_DPSA3',
                                    'adsorbate_FNSA1',
                                    'adsorbate_FNSA2',
                                    'adsorbate_FNSA3',
                                    'adsorbate_FPSA1',
                                    'adsorbate_FPSA2',
                                    'adsorbate_FPSA3',
                                    'adsorbate_WNSA1',
                                    'adsorbate_WNSA2',
                                    'adsorbate_WNSA3',
                                    'adsorbate_WPSA1',
                                    'adsorbate_WPSA2',
                                    'adsorbate_WPSA3',
                                    'adsorbate_TASA',
                                    'adsorbate_FrTATP',
                                    'adsorbate_RASA',
                                    'adsorbate_RPSA',
                                    'adsorbate_RNCS',
                                    # 'adsorbate_RPCS',
                                    ],
    
    'E_int_90_adsorbate_descriptors.csv':[
                                        'adsorbate_MW',
                                        'adsorbate_C_count',
                                        'adsorbate_H_count',
                                        'adsorbate_O_count',
                                        'adsorbate_non_H_count',
                                        'adsorbate_hydroxyl_count',
                                        'adsorbate_oxygen_distance',
                                        'adsorbate_radius_of_gyration',
                                        'adsorbate_gyration_moments',
                                        'adsorbate_shape_parameter',
                                        'adsorbate_dipole_mdanalysis',
                                        'adsorbate_quadrupole_mdanalysis',
                                        'adsorbate_dispersion_p',
                                        'adsorbate_ionization_potential',
                                        'adsorbate_ionization_potential_corr',
                                        'adsorbate_electron_affinity',
                                        'adsorbate_homo',
                                        'adsorbate_lumo',
                                        'adsorbate_electrophilicity',
                                        'adsorbate_nucleophilicity',
                                        'adsorbate_charge_max',
                                        'adsorbate_charge_min',
                                        'adsorbate_sum_negative_charges',
                                        'adsorbate_sum_abs_qmin_qmax',
                                        'adsorbate_sum_abs_charges',
                                        ],
    
    'E_int_90_adsorbate_Ewald_Sum_Matrix-reduced-sorted_l2.csv': [ 'adsorbate_fp_1',
                                                                   'adsorbate_fp_2',
                                                                   'adsorbate_fp_3',
                                                                   'adsorbate_fp_4',
                                                                   'adsorbate_fp_5',
                                                                   'adsorbate_fp_6',
                                                                   'adsorbate_fp_7',
                                                                   'adsorbate_fp_8',
                                                                   'adsorbate_fp_9',
                                                                   'adsorbate_fp_10',
                                                                   'adsorbate_fp_11',
                                                                   'adsorbate_fp_12',
                                                                   'adsorbate_fp_13',
                                                                   'adsorbate_fp_14',
                                                                   'adsorbate_fp_15',
                                                                    ],
    
    'E_int_90_adsorbate_hbonds_donor_acceptor.csv': [
                                                    'adsorbate_donor_count',
                                                    'adsorbate_donor_sites',
                                                    'adsorbate_acceptor_count',
                                                    'adsorbate_acceptor_sites',
                                                    ],
    
    
    'E_int_90_adsorbate_projection_area.csv': [
                                                'adsorbate_xy_area',
                                                'adsorbate_xz_area',
                                                'adsorbate_yz_area',
                                                'adsorbate_max_area',
                                                'adsorbate_min_area',
                                                ],
    
    
    'E_int_90_adsorbate_pybel.csv': [
                                    # 'adsorbate_logP_sdf',
                                    'adsorbate_rotors',
                                    'adsorbate_molar_refractivity',
                                    # 'adsorbate_hba1',
                                    ],
    

    'E_int_90_adsorbate_pymol.csv':[
                                    'adsorbate_vdw_sa_pymol',
                                    'adsorbate_vdw_sa_polar_pymol',
                                    'adsorbate_sasa_pymol',
                                    'adsorbate_sasa_polar_pymol',
                                    'adsorbate_sasa_hydrophobic_pymol',
                                    'adsorbate_sa_oxy_area_pymol',
                                    'adsorbate_hydroxyl_fraction_pymol',
                                    'adsorbate_neg_sasa_pymol',
                                    'adsorbate_neg_sasa_fraction_pymol',
                                    ],
    
    
    'E_int_90_adsorbate_rdkit.csv':[
                                    'MaxAbsEStateIndex',
                                    'MaxEStateIndex',
                                    'MinAbsEStateIndex',
                                    'MinEStateIndex',
                                    'qed',
                                    'SPS',
                                    'HeavyAtomMolWt',
                                    'NumValenceElectrons',
                                    'NumRadicalElectrons',
                                    'FpDensityMorgan1',
                                    'FpDensityMorgan2',
                                    'FpDensityMorgan3',
                                    'BCUT2D_MWHI',
                                    'BCUT2D_MWLOW',
                                    'BCUT2D_CHGHI',
                                    'BCUT2D_CHGLO',
                                    'BCUT2D_LOGPHI',
                                    'BCUT2D_LOGPLOW',
                                    'BCUT2D_MRHI',
                                    'BCUT2D_MRLOW',
                                    'AvgIpc',
                                    'BalabanJ',
                                    'BertzCT',
                                    'Chi0',
                                    'Chi0n',
                                    'Chi0v',
                                    'Chi1',
                                    'Chi1n',
                                    'Chi1v',
                                    'Chi2n',
                                    'Chi2v',
                                    'Chi3n',
                                    'Chi3v',
                                    'Chi4n',
                                    'Chi4v',
                                    'HallKierAlpha',
                                    'Ipc',
                                    'Kappa1',
                                    'Kappa2',
                                    'Kappa3',
                                    'LabuteASA',
                                    'PEOE_VSA1',
                                    'PEOE_VSA10',
                                    'PEOE_VSA11',
                                    'PEOE_VSA12',
                                    'PEOE_VSA13',
                                    'PEOE_VSA14',
                                    'PEOE_VSA2',
                                    'PEOE_VSA3',
                                    'PEOE_VSA6',
                                    'PEOE_VSA7',
                                    'PEOE_VSA8',
                                    'PEOE_VSA9',
                                    'SMR_VSA1',
                                    'SMR_VSA10',
                                    'SMR_VSA5',
                                    'SMR_VSA6',
                                    'SMR_VSA7',
                                    'SlogP_VSA2',
                                    'SlogP_VSA3',
                                    'SlogP_VSA4',
                                    'SlogP_VSA5',
                                    'SlogP_VSA6',
                                    # 'EState_VSA1',
                                    # 'EState_VSA10',
                                    # 'EState_VSA2',
                                    # 'EState_VSA3',
                                    # 'EState_VSA4',
                                    # 'EState_VSA5',
                                    # 'EState_VSA6',
                                    # 'EState_VSA7',
                                    # 'EState_VSA8',
                                    # 'EState_VSA9',
                                    'VSA_EState1',
                                    'VSA_EState2',
                                    'VSA_EState3',
                                    'VSA_EState5',
                                    'VSA_EState7',
                                    'VSA_EState8',
                                    'VSA_EState9',
                                    'FractionCSP3',
                                    # 'HeavyAtomCount',
                                    'NHOHCount',
                                    'NumHAcceptors',
                                    'NumHDonors',
                                    'NumHeteroatoms',
                                    'fr_Al_OH',
                                    'fr_Al_OH_noTert',
                                    'fr_COO2',
                                    'fr_C_O',
                                    'fr_C_O_noCOO',
                                    'fr_aldehyde',
                                    'fr_allylic_oxid',
                                    'fr_ketone',
                                    'fr_ketone_Topliss',
                                    'PMI1',
                                    'PMI2',
                                    'PMI3',
                                    'NPR1',
                                    'NPR2',
                                    'InertialShapeFactor',
                                    'Eccentricity',
                                    'Asphericity',
                                    'SpherocityIndex',
                                    'PBF',
                                    ],
    
    
    'E_int_90_hbonds.csv':[
                            # 'hbonds_avg_120',
                            # 'hbonds_avg_125',
                            # 'hbonds_avg_130',
                            # 'hbonds_avg_135',
                            'hbonds_avg_140',
                            # 'hbonds_avg_145',
                            # 'hbonds_avg_150',
                            # 'hbonds_avg_155',
                            ],
    
    
    'E_int_90_water.csv':[
                            'water_nearest_distance_all_atoms_3',
                            'water_nearest_distance_oxygen_3',
                            'coordination_number_oxygen',
                            'coordination_number_oxygen_std',
                            'coordination_number_all_atoms',
                            'coordination_number_all_atoms_std',
                            'water_density_all_atoms',
                            'water_density_oxygen',
                            'water_enrichment',
                            'rdf_peak_position',
                            'rdf_peak_height',
                            'rdf_total_area',
                            'rdf_area_within_5A',
                            'water_dipole_vs_ads_com',
                            'water_dipole_vs_ads_close_atom',
                            ],

}
