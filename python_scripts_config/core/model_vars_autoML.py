## Import Machine Learning Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


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
    'LightGBM':         LGBMRegressor(),
    'MLP':              MLPRegressor(),
        }


## Extraction Of Molecular Descriptors Numbers
DESCRIPTORS_CONFIG_AUTOML = {
    
    'E_int_450_adsorbate_3d_d_moments.csv':[
                                            'adsorbate_Ctd-Mean',
                                            'adsorbate_Ctd-Sigma',
                                            
                                            'adsorbate_Cst-Mean',
                                            'adsorbate_Cst-Skewness',
                                            
                                            'adsorbate_Fct-Mean',
                                            'adsorbate_Fct-Sigma',
                                            'adsorbate_Fct-Skewness',
                                            
                                            'adsorbate_Ftf-Mean',
                                            'adsorbate_Ftf-Sigma',
                                            'adsorbate_Ftf-Skewness',
                                            ],
    
    
    'E_int_450_adsorbate_3d_whim_vdw.csv':[
                                            'adsorbate_3d_whim_vdw_Wlambda1',
                                            'adsorbate_3d_whim_vdw_Wlambda2',
                                            'adsorbate_3d_whim_vdw_Wlambda3',
                                            
                                            'adsorbate_3d_whim_vdw_WT',
                                            'adsorbate_3d_whim_vdw_WA',
                                            'adsorbate_3d_whim_vdw_WV',
                                            ],
    
    
    'E_int_450_adsorbate_chemaxon.csv': [
                                        'adsorbate_logP_chemaxon',
                                        'adsorbate_HLB',
                                        'adsorbate_logS_chemaxon',
                                        'adsorbate_polarizability_chemaxon',
                                        ],
    
    
    'E_int_450_adsorbate_CPSA.csv': [
                                    'adsorbate_MSA',
                                    'adsorbate_ASA',
                                    'adsorbate_PNSA1',
                                    'adsorbate_PPSA1',
                                    'adsorbate_PNSA2',
                                    'adsorbate_PPSA2',
                                    'adsorbate_PNSA3',
                                    'adsorbate_PPSA3',
                                    'adsorbate_DPSA1',
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
                                    'adsorbate_TPSA',
                                    'adsorbate_FrTATP',
                                    'adsorbate_RASA',
                                    'adsorbate_RPSA',
                                    'adsorbate_RNCS',
                                    ],
    
    'E_int_450_adsorbate_descriptors.csv':[
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
                                            'adsorbate_electron_affinity',
                                            'adsorbate_homo',
                                            'adsorbate_lumo',
                                            'adsorbate_dipole_morfeus',
                                            'adsorbate_electrophilicity',
                                            'adsorbate_nucleophilicity',
                                            'adsorbate_charge_max',
                                            'adsorbate_charge_min',
                                            'adsorbate_sum_negative_charges',
                                            'adsorbate_sum_abs_qmin_qmax',
                                            'adsorbate_sum_abs_charges',
                                            'adsorbate_LJ_sigma',
                                            'adsorbate_LJ_epsilon',
                                            ],
    
    
    'E_int_450_adsorbate_Ewald_Sum_Matrix-reduced-sorted_l2.csv': ['adsorbate_fp_1',
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
    
    
    
    'E_int_450_adsorbate_hbonds_donor_acceptor.csv': [
                                                    'adsorbate_donor_count',
                                                    'adsorbate_donor_sites',
                                                    'adsorbate_acceptor_count',
                                                    'adsorbate_acceptor_sites',
                                                    ],
    
    
    'E_int_450_adsorbate_projection_area.csv': [
                                                'adsorbate_xy_area',
                                                'adsorbate_xz_area',
                                                'adsorbate_yz_area',
                                                'adsorbate_max_area',
                                                'adsorbate_min_area',
                                                ],
    
    
    'E_int_450_adsorbate_pybel.csv': [
                                    'adsorbate_logP_sdf',
                                    'adsorbate_rotors',
                                    'adsorbate_molar_refractivity',
                                    'adsorbate_hba1',
                                    ],
    

    'E_int_450_adsorbate_pymol.csv':[
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
    

    'E_int_450_energy_descriptors.csv':[
                                        'energy_vdw_energy',
                                        'energy_elec_energy',
                                        # 'energy_vdw_elec_energy',
                                        ],
    
    
    'E_int_450_hbonds_descriptors.csv':[
                                        'hbonds',             # 0.6188
                                        'hbonds_mdanalysis',  # 0.6270
                                        # 'hb_score_slick',     # 0.4461
                                        'hb_score_aa_score',  # 0.4729
                                        # 'hb_score_id_score',  # 0.5543
                                        # 'hb_score_Goodford',  # 0.5202
                                        ],

    
    'E_int_450_water_descriptors.csv':[
                                        'water_nearest_distance_all_atoms',
                                        'water_nearest_distance_oxygen',
                                        
                                        'water_coordination_number_all_atoms',
                                        'water_coordination_number_oxygen',
                                        
                                        'water_density_all_atoms',
                                        'water_density_oxygen',
                                        
                                        'water_enrichment',
                                        'rdf_peak_position',
                                        'rdf_peak_height',
                                        'rdf_total_area',
                                        'rdf_area_first_three_peaks',
                                        'rdf_area_within_5A',
                                        ],
}
