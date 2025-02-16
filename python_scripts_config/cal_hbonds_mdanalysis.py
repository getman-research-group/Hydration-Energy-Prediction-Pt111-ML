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
from core.path import get_paths

from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis import transformations

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No hydrogen bonds were found given angle of")

class HBA_MDAnalysis:

    def __init__(   self,
                    adsorbate_list = ['A01'],
                    d_a_cutoff = 3.5,
                    d_h_cutoff = 1.0,
                    d_h_a_angle_cutoff = 150,
                    make_plots = False,
                    save_data = False,
                    ):
        """

        make_plots : Bool, if we want to make plots of hydrogen bond numbers vs configs
        config_mda : MDAnalysis object
        """
        
        ## Storing Initial Information
        self.adsorbate_list = adsorbate_list

        self.d_a_cutoff = d_a_cutoff
        self.d_h_cutoff = d_h_cutoff
        self.d_h_a_angle_cutoff = d_h_a_angle_cutoff

        self.make_plots = make_plots
        self.save_data = save_data

        self.dict_mdanalysis_config = {}

        ## Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
                
            ## Load the configuration
            config_mda = configMDAnalysis(adsorbate)
            
            ## Store Configurations for adsorbates
            self.dict_mdanalysis_config[adsorbate] = config_mda

        
        ## Find Hydrogen Bonds
        self.hbonds_overall()
        
        self.get_hbonds_final()
        
        
        if self.save_data == True and len(self.df_final) > 449:
            
            csv_name = f'E_int_450_hbonds_mdanalysis_{int(self.d_h_a_angle_cutoff)}.csv'
            
            self.df_final.to_csv(os.path.join(get_paths("label_data_path"), csv_name),
                                  index = False,
                                  )


    def hbonds_overall(self):
        
        # Frame indices of interest
        frame_indices = [230, 290, 350, 410, 470]
        frame_to_config = {230: 0, 290: 1, 350: 2, 410: 3, 470: 4}
        # create a list to store the dataframes
        dataframes = []
        
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            
            print(f"\n--- Analyzing Hydrogen Bonds for {adsorbate} at specified frames:")
            
            # adsorbate donor selection
            # donor_atoms = [atom for atom in config_mda.atom_types if atom.startswith("O") and "adsorbate" in atom]
            # ad_donors_sel = "resname ADS and name " + " ".join(donor_atoms) if donor_atoms else "resname ADS"
            ad_donors_sel = "resname ADS and name O*"
            print('    ad_donors_sel:       ', ad_donors_sel)
            
            # adsorbate hydrogen selection
            # hydrogen_atoms = [atom for atom in config_mda.atom_types if atom.startswith("H") and "adsorbate" in atom]
            # ad_hydrogens_sel = "resname ADS and name " + " ".join(hydrogen_atoms) if hydrogen_atoms else "resname ADS"
            ad_hydrogens_sel = "resname ADS and name H*"
            print('    ad_hydrogens_sel:    ', ad_hydrogens_sel)
            
            # adsorbate acceptor selection
            # acceptor_atoms = [atom for atom in config_mda.atom_types if atom.startswith("O") and "adsorbate" in atom]
            # ad_acceptors_sel = "resname ADS and name " + " ".join(acceptor_atoms) if acceptor_atoms else "resname ADS"
            ad_acceptors_sel = "resname ADS and name O*"
            print('    ad_acceptors_sel:    ', ad_acceptors_sel)
            
            
            # water donor selection
            water_donors_sel = "resname HOH and name O*"
            print('    water_donors_sel:    ', water_donors_sel)
            # water hydrogen selection
            water_hydrogens_sel = "resname HOH and name H*"
            print('    water_hydrogens_sel: ', water_hydrogens_sel)
            # water acceptor selection
            water_acceptors_sel = "resname HOH and name O*"
            print('    water_acceptors_sel: ', water_acceptors_sel)
            
            # https://www.mdanalysis.org/2020/03/09/on-the-fly-transformations/
            u = config_mda.universe
            workflow = [transformations.unwrap(u.atoms)]
            u.trajectory.add_transformations(*workflow)
            
            # Loop through the frame indices
            for frame_index in frame_indices:
                
                # Locate the frame
                u.trajectory[frame_index]

                hbonds = HydrogenBondAnalysis(universe = u,
                                              d_a_cutoff = self.d_a_cutoff,
                                              d_h_cutoff = self.d_h_cutoff,
                                              d_h_a_angle_cutoff = self.d_h_a_angle_cutoff,
                                              between = ['resname ADS', 'resname HOH'],
                                              )
                
                hbonds.donors_sel = f"({ad_donors_sel}) or ({water_donors_sel})"
                hbonds.hydrogens_sel = f"({ad_hydrogens_sel}) or ({water_hydrogens_sel})"
                hbonds.acceptors_sel = f"({ad_acceptors_sel}) or ({water_acceptors_sel})"

                ## run the analysis
                hbonds.run(start = frame_index, stop = frame_index + 1, step = 1, verbose = False)
            
                ## Process Data For Hydrogen Bonds That Adsorbate As Donor
                hbonds = hbonds.results.hbonds
                results_df = pd.DataFrame(hbonds, columns = ['frame', 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'distance', 'angle'])
                results_df['config'] = results_df['frame'].map(frame_to_config)
                results_df.drop(columns=['frame'], inplace=True)
                
                # Add charges to the dataframe
                charges_dict = {atom.ix: atom.charge for atom in u.atoms}
                results_df['donor_charge'] = results_df['donor_idx'].map(charges_dict)
                results_df['hydrogen_charge'] = results_df['hydrogen_idx'].map(charges_dict)
                results_df['acceptor_charge'] = results_df['acceptor_idx'].map(charges_dict)

                # Drop the original index columns
                results_df.drop(columns=['donor_idx', 'hydrogen_idx', 'acceptor_idx'], inplace=True)
                
                # add adsorbate name
                results_df.insert(0, 'adsorbate', adsorbate)
                results_df.insert(1, 'config', results_df.pop('config'))
                
                dataframes.append(results_df)
                
        # Concatenate all results
        self.df_hbonds = pd.concat(dataframes, ignore_index=True)
        
        
        return self.df_hbonds
            
            
            

    def get_hbonds_final(self):
        # Ensure hbonds_overall has been called
        if not hasattr(self, 'df_hbonds'):
            self.hbonds_overall()
        print(f"\n--- Analyzing Hydrogen Bonds")

        # Dynamically generate the column name for the number of hydrogen bonds
        num_hbonds_col = f'hbonds_mdanalysis_{int(self.d_h_a_angle_cutoff)}'
        
        if self.df_hbonds.empty:
            print("No hydrogen bonds found in any configuration.")
            columns = ['adsorbate', 'config', num_hbonds_col]
            self.df_final = pd.DataFrame(columns=columns)
            return self.df_final

        # Initialize the final DataFrame
        max_hbonds = self.df_hbonds.groupby(['adsorbate', 'config']).size().max()
        columns = ['adsorbate', 'config', num_hbonds_col] + [f'hbond_{i+1}_{attr}' for i in range(max_hbonds) for attr in ['distance', 'angle', 'donor_charge', 'hydrogen_charge', 'acceptor_charge']]
        self.df_final = pd.DataFrame(columns=columns)

        # Populate the final DataFrame
        rows = []
        adsorbates = self.adsorbate_list  # Use self.adsorbate_list instead of unique adsorbates from df_hbonds

        for adsorbate in adsorbates:
            config_hbond_counts = []

            for config in range(5):  # As each adsorbate has 5 configs
                subset = self.df_hbonds[(self.df_hbonds['adsorbate'] == adsorbate) & (self.df_hbonds['config'] == config)]
                subset = subset.sort_values(by='distance').reset_index(drop=True)
                num_hbonds = len(subset)
                config_hbond_counts.append(num_hbonds)

                row = {'adsorbate': adsorbate, 'config': config, num_hbonds_col: num_hbonds}
                for i in range(num_hbonds):
                    row[f'hbond_{i+1}_distance'] = subset.loc[i, 'distance']
                    row[f'hbond_{i+1}_angle'] = subset.loc[i, 'angle']
                    row[f'hbond_{i+1}_donor_charge'] = subset.loc[i, 'donor_charge']
                    row[f'hbond_{i+1}_hydrogen_charge'] = subset.loc[i, 'hydrogen_charge']
                    row[f'hbond_{i+1}_acceptor_charge'] = subset.loc[i, 'acceptor_charge']

                rows.append(row)

            print(f"    Hydrogen Bonds Found for adsorbate {adsorbate}: configurations: {config_hbond_counts}")

        # Ensure all adsorbates and their configs are included, even if they have no hbonds
        for adsorbate in adsorbates:
            for config in range(5):
                if not any((row['adsorbate'] == adsorbate and row['config'] == config) for row in rows):
                    row = {'adsorbate': adsorbate, 'config': config, num_hbonds_col: 0}
                    for i in range(max_hbonds):
                        row[f'hbond_{i+1}_distance'] = 0
                        row[f'hbond_{i+1}_angle'] = 0
                        row[f'hbond_{i+1}_donor_charge'] = 0
                        row[f'hbond_{i+1}_hydrogen_charge'] = 0
                        row[f'hbond_{i+1}_acceptor_charge'] = 0
                    rows.append(row)

        # Convert rows to DataFrame
        self.df_final = pd.DataFrame(rows, columns=columns)

        # Fill NaN values with 0 (just in case)
        self.df_final.fillna(0, inplace=True)

        return self.df_final



if __name__ == "__main__":

    ## Defining List of Adsorbates
    adsorbate_list = [
                        'A01',      # [1, 0, 0, 1, 1],
                        'A02',      # [0, 1, 1, 0, 0],
                        'A03',      # [0, 3, 2, 0, 0],
                        'A04',      # [3, 2, 2, 3, 1],
                        'A05',      # [0, 0, 1, 1, 1],
                        'A06',      # [0, 1, 2, 1, 2],
                        'A07',      # [2, 1, 3, 3, 1],
                        'A08',      # [2, 3, 1, 1, 0],
                        'A09',      # [2, 1, 1, 1, 1],
                        'A10',      # [1, 2, 1, 1, 1],
                      ]

    adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    for cutoff in [130 ,135, 140, 145, 150, 155]:
        
        hba_mdanalysis = HBA_MDAnalysis(adsorbate_list = adsorbate_list,
                                        d_a_cutoff = 3.5,               # 3.5 in xiaohong paper
                                        d_h_cutoff = 1.0,               # 1.0 in xiaohong paper
                                        d_h_a_angle_cutoff = cutoff,    # 150 in xiaohong paper
                                        make_plots = False,             # False # True
                                        save_data = True,               # False # True
                                        )
        