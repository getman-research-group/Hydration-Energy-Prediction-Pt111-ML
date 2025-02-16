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

from read_md_traj_mdanalysis import configMDAnalysis
from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths

from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis import transformations

# Ignore specific warnings
warnings.filterwarnings('ignore', 'No hydrogen bonds were found.*')


class CalHbondsTraj:

    def __init__(   self,
                    adsorbate_list,
                    d_a_cutoff,
                    d_h_cutoff,
                    d_h_a_angle_cutoff,
                    csv_name = None,
                    cal_type = 'overall',
                    make_plots = False,
                    save_data = False,
                    font_size = 20,
                    ):
        
        # Storing Initial Information
        self.adsorbate_list = adsorbate_list

        self.d_a_cutoff = d_a_cutoff
        self.d_h_cutoff = d_h_cutoff
        self.d_h_a_angle_cutoff = d_h_a_angle_cutoff

        self.csv_name = csv_name
        
        self.cal_type = cal_type
        self.make_plots = make_plots
        self.save_data = save_data
        self.font_size = font_size
        
        self.mdanalysis_config_dict = {}

        self.hbonds_run_dict = {}
        
        # Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
                
            # Load the configuration
            config_mda = configMDAnalysis(adsorbate)
            
            # Store Configurations for adsorbates
            self.mdanalysis_config_dict[adsorbate] = config_mda
        
        if self.cal_type == 'separate':
            # Find Hydrogen Bonds separately
            self.df_hbonds_ad_as_donor = self.hbonds_adsorbate_as_donor()
            self.df_hbonds_water_as_donor = self.hbonds_water_as_donor()
            self.df_hbonds = self.combine_and_sort_hbonds()
        
        elif self.cal_type == 'overall':
            # Find Hydrogen Bonds overall
            self.df_hbonds = self.bonds_overall()
        

        # Get the final dataframe
        self.df_final = self.get_hbonds_final()
        
        if self.save_data == True:
        # and len(self.df_final) == 90:
            
            self.df_final.to_csv(os.path.join(get_paths('label_data_path'), csv_name),
                                 index = False,
                                 )

        
        # # Calculate the lifetime of hydrogen bonds
        # self.df_hbonds_lifetime = self.cal_hbonds_lifetime()
           
        if self.make_plots == True:
            # Plot the data
            # self.plot_hbonds_avg_vs_eint()
            self.plot_hbonds_avg_vs_eint_by_csv()

    
    def hbonds_adsorbate_as_donor(self):
        # Hydrogen Bonds between Adsorbate and Water
        # 1. Adsorbate As Donor
        print ("\n--- Find Hydrogen Bonds between Ads and Water (Adsorbate As Donor)")

        # create a list to store the dataframes
        dataframes = []
        
        for adsorbate, config_mda in self.mdanalysis_config_dict.items():
            
            print ("\n    %s: (Adsorbate As Donor)" % adsorbate)
            
            # Donor Selection (Adsorbate)
            # donors_sel = "resname ADS and name " + " ".join(atom for atom in config_mda.atom_types if atom.startswith("O") and "adsorbate" in atom)
            donor_atoms = [atom for atom in config_mda.atom_types if atom.startswith("O") and "adsorbate" in atom]
            donors_sel = "resname ADS and name " + " ".join(donor_atoms) if donor_atoms else 'resname ADS'
            print('    donors_sel:    ', donors_sel)
            
            # Hydrogen Selection (Adsorbate)
            # hydrogens_sel = "resname ADS and name " + " ".join(atom for atom in config_mda.atom_types if atom.startswith("H") and "adsorbate" in atom)
            hydrogen_atoms = [atom for atom in config_mda.atom_types if atom.startswith("H") and "adsorbate" in atom]
            hydrogens_sel = "resname ADS and name " + " ".join(hydrogen_atoms) if hydrogen_atoms else 'resname ADS'
            print('    hydrogens_sel: ', hydrogens_sel)
            
            # Acceptor Selection (Water)
            acceptors_sel = "resname HOH and name O*"
            print('    acceptors_sel: ', acceptors_sel)
            
            hbonds_ads_donor = HydrogenBondAnalysis(universe = config_mda.universe,
                                                    d_a_cutoff = self.d_a_cutoff,
                                                    d_h_cutoff = self.d_h_cutoff,
                                                    d_h_a_angle_cutoff = self.d_h_a_angle_cutoff,
                                                    between = ['resname ADS', 'resname HOH'],
                                                    donors_sel = donors_sel,         # Donor   : Adsorbate Oxygen
                                                    hydrogens_sel = hydrogens_sel,   # Hydrogen: Adsorbate Hydrogen
                                                    acceptors_sel = acceptors_sel,   # Acceptor: Water Oxygen
                                                    )

            hbonds_ads_donor.run(verbose = False)
            
            # Process Data For Hydrogen Bonds That Adsorbate As Donor
            hbonds_ads_donor = hbonds_ads_donor.results.hbonds
            results_df = pd.DataFrame(hbonds_ads_donor, columns=['frame', 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'distance', 'angle'])
            results_df.insert(0, 'adsorbate', adsorbate)
            
            dataframes.append(results_df)
                
        # Concatenate all results
        return pd.concat(dataframes, ignore_index=True)

        
    def hbonds_water_as_donor(self):
        
        # 2. Water As Donor
        print ("\n--- Find Hydrogen Bonds between Ads and Water (Water As Donor)")

        # create a list to store the dataframes
        dataframes = []
        
        for adsorbate, config_mda in self.mdanalysis_config_dict.items():
            
            print ("\n    %s: (Water As Donor)" % adsorbate)
            
            # Donor Selection (Water)
            donors_sel = "resname HOH and name O*"
            print('    donors_sel:    ', donors_sel)
            
            # Hydrogen Selection (Water)
            hydrogens_sel = "resname HOH and name H*"
            print('    hydrogens_sel: ', hydrogens_sel)
            
            # Acceptor Selection (Adsorbate)
            # acceptors_sel = "resname ADS and name " + " ".join(atom for atom in config_mda.atom_types if atom.startswith("O") and "adsorbate" in atom)
            acceptor_atoms = [atom for atom in config_mda.atom_types if atom.startswith("O") and "adsorbate" in atom]
            acceptors_sel = "resname ADS and name " + " ".join(acceptor_atoms) if acceptor_atoms else 'resname ADS'
            print('    acceptors_sel: ', acceptors_sel)
            
            hbonds_water_donor = HydrogenBondAnalysis(  universe = config_mda.universe,
                                                        d_a_cutoff = self.d_a_cutoff,
                                                        d_h_cutoff = self.d_h_cutoff,
                                                        d_h_a_angle_cutoff = self.d_h_a_angle_cutoff,
                                                        between = ['resname ADS', 'resname HOH'],
                                                        donors_sel = donors_sel,           # Donor   : Water Oxygen
                                                        hydrogens_sel = hydrogens_sel,     # Hydrogen: Water Hydrogen
                                                        acceptors_sel = acceptors_sel,     # Acceptor: Adsorbate Oxygen
                                                        )
            
            hbonds_water_donor.run(verbose=False)
            
            # Process Data For Hydrogen Bonds That Water As Donor
            hbonds_water_donor = hbonds_water_donor.results.hbonds
            results_df = pd.DataFrame(hbonds_water_donor, columns=['frame', 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'distance', 'angle'])
            results_df.insert(0, 'adsorbate', adsorbate)
            
            dataframes.append(results_df)

        return pd.concat(dataframes, ignore_index=True)
    
    
    
    
    def bonds_overall(self):
        
        print ("\n--- Find Hydrogen Bonds between Ads and Water")

        # create a list to store the dataframes
        dataframes = []
        
        for adsorbate, config_mda in self.mdanalysis_config_dict.items():
            
            print ("\n    %s: (Overall)" % adsorbate)
            
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
            
            # # https://www.mdanalysis.org/2020/03/09/on-the-fly-transformations/
            # u = config_mda.universe
            # workflow = [transformations.unwrap(u.atoms)]
            # u.trajectory.add_transformations(*workflow)
            
            # https://docs.mdanalysis.org/2.7.0/documentation_pages/analysis/hydrogenbonds.html
            hbonds = HydrogenBondAnalysis(universe = config_mda.universe,
                                          d_a_cutoff = self.d_a_cutoff,
                                          d_h_cutoff = self.d_h_cutoff,
                                          d_h_a_angle_cutoff = self.d_h_a_angle_cutoff,
                                          between = ['resname ADS', 'resname HOH'],
                                          )
            
            hbonds.donors_sel = f"({ad_donors_sel}) or ({water_donors_sel})"
            hbonds.hydrogens_sel = f"({ad_hydrogens_sel}) or ({water_hydrogens_sel})"
            hbonds.acceptors_sel = f"({ad_acceptors_sel}) or ({water_acceptors_sel})"
            
            # run the analysis
            hbonds.run(verbose=False)
            self.hbonds_run_dict[adsorbate] = hbonds
            
            # Process Data For Hydrogen Bonds That Water As Donor
            hbonds = hbonds.results.hbonds
            results_df = pd.DataFrame(hbonds, columns=['frame', 'donor_idx', 'hydrogen_idx', 'acceptor_idx', 'distance', 'angle'])
            results_df.insert(0, 'adsorbate', adsorbate)
            
            dataframes.append(results_df)

        return pd.concat(dataframes, ignore_index=True)
    
    
    
    def combine_and_sort_hbonds(self):
        combined_df = pd.concat([self.df_hbonds_ad_as_donor, self.df_hbonds_water_as_donor], ignore_index=True)
        combined_df.sort_values(by=['adsorbate', 'frame'], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        return combined_df
    
    
    
    def get_hbonds_final(self):
        print(f"\n--- Analyzing Hydrogen Bonds")

        if self.df_hbonds.empty:
            print("No hydrogen bonds found in any configuration.")
            columns = ['adsorbate', 'hbonds_avg']
            self.df_final = pd.DataFrame(columns=columns)
            return self.df_final

        # Calculate the number of hydrogen bonds for each adsorbate
        df_final = self.df_hbonds.groupby('adsorbate').size().reset_index(name='hbonds')

        # Create a DataFrame with all adsorbates
        all_adsorbates = pd.DataFrame(self.adsorbate_list, columns=['adsorbate'])

        # Merge all_adsorbates with df_final, ensuring all adsorbates are included
        df_final = pd.merge(all_adsorbates, df_final, on='adsorbate', how='left').fillna(0)

        # Calculate the average number of hydrogen bonds
        df_final['hbonds_avg'] = df_final['hbonds'] / 501

        # Sort by adsorbate and reset index
        df_final.sort_values(by='adsorbate', inplace=True)
        df_final.reset_index(drop=True, inplace=True)

        return df_final


    
    def plot_hbonds_avg_vs_eint(self):
        # Read csv file
        self.df_e_int = pd.read_csv(get_paths('md_descriptors_90'))

        # Merge two dataframes
        self.df_plot_hbonds_vs_eint = pd.merge(self.df_e_int,
                                            self.df_final,
                                            on='adsorbate',
                                            how='inner')

        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(8, 8))

        # Create a scatter plot
        ax.scatter(self.df_plot_hbonds_vs_eint['hbonds_avg'], self.df_plot_hbonds_vs_eint['e_int_dft'])

        # Perform linear regression
        z = np.polyfit(self.df_plot_hbonds_vs_eint['hbonds_avg'], self.df_plot_hbonds_vs_eint['e_int_dft'], 1)
        p = np.poly1d(z)

        # Add trendline
        xp = np.linspace(self.df_plot_hbonds_vs_eint['hbonds_avg'].min(), self.df_plot_hbonds_vs_eint['hbonds_avg'].max(), 100)
        ax.plot(xp, p(xp), '-', color='red')

        # Set labels
        ax.set_xlabel('Number of hydrogen bonds per frame', fontsize=self.font_size)
        ax.set_ylabel('Interaction energy', fontsize=self.font_size)

        # Set aspect ratio to make the plot a square
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

        # Remove title
        # ax.set_title('')

        # Adjust font size for ticks
        ax.tick_params(axis='both', which='major', labelsize=self.font_size)

        # Save the figure
        output_filename = "avg_hbonds_vs_eint_da_{}_dh_{}_dha_{}.png".format(
            self.d_a_cutoff,
            self.d_h_cutoff,
            self.d_h_a_angle_cutoff,
        )

        fig.savefig(os.path.join(get_paths('output_figure_path'), 'hbonds', output_filename), dpi=1000)

        plt.show()
    
    
    
    def plot_hbonds_avg_vs_eint_by_csv(self):
        # Read csv file
        csv_file_path = os.path.join(get_paths('label_data_path'), 'E_int_90_hbonds.csv')
        df = pd.read_csv(csv_file_path)

        # Determine the column name based on self.d_h_a_angle_cutoff
        hbonds_avg_column = f'hbonds_avg_{self.d_h_a_angle_cutoff}'

        if hbonds_avg_column not in df.columns:
            raise ValueError(f"Column {hbonds_avg_column} not found in CSV file")

        # Create figure and axes objects with specified size
        fig, ax = plt.subplots(figsize=(10, 8))  # Set the size of the figure (width, height)

        # Create a scatter plot
        ax.scatter(df[hbonds_avg_column], df['e_int_dft'])

        # Perform linear regression
        z = np.polyfit(df[hbonds_avg_column], df['e_int_dft'], 1)
        p = np.poly1d(z)

        # Add trendline
        xp = np.linspace(df[hbonds_avg_column].min(), df[hbonds_avg_column].max(), 100)
        ax.plot(xp, p(xp), '-', color='red')

        # Set labels
        ax.set_xlabel('Number of hydrogen bonds per frame', fontsize=self.font_size)
        ax.set_ylabel('Interaction energy', fontsize=self.font_size)

        # Set aspect ratio to make the plot a square
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

        # Adjust font size for ticks
        ax.tick_params(axis='both', which='major', labelsize=self.font_size)

        # Save the figure with tight bounding box
        output_filename = "avg_hbonds_vs_eint_da_{}_dh_{}_dha_{}.png".format(
            self.d_a_cutoff,
            self.d_h_cutoff,
            self.d_h_a_angle_cutoff,
        )

        fig.savefig(os.path.join(get_paths('output_figure_path'), 'hbonds', output_filename), dpi=1000, bbox_inches='tight')
        print(f"Figure saved as {output_filename}")
        plt.show()
    
    
    
    def cal_hbonds_lifetime(self):
        def fit_biexponential(tau_timeseries, ac_timeseries):
            """
            Fit a biexponential function to a hydrogen bond time autocorrelation function
            Return the two time constants
            """
            from scipy.optimize import curve_fit

            def model(t, A, tau1, B, tau2):
                """Fit data to a biexponential function."""
                return A * np.exp(-t / tau1) + B * np.exp(-t / tau2)

            params, params_covariance = curve_fit(model, tau_timeseries, ac_timeseries, [1, 0.5, 1, 2], maxfev=5000)

            fit_t = np.linspace(tau_timeseries[0], tau_timeseries[-1], 1000)
            fit_ac = model(fit_t, *params)

            return params, fit_t, fit_ac

        tau_max = 25
        window_step = 1

        # create a list to store the dataframes
        dataframes = []

        for adsorbate, hbonds in self.hbonds_run_dict.items():
            tau_frames, hbond_lifetime = hbonds.lifetime(tau_max=tau_max, window_step=window_step)

            u = self.mdanalysis_config_dict[adsorbate].universe
            tau_times = tau_frames * u.trajectory.dt
            
            params, fit_t, fit_ac = fit_biexponential(tau_times, hbond_lifetime)

            plt.rcParams.update({'font.size': 14})
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.plot(tau_times, hbond_lifetime, label="data")
            ax.plot(fit_t, fit_ac, label="fit")

            ax.set_xlabel(r"$\tau\ \rm (ps)$")
            ax.set_ylabel(r"$C(\tau)$")
            ax.legend()
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

            output_filename = "hbond_lifetime"
            fig.savefig(os.path.join(get_paths('output_figure_path'), 'hbonds', output_filename + "_" + adsorbate + ".png"), dpi=600)
            plt.show()

            A, tau1, B, tau2 = params
            time_constant = A * tau1 + B * tau2
            print(f"    time_constant = {time_constant:.2f} ps")

            lifetime = pd.DataFrame({
                "adsorbate": [adsorbate],
                "avg_hbonds": [time_constant],
            })

            dataframes.append(lifetime)

        self.df_hbonds_lifetime = pd.concat(dataframes)
        self.df_hbonds_lifetime = self.df_hbonds_lifetime.reset_index(drop=True)
        
        return self.df_hbonds_lifetime


        
if __name__ == "__main__":

    ## Defining List of Adsorbates
    adsorbate_list = [
                    # 'A01',
                    # 'A02',
                    # 'A03',
                    # 'A04',
                    # 'A05',
                    # 'A29',
                    # 'A40',
                    # 'A44',
                    # 'A58',
                    # 'A09',
                    # 'A88',
                    # 'A87',
                    
                    # '252',
                    # '253',
                    '254',
                    
                    # '262',
                    # '263',
                    '264',
                      ]

    # adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    for cutoff in [
                    120,
                    # 125,
                    # 130,
                    # 135,
                    # 140,
                    # 145,
                    # 150,
                    # 155,
                   ]:
        
        csv_name = f'E_int_90_hbonds_mdanalysis_{int(cutoff)}_temp.csv'
        
        hba_mdanalysis_traj = CalHbondsTraj(adsorbate_list = adsorbate_list,
                                            d_a_cutoff = 3.5,               # 3.5 in xiaohong paper
                                            d_h_cutoff = 1.0,               # 1.0 in xiaohong paper
                                            d_h_a_angle_cutoff = cutoff,    # 150 in xiaohong paper
                                            cal_type = 'overall',
                                            make_plots = False,  # False # True
                                            save_data = True,   # False # True
                                            csv_name = csv_name,
                                            font_size = 20,
                                            )