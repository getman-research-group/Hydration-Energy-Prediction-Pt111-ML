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

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import distance_array, minimize_vectors
from MDAnalysis import transformations
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis.waterdynamics import AngularDistribution as AD
from MDAnalysis.analysis.rms import RMSD, RMSF
from ase.geometry import get_distances

from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths

from read_md_traj_mdanalysis import configMDAnalysis
from read_md_traj_mdtraj import configMDTraj
from read_md_traj_ase import universeAse

# https://docs.mdanalysis.org/2.7.0/documentation_pages/analysis/wbridge_analysis.html

class calculateWaterDescriptorsTraj:

    def __init__(   self,
                    adsorbate_list,
                    verbose,
                    save_data,
                    csv_file,
                    plot_rdf,
                    font_size,
                    ):

        ## Storing Initial Information
        self.adsorbate_list = adsorbate_list
        self.verbose = verbose
        self.save_data = save_data
        self.plot_rdf = plot_rdf
        self.font_size = font_size
        
        ## File to store descriptors Value
        self.csv_file = csv_file
        
        # Dictionaries to store MDAnalysis, MDTraj, ASE objects
        self.dict_mdanalysis_config = {}
        self.dict_ase_universe = {}
        
        self.descriptor_dataframes = []
        
        ## Loop Through adsorbate_list
        for adsorbate in self.adsorbate_list:
            
            # mdanalysis universe object
            config_mda = configMDAnalysis(adsorbate = adsorbate)
            self.dict_mdanalysis_config[adsorbate] = config_mda
            
            # ase atoms object
            config_ase = universeAse(adsorbate = adsorbate)
            self.dict_ase_universe[adsorbate] = config_ase
        

        # calculate nearest water distance
        self.calculate_nearest_water_distances_mdanalysis(n = 3)
        
        # calculate coordination number of water
        self.calculate_coordination_number_mdanalysis(cutoff=3.5)
        
        # calculate water density
        self.calculate_water_density_mdanalysis(radius = 5.0)
        
        # calculate water enrichment
        self.calculate_water_enrichment_mdanalysis(local_radius = 5.0)
        
        # calculate radial distribution function features
        self.calculate_rdf_features_mdanalysis(nbins = 100, rdf_range = (0.0, 8.0), sigma = 2.0, plot = self.plot_rdf)

        # calculate water distance to surface
        self.calculate_water_distance_to_surface(surface_z = 4.581043)
        
        # calculate average dipole orientation within a radius 3.5
        self.calculate_avg_water_dipole_angle_vs_ads_com(radius = 3.5)
        self.calculate_avg_water_dipole_angle_vs_ads_close_atom(radius = 3.5)
                
        # # calculate water angular distribution
        self.calculate_water_angular_distribution(plot = False)
        
        # # calculate rmsf for water molecules
        self.calculate_rmsd_and_rmsf()
        # self.plot_rmsd_and_rmsf()
        
        # # combine all dataframes
        # self.combine_dataframes()

        
        if self.save_data == True:
            self.df.to_csv(self.csv_file, index = False)
            print(f"Descriptors Data Saved in {self.csv_file}")
    
    
    
    
    def calculate_nearest_water_distances_mdanalysis(self, n=3):
        descriptor_data = []
        print("\n--- Calculating Nearest Water Distances ---")
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values_all_atoms = []
            values_oxygen = []
            for ts in config_mda.universe.trajectory:
                box = ts.dimensions
                
                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
                water_atoms = config_mda.universe.select_atoms('resname HOH')
                water_oxygen_atoms = config_mda.universe.select_atoms('resname HOH and name O*')

                distances_all_atoms = distance_array(adsorbate_atoms.positions, water_atoms.positions, box=box)
                distances_oxygen = distance_array(adsorbate_atoms.positions, water_oxygen_atoms.positions, box=box)

                nearest_distances_all_atoms = np.sort(distances_all_atoms, axis=None)[:n]
                nearest_distances_oxygen = np.sort(distances_oxygen, axis=None)[:n]

                avg_nearest_distances_all_atoms = np.mean(nearest_distances_all_atoms)
                avg_nearest_distances_oxygen = np.mean(nearest_distances_oxygen)

                values_all_atoms.append(avg_nearest_distances_all_atoms)
                values_oxygen.append(avg_nearest_distances_oxygen)

            overall_avg_all_atoms = np.mean(values_all_atoms)
            overall_avg_oxygen = np.mean(values_oxygen)
            print(f"    adsorbate: {adsorbate}, avg_all_atoms: {overall_avg_all_atoms:.2f}, avg_oxygen: {overall_avg_oxygen:.2f}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate],
                'adsorbate_name': [adsorbate_name],
                f'water_nearest_distance_all_atoms_{n}': [overall_avg_all_atoms],
                f'water_nearest_distance_oxygen_{n}': [overall_avg_oxygen]
            })
            descriptor_data.append(df)

        self.df_nearest_water_distances_mdanalysis = pd.concat(descriptor_data, ignore_index=True)
        self.descriptor_dataframes.append(self.df_nearest_water_distances_mdanalysis)
    
    
    def calculate_coordination_number_mdanalysis(self, cutoff = 3.5):
        descriptor_data = []
        print("\n--- Calculating Coordination Number ---")
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values_oxygen = []
            values_all_atoms = []
            for ts in config_mda.universe.trajectory:
                box = ts.dimensions
                
                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
                water_atoms = config_mda.universe.select_atoms('resname HOH')
                water_oxygen_atoms = config_mda.universe.select_atoms('resname HOH and name O*')
                
                distances_oxygen = distance_array(adsorbate_atoms.positions, water_oxygen_atoms.positions, box=box)
                distances_all_atoms = distance_array(adsorbate_atoms.positions, water_atoms.positions, box=box)
                
                coordination_number_oxygen = np.sum(distances_oxygen < cutoff)
                coordination_number_all_atoms = np.sum(distances_all_atoms < cutoff)
                
                values_oxygen.append(coordination_number_oxygen)
                values_all_atoms.append(coordination_number_all_atoms)

            avg_coordination_number_oxygen = np.mean(values_oxygen)
            std_coordination_number_oxygen = np.std(values_oxygen)
            avg_coordination_number_all_atoms = np.mean(values_all_atoms)
            std_coordination_number_all_atoms = np.std(values_all_atoms)

            print(f"    adsorbate: {adsorbate}, coordination_O: {avg_coordination_number_oxygen:.2f} ± {std_coordination_number_oxygen:.2f}, coordination_all: {avg_coordination_number_all_atoms:.2f} ± {std_coordination_number_all_atoms:.2f}")

            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate],
                'adsorbate_name': [adsorbate_name],
                'coordination_number_oxygen': [avg_coordination_number_oxygen],
                'coordination_number_oxygen_std': [std_coordination_number_oxygen],
                'coordination_number_all_atoms': [avg_coordination_number_all_atoms],
                'coordination_number_all_atoms_std': [std_coordination_number_all_atoms]
            })
            descriptor_data.append(df)

        self.df_coordination_number_mdanalysis = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_coordination_number_mdanalysis)


    def calculate_water_density_mdanalysis(self, radius=5.0):
        descriptor_data = []
        print("\n--- Calculating Water Density ---")
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values_all_atoms = []
            values_oxygen = []
            for ts in config_mda.universe.trajectory:
                box = ts.dimensions
                
                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
                water_atoms = config_mda.universe.select_atoms('resname HOH')
                water_oxygen_atoms = config_mda.universe.select_atoms('resname HOH and name O*')

                distances_all_atoms = distance_array(adsorbate_atoms.positions, water_atoms.positions, box=box)
                distances_oxygen = distance_array(adsorbate_atoms.positions, water_oxygen_atoms.positions, box=box)

                within_radius_all_atoms = np.sum(distances_all_atoms < radius)
                within_radius_oxygen = np.sum(distances_oxygen < radius)

                volume = (4/3) * np.pi * radius**3

                density_all_atoms = within_radius_all_atoms / volume
                density_oxygen = within_radius_oxygen / volume

                values_all_atoms.append(density_all_atoms)
                values_oxygen.append(density_oxygen)

            avg_density_all_atoms = np.mean(values_all_atoms)
            avg_density_oxygen = np.mean(values_oxygen)
            print(f"    adsorbate: {adsorbate}, density_all: {avg_density_all_atoms:.4f}, density_oxygen: {avg_density_oxygen:.4f}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate],
                'adsorbate_name': [adsorbate_name],
                'water_density_all_atoms': [avg_density_all_atoms],
                'water_density_oxygen': [avg_density_oxygen]
            })
            descriptor_data.append(df)

        self.df_water_density_mdanalysis = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_water_density_mdanalysis)


    def calculate_water_enrichment_mdanalysis(self, local_radius = 5.0):
        descriptor_data = []
        print("\n--- Calculating Water Enrichment ---")
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            values = []
            for ts in config_mda.universe.trajectory:
                adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
                water_atoms = config_mda.universe.select_atoms('resname HOH')
                box = ts.dimensions
                
                distances = distance_array(adsorbate_atoms.positions, water_atoms.positions, box=box)
                local_volume = (4/3) * np.pi * local_radius**3
                num_within_radius = np.sum(distances < local_radius)
                local_density = num_within_radius / local_volume
                bulk_density = len(water_atoms) / (box[0] * box[1] * box[2])
                enrichment = local_density / bulk_density

                values.append(enrichment)

            avg_enrichment = np.mean(values)
            print(f"    adsorbate: {adsorbate}, enrichment: {avg_enrichment:.4f}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame({
                'adsorbate': [adsorbate],
                'adsorbate_name': [adsorbate_name],
                'water_enrichment': [avg_enrichment]
            })
            descriptor_data.append(df)

        self.df_water_enrichment_mdanalysis = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_water_enrichment_mdanalysis)



    def calculate_rdf_features_mdanalysis(self, nbins=100, rdf_range=(0.0, 8.0), sigma=1.0, plot=True, fontsize=20):
        descriptor_data = []
        print("\n--- Calculating Radial Distribution Function (RDF) Features ---")
        
        # Load e_int_dft data from CSV
        e_int_dft_df = pd.read_csv(os.path.join(get_paths("database_path"), "label_data", "E_int_90.csv"))
        
        # Prepare plot if plotting is enabled
        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size to 8x8 inches
        
        # Use a color map for different e_int_dft values
        from matplotlib import cm
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=e_int_dft_df['e_int_dft'].min(), vmax=e_int_dft_df['e_int_dft'].max())
        
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            all_rdf_values = []
            bins = None

            adsorbate_atoms = config_mda.universe.select_atoms('resname ADS')
            water_atoms = config_mda.universe.select_atoms('resname HOH')

            rdf = InterRDF(adsorbate_atoms, water_atoms, nbins=nbins, range=rdf_range)
            rdf.run()
            
            bins = rdf.results.bins
            rdf_values = rdf.results.rdf
            all_rdf_values.append(rdf_values)
            
            avg_rdf_values = np.mean(all_rdf_values, axis=0)
            smoothed_rdf_values = gaussian_filter1d(avg_rdf_values, sigma=sigma)
            
            # Highest peak position
            peak_position = bins[np.argmax(smoothed_rdf_values)]
            
            # Highest peak height
            peak_height = np.max(smoothed_rdf_values)
            
            # Area under the curve
            total_area = simpson(y=smoothed_rdf_values, x=bins)
            
            # Area under the curve within 5A
            area_within_5A = simpson(y=smoothed_rdf_values[bins <= 5.0], x=bins[bins <= 5.0])
            print(f"    adsorbate: {adsorbate}, peak_position: {peak_position:.2f}, peak_height: {peak_height:.2f}, area: {total_area:.2f}, area_within_5A: {area_within_5A:.2f}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame([[peak_position, peak_height, total_area, area_within_5A]],
                            columns=['rdf_peak_position', 'rdf_peak_height', 'rdf_total_area', 'rdf_area_within_5A'])
            df['adsorbate'] = adsorbate
            df['adsorbate_name'] = adsorbate_name
            descriptor_data.append(df)
            
            if plot:
                # Get e_int_dft value for the adsorbate
                e_int_dft_value = e_int_dft_df.loc[e_int_dft_df['adsorbate'] == adsorbate, 'e_int_dft'].values[0]
                
                # Assign color based on e_int_dft value
                color = cmap(norm(e_int_dft_value))
                ax.plot(bins, smoothed_rdf_values, color=color, alpha=0.4)  # Plot with color based on e_int_dft value
        
        # Concatenate all descriptor data into a DataFrame
        self.df_rdf_features_mdanalysis = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_rdf_features_mdanalysis)
        
        # Finalize and show plot if plot is True
        if self.plot_rdf:
            ax.set_xlabel('distance (Å)', fontsize=self.font_size)
            ax.set_ylabel('g(r)', fontsize=self.font_size)
            ax.tick_params(axis='both', which='major', labelsize=self.font_size)
            
            # Set box aspect to ensure square plot area
            ax.set_box_aspect(1)  # This ensures the plot area is square
            
            # Set y-axis limits from 0 to 2
            import matplotlib.ticker as mticker
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

            # Add the label (a) at the top left corner
            ax.text(-0.15, 1.06, '(b)', transform=ax.transAxes, fontsize=self.font_size + 2, fontweight='bold', va='top', ha='left')

            # Add a colorbar
            norm = plt.Normalize(vmin=e_int_dft_df['e_int_dft'].min(), vmax=e_int_dft_df['e_int_dft'].max())
            cmap = plt.get_cmap('viridis')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Only needed for older versions of matplotlib
            cbar = fig.colorbar(sm, ax=ax, shrink=0.68)  # Adjust shrink to shorten the colorbar
            cbar.set_label(r'$\Delta E_{\mathit{sol}}^{\mathit{DFT}}$', fontsize=self.font_size)
            cbar.ax.tick_params(labelsize=self.font_size)

            plt.tight_layout()
            
            # Save the plot
            output_path = os.path.join(get_paths('output_figure_path'), 'rdf_figures', 'rdf_figure_traj.png')
            fig.savefig(output_path, dpi=1000, bbox_inches='tight', pad_inches=0.1)
            plt.show()
    

    def calculate_avg_water_dipole_angle_vs_ads_com(self, radius=3.5):
        descriptor_data = []
        print("\n--- Calculating Average Dipole Orientation Within Radius ---")
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            all_cosine_angles = []
            
            # https://www.mdanalysis.org/2020/03/09/on-the-fly-transformations/
            u = config_mda.universe
            workflow = [transformations.unwrap(u.atoms)]
            u.trajectory.add_transformations(*workflow)
            
            for ts in u.trajectory:
                box = ts.dimensions
                
                adsorbate_atoms = u.select_atoms('resname ADS')
                water_atoms = u.select_atoms('resname HOH')
                water_oxygen_atoms = water_atoms.select_atoms('name O*')
                
                distances = distance_array(adsorbate_atoms.positions, water_oxygen_atoms.positions, box=box)
                within_radius_indices = np.any(distances <= radius, axis=0)
                close_oxygen_atoms = water_oxygen_atoms[within_radius_indices]
                
                adsorbate_com = adsorbate_atoms.center_of_mass()
                cosine_angles = []
                
                for oxygen in close_oxygen_atoms:
                    residue_id = oxygen.resid
                    water = water_atoms.select_atoms(f"resid {residue_id}")
                    
                    # https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup.dipole_vector
                    dipole_vector = water.dipole_vector(unwrap = True, compound='group', center='mass')
                    
                    oxygen_position = oxygen.position
                    
                    # https://docs.mdanalysis.org/stable/documentation_pages/lib/distances.html#MDAnalysis.lib.distances.minimize_vectors
                    vector = oxygen_position - adsorbate_com
                    distance_vector = minimize_vectors(vector, box)
                    
                    cosine_angle = np.dot(dipole_vector, distance_vector) / (np.linalg.norm(dipole_vector) * np.linalg.norm(distance_vector))
                    cosine_angles.append(cosine_angle)
                    
                avg_cosine_angle = np.mean(cosine_angles) if cosine_angles else np.nan
                all_cosine_angles.append(avg_cosine_angle)
            
            filtered_cosine_angles = [cosine for cosine in all_cosine_angles if not np.isnan(cosine)]
            avg_cosine = np.mean(filtered_cosine_angles)
            print(f"    adsorbate: {adsorbate}, Avg Water Dipole Cosine vs adsorbate COM: {avg_cosine:.2f}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame([avg_cosine], columns=['water_dipole_cosine_vs_ads_com'])
            df['adsorbate'] = adsorbate
            df['adsorbate_name'] = adsorbate_name
            descriptor_data.append(df)

        self.df_avg_dipole_cosine_orientation_within_radius = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_avg_dipole_cosine_orientation_within_radius)
    

    def calculate_avg_water_dipole_angle_vs_ads_close_atom(self, radius=3.5):
        descriptor_data = []
        print("\n--- Calculating Average Dipole Orientation Within Radius ---")
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            all_cosines = []
            
            # https://www.mdanalysis.org/2020/03/09/on-the-fly-transformations/
            u = config_mda.universe
            # workflow = [transformations.unwrap(u.atoms)]
            # u.trajectory.add_transformations(*workflow)
            
            for ts in u.trajectory:
                box = ts.dimensions
                
                adsorbate_atoms = u.select_atoms('resname ADS')
                water_atoms = u.select_atoms('resname HOH')
                water_oxygen_atoms = water_atoms.select_atoms('name O*')
                
                distances = distance_array(adsorbate_atoms.positions, water_oxygen_atoms.positions, box=box)
                within_radius_indices = np.any(distances <= radius, axis=0)
                close_oxygen_atoms = water_oxygen_atoms[within_radius_indices]
                
                # Calculate dipole orientations for water molecules within the radius
                dipole_cosines = []
                
                for oxygen in close_oxygen_atoms:
                    
                    residue_id = oxygen.resid
                    water = water_atoms.select_atoms(f"resid {residue_id}")
                    
                    # https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup.dipole_vector
                    dipole_vector = water.dipole_vector(unwrap = True, compound='group', center='mass')
                    
                    # Find the nearest adsorbate atom to this oxygen atom
                    distances_to_adsorbate = distance_array(oxygen.position, adsorbate_atoms.positions, box=box)
                    nearest_adsorbate_atom_index = np.argmin(distances_to_adsorbate)
                    nearest_adsorbate_atom_position = adsorbate_atoms.positions[nearest_adsorbate_atom_index]
                    
                    oxygen_position = oxygen.position
                    
                    # https://docs.mdanalysis.org/stable/documentation_pages/lib/distances.html#MDAnalysis.lib.distances.minimize_vectors
                    vector = oxygen_position - nearest_adsorbate_atom_position
                    distance_vector = minimize_vectors(vector, box)
                    
                    cosine_angle = np.dot(dipole_vector, distance_vector) / (np.linalg.norm(dipole_vector) * np.linalg.norm(distance_vector))
                    dipole_cosines.append(cosine_angle)
                    
                avg_dipole_cosine = np.mean(dipole_cosines) if dipole_cosines else np.nan
                all_cosines.append(avg_dipole_cosine)
            
            filtered_cosines = [cosine for cosine in all_cosines if not np.isnan(cosine)]
            avg_cosine = np.mean(filtered_cosines)
            print(f"    adsorbate: {adsorbate}, Avg Water Dipole Cosine vs adsorbate closest atom: {avg_cosine:.2f}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame([avg_cosine], columns=['water_dipole_cosine_vs_ads_close_atom'])
            df['adsorbate'] = adsorbate
            df['adsorbate_name'] = adsorbate_name
            descriptor_data.append(df)

        self.df_avg_dipole_orientation_within_radius = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_avg_dipole_orientation_within_radius)
        
        
    def calculate_water_distance_to_surface(self, surface_z = 4.581043):
        descriptor_data = []
        print("\n--- Calculating Water Distance to Surface ---")
        
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            all_distances = []
            for ts in config_mda.universe.trajectory:
                box = ts.dimensions
                
                water_atoms = config_mda.universe.select_atoms('resname HOH')
                water_oxygen_atoms = water_atoms.select_atoms('name O*')
                pt_atoms = config_mda.universe.select_atoms('resname PT')
                
                distances = np.abs(water_oxygen_atoms.positions[:, 2] - surface_z)
                nearest_distances = np.sort(distances)[:3]
                
                avg_distance = np.mean(nearest_distances)
                all_distances.append(avg_distance)

            avg_water_distance = np.nanmean(all_distances) if all_distances else np.nan
            print(f"    adsorbate: {adsorbate}, avg_water_distance: {avg_water_distance:.2f}")
            
            adsorbate_name = ADSORBATE_TO_NAME_DICT[adsorbate]
            df = pd.DataFrame([[avg_water_distance]], columns=['avg_water_distance'])
            df['adsorbate'] = adsorbate
            df['adsorbate_name'] = adsorbate_name
            descriptor_data.append(df)

        self.df_water_distance_to_surface = pd.concat(descriptor_data)
        self.descriptor_dataframes.append(self.df_water_distance_to_surface)
    
    
    def calculate_water_angular_distribution(self, bins = 20, plot = False):
        descriptor_data = []
        print("\n--- Analyzing Angular Distribution ---")
        
        # Prepare figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            
            # https://www.mdanalysis.org/2020/03/09/on-the-fly-transformations/
            u = config_mda.universe
            # workflow = [transformations.unwrap(u.atoms)]
            # u.trajectory.add_transformations(*workflow)
            
            selection = f"byres name O_H_TIP3PCHARMM and sphzone 4.0 resname ADS"
            # selection = 'resname HOH'
            
            AD_analysis = AD(u, selection, bins)
            AD_analysis.run()
            self.AD_analysis = AD_analysis
            # AD OH
            axes[0].plot([float(column.split()[0]) for column in AD_analysis.graph[0][:-1]],
                        [float(column.split()[1]) for column in AD_analysis.graph[0][:-1]],
                        label=f'{adsorbate}')
            
            # AD HH
            axes[1].plot([float(column.split()[0]) for column in AD_analysis.graph[1][:-1]],
                        [float(column.split()[1]) for column in AD_analysis.graph[1][:-1]],
                        label=f'{adsorbate}')
            
            # AD dipole
            axes[2].plot([float(column.split()[0]) for column in AD_analysis.graph[2][:-1]],
                        [float(column.split()[1]) for column in AD_analysis.graph[2][:-1]],
                        label=f'{adsorbate}')
            
        # Set labels and titles
        axes[0].set_xlabel('cos θ', fontsize=14)
        axes[0].set_ylabel('P(cos θ)', fontsize=14)
        axes[0].set_title('PDF cos θ for OH', fontsize=14)

        axes[1].set_xlabel('cos θ', fontsize=14)
        axes[1].set_ylabel('P(cos θ)', fontsize=14)
        axes[1].set_title('PDF cos θ for HH', fontsize=14)

        axes[2].set_xlabel('cos θ', fontsize=14)
        axes[2].set_ylabel('P(cos θ)', fontsize=14)
        axes[2].set_title('PDF cos θ for dipole', fontsize=14)
        
        if len(self.dict_mdanalysis_config) <= 10:
            for ax in axes:
                ax.legend()

        plt.tight_layout()
        plt.show()

        # self.df_angular_distribution = pd.concat(descriptor_data, ignore_index=True)
        # self.descriptor_dataframes.append(self.df_angular_distribution)
    
    
    def calculate_rmsd_and_rmsf(self):
        descriptors = []
        self.rmsd_data = []  # Store RMSD data for plotting
        self.rmsf_data = []  # Store RMSF data for plotting
        print("\n--- Calculating RMSD and RMSF ---")
        
        for adsorbate, config_mda in self.dict_mdanalysis_config.items():
            # https://docs.mdanalysis.org/2.7.0/documentation_pages/transformations/wrap.html
            # https://www.mdanalysis.org/2020/03/09/on-the-fly-transformations/
            u = config_mda.universe
            water = u.select_atoms('resname HOH')
            
            # Center and unwrap the water molecules
            # workflow = [transformations.unwrap(water)]
            # u.trajectory.add_transformations(*workflow)
            
            # Calculate the average structure of water molecules
            ref_coordinates = u.trajectory.timeseries(asel=water).mean(axis=0)  # # (501, 72, 3) to (72, 3)
            reference = mda.Merge(water).load_new(ref_coordinates[None, :, :], format=mda.coordinates.memory.MemoryReader)
            
            # Fit the trajectory to the reference
            align.AlignTraj(u, reference, select='resname HOH', in_memory=True).run()
            
            # Calculate RMSD
            rmsd = RMSD(water, reference=reference).run()
            rmsd_values = rmsd.rmsd[:, 2]  # Use the RMSD values for all frames
            mean_rmsd = np.mean(rmsd_values)
            
            # Calculate RMSF
            rmsf = RMSF(water).run()
            rmsf_values = rmsf.rmsf
            
            # Calculate RMSF descriptors
            mean_rmsf = np.mean(rmsf_values)
            print('mean_rmsd: ', mean_rmsd)
            print('mean_rmsf: ', mean_rmsf)
            
            mean_rmsf_max = np.mean(np.sort(rmsf_values)[-9:])    # value of top 9
            print('mean_rmsf_max: ', mean_rmsf_max)
            mean_rmsf_min = np.mean(np.sort(rmsf_values)[:9])  # value of bottom 9
            print('mean_rmsf_min: ', mean_rmsf_min)

            descriptor = {
                'mean_rmsd': mean_rmsd,
                'mean_rmsf': mean_rmsf,
                'mean_rmsf_max': mean_rmsf_max,
                'mean_rmsf_min': mean_rmsf_min,
                'adsorbate': adsorbate,
            }
            
            descriptors.append(descriptor)
            self.rmsd_data.append(rmsd_values)  # Store RMSD values for plotting
            self.rmsf_data.append(rmsf_values)  # Store RMSF values for plotting
            print(f"    adsorbate: {adsorbate}, RMSD and RMSF descriptors calculated")
        
        self.df_descriptors = pd.DataFrame(descriptors)
        self.descriptor_dataframes.append(self.df_descriptors)
    
    
    def plot_rmsd_and_rmsf(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot RMSD data
        for i, rmsd_values in enumerate(self.rmsd_data):
            ax1.plot(rmsd_values, label=f'Adsorbate {i+1}')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('RMSD [nm]')
        ax1.legend()
        
        # Plot RMSF data
        for i, rmsf_values in enumerate(self.rmsf_data):
            ax2.plot(rmsf_values, label=f'Adsorbate {i+1}')
        ax2.set_xlabel('Atom Index')
        ax2.set_ylabel('RMSF [nm]')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('rmsd_rmsf_plot.png')
        plt.show()
    
    
    
    
    
    def combine_dataframes(self):
        if self.verbose:
            print("\n--- Combining DataFrames for All Descriptors")

        if not self.descriptor_dataframes:
            raise ValueError("No descriptor dataframes to combine")

        self.df = reduce(lambda left, right: pd.merge(left, right, on=['adsorbate'], how='outer'), self.descriptor_dataframes)
        
        if self.save_data:
            self.df.to_csv(self.csv_file, index=False)
        if self.verbose:
            print(self.df)
        return self.df



if __name__ == "__main__":
    
    csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_90_water_raw.csv")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    # 'A44',
                    
                    # '262',
                    # '263',
                    
                    '254',
                    '264',
                      ]

    # adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    water_descriptors = calculateWaterDescriptorsTraj(adsorbate_list = adsorbate_list,
                                                      verbose = True,
                                                      save_data = True,  # True, False
                                                      csv_file = csv_file,
                                                      plot_rdf = False,
                                                      font_size = 20,
                                                      )




# average distance between solvent molecules and solute atoms