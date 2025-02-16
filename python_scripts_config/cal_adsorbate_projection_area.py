"""
calculate_adsorbate_projection_area.py
This script is used to calculate the projection area of adsorbate on the surface.

"""
import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths


class Adsorbate_Projection:
    def __init__(
                self,
                adsorbate_list,
                csv_file = None,
                save_data = False,
                ):
        
        self.adsorbate_list = adsorbate_list
        self.save_data = save_data
        
        self.csv_file = csv_file
        self.file_dir = get_paths("simulation_path")
        
        self.number_points = 2000
        self.radiiDict = {'H': 1.2, 'C': 1.7, 'O': 1.52}
        
        self.df_projection_area_opencv = []

        self.calculate_projection_area_opencv()

        if self.save_data:
            self.df_projection_area_opencv.to_csv(self.csv_file, index=False)
            print(f"Data saved to {self.csv_file}")
    
    
    def calculate_projection_area_opencv(self):
        surface_points_path = os.path.join(self.file_dir, "vdw_points")
        vdw_points_images_path = os.path.join(get_paths("output_figure_path"), "vdw_points_images")

        for adsorbate in self.adsorbate_list:
            print(f"\n--- Loading xyz file and Computing projection area for adsorbate {adsorbate}")
            xyz_file = os.path.join(self.file_dir, 'xyz_file_adsorbate_only', adsorbate + '.xyz')
            
            print("    Getting surface points for adsorbate %s" % adsorbate)
            x, y, z = self.get_points(xyz=xyz_file)
            
            print("    Saving surface points information for adsorbate %s" % adsorbate)
            self.save_surface_points(surface_points_path, adsorbate, x, y, z)
            
            print("    Saving images for adsorbate %s" % adsorbate)
            self.save_projection_images(vdw_points_images_path, adsorbate, x, y, z)
            
            results1 = self.shadow_info(os.path.join(vdw_points_images_path, adsorbate + "_1.png"))  # xy surface
            print('    Projection area for xy surface:', results1[0])
            
            results2 = self.shadow_info(os.path.join(vdw_points_images_path, adsorbate + "_2.png"))  # xz surface
            print('    Projection area for xz surface:', results2[0])
            
            results3 = self.shadow_info(os.path.join(vdw_points_images_path, adsorbate + "_3.png"))  # yz surface
            print('    Projection area for yz surface:', results3[0])
            
            xy_area = results1[0] / 100
            xz_area = results2[0] / 100
            yz_area = results3[0] / 100
            
            max_area = max(xy_area, xz_area, yz_area)
            min_area = min(xy_area, xz_area, yz_area)

            df = pd.DataFrame({
                'adsorbate': [adsorbate] * 5,
                'config': list(range(5)),
                'adsorbate_xy_area': [xy_area] * 5,
                'adsorbate_xz_area': [xz_area] * 5,
                'adsorbate_yz_area': [yz_area] * 5,
                'adsorbate_max_area': [max_area] * 5,
                'adsorbate_min_area': [min_area] * 5,
            })

            self.df_projection_area_opencv.append(df)

        self.df_projection_area_opencv = pd.concat(self.df_projection_area_opencv)
        return self.df_projection_area_opencv

    
    def get_points(self, xyz):
        with open(xyz, "r") as f:
            data = f.readlines()[2:]

        atom_names = [line.split()[0] for line in data]
        atom_pos = np.array([[float(coord) for coord in line.split()[1:4]] for line in data])
        atom_radii = [self.radiiDict[name] for name in atom_names]

        x, y, z = [], [], []
        for pos, radius in zip(atom_pos, atom_radii):
            points = (2 * radius * np.random.rand(self.number_points, 3)) - radius
            points = points[np.linalg.norm(points, axis=1) < radius]
            points = pos + points * (radius / np.linalg.norm(points, axis=1)[:, None])
            x.extend(points[:, 0])
            y.extend(points[:, 1])
            z.extend(points[:, 2])

        return x, y, z


    def save_surface_points(self, surface_points_path, adsorbate, x, y, z):
        os.makedirs(surface_points_path, exist_ok=True)
        with open(os.path.join(surface_points_path, adsorbate + ".csv"), "w") as f:
            for xi, yi, zi in zip(x, y, z):
                f.write(f"{xi},{yi},{zi}\n")


    def save_projection_images(self, vdw_points_images_path, adsorbate, x, y, z):
        os.makedirs(vdw_points_images_path, exist_ok=True)
        
        angles = [(90, 90), (0, 0), (90, 0)]
        for i, (az, el) in enumerate(angles):
            self.plot_projection_graph(x, y, z, az, el)
            plt.savefig(os.path.join(vdw_points_images_path, f"{adsorbate}_{i+1}.png"))
            plt.close()

    
    def plot_projection_graph(self, x, y, z, az, el):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, color="black", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        plt.axis('off')
        ax.view_init(azim=az, elev=el)

    
    def shadow_info(self, image):
        img = cv.imread(image, 0)
        _, thresh = cv.threshold(img, 127, 255, 0)
        contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        if len(contours) < 2:
            raise ValueError("Not enough contours found in image")
        cnt = contours[-2]
        area = cv.contourArea(cnt)
        rect = cv.minAreaRect(cnt)
        width, length = rect[1]
        aspect_ratio = min(width, length) / max(width, length)
        return area, aspect_ratio

    
    

if __name__ == "__main__":
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    # 'A01',
                    # 'A13',
                    # 'A29',
                    # 'A44',
                    # 'A60',
                    '254',
                    '264',
                      ]

    # adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_450_adsorbate_projection_area_new.csv")
    
    projection = Adsorbate_Projection(
                                        adsorbate_list = adsorbate_list,
                                        csv_file = csv_file,
                                        save_data = True,   # True or False
                                        )
    
