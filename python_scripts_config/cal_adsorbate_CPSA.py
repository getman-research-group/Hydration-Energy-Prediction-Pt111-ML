import os
import pandas as pd
import numpy as np
from math import pi, sqrt
from numba import njit
from core.path import get_paths
from core.global_vars import ADSORBATE_TO_NAME_DICT



"""
Get 3D CPSA descriptors
    ASA = solvent-accessible surface area
    MSA = molecular surface area
    
    PNSA1 = partial negative surface area
    PNSA2 = total charge weighted negative surface area
    PNSA3 = atom charge weighted negative surface area
    
    PPSA1 = partial positive area
    PPSA2 = total charge weighted positive surface area
    PPSA3 = atom charge weighted positive surface area
    
    DPSA1 = difference in charged partial surface area
    DPSA2 = total charge weighted difference in charged partial surface area
    DPSA3 = atom charge weighted difference in charged partial surface area
    
    FNSA1 = fractional charged partial negative surface area
    FNSA2 = total charge weighted fractional charged partial negative surface area
    FNSA3 = atom charge weighted fractional charged partial negative surface area
    
    FPSA1 = fractional charged partial positive surface area
    FPSA2 = total charge weighted fractional charged partial positive surface area
    FPSA3 = atom charge weighted fractional charged partial positive surface area
    
    WNSA1 = surface weighted charged partial negative surface area 1
    WNSA2 = surface weighted charged partial negative surface area 2
    WNSA3 = surface weighted charged partial negative surface area 3
    
    WPSA1 = surface weighted charged partial positive surface area 1
    WPSA2 = surface weighted charged partial positive surface area 2
    WPSA3 = surface weighted charged partial positive surface area 3
    
    TASA = total hydrophobic surface area
    TPSA = total polar surface area
    
    FrTATP = TASA/TPSA
    RASA = relative hydrophobic surface area        =TASA/SA
    RPSA = relative polar surface area              =TPSA/SA
    
    RNCS = relative negative charge surface area
    RPCS = relative positive charge surface area

"""
        
atomic_covalent_radius = {
                    'H': 0.37,
                    'C': 0.77,
                    'O': 0.73,
                }

 
inc = pi * (3 - sqrt(5))

@njit
def generate_sphere_points(n):
    offset = 2.0 / n
    inc = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    phi = i * inc
    y = i * offset - 1.0 + (offset / 2.0)
    temp = np.sqrt(1 - y * y)
    x = np.cos(phi) * temp
    z = np.sin(phi) * temp
    points = np.empty((n, 3))  # Preallocate the array
    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = z
    return points

@njit
def find_neighbor_indices(xyz, Rc, RadiusProbe, k):
    dist = np.sqrt(np.sum((xyz - xyz[k])**2, axis=1))
    temp = Rc[k] + Rc + 2 * RadiusProbe
    indices = np.arange(xyz.shape[0], dtype=np.int64)
    return indices[(indices != k) & (dist < temp)]

@njit
def calculate_asa(xyz, Rc, RadiusProbe, n_sphere_point):
    """
    Partial accessible surface areas of the atoms, using the probe and atom radius
    which were used todefine the surface
    """
    areas = []
    radius = RadiusProbe + Rc
    constant = 4.0*pi/n_sphere_point
    sphere_points = generate_sphere_points(n_sphere_point)
    areas = np.zeros(xyz.shape[0])
    for i in range(len(xyz)):
        neighbor_indices = find_neighbor_indices(xyz, Rc, RadiusProbe, i)
        r = Rc[neighbor_indices] + RadiusProbe
        testpoint = sphere_points*radius[i] + xyz[i, :]
        n_accessible_point = sum([1.0 if np.all(np.sqrt(np.sum((xyz[neighbor_indices] - testpoint[ii])**2, axis=1)) >= r) 
                                  else 0.0 for ii in range(n_sphere_point)])
        areas[i] = constant * (radius[i]**2) * n_accessible_point 
    return areas






class calculateCPSAdescriptors:
    
    def __init__(self, adsorbate_list, verbose=False, save_data=False, csv_file=None):
        self.adsorbate_list = adsorbate_list
        self.verbose = verbose
        self.save_data = save_data
        self.csv_file = csv_file

        self.dict_xyz = {}
        self.descriptor_dataframes = []

        for adsorbate in self.adsorbate_list:
            xyz_file = os.path.join(get_paths('simulation_path'), 'xyz_file_adsorbate_only', adsorbate + '.xyz')
            self.dict_xyz[adsorbate] = xyz_file


        self.calculate_all_descriptors()
        
        self.combine_dataframes()


    def read_xyz(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            num_atoms = int(lines[0])
            atoms = []
            coords = []
            charges = []
            for line in lines[2:2 + num_atoms]:
                parts = line.split()
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
                charges.append(float(parts[4]))
            return np.array(atoms), np.array(coords), np.array(charges)


    def calculate_all_descriptors(self):
        for adsorbate, xyz_file in self.dict_xyz.items():
            atoms, coords, charge = self.read_xyz(xyz_file)
            Rc = np.array([atomic_covalent_radius[atom] for atom in atoms])
            descriptors = self.get_cpsa_descriptors(coords, charge, Rc)
            descriptors['adsorbate'] = adsorbate
            self.descriptor_dataframes.append(pd.DataFrame(descriptors, index=[0]))


    def get_cpsa_descriptors(self, xyz, charge, Rc):
        
        CPSA = {}
        Rc = Rc * 1.75

        # Molecular surface areas (MSA)
        RadiusProbe = 0.0
        n_sphere_point = 500
        SA = calculate_asa(xyz, Rc, RadiusProbe, n_sphere_point)
        CPSA['MSA'] = np.sum(SA)
        
        # Solvent-accessible surface areas (ASA)
        RadiusProbe = 1.4
        n_sphere_point = 1500
        SA = calculate_asa(xyz, Rc, RadiusProbe, n_sphere_point)
        CPSA['ASA'] = np.sum(SA)

        idxNeg = np.where(charge < 0.0)[0]
        # print('idxNeg: ', idxNeg)
        idxPos = np.where(charge > 0.0)[0]
        # print('idxPos: ', idxPos)
        
        idx1 = np.where(np.abs(charge) < 0.2)[0]
        idx2 = np.where(np.abs(charge) >= 0.2)[0]

        CPSA['PNSA1'] = np.sum(SA[idxNeg])
        CPSA['PPSA1'] = np.sum(SA[idxPos])

        CPSA['PNSA2'] = np.sum(charge[idxNeg]) * np.sum(SA[idxNeg])
        CPSA['PPSA2'] = np.sum(charge[idxPos]) * np.sum(SA[idxPos])

        CPSA['PNSA3'] = np.sum(charge[idxNeg] * SA[idxNeg])
        CPSA['PPSA3'] = np.sum(charge[idxPos] * SA[idxPos])

        CPSA['DPSA1'] = CPSA['PPSA1'] - CPSA['PNSA1']
        CPSA['DPSA2'] = CPSA['PPSA2'] - CPSA['PNSA2']
        CPSA['DPSA3'] = CPSA['PPSA3'] - CPSA['PNSA3']

        temp = np.sum(SA)
        CPSA['FNSA1'] = CPSA['PNSA1'] / temp
        CPSA['FNSA2'] = CPSA['PNSA2'] / temp
        CPSA['FNSA3'] = CPSA['PNSA3'] / temp
        
        CPSA['FPSA1'] = CPSA['PPSA1'] / temp
        CPSA['FPSA2'] = CPSA['PPSA2'] / temp
        CPSA['FPSA3'] = CPSA['PPSA3'] / temp

        CPSA['WNSA1'] = CPSA['PNSA1'] * temp / 1000
        CPSA['WNSA2'] = CPSA['PNSA2'] * temp / 1000
        CPSA['WNSA3'] = CPSA['PNSA3'] * temp / 1000
        
        CPSA['WPSA1'] = CPSA['PPSA1'] * temp / 1000
        CPSA['WPSA2'] = CPSA['PPSA2'] * temp / 1000
        CPSA['WPSA3'] = CPSA['PPSA3'] * temp / 1000

        CPSA['TASA'] = np.sum(SA[idx1])
        CPSA['TPSA'] = np.sum(SA[idx2])

        CPSA['FrTATP'] = 0.0 if CPSA['TPSA'] == 0 else CPSA['TASA'] / CPSA['TPSA']
        CPSA['RASA'] = CPSA['TASA'] / temp
        CPSA['RPSA'] = CPSA['TPSA'] / temp

        idxmincharge = np.where(charge == np.min(charge))[0]
        RNCG = np.min(charge) / np.sum(charge[idxNeg])
        RPCG = np.max(charge) / np.sum(charge[idxPos])
        CPSA['RNCS'] = np.mean(SA[idxmincharge]) / RNCG
        CPSA['RPCS'] = np.mean(SA[idxmincharge]) / RPCG

        return CPSA
            

    def combine_dataframes(self):
        self.df = pd.concat(self.descriptor_dataframes, ignore_index=True)
        cols = ['adsorbate'] + [col for col in self.df.columns if col != 'adsorbate']
        self.df = self.df[cols]
        if self.save_data and self.csv_file:
            self.df.to_csv(self.csv_file, index=False)
        if self.verbose:
            print(self.df)
            

if __name__ == "__main__":
    
    csv_file = os.path.join(get_paths("database_path"), "label_data", "E_int_90_adsorbate_CPSA_new.csv")
    
    ## Defining List Of Adsorbates
    adsorbate_list = [
                    # 'A01',
                    # 'A02',
                    # 'A30',
                    # 'A27',
                    # 'A44',
                    # 'A83',
                    '254',
                    '264',
                      ]
    #    'A01':  '1-CH2OH-CHOH-CH2OH',
    #    'A02':  '2-CH2OH-COH-CH2OH',
    
    # adsorbate_list = list(ADSORBATE_TO_NAME_DICT)
    
    descriptors = calculateCPSAdescriptors(adsorbate_list = adsorbate_list,
                                        verbose = True,
                                        save_data = True,  # True, False
                                        csv_file = csv_file,
                                        )

