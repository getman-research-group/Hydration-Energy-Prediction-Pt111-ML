# -*- coding: utf-8 -*-
"""
import_tools.py
This contains codes on importing functions.

Classes:
    import_traj: class that can import trajectory information (uses md.load from mdtraj module)
        
"""

import os
import time

# Mdtraj to Read Trajectories
import mdtraj as md
from core.path import get_paths


# This Class Imports All Trajectory Information
class import_traj_mdtraj:
    '''
    INPUTS:
        directory: directory where the files located
        self.file_structure: name of the structure file
        lammpstrj_file: name of lammpstrj dump file
        want_only_directories: [bool, default = False] If True, this function will no longer load the trajectory. It will simply get the directory information
    
    OUTPUTS:
        # File Structure
            self.directory: directory the file is in
            
        # Trajectory Information
            self.traj: trajectory from md.traj
            self.topology: toplogy from traj
            self.residues: Total residues as a dictionary, {residue_name: number}
                e.g. {'HOH': 35}, 35 water molecules
            self.num_frames: [int] total number of frames
        
    Functions:
        load_traj_from_dir: Load trajectory from a directory
        print_traj_general_info: print the trajectory information
    '''
    
    ### Initializing
    def __init__(   self,
                    directory,
                    structure_file,
                    lammpstrj_file,
                    ):
        
        ## Storing Information
        self.directory = directory
        self.file_structure = structure_file
        self.file_lammpstrj = lammpstrj_file
        
        ## Start by Loading The Directory
        self.load_traj_from_dir()
        
        ## Print General Trajectory Information
        self.print_traj_general_info()
            
    ### Function to Load Trajectories
    def load_traj_from_dir(self):
        '''
        This function loads a trajectory given an lammpstrj, structure file, and a directory path
        INPUTS:
            self: class object
        OUTPUTS:
            self.traj: [class] trajectory from md.traj
            self.topology: [class] toplogy from traj
            self.num_frames: [int] total number of frames
        '''
        ## Checking Paths: if the Directory as a Slash at the End
        self.path_lammpstrj = os.path.join(self.directory, self.file_lammpstrj)
        self.path_structure = os.path.join(self.directory, self.file_structure)
    
        ## Print Loading Trajectory
        print('\n--- Loading trajectories from: %s'%(self.directory))
        print('    lammpstrj File: %s' %(self.file_lammpstrj))
        print('    Structure File: %s' %(self.file_structure) )
        
        ## Loading Trajectories
        start_time = time.time()
        self.traj =  md.load_lammpstrj(self.path_lammpstrj, top = self.path_structure)
        print("\n--- Total Time for MD Load is %.3f seconds ---" % (time.time() - start_time))
        
        ## Getting Topology
        self.topology = self.traj.topology
        
        ## Getting Total Time
        self.num_frames = len(self.traj)
        
        return
    
    ### Function to Print General Trajectory Information
    def print_traj_general_info(self):
        '''
        The function takes the trajectory and prints the residue names, corresponding number, and time length of the trajectory
        INPUTS:
            self: class object
        OUTPUTS:
            Printed output
        '''

        def findUniqueResNames(traj):
            ''' This function finds all the residues in the trajectory and outputs its unique residue name
            INPUTS:
                traj: trajectory from md.traj
            OUTPUTS:
                List of unique residues
            '''
            return list(set([ residue.name for residue in traj.topology.residues ]))

        def findTotalResidues(traj, resname):
            ''' This function takes the residue name and finds the residue indexes and the total number of residues
            INPUTS:
                traj: trajectory from md.traj
                resname: Name of the residue
            OUTPUTS:
                num_residues, index_residues
            '''
            # Finding residue index
            index_residues = [ residue.index for residue in traj.topology.residues if residue.name == resname ]
            
            # Finding total number of residues
            num_residues = len(index_residues)
            
            return num_residues, index_residues

        print("\n--- General Information about the Trajectory ---")
        print("%s\n"%(self.traj))
                  
        # The traj time given by MDTraj is wrong, is one-tenth of the correct length
        print("--- Time length of trajectory: %s ns"%(((self.traj.time[-1] - self.traj.time[0]) / 100)))
        
        print("\n--- Unit Cell Volumes: %.3f nm^3" % self.traj.unitcell_volumes[0])
        print("    Unit Cell Angles:  %i, %i, %i" % (self.traj.unitcell_angles[0][0], self.traj.unitcell_angles[0][1], self.traj.unitcell_angles[0][2]))
        print("    Unit Cell Lengths: %.3f nm, %.3f nm, %.3f nm" % (self.traj.unitcell_lengths[0][0], self.traj.unitcell_lengths[0][1], self.traj.unitcell_lengths[0][2]))
        
        ## Storing Total Residues
        self.residues = {}
        
        # Finding unique residues
        unique_res_names = findUniqueResNames(self.traj)
        
        for currentResidueName in unique_res_names:
            # Finding total number of residues, and their indexes
            num_residues, index_residues = findTotalResidues(self.traj, resname = currentResidueName)
            
            ## Storing
            self.residues[currentResidueName] = num_residues
            
            # Printing an output
            print("    Total number of residues for %s is: %s"%(currentResidueName, num_residues))

        return




if __name__ == "__main__":
    
    # Defining Path Information
    specific_dir = "A81"
    
    ### Defining File Names
    structure_file = r"prod2.pdb" # Structural file
    lammpstrj_file = r"prod.lammpstrj"
    
    ### Defining Path
    directory = os.path.join(get_paths('simulation_path'), 'lammps_data', specific_dir)
    
    ### Loading Trajectory
    traj_data = import_traj_mdtraj(directory = directory,              # Directory to analysis
                            structure_file = structure_file,    # Structure File
                            lammpstrj_file = lammpstrj_file,    # Trajectory
                            )

    