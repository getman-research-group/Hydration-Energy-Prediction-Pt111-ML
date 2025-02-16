# -*- coding: utf-8 -*-
"""
calc_tools.py

This script has functions that can operate across trajectories. General functions are listed below:
    ## Trajectory Tools
        find_total_residues: Finds the total number of residues and the corresponding indexes to them
        find_atom_index: Finds the atom index based on residue name and atom name
        find_atom_names: Finds atom names given the residue name
        find_specific_atom_index_from_residue: finds atom index from residue name
        find_residue_atom_index: Outputs residue and atom index for a residue of interest
        find_multiple_residue_index: finds multiple residues given a list -- outputs index and total residues
        find_center_of_mass: Calculates the center of mass of residue
        calc_ensemble_vol: calculates ensemble volume given the trajectory
        create_atom_pairs_list: creates atom pair list between two atom lists
        create_atom_pairs_with_self: creates atom pair list for a single atom list (you are interested in atom-atom interactions with itself)
        find_water_index: finds all water index (atom indices)
        calc_pair_distances_with_self_single_frame: calculates pair distances for a single frame
        calc_pair_distances_between_two_atom_index_list: calculates pair distances given two list of atom indices
        
    ## Splitting Trajectory Functions
        split_traj_function: splits the trajectory and calculates a value. This works well if you are receiving memory errors
        split_traj_for_avg_std: splits the trajectory so we can calculate an average and standard deviation
        calc_avg_std: calculates average and std of a list of dictionaries
        calc_avg_std_of_list: calculates average and standard deviation of a list
        split_list: splits list
        split_general_functions: splits calculations based on generalized inputs *** useful for cutting trajectories and computing X.
        
    ## Vector Algebra
        unit_vector: converts vectors to unit vectors
        angle_between: finds the angles between any two vectors in radians
        rescale_vector: rescales vectors and arrays from 0 to 1
        
    ## Equilibria
        find_equilibrium_point: finds equilibrium points for a given list
        
    ## Dictionary Functions
        merge_two_dicts: merges two dictionaries together
    
    ## Distances Between Atoms [ NOTE: These do not yet account for periodic boundary conditions! ]
        calc_xyz_dist_matrix: calculates xyz distance matrix given the coordinates
        calc_dist2_btn_pairs: calculates distance^2 between two pairs (taken from md.traj's numpy distances)
        calc_total_distance2_matrix: calculates total distance matrix^2. Note, distance^2 is useful if you only care about the minimum / maximum distances (avoiding sqrt function!)
            In addition, this function prevents numpy memory error by partitioning the atoms list based on how many you have. This is important for larger system sizes.
            
    ## Similarity Functions
        common_member_length: calculate common members between two arrays
    
"""

### Importing Functions
import time
import numpy as np
# import mdtraj as md
import core.initialize as initialize # Checks file path

### Function To Find Total Residues And The Index Of Those Residues
def find_total_residues(traj, resname):
    '''
    Take the residue name and find the residue indexes and the total number of residues
    INPUTS:
        traj: trajectory from md.traj
        resname: Name of the residue
    OUTPUTS:
        num_residues, index_residues
    '''
    # Finding Residue Index
    index_residues = [ residue.index for residue in traj.topology.residues if residue.name == resname ]
    
    # Finding Total Number of Residues
    num_residues = len(index_residues)
    
    return num_residues, index_residues

### Function To Find The Index Of Atom Names
def find_atom_index(traj, atom_name, resname = None):
    '''
    Find the atom index based on the residue name and atom name
    INPUTS:
        traj: trajectory from md.traj
        atom_name: [str] Name of the atom, e.g. 'O2'
        resname: [str, OPTIONAL, default = None] Name of the residue, e.g. 'HOH'
            NOTE: If resname is None, then this will look for all atomnames that match "atom_name" variable
    OUTPUTS:
        atom_index: [list] atom index corresponding to the residue name and atom name
    '''
    if resname is None:
        atom_index = [ each_atom.index for each_atom in traj.topology.atoms if each_atom.name == atom_name]
    else:
        atom_index = [ each_atom.index for each_atom in traj.topology.atoms if each_atom.residue.name == resname and each_atom.name == atom_name]
    return atom_index

### Function To Find Total Atoms And The Index Of Those Atoms Given Residue Name
def find_total_atoms(traj, resname):
    '''
    Take the residue names, find the atom indexes and the total number of atoms
    INPUTS:
        traj: [class] A trajectory loaded from md.load
        resname: [str] residue name, e.g. 'HOH'
    OUTPUTS:
        num_atoms: [int] Total number of atoms
        atom_index: [list] index of the atoms
    '''
    ## Finding Atom Index
    atom_index = [ each_atom.index for each_atom in traj.topology.atoms if each_atom.residue.name == resname]
    
    ## Finding Total Number Of Atoms
    num_atoms = len(atom_index)
    
    return num_atoms, atom_index

def find_adsorbate_oxygen_atoms(traj, residue_name, atom_type = 'O' ):
    ## Finding All Residue Indices
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    
    ## Finding All Atom Index If Matching
    atom_index = [ atom.index for each_residue_index in residue_index
                              for atom in traj.topology.residue(each_residue_index).atoms
                              if atom.element.symbol == atom_type ]
    
    num_atoms = len(atom_index)
    return num_atoms, atom_index

def find_water_hydrogen_atoms(traj, residue_name, atom_type = 'H' ):
    ## Finding All Residue Indices
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    
    ## Finding All Atom Index If Matching
    atom_index = [ atom.index for each_residue_index in residue_index
                              for atom in traj.topology.residue(each_residue_index).atoms
                              if atom.element.symbol == atom_type ]
    
    num_atoms = len(atom_index)
    print ('Number of Hydrogen atoms Found in Water: %s'% num_atoms)
    return num_atoms, atom_index

def find_water_oxygen_atoms(traj, residue_name, atom_type = 'O' ):
    ## Finding All Residue Indices
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    
    ## Finding All Atom Index If Matching
    atom_index = [ atom.index for each_residue_index in residue_index
                              for atom in traj.topology.residue(each_residue_index).atoms
                              if atom.element.symbol == atom_type ]
    
    num_atoms = len(atom_index)
    print ('Number of  Oxygen  atoms Found in Water: %s'% num_atoms)
    return num_atoms, atom_index
    
### Function To Find Residue Names
def find_residue_names( traj, ):
    '''
    Find the residue names of the molecules within a MD trajectory
    INPUTS:
        traj: [class] A trajectory loaded from md.load
    
    OUTPUTS:
        res_name: [str] list of strings of all the residue names
    
    '''
    return list(set([ residue.name for residue in traj.topology.residues ]))

## Function To Find All Atom Types For A Given Residue
def find_specific_atom_index_from_residue( traj, residue_name, atom_type = 'O' ):
    '''
    Find all atom indexes of a type of a specific residue.
    For instance, find all the oxygens for a given residue.
    INPUTS:
        traj: [md.traj]
            trajectory file from md.load
        residue_name: [str]
            name of the residue
        atom_type: [str, default = 'O']
            atom type we are interest in. Use the chemical symbol
        
    '''
    ## Finding All Residue Indices
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    ## Finding All Atom Index If Matching
    atom_index = [ atom.index for each_residue_index in residue_index
                              for atom in traj.topology.residue(each_residue_index).atoms
                              if atom.element.symbol == atom_type ]
    return atom_index

### Function To Find Total Adsorbates And Residues Given A Trajectory
def find_multiple_residue_index( traj, residue_name_list ):
    '''
    Find multiple residue indices and total number of residues given a list of residue name list
    INPUTS:
        traj: [md.traj]
            trajectory from md.traj
        residue_name_list: [list]
            residue names in a form of a list that is within the trajectory
    OUTPUTS:
        total_residues: [list]
            total residues of each residue name list
        residue_index: [list]
            list of residue indices
    '''
    # Creating Empty Array To Store
    total_residues, residue_index = [], []
    # Looping Through Each Possible Solvent
    for each_solvent_name in residue_name_list:
        ## Finding Total Residues
        each_solvent_total_residue, each_solvent_residue_index= find_total_residues(traj, resname = each_solvent_name)
        ## Storing
        total_residues.append(each_solvent_total_residue)
        residue_index.append(each_solvent_residue_index)
    return total_residues, residue_index

### Function To Find Atom Names
def find_atom_names(traj, residue_name):
    '''
    Find the atom names given the residue name
    INPUTS:
        traj: trajectory file from md.load
        residue_name: [str] name of the residue
    OUTPUTS:
        atom_names: [list] list of strings of the atom names
    '''
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name][0]
    atom_names = [ atom.name for atom in traj.topology.residue(residue_index).atoms ]
    return atom_names

### Function To Find Unique Residue Names
def find_unique_residue_names(traj):
    '''
    Find all the residues in the trajectory and outputs its unique residue name
    INPUTS:
        traj: trajectory from md.traj
    OUTPUTS:
        List of unique residues
    '''
    return list(set([ residue.name for residue in traj.topology.residues ]))

### Function To Find Residue / Atom Index Given Residue Name And Atom Names
def find_residue_atom_index(traj, residue_name = 'HOH', atom_names = None):
    '''
    Look at the trajectory's topology and find the atom index that we care about.
    INPUTS:
        traj: trajectory from md.traj
        residue_name: [str] residue name (i.e. 'HOH')
        atom_names: [str, default = None]
            list of atom names within the residue (i.e. ['O','H1','H2'])
            If None, then just find all possible atoms from the residue index
    OUTPUTS:
        residue_index: list of residue index
        atom_index: list of atom index
    '''
    ## Finding All Residues Of The Type
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    ## Finding All Atoms That We Care About
    if atom_names == None:
        atom_index = [ [ atom.index for atom in traj.topology.residue(res).atoms ] for res in residue_index ]
    else:
        atom_index = [ [ atom.index for atom in traj.topology.residue(res).atoms if atom.name in atom_names ] for res in residue_index ]
    
    return residue_index, atom_index


### Function To Find Center Of Mass Of The Residues / Atoms
def find_center_of_mass( traj, residue_name = 'HOH', atom_names = ['O','H1','H2'] ):
    '''
    Find the center of mass of the residues given the residue name and atom names.
    atom_names is a list.
    INPUTS:
        traj: trajectory from md.traj
        residue_name: [str] residue name (i.e. 'HOH')
        atom_names: list of atom names within the residue (i.e. ['O','H1','H2'])
    OUTPUTS:
        center_of_mass: Numpy array of the center of masses ( Num Frames × Num Atoms × 3 (X,Y,Z))
    NOTES:
        This function may have issues later due to the high vectorization approach!
    '''
    ## Keeping Track Of Time
    COM_time = time.time()
    ## Initialization Of COM
    ## Finding Atom And Residue Index
    residue_index, atom_index = find_residue_atom_index(traj, residue_name, atom_names)
    ## Finding Mass Of a Single Group
    atom_mass = [ traj.topology.atom(atom_ind).element.mass for atom_ind in atom_index[0]]
    ## Finding Total Mass
    totalMass = np.sum(atom_mass)
    print("--- COM Calculation for %s Frames, %s Residue (%s residues), and %s Atom Types (%s atoms) ---" %
                                                        (   len(traj),                              # Number of Frames
                                                            residue_name,                           # Residue Name
                                                            len(residue_index),                     # Number of Residue
                                                            atom_names,                             # Atom Names (list)
                                                            len(residue_index) * len(atom_names)    # Total Number Of Atoms In Selected Residues
                                                        )   )
    
    ### Center of Mass (COM) Calculation
    
    ## Finding Position Of All Atoms
    position_all_atoms = traj.xyz[:, atom_index] # select atom_index column; Frame × atom × positions
    
    ## Getting Shape Of All Positions
    n_frames, n_residues, n_atoms, n_coordinates = position_all_atoms.shape
    
    ## Replicating Masses For Matrix Multiplication
    rep_mass = np.tile(np.transpose(atom_mass).reshape((n_atoms, 1)), (n_frames, n_residues, 1, 1)) # 1 for x,y,z coordinates already defined
    
    ## Multiplying To Get M_i * x_i
    multiplied_numerator = position_all_atoms * rep_mass
    
    ## Summing All M_i * X_i
    summed_numerator = np.sum(multiplied_numerator, axis=2 ) # Summing within each of the residues
    
    ## Dividing Numerator To Get COM
    center_of_mass = summed_numerator / totalMass
    
    ## Printing Total Time Taken
    h, m, s = initialize.convert2HoursMinSec( time.time() - COM_time )
    print('Total time for COM calculation was: %d hours, %d minutes, %d seconds \n' %(h, m, s))
    
    return center_of_mass

### Function To Calculate The Ensemble Volume Of A Trajectory
def calc_ensemble_vol( traj ):
    '''
    Take the trajectory and find the ensemble average volume.
    INPUTS:
        traj: trajectory
    OUTPUTS:
        ensemVol: Ensemble volume, typically nm ^ 3
    '''
    vol = np.mean(traj.unitcell_volumes)
    return vol
    # # For Cube Cell Calculation:
    # # List of all unit cell lengths
    # unitCellLengths = traj.unitcell_lengths
    # unitCellVolumes = unitCellLengths * unitCellLengths * unitCellLengths # Assuming cubic
    # # Now, Using Numpy To Find Average
    # vol = np.mean(unitCellVolumes)
    # return vol

### Function To Create Atom Pairs List
def create_atom_pairs_list(atom_1_index_list, atom_2_index_list):
    '''
    Create all possible atom pairs between two lists, it uses numpy to speed up atom pair generation list
    This function is especially useful when we are generating atom lists for distances, etc.
    This function is way faster than using list comprehensions. (due to numpy using C++ to quickly do computations)
    INPUTS:
        atom_1_index_list: [np.array, shape = (N, 1)] index list 1, e.g. [ 0, 1, 4, .... ]
        atom_2_index_list: [np.array, shape = (N, 1)] index list 2, e.g. [ 9231, ...]
    OUTPUTS:
        atom_pairs: [np.array, shape = (N_pairs, 2)] atom pairs when varying index 1 and 2
            e.g.:
                [ [0, 9231 ],
                  [1, 9321 ], ...
                  ]
    '''
    ## Creating Meshgrid Between The Atom Indexes
    xv, yv = np.meshgrid(atom_1_index_list,  atom_2_index_list)

    ## Stacking The Arrays
    array = np.stack( (xv, yv), axis = 1)

    ## Transposing Array
    array_transpose = array.transpose(0, 2, 1) #  np.transpose(array)

    ## Concatenating All Atom Pairs
    atom_pairs = np.concatenate(array_transpose, axis = 0)
    
    ## RETURNS: (N_PAIRS, 2)
    return atom_pairs

### Function To Create Atom Pairs With Self Atoms, such as Pt-Pt, Or Any Other Structural Objects
def create_atom_pairs_with_self(indices):
    '''
    Create atom pairs for a set of atoms with itself.
    For example, we may want the atom indices of Pt atoms to Pt atoms, but do not want the distance calculations to repeat.
    This script is useful for those interested in generating atom pairs for a list with the input list itself.
    There are no repeats in the atom indices here. an atom cannot interact with itself.
    
    INPUTS:
        indices: [np.array, shape = (num_atoms, 1)] Index of the atoms
    
    OUTPUTS:
        atom_pairs: [np.array, shape = (N_pairs,2)] atom pairs when varying indices, but NO repeats!
        e.g.: Suppose we have atoms [0, 1, 2], then the list will be:
            [[0, 1],
             [0, 2],
             [1, 2]]
        upper_triangular: [np.array] indices of the upper triangular matrix, which you can use to create matrix
    '''
    ## Finding Number Of Atoms
    num_atoms = len(indices)
    
    ## Defining A Upper Trianglar Matrix
    upper_triangular = np.triu_indices(num_atoms, k = 1)
    
    ## Finding Atom Indices
    atom_indices = np.array(upper_triangular).T

    ## Correcting Atom Indices Based On Input Atom Index
    atom_pairs = indices[atom_indices]
    
    return atom_pairs, upper_triangular


### Function To Find The Distances For Single Frame
def calc_pair_distances_with_self_single_frame(traj, atom_index, frame = -1, periodic = True):
    '''
    Calculate the pair distances based on a trajectory of coordinates
    
    NOTES:
        - This function finds the pair distances based on the very last frame. (editable by changing frame)
        - Therefore, this function only calculates pair distances for a single frame to improve memory and processing requirements
        - The assumption here is that the pair distances does not significantly change over time. In fact, we assume no changes with distances.
        - This was developed for gold-gold distances, but applicable to any metallic or strong bonded systems
    INPUTS:
        traj: trajectory from md.traj
        atom_index: [np.array, shape=(num_atoms, 1)] atom indices that you want to develop a pair distance matrix for
        frame: [int, default=-1] frame to calculate gold-gold distances
        periodic: [bool, default=True] True if you want PBCs to be accounted for
    OUTPUTS:
        distance_matrix: [np.array, shape=(num_atom,num_atom)] distance matrix of gold-gold, e.g.
        e.g.
            [ 0, 0.15, ....]
            [ 0, 0   , 0.23, ...]
            [ 0, ... , 0]
    '''
    ## Finding Total Number Of Atoms
    total_atoms = len(atom_index)
    ## Creating Atom Pairs
    atom_pairs, upper_triangular_indices = create_atom_pairs_with_self( atom_index )
    ## Calculating Distances
    distances = md.compute_distances( traj = traj[frame], atom_pairs = atom_pairs, periodic = periodic, opt = True )
    ## Reshaping Distances Array To Make a Matrix
    ## Creating Zeros Matrix
    distances_matrix = np.zeros( (total_atoms, total_atoms) )
    ## Filling Distance Matrix
    distances_matrix[upper_triangular_indices] = distances[0]
    return distances_matrix

### Function To Calculate Distances For A Single Frame Using md.traj
def calc_pair_distances_between_two_atom_index_list(traj, atom_1_index, atom_2_index, periodic = True):
    '''
    Calculate distances between two atom indexes
    NOTES:
        - This function by default calculates pair distances of the last frame
        - This is designed to quickly get atom indices
        - This function is expandable to multiple frames
    INPUTS:
        traj: [class]
            trajectory from md.traj
        atom_1_index: [np.array, shape = (num_atoms, 1)]
            atom_1 type indices
        atom_2_index: [np.array, shape = (num_atoms, 1)]
            atom_2 type indices
        periodic: [bool, default = True]
            True if want PBCs to be accounted for
    OUTPUTS:
        distances: [np.array, shape=(num_frame, num_atom_1, num_atom_2)]
        distance matrix with rows as atom_1 and col as atom_2.
    '''
    ## Finding Total Number Of Atoms
    total_atom_1 = len(atom_1_index)
    total_atom_2 = len(atom_2_index)
    
    ## Finding Total Frames
    total_frames = len(traj)
    
    ## Generating Atom Pairs
    atom_pairs = create_atom_pairs_list(atom_2_index, atom_1_index)
    ''' RETURNS ATOM PAIRS LIKE:
            [
                [ATOM_1_IDX_1, ATOM2_IDX_1],
                [ATOM_1_IDX_1, ATOM2_IDX_2],
                                ...
                [ATOM_1_IDX_N, ATOM2_IDX_N],
            ]
    '''
    ## Calculating Distances
    distances = md.compute_distances(   traj = traj,
                                        atom_pairs = atom_pairs,
                                        periodic = periodic )
    ## Returns Timeframe × (Num_ATOM_1 × Num_GOLD) Numpy Array
    ## Reshaping The Distances
    distances = distances.reshape(total_frames, total_atom_1, total_atom_2)
    ## Returns: Timeframe × Num_Atom_1 × Num_Atom 2 (3D Matrix)
    return distances
    

### Function To Find Water Residue Index And Atom Index
def find_water_index(traj, water_residue_name = 'HOH'):
    '''
    Find the residue index and atom index of water
    INPUTS:
        traj: [class] trajectory from md.traj
        water_residue_name: [str] residue name for water
    OUTPUTS:
        num_water_residues: [int] total number of water molecules
        water_residue_index: [list] list of water residue index
        water_oxygen_index: [np.array] atom list index of water oxygen
    '''
    ## Finding Residue Index Of Water
    num_water_residues, water_residue_index = find_total_residues(traj = traj, resname = water_residue_name)
    ## Finding All Oxygens Indexes Of Water
    water_oxygen_index = np.array(find_atom_index( traj, resname = water_residue_name, atom_name = 'O'))
    
    return num_water_residues, water_residue_index, water_oxygen_index


#######################################
### Splitting Trajectory Functions ####
#######################################

### Function That Splits The Trajectory, Calculates Using Some Function, Then Outputs as Numpy Array
def split_traj_function( traj, split_traj = 50, input_function = None, optimize_memory = False, **input_variables):
    '''
    Split the trajectory up assuming that the input function is way too expensive to calculate via vectors
    INPUTS:
        traj: trajectory from md.traj
        input_function: input function. The input function assumes to have a trajectory input.
        Furthermore, the output of the function is a numpy array, which will be the same length as the trajectory.
        input_variables: input variables for the function
        optimize_memory: [bool, default = False] If True, we will assume the output is a numpy array. Then, we will pre-allocate memory so you do not raise MemoryError.
            NOTES: If this does not work, then you have a difficult problem! You have way too much data to appropriately do the calculations. If this is the case, then:
                    - pickle the output into segments
                    - Do analysis for each segment (hopefully, you will not have to do this!)
    OUTPUTS:
        output_concatenated: (numpy array) Contains the output values from the input functions
    SPECIAL NOTES:
        The function you input should be a static function! (if within class) The main idea here is that we want to split the trajectory and calculate something.
        The output should be simply a numpy array with the same length as the trajectory!
    '''
    ## Importing Modules
    import sys
    import time
    
    ## Printing
    from core.initialize import convert2HoursMinSec
    ## Checking Input Function
    if input_function is None:
        print("Error in using split_traj_function! Please check if you correctly split the trajectory!")
        sys.exit()
    else:
        ## Finding Trajectory Length
        traj_length = len(traj)
        ## Printing
        print("*** split_traj_function for function: %s ***"%(input_function.__name__))
        print("Splitting trajectories for each %d intervals out of a total of %d frames"%(split_traj, traj_length))
        ## Storing Time
        start_time = time.time()
        ## Creating Split Regions
        split_regions = [[i,i + split_traj] for i in range(0, traj_length, split_traj)]
        ## Splitting Trajectory Based On Inputs
        traj_list = [ traj[regions[0]:regions[1]] for regions in split_regions]

        ## Creating Blank List
        if optimize_memory == False:
            output_storage = []
        else:
            print("Optimization memory has been enabled! Creating empty array to fit the matrix!")
        ## Looping Through The Trajectory
        for index, current_traj in enumerate(traj_list):
            ## Keep Track Of Current Time
            current_time = time.time()
            ## Printing And Keeping Track Of Time
            print("%s: WORKING ON TRAJECTORIES %d ps TO %d ps OUT OF %d ps"%(input_function.__name__,current_traj.time[0], current_traj.time[-1], traj.time[-1]))
            ## Running Input Functions
            output = input_function(current_traj, **input_variables)
            
            ## Storing Output To Corresponding Output Storage Space
            if optimize_memory == False:
                output_storage.append(output)
            else:
                ## If First Frame, Create The Matrix
                if index == 0:
                    ## FINDING SHAPES
                    output_shape = output.shape[1:] ## OMITTING TIME
                    ## FINDING FULL SHAPE
                    full_shape = tuple([traj_length] + list(output_shape))
                    print("CREATING MATRIX OF ARRAY SIZE: %s"%(', '.join([str(each) for each in full_shape ]) ) )
                    ## CREATING EMPTY ARRAY
                    output_storage = np.empty( full_shape )
                ## STORING ARRAY
                output_storage[ split_regions[index][0]:split_regions[index][1], : ] = output[:]
                                
            h, m, s = convert2HoursMinSec(time.time() - current_time)
            ## PRINTING TOTAL TIME
            print("---------> %d hours, %d minutes, %d seconds"%(h,m,s))
        if optimize_memory == False:
            ## FINALLY, Concatenating To Combine All The Output
            output_storage = np.concatenate(output_storage, axis = 0)
        ## Writing Total Time
        h, m, s = convert2HoursMinSec( time.time() - start_time)
        print('TOTAL TIME ELAPSED FOR %s: %d hours, %d minutes, %d seconds  '%(input_function.__name__, h, m, s))
        ##TODO: can re-adjust script to increase intervals if it feels confident -- could be a while loop criteria
    return output_storage


### Function That Splits A List Into Multiple Parts
def split_list(alist, wanted_parts = 1):
    '''
    Split a larger list into multiple parts
    INPUTS:
        alist: [list] original list
        wanted_parts: [int] number of splits
    OUTPUTS:
        List containing chunks of the list
    Reference:
        https://stackoverflow.com/questions/752308/split-list-into-smaller-lists?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    '''
    length = len(alist)
    return [ alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
             for i in range(wanted_parts) ]

### Function That Splits The Trajectory, Runs Calculations, Then Takes Average And std Of Each Calculation
def split_traj_for_avg_std( traj, num_split, input_function, split_variables_dict, **static_variables):
    '''
    Split a trajectory into multiple parts, use a function to do some calculations, then average/std the results
    INPUTS:
        traj: trajectory from md.traj
        num_split: number of times to split the trajectory
        input_function: input function. Note that the input function assumes to have a trajectory input.
        split_variables_dict: variables that want to split. This for example could be center of masses across trajectory (needs to be split!)
            NOTE: This is a dictionary. The dictionary should coincide with the input names of the function
            e.g.:
                input_split_vars = { 'COM_Adsorbate'   : self.rdf.adsorbate_COM,                          # adsorbate center of mass
        **static_variables: variables that does not change when you split the trajectory
            e.g.
                input_static_vars = { 'adsorbate_res_index'    : self.rdf.adsorbate_res_index,            # Residue index for the adsorbate
    OUTPUTS:
        output: results as a form of a list
    '''
    ## Printing
    print("----- split_traj_avg_std -----")
    print("Working On Trajectory With Time Length Of: %d ps"%(traj.time[-1] - traj.time[0]))
    print("Splitting Trajectory In %d Pieces:"%(num_split))
    ## Splitting The Trajectory
    split_traj = split_list(traj, num_split)
    ## Splitting The Variables
    for each_variable in split_variables_dict.keys():
        print("--> Splitting Variable %s"%(each_variable))
        split_variables_dict[each_variable] = split_list(split_variables_dict[each_variable], num_split)
    
    ## Creating List To Store Output
    output = []
    
    ## Looping Through Each Trajectory
    for index, each_traj in enumerate(split_traj):
        ## Printing
        print("WORKING ON TRAJECTORY: %s : %d ps to %d ps"%(input_function.__name__, each_traj.time[0], each_traj.time[-1] ))
        ## Getting The Variables
        current_split_variables_dict = {key:value[index] for (key,value) in split_variables_dict.items()}
        ## Inputting To Function
        current_output = input_function( each_traj, **merge_two_dicts(current_split_variables_dict, static_variables) )
        ## Storing Output
        output.append(current_output)

    return output
    
### Function That Splits Dictionary Based On Trajectory And Runs Variables
def split_general_functions( input_function, split_variable_dict, static_variable_dict, num_split = 1 ):
    '''
    Split a trajectory, corresponding variables, and run the function again. The outputs of the functions will be stored into a list.
    INPUTS:
        input_function: [function]
            input function. Note that the input function assumes you will import split variables and static variables
        num_split: [int, default = 1]
            number of times to split the trajectory. Default =  1 means no splitting
        split_variables_dict: variables that you WANT to split. This for example could be center of masses across trajectory (needs to be split!)
            NOTE: This is a dictionary. The dictionary should coincide with the input names of the function
            e.g.:
                input_split_vars = { 'COM_Solute'   : self.rdf.solute_COM,                          # Solute center of mass
        static_variables_dict: variables that does not change when you split the trajectory
            e.g.
                input_static_vars = { 'solute_res_index'    : self.rdf.solute_res_index,                        # Residue index for the solute
    OUTPUTS:
        output: [list]
            output in a form of a list. Note that if you had multiple arguments, it will output as a list of tuples.
    '''
    ## Splitting Variables
    for each_variable in split_variable_dict.keys():
        print("--> SPLITTING VARIABLE %s"%(each_variable))
        split_variable_dict[each_variable] = split_list(split_variable_dict[each_variable], num_split)
        
    ## Creating List To Store The Outputs
    output = []
    
    ## Looping Through The Splitted Files
    for index in range(num_split):
        ## GETTING VARIABLES AND COMBINING
        current_split_variables_dict = {key:value[index] for (key,value) in split_variable_dict.items()}
        ## MERGING TWO DICTIONARIES
        merged_dicts = merge_two_dicts(current_split_variables_dict, static_variable_dict)
        ## RUNNING INPUT FUNCTION AND STORING
        output.append( input_function( **merged_dicts ) )
        
    return output

### Function To Calculage Avg And Std Accordingly
def calc_avg_std(list_of_dicts):
    '''
    Calculate the average and standard deviation of several values.
    ASSUMPTION:
        We are assuming that the input is a list of dictionaries:
            [{'var1': 2}, {'var1':3} ] <-- We want to average var1, etc.
        Furthermore, we are assuming that each dictionaries should have more or less the same keys (otherwise, averaging makes no sense.)
    INPUTS:
        list_of_dicts: [list] List of dictionaries
    OUTPUTS:
        avg_std_dict: [dict] Dictionary with the same keys, but each key has the following:
            'avg': average value
            'std': standard deviation
    '''
    avg_std_dict = {}
    ## Looping Through The Dictionary
    for dict_index, each_dict in enumerate(list_of_dicts):
        ## GETTING THE KEYS
        current_keys = each_dict.keys()
        ## Looping Through Each Key
        for each_key in current_keys:
            ## ADDING KEY IF IT IS NOT INSIDE
            if each_key not in avg_std_dict.keys():
                avg_std_dict[each_key] = [list_of_dicts[dict_index][each_key]]
            ## APPENDING IF WE ALREADY HAVE THE KEY
            else:
                avg_std_dict[each_key].append(list_of_dicts[dict_index][each_key])
    
    ## ADD THE END, TAKE AVERAGE AND STANDARD DEVIATION
    avg_std_dict = {key: {'avg': np.mean(value), 'std':np.std(value)} for (key,value) in avg_std_dict.items()}
    return avg_std_dict

### FUNCTION TO CALCULATE AVG AND STANDARD DEVIATION OF A VARIABLE ACROSS MULTIPLE FRAMES
def calc_avg_std_of_list( traj_list ):
    '''
    Calculate the average and standard deviation of a list (e.g. a value that fluctuates across the trajectory)
    INPUTS:
        traj_list: [np.array or list] list that you want average and std for
    OUTPUTS:
        avg_std_dict: [dict] dictionary with the average ('avg') and standard deviation ('std')
    NOTES:
        - This function takes into account 'nan', where non existent numbers are not considered in the mean or std
    '''
    ## TESTING IF LIST HAS NON EXISTING NUMBERS (NANS). IF SO, AVERAGE + STD WITHOUT THE NANs
    if np.any(np.isnan(traj_list)) == True:
        ## NON EXISTING NUMBERS EXIST
        avg = np.nanmean(traj_list)
        std = np.nanstd(traj_list)
    else:
        ## FINDING AVERAGE AND STANDARD DEVIATION
        avg = np.mean(traj_list)
        std = np.std(traj_list)
    
    ## STORING AVERAGE AND STD TO A DICTIONARY
    avg_std_dict = { 'avg': avg,
                     'std': std,}
    return avg_std_dict



####################################
########## Vector Algebra ##########
####################################

### Function To Convert A Vector Of Any Length To A Unit Vector
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

### Function To Get The Angle Between Two Vectors In Radians
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
        Uses dot product , where theta = arccos ( unitVec(A) dot unitVec(B))
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

### Function To Rescale Vector Between 0 And 1
def rescale_vector(vec):
    '''
    Rescale a vector between 0 and 1
    INPUTS:
        vec: [np.array, shape = (N,1)] numpy array that we want to be normalized
    OUTPUTS:
        rescaled_vec: [np.array, shape = (N,1)] numpy array that has been rescaled between 0 and 1
    '''
    ## Finding New Vector Using Minima And Maxima
    rescaled_vec =  ( vec - np.min(vec) ) / (np.max(vec) - np.min(vec))
    return rescaled_vec

### Function To Find Equilibrium Point
def find_equilibrium_point(ylist = [], tolerance = 0.015 ):
    '''
    Take the y-values, and find some equilibrium point based on some tolerance. This does a running average and sees if the value deviates too far.
    INPUTS:
        ylist: yvalues as a list
        tolerance: tolerance for the running average
    OUTPUTS:
        index_of_equil: Index of the ylist where it has equilibrated
    EXAMPLE:
        Suppose we have a radial distribution function and we want to find when it equilibrated. We do this by:
        - Reverse the list of the y-values
        - Find when the average is off a tolerance
    '''
    # Reversing list
    ylist_rev = list(reversed(ylist[:]))
    
    # Pre-for loop
    # Starting counter
    counter = 1
    endPoint = len(ylist_rev)
    
    # Assuming initial values
    runningAvg = ylist_rev[0] # First value
    nextValue = ylist_rev[counter] # Next Value
    
    while abs(runningAvg-nextValue) < tolerance and counter < endPoint - 1:
        # Finding new running average
        runningAvg = (runningAvg * counter + nextValue) / (counter + 1)
        
        # Adding to counter
        counter += 1
               
        # Finding the next energy
        nextValue = ylist_rev[counter]
        
    # Going back one counter, clearly the one that worked last
    correct_counter = counter - 1
    
    # Getting index of the correct list
    index_of_equil = endPoint - correct_counter - 1 # Subtracting 1 because we count from zero
    
    return index_of_equil


### Function To Merge Two Dictionaries
def merge_two_dicts(x, y):
    '''
    The purpose of this function is to merge two dictionaries
    INPUTS:
        x, y: dictionaries
    OUTPUTS:
        z: merged dictionary
    '''
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

### Function To Calculate XYZ Distance Matrix Between All Coordinate Atoms
def calc_xyz_dist_matrix(coordinates):
    '''
    Take coordinates and find the difference between i and j
    INPUTS:
        Coordinates - Numpy array
    OUTPUTS:
        ΔX array - Array for x-differences (j-i)
        ΔY array - Array for y-differences (j-i)
        ΔZ array - Array for z-differences (j-i)
    '''
    def makeDistArray(Vector):
        '''
        Take a vector of x's, y's, or z's and creates a distance matrix for them
        INPUTS:
            Vector - A list of x coordinates for example
        OUTPUTS:
            Array - Distance matrix j - i type
        '''
        vectorSize = len(Vector)
        
        # Start by creating a blank matrix
        myArray = np.zeros( (vectorSize, vectorSize ) )
        
        # Use for-loop to input into array and find distances
        for i in range(0, vectorSize - 1):
            for j in range(i, vectorSize):
                myArray[i, j] = Vector[j] - Vector[i]
        return myArray

    deltaXarray = makeDistArray(coordinates.T[0])  # X-values
    deltaYarray = makeDistArray(coordinates.T[1])  # Y-values
    deltaZarray = makeDistArray(coordinates.T[2])  # Z-values
    return deltaXarray, deltaYarray, deltaZarray


### Function To Calculate Distance Between Pairs (TAKEN FROM MD.TRAJ)
def calc_dist2_btn_pairs(coordinates, pairs):
    """
    Distance squared between pairs of points in each coordinate
    INPUTS:
        coordinates: N × 3 numpy array
        pairs: M × 2 numpy array, which are pairs of atoms
    OUTPUTS:
        distances: distances in the form of a M × 1 array
    """
    delta = np.diff(coordinates[pairs], axis = 1)[:, 0]
    return (delta ** 2.).sum(-1)

### Function to Calculate Total Distance Matrix
def calc_total_distance2_matrix(coordinates, force_vectorization = False, atom_threshold = 2000):
    '''
    This function calls for calc_xyz_dist_matrix and simply uses its outputs to calculate a total distance matrix that is squared
    INPUTS:
        coordinates: numpy array (n × 3)
        force_vectorization: If True, it will force vectorization every time
        atom_threshold: threshold of atoms, if larger than this, we will use for loops for vectorization
    OUTPUTS:
        dist2: distance matrix (N × N)
    '''
    ## Finding Total Length Of Coordinates
    total_atoms = len(coordinates)
    
    ## Seeing If We Need To Use Loops For Total Distance Matrix
    if total_atoms < atom_threshold:
        num_split = 0
    else:
        num_split = int(np.ceil(total_atoms / atom_threshold))
        
    if num_split == 0 or force_vectorization is True:
        deltaXarray, deltaYarray, deltaZarray = calc_xyz_dist_matrix(coordinates)
        # Finding total distance^2
        dist2 = deltaXarray*deltaXarray + deltaYarray*deltaYarray + deltaZarray*deltaZarray
    else:
        print("Since number of atoms > %s, we are shortening the atom list to prevent memory error!"%(atom_threshold))
        ## SPLITTING ATOM LIST BASED ON THE SPLITTING
        atom_list = np.array_split( np.arange(total_atoms), num_split )
        ## CREATING EMPTY ARRAY
        dist2 = np.zeros((total_atoms, total_atoms))
        total_atoms_done = 0
        ## LOOPING THROUGH EACH ATOM LIST
        for index, current_atom_list in enumerate(atom_list):
            ## FINDING MAX AN MINS OF ATOM LIST
            atom_range =[ current_atom_list[0], current_atom_list[-1] ]
            ## FINDING CURRENT TOTAL ATOMS
            current_total_atoms = len(current_atom_list)
            ## PRINTING
            print("--> WORKING ON %d ATOMS, ATOMS LEFT: %d"%(current_total_atoms, total_atoms - total_atoms_done) )
            ## GETTING ATOM PAIRS
            pairs = np.array([[y, x] for y in range(atom_range[0], atom_range[1]+1 ) for x in range(y+1, total_atoms) ] )
            ## CALCULATING DISTANCES
            current_distances = calc_dist2_btn_pairs(coordinates, pairs)
            ## ADDING TO TOTAL ATOMS
            total_atoms_done += current_total_atoms
            ## STORING DISTANCES
            dist2[pairs[:,0],pairs[:,1]] = current_distances
            
            ##TODO: inclusion of total time
            ##TODO: optimization of atom distances
    return dist2

##########################################
########## SIMILARITY FUNCTIONS ##########
##########################################

### Function To Find The Length Of Total Members
def common_member_length(a, b):
    return len(np.intersect1d( a, b ))

### Function To Flatten List Of List
def flatten_list_of_list( my_list ):
    ''' This flattens list of list '''
    return [item for sublist in my_list for item in sublist]