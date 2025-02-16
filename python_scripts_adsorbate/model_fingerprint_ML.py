'''
This script is used to run the machine learning models on the molecular descriptors

'''
import os
import csv

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul
import shap

# Import Machine Learning Algorithms
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA

# Importing sklearn Libraries
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GroupKFold, learning_curve
from sklearn.inspection import permutation_importance

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV
from sklearn.model_selection import GridSearchCV
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
# https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV
from skopt import BayesSearchCV

# Importing Custom Libraries
from core.path import get_paths
from core.model_vars import MODEL_DICT
from core.parameter_dict_bayesian import PARAMETER_DICT_BAYESIAN
from core.parameter_dict_grid import PARAMETER_DICT_GRID
from core.parameter_dict_random import PARAMETER_DICT_RANDOM
from core.parameter_dict_best import PARAMETER_BEST

# Class to run the machine learning models on the molecular descriptors
class Adsorbate_FP_Model:
    
    ## Initializing
    def __init__(self,
                 path_input,
                 path_label,
                 path_output_csv,
                 path_output_figures,
                 algorithm_list,
                 fingerprint_type,
                 sort_type,
                 search_type,
                 plot_learning_curve = False,
                 save_predictions_csv = False,
                 plot_explained_variance = False,
                 ):
        
        # Initializing
        self.path_input = path_input
        self.path_label = path_label
        self.path_output_csv = path_output_csv
        self.path_output_figures = path_output_figures
        
        self.algorithm_list = algorithm_list
        self.fingerprint_type = fingerprint_type
        self.sort_type = sort_type
        self.search_type = search_type
        self.plot_learning_curve = plot_learning_curve
        self.save_predictions_csv = save_predictions_csv
        self.plot_explained_variance = plot_explained_variance
        
        # Model Storage
        self.model_storage = {}
        
        # Using Pandas To Read File
        self.file_matrix = pd.read_csv(os.path.join(self.path_input,
                                                 f'{self.fingerprint_type}-flattened-{self.sort_type}.csv'))

        self.file_label = pd.read_csv(os.path.join(self.path_label, 'E_int_90.csv'))
        self.file_label = self.file_label[['adsorbate', 'e_int_dft']]
        
        # Defining X Data
        self.X = self.file_matrix.drop('adsorbate', axis=1).to_numpy()
        self.y = self.file_label['e_int_dft'].to_numpy()
        
        self.names = self.file_matrix['adsorbate'].to_numpy()
        
        ## get indicdes of training and testing set
        self.cv_indices_dict = self.get_cross_validation_indices()

        # Run nested cross-validation
        for algorithm in self.algorithm_list:
            self.cross_validation_SearchCV(algorithm,
                                           cv_indices = self.cv_indices_dict,
                                           n_splits_inner = 5
                                           )
        
        # Combine the results from the 5 folds
        self.combined_model_storage = self.combine_model_storages()
            
        # Save the predictions to CSV files
        if self.save_predictions_csv:
            self.save_predictions_to_csv()
        
        # Plot explained variance
        if self.plot_explained_variance:
            self.plot_PCA_explained_variance()
    
    ## Cross Validation Based on Random
    def get_cross_validation_indices(self, n_splits = 5):
        # Create a KFold object with the specified number of splits
        kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

        # Dictionary to store the indices of the training and testing sets for each fold
        cv_indices = {}

        # Iterate over the indices of the training and testing sets for each fold
        for index, (train_index, test_index) in enumerate(kf.split(self.X)):
            fold_key = f"fold_{index}"  # Creating a unique key for each fold
            cv_indices[fold_key] = {
                'train_index': train_index,
                'test_index': test_index,
                'train_names': self.names[train_index],
                'test_names': self.names[test_index]
            }

        return cv_indices
    
    
    # Nested Cross Validation
    def cross_validation_SearchCV(self,
                                algorithm,
                                cv_indices,
                                n_splits_inner = 5,
                                ):

        # Load the parameter dict for the specified algorithm based on the search type
        if self.search_type == 'grid':
            param_grid = PARAMETER_DICT_GRID[algorithm]
        elif self.search_type == 'random':
            param_grid = PARAMETER_DICT_RANDOM[algorithm]
        elif self.search_type == 'bayesian':
            param_grid = PARAMETER_DICT_BAYESIAN[algorithm]
        elif self.search_type == 'best_model':
            param_grid = PARAMETER_BEST[algorithm]

        # Calculate the number of combinations, if not using Bayesian search
        if self.search_type != 'bayesian':
            num_combinations = reduce(mul, (len(v) for v in param_grid.values()), 1)
            print(f"--- Total number of hyperparameter combinations for {algorithm}: {num_combinations}")

        # Add the 'model__' prefix to the keys in the parameter grid
        param_grid = {'model__' + key: value for key, value in param_grid.items()}

        # Add parameters for the PCA
        param_grid['pca__n_components'] = [10, 15, 20, 25, 30]  # Use 0.95 to automatically select the number of components to preserve 95% variance

        # Add parameter for the scaler
        param_grid['scaler'] = [
                                # StandardScaler(),
                                MinMaxScaler()
                                ]

        print(f"--- Hyperparameters Grid for {algorithm}: ")
        for key, value in param_grid.items():
            print(f"    '{key}': {value},")

        # Print the running message based on the cv_indices reference
        print('\n--- Running nested cross-validation ---')

        if self.plot_learning_curve:
            # Lists to store the learning curve data
            train_sizes_total = []
            train_scores_total = []
            test_scores_total = []

        # Initialize a temporary storage for the current cv_indices
        model_storage = {}

        # Loop over each fold in the cross-validation indices
        for fold_key, fold in cv_indices.items():

            # Extract the training and testing data from the original DataFrame using indices
            df_train = self.file_label.iloc[fold['train_index']][['adsorbate', 'e_int_dft']]
            df_test = self.file_label.iloc[fold['test_index']][['adsorbate', 'e_int_dft']]

            # Split the data into training and testing sets
            X_train = self.X[fold['train_index']].copy()
            y_train = self.y[fold['train_index']].copy()
            X_test = self.X[fold['test_index']].copy()
            y_test = self.y[fold['test_index']].copy()

            # Create a Scaler object
            scaler = MinMaxScaler()

            # Load the model for the specified algorithm
            model = MODEL_DICT[algorithm]

            # Create a pipeline object
            pipeline = Pipeline(steps = [
                                        ('scaler', scaler),
                                        ('pca', PCA(n_components=0.98)),  # Use PCA to retain 95% explained variance
                                        ('model', model)
                                        ])

            if self.search_type == 'grid' or self.search_type == 'best_model':
                # Use GridSearchCV to find the best hyperparameters
                grid = GridSearchCV(estimator = pipeline,
                                    param_grid = param_grid,
                                    cv = n_splits_inner,
                                    scoring = 'neg_root_mean_squared_error',
                                    verbose = 2,
                                    n_jobs = -1,
                                    return_train_score = True,
                                    )

            elif self.search_type == 'random':
                # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
                grid = RandomizedSearchCV(estimator = pipeline,
                                        param_distributions = param_grid,
                                        cv = n_splits_inner,
                                        scoring = 'neg_root_mean_squared_error',
                                        verbose = 2,
                                        n_jobs = -1,
                                        return_train_score = True,
                                        n_iter = 100,
                                        )

            elif self.search_type == 'bayesian':
                # Use BayesSearchCV to find the best hyperparameters
                grid = BayesSearchCV(estimator = pipeline,
                                    search_spaces = param_grid,
                                    cv = n_splits_inner,
                                    scoring = 'neg_root_mean_squared_error',
                                    verbose = 2,
                                    n_jobs = -1,
                                    n_iter = 50,
                                    return_train_score = True,
                                    )

            print(f'\n--- Looking for best hyperparameters for {algorithm}, fold {fold_key} ---')
            grid.fit(X_train, y_train)

            if self.plot_learning_curve:
                # Calculate the learning curve data
                train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_,   # pipeline object
                                                                        X_train,
                                                                        y_train,
                                                                        cv = n_splits_inner,
                                                                        n_jobs = -1,
                                                                        train_sizes = np.linspace(0.1, 1.0, 5),
                                                                        scoring = 'neg_root_mean_squared_error'
                                                                        )

                # Store the learning curve data, here, we store the data for each fold
                train_sizes_total.append(train_sizes)
                train_scores_total.append(train_scores)
                test_scores_total.append(test_scores)

            # Print best hyperparameters for each fold
            print(f"--- Best hyperparameters for fold {fold_key}:")
            for parameter, value in grid.best_params_.items():
                print('   ', f'{parameter:25}{value}')

            # Extract the best model from the grid search (pipeline object)
            best_model = grid.best_estimator_

            # Accessing the PCA component after fitting
            best_pca = grid.best_estimator_.named_steps['pca']
            explained_variance_ratio = best_pca.explained_variance_ratio_
            n_components_retained = best_pca.n_components_
            print(f"--- PCA retained {n_components_retained} dimensions to explain {explained_variance_ratio.sum() * 100:.2f}% of variance.")

            # Save explained variance ratio for plotting
            self.explained_variance_ratio_ = explained_variance_ratio

            # Predict on the training data using the best model
            y_train_pred = best_model.predict(X_train)
            # Calculate the evaluation metrics for the training set
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            print(f'\n    Train rMSE: {train_rmse:.4f}')
            train_r2 = r2_score(y_train, y_train_pred)
            print(f'    Train R2:   {train_r2:.4f}')
            train_mae = mean_absolute_error(y_train, y_train_pred)
            print(f'    Train MAE:  {train_mae:.4f}')

            # Predict using the pipeline (which includes the scaler)
            y_test_pred = best_model.predict(X_test)
            # Calculate the evaluation metrics for the test set
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            print(f'\n    Test rMSE: {test_rmse:.4f}')
            test_r2 = r2_score(y_test, y_test_pred)
            print(f'    Test R2:   {test_r2:.4f}')
            test_mae = mean_absolute_error(y_test, y_test_pred)
            print(f'    Test MAE:  {test_mae:.4f}')

            # Extract the best regressor from the pipeline
            best_regressor = grid.best_estimator_.named_steps['model']

            # Add predicted values to the DataFrame
            df_train['e_int_pred'] = y_train_pred
            df_test['e_int_pred'] = y_test_pred

            # Store the results in the model_storage dictionary
            model_storage[fold_key] = {
                                        'algorithm':                algorithm,
                                        'model':                    best_model,
                                        'regressor':                best_regressor,
                                        'train_index':              fold['train_index'],
                                        'test_index':               fold['test_index'],
                                        'y_train_predicted':        y_train_pred,
                                        'y_test_predicted':         y_test_pred,
                                        'y_train_actual':           y_train,
                                        'y_test_actual':            y_test,
                                        'df_train':                 df_train,
                                        'df_test':                  df_test,
                                        'train_rmse':               train_rmse,
                                        'test_rmse':                test_rmse,
                                        'train_r2':                 train_r2,
                                        'test_r2':                  test_r2,
                                        'train_mae':                train_mae,
                                        'test_mae':                 test_mae,
                                        'scaler':                   scaler,
                                        'best_params':              grid.best_params_,
                                        'search_type':              self.search_type,
                                        }

        # Plot the learning curve
        if self.plot_learning_curve:
            self.plot_cv_learning_curve(train_sizes_total, train_scores_total, test_scores_total, algorithm)

        # Use weighted performance to select the best hyperparameters
        performance_weighted_params = {}

        # Loop over each fold in the model_storage dictionary
        for fold_info in model_storage.values():
            # Extract the tuple of sorted hyperparameters
            params_tuple = tuple(sorted(fold_info['best_params'].items()))
            # Get the test RMSE score for current fold
            score = fold_info['test_rmse']

            performance_weighted_params[params_tuple] = performance_weighted_params.get(params_tuple, 0) + score

        # Choose the hyperparameters with the best weighted performance
        best_weighted_params = dict(max(performance_weighted_params, key = performance_weighted_params.get))
        print("\n--- Best Weighted Hyperparameters Based on Performance ---")
        for param, value in best_weighted_params.items():
            print(f"    {param}: {value}")

        # Based on the cv_indices, decide which main storage to update
        self.model_storage[algorithm] = model_storage



    # Function to plot explained variance ratio
    def plot_PCA_explained_variance(self):
        explained_variance_ratio_cumsum = np.cumsum(self.explained_variance_ratio_)
        print("Explained variance ratio:", explained_variance_ratio_cumsum)

        # Extend the explained variance ratio array to 30 components if it has less than 30
        if len(explained_variance_ratio_cumsum) < 30:
            explained_variance_ratio_cumsum = np.pad(explained_variance_ratio_cumsum, (0, 30 - len(explained_variance_ratio_cumsum)), 'edge')

        plt.figure(figsize=(8, 8), facecolor='white')
        plt.plot(explained_variance_ratio_cumsum, marker='o', linestyle='--', color='b')
        plt.xlim(0, 30)  # Set x-axis limit to 30
        plt.xlabel('Number of Components', fontsize=18)
        plt.ylabel('Cumulative Explained Variance', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(self.path_output_figures, 'explained_variance.png'), dpi=1000, bbox_inches='tight')
        plt.show()





    
    def plot_cv_learning_curve(self, train_sizes, train_scores, test_scores, algorithm):
        train_sizes = np.mean(train_sizes, axis=0)
        train_scores_mean = -np.mean(np.mean(train_scores, axis=1), axis=0)
        train_scores_std = np.std(np.mean(train_scores, axis=1), axis=0)
        test_scores_mean = -np.mean(np.mean(test_scores, axis=1), axis=0)
        test_scores_std = np.std(np.mean(test_scores, axis=1), axis=0)
        
        plt.figure()
        plt.title(f"Learning Curve for {algorithm}")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.path_output_figures, 'learning_curves', f"{algorithm}_learning_curve.png"))
        plt.show()
    
    
    # Function to combine model storages from nested CV
    def combine_model_storages(self):
        # Determine which model storage dictionary to use
        model_storage_dict = self.model_storage

        # Initialize a dictionary to store the combined model information
        combined_model_storage = {}

        # Loop over each algorithm in the model storage dictionary
        for algorithm, folds in model_storage_dict.items():
            # Initialize a dictionary to store the aggregated information
            aggregated_info = {
                        'algorithm':                    algorithm,
                        
                        'best_model':                   None,
                        'best_params':                  {},
                        
                        'best_mae':                     float('inf'),
                        'best_r2':                      float('-inf'),
                        'best_rmse':                    float('inf'),
                        
                        'test_mae_avg':                 0,
                        'test_r2_avg':                  0,
                        'test_rmse_avg':                0,
                        
                        'train_mae_avg':                0,
                        'train_r2_avg':                 0,
                        'train_rmse_avg':               0,
                        
                        'y_test_predicted_combined':    [],
                        'y_train_predicted_combined':   [],
                        
                        'train_index_best_model':       [], # Add train index for the best model
                        'test_index_best_model':        [], # Add test index for the best model
                        
                        'df_train':                     pd.DataFrame(),
                        'df_test':                      pd.DataFrame(),
                            }

            df_train_list = []
            df_test_list = []

            # Loop over each fold, key is the fold number and value is the fold data
            # fold_key, fold_data = 'fold_0', dict
            for fold_key, fold_data in folds.items():
                # Update best model and its information if current fold has a better RMSE
                if fold_data['test_rmse'] < aggregated_info['best_rmse']:
                    aggregated_info['best_rmse'] = fold_data['test_rmse']
                    aggregated_info['best_model'] = fold_data['model']
                    aggregated_info['best_params'] = fold_data['best_params']
                    aggregated_info['train_index_best_model'] = fold_data['train_index']
                    aggregated_info['test_index_best_model'] = fold_data['test_index']

                # Update best MAE if current fold has a better MAE
                if fold_data['test_mae'] < aggregated_info['best_mae']:
                    aggregated_info['best_mae'] = fold_data['test_mae']

                # Update best R2 if current fold has a better R2
                if fold_data['test_r2'] > aggregated_info['best_r2']:
                    aggregated_info['best_r2'] = fold_data['test_r2']
                
                # Update the combined predictions for the training and testing sets
                aggregated_info['y_test_predicted_combined'].extend(fold_data['y_test_predicted'])
                aggregated_info['y_train_predicted_combined'].extend(fold_data['y_train_predicted'])
                
                # Append the DataFrames from the current fold to the lists
                df_train_list.append(fold_data['df_train'].assign(fold = fold_key))
                df_test_list.append(fold_data['df_test'].assign(fold = fold_key))

            # Combine all train and test DataFrames from different folds
            aggregated_info['df_train'] = pd.concat(df_train_list, ignore_index=True)
            aggregated_info['df_test'] = pd.concat(df_test_list, ignore_index=True)

            # Compute averages of scores and other metrics
            aggregated_info['test_mae_avg'] = np.mean([fold['test_mae'] for fold in folds.values()])
            aggregated_info['test_r2_avg'] = np.mean([fold['test_r2'] for fold in folds.values()])
            aggregated_info['test_rmse_avg'] = np.mean([fold['test_rmse'] for fold in folds.values()])
            aggregated_info['train_mae_avg'] = np.mean([fold['train_mae'] for fold in folds.values()])
            aggregated_info['train_r2_avg'] = np.mean([fold['train_r2'] for fold in folds.values()])
            aggregated_info['train_rmse_avg'] = np.mean([fold['train_rmse'] for fold in folds.values()])
            
            # Store the aggregated information in the combined model storage
            combined_model_storage[algorithm] = aggregated_info

        # Assign combined model storage based on cv type
        self.combined_model_storage = combined_model_storage

        return combined_model_storage

    
    # Function to save the predictions to CSV files
    def save_predictions_to_csv(self):
        # Loop over each algorithm in the combined model storage
        for algorithm, data in self.combined_model_storage.items():
            
            # Define the filenames for the train and test predictions
            train_rmse = round(data['train_rmse_avg'], 3)
            test_rmse = round(data['test_rmse_avg'], 3)
            
            train_rmse = data['train_rmse_avg']
            test_rmse = data['test_rmse_avg']
            
            train_filename = f'{self.fingerprint_type}-{self.sort_type}-{algorithm}-{self.search_type}-train-{train_rmse:.4f}.csv'
            test_filename = f'{self.fingerprint_type}-{self.sort_type}-{algorithm}-{self.search_type}-test-{test_rmse:.4f}.csv'

            # Construct full paths for the CSV files
            train_file_path = os.path.join(self.path_output_csv, 'predictions_adsorbate_fingerprint', train_filename)
            test_file_path = os.path.join(self.path_output_csv, 'predictions_adsorbate_fingerprint', test_filename)
            
            # Extract the DataFrames from the combined model storage
            df_train = data['df_train']
            df_test = data['df_test']

            # Save the DataFrames to CSV files
            df_train.to_csv(train_file_path, index=False)
            df_test.to_csv(test_file_path, index=False)

            # Print a message to confirm the saving of the files
            print(f'--- Saved train predictions to {train_file_path}')
            print(f'--- Saved test predictions to {test_file_path}')



## Main Function
if __name__ == "__main__":

    # Defining path to input data
    path_input = os.path.join(get_paths("database_path"), 'adsorbate_fingerprints')
    
    # Defining path to label data
    path_label = get_paths("label_data_path")   # label_data folder
    
    # Defining Output Path for CSV Files
    path_output_csv = get_paths("csv_file_path")
    
    # Defining Output Path for Figures
    path_output_figures = get_paths("output_figure_path")
    
    # Defining Algorithm List
    algorithm_list = [
                    # 'Multi_Linear',
                    # 'SVR',
                    # 'LASSO',
                    # 'Ridge',
                    'SGD',
                    # 'Bayesian_Ridge',
                    # 'K_Neighbors',
                    # 'Decision_Tree',
                    # 'Random_Forest',    # Checked
                    # 'Gradient_Boost',
                    # 'XGBoost',
                    ]
    
    fingerprint_types = [
                        # 'Bag_of_Bonds',
                        # 'Coulomb_Matrix',
                        # 'Sine_Matrix',
                        'Ewald_Sum_Matrix',
                        ]
    
    sort_types = [
                #   'none',
                #   'sorted_l2',
                  'sort_all',
                #   'eigenspectrum',
                  ]

    search_types = [
                    'grid',
                    # 'random',
                    # 'bayesian',
                    # 'best_model',
                    ]
    
    # Loop over the fingerprint types
    for fingerprint_type in fingerprint_types:
        # Loop over the sort types
        for sort_type in sort_types:
            # Loop over the search types
            for search_type in search_types:
                # Run Models
                adsorbate_fp_model = Adsorbate_FP_Model(
                                                        path_input = path_input,
                                                        path_label = path_label,
                                                        path_output_csv = path_output_csv,
                                                        path_output_figures = path_output_figures,
                                                        algorithm_list = algorithm_list,
                                                        fingerprint_type = fingerprint_type,
                                                        sort_type = sort_type,
                                                        search_type = search_type,
                                                        plot_learning_curve = False,
                                                        save_predictions_csv = True,
                                                        plot_explained_variance = False,
                                                        )