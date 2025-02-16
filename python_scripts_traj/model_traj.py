'''
This script is used to run the multi-descriptor model.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul
import shap
import pickle

# Importing sklearn Libraries
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GroupKFold, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV
from sklearn.model_selection import GridSearchCV
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
# https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV
from skopt import BayesSearchCV

# Importing Custom Libraries
from core.path import get_paths
from core.model_vars import DESCRIPTORS_TRAJ, MODEL_DICT
from core.parameter_dict_bayesian import PARAMETER_DICT_BAYESIAN
from core.parameter_dict_grid import PARAMETER_DICT_GRID
from core.parameter_dict_random import PARAMETER_DICT_RANDOM
from core.parameter_dict_best import PARAMETER_BEST


# Class to run the machine learning models on the molecular descriptors
class Descriptor_Model_Config:
    
    def __init__(self,
                algorithm_list,
                search_type,
                plot_learning_curve=False,
                plot_feature_importance=False,
                save_predictions_csv=False,
                verbose = 1,
                output_label = 'e_int_dft',
                ):
        
        self.algorithm_list = algorithm_list
        self.search_type = search_type
        self.plot_learning_curve = plot_learning_curve
        self.plot_feature_importance = plot_feature_importance
        self.save_predictions_csv = save_predictions_csv
        self.verbose = verbose
        self.output_label = output_label
        
        self.path_output_csv = get_paths("csv_file_path")
        self.path_output_figures = get_paths("output_figure_path")
        self.path_output_models = get_paths("output_model_path")

        ## Model Storage
        self.model_storage_random = {}
        self.model_storage_adsorb = {}

        ## Using Pandas To Read File
        self.csv_file = self.load_and_combine_features()

        ## Adding Labels To Dataframe
        self.csv_file = self.add_labels_to_df(df=self.csv_file,
                                            columns=['adsorbate', 'config'],
                                            output_key='label',)

        ## Defining X Data
        self.molecular_descriptors = [descriptor for sublist in DESCRIPTORS_CONFIG.values() for descriptor in sublist]
        self.X = self.csv_file[self.molecular_descriptors].to_numpy()
        self.y = self.csv_file[self.output_label].to_numpy()
        self.names = self.csv_file['label'].to_numpy()
        self.num_features = len(self.molecular_descriptors)
        print('--- Number of total features: ', self.num_features)
        
        
        
        for algorithm in self.algorithm_list:
            self.model_storage_random_pkl_path = os.path.join(self.path_output_models,
                                                              'model_config',
                                                              f'{algorithm}-random-{self.num_features}_feature-{self.search_type}.pkl')
            self.model_storage_adsorb_pkl_path = os.path.join(self.path_output_models,
                                                              'model_config',
                                                              f'{algorithm}-adsorb-{self.num_features}_feature-{self.search_type}.pkl')
            
            # Random split
            if os.path.exists(self.model_storage_random_pkl_path):
                self.model_storage_random[algorithm] = self.load_model_from_pkl(self.model_storage_random_pkl_path)
            else:
                ## get indicdes of training and testing set, randomly
                self.cv_indices_dict_random = self.get_cross_validation_indices_random()

                # Run nested cross-validation of random data split
                self.model_storage_random[algorithm] = self.cross_validation_SearchCV(algorithm,
                                                                                      cv_indices = self.cv_indices_dict_random,
                                                                                      n_splits_inner = 5)

                # Calculate the average test_rmse
                avg_test_rmse = np.mean([fold['test_rmse'] for fold in self.model_storage_random[algorithm].values()])
                
                # Save the random model storage to a pkl file with average test_rmse in the filename
                self.save_model_to_pkl(self.model_storage_random[algorithm],
                                       f"{self.model_storage_random_pkl_path[:-4]}-{avg_test_rmse:.3f}.pkl")

            # Split based on adsorbate
            if os.path.exists(self.model_storage_adsorb_pkl_path):
                self.model_storage_adsorb[algorithm] = self.load_model_from_pkl(self.model_storage_adsorb_pkl_path)
            else:
                ## get indicdes of training and testing set, based on adsorbate
                self.cv_indices_dict_adsorb = self.get_cross_validation_indices_adsorb()

                # Run nested cross-validation of adsorbate-based data split
                self.model_storage_adsorb[algorithm] = self.cross_validation_SearchCV(algorithm,
                                                                                      cv_indices = self.cv_indices_dict_adsorb,
                                                                                      n_splits_inner = 5)

                # Calculate the average test_rmse
                avg_test_rmse = np.mean([fold['test_rmse'] for fold in self.model_storage_adsorb[algorithm].values()])

                # Save the adsorb model storage to a pkl file with average test_rmse in the filename
                self.save_model_to_pkl(self.model_storage_adsorb[algorithm],
                                       f"{self.model_storage_adsorb_pkl_path[:-4]}-{avg_test_rmse:.3f}.pkl")
                
        
        # Combine the results from the 5 folds, for random and adsorbate-based data splits
        self.combined_model_storage_random = self.combine_model_storages(cv_type='random')
        self.combined_model_storage_adsorb = self.combine_model_storages(cv_type='adsorb')
        
        # Plot the feature importance for the best model
        if self.plot_feature_importance:
            print("\n--- Plotting Feature Importance ---")
            
            # Plot feature importance using sklearn feature_importances_ attribute
            # self.plot_feature_importance_sklearn(best_or_avg = 'avg')
            # self.plot_feature_importance_sklearn(best_or_avg = 'best')

            # Plot feature importance using permutation test
            # self.plot_feature_importance_permutation(n_repeats=10, type='boxplot', top_n_features=30)
            self.plot_feature_importance_permutation(n_repeats=10, type='barplot', top_n_features=30)
            
            # self.plot_feature_importance_shap(plot_type = 'barplot')
            
        # Save the predictions to CSV files
        if self.save_predictions_csv:
            self.save_predictions_to_csv()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Save model storage
    def save_model_to_pkl(self, model_storage, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model_storage, file)
        print(f"Model storage saved to {filename}")

    # Load model storage
    def load_model_from_pkl(self, filename):
        with open(filename, 'rb') as file:
            model_storage = pickle.load(file)
        print(f"Model storage loaded from {filename}")
        return model_storage
    
    
    
    # Function to load and combine features from multiple CSV files
    def load_and_combine_features(self):
        df_list = []
        label_data_path = get_paths('label_data_path')
        
        # Read the data from each CSV file and store in a list
        for csv_file, descriptors in DESCRIPTORS_CONFIG.items():
            file_path = os.path.join(label_data_path, csv_file)
            df_temp = pd.read_csv(file_path, usecols=['adsorbate', 'config'] + descriptors)
            df_list.append(df_temp)
        
        # Combine the data from all CSV files
        df_combined = df_list[0]
        for df in df_list[1:]:
            df_combined = pd.merge(df_combined, df, on=['adsorbate', 'config'], how='inner')
        
        # Read the label data from the CSV file
        label_file_path = os.path.join(label_data_path, 'E_int_450.csv')
        df_labels = pd.read_csv(label_file_path, usecols=['adsorbate', 'config', 'e_int_dft'])
        
        # Put the labels in the combined DataFrame
        df_combined = pd.merge(df_combined, df_labels, on=['adsorbate', 'config'], how='inner')
        
        return df_combined
    
    
    # Function to add labels to the dataframe
    def add_labels_to_df(self,
                        df,
                        columns = ['adsorbate', 'config'],
                        output_key = 'label'):
        # 1. Create a new empty column in the dataframe for storing the generated labels
        df[output_key] = ''

        # 2. Iterate over each row in the dataframe
        for index, row in df.iterrows():
            # Generate a label by concatenating the values of specified columns
            label = "_".join(row[columns].astype(str))
            # Assign the generated label to the new column for the current row
            df.at[index, output_key] = label

        # 3. Rearrange the columns such that the new label column is the first column
        columns_order = [output_key] + [col for col in df.columns if col != output_key]
        df = df[columns_order]

        return df
    

    ## Cross Validation Based on Random
    def get_cross_validation_indices_random(self, n_splits=5):
        path_to_random_indices_pickle = os.path.join(self.path_output_models, 'cv_indices_random.pkl')
        
        if os.path.exists(path_to_random_indices_pickle):
            print(f"    Found cv indices pickle file")
            # Load the pickle if it exists
            with open(path_to_random_indices_pickle, 'rb') as file:
                print(f"    Loading cv indices pickle file from {path_to_random_indices_pickle}")
                cv_indices = pickle.load(file)
        else:
            print(f"    Could not find cv indices pickle file, generating new indices")
            # Generate indices and save to pickle if it does not exist
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_indices = {}
            for index, (train_index, test_index) in enumerate(kf.split(self.X)):
                fold_key = f"fold_{index}"  # Creating a unique key for each fold
                cv_indices[fold_key] = {
                    'train_index': train_index,
                    'test_index': test_index,
                    'train_names': self.names[train_index],
                    'test_names': self.names[test_index]
                }
            os.makedirs(os.path.dirname(path_to_random_indices_pickle), exist_ok=True)
            with open(path_to_random_indices_pickle, 'wb') as file:
                pickle.dump(cv_indices, file)
                print(f"    New indices saved to {path_to_random_indices_pickle}")

        return cv_indices
    
    
    # Cross Validation Based on adsorbate
    def get_cross_validation_indices_adsorb(self, column_name = 'adsorbate', n_splits = 5):
        path_to_adsorb_indices_pickle = os.path.join(self.path_output_models, 'cv_indices_adsorb.pkl')

        if os.path.exists(path_to_adsorb_indices_pickle):
            print(f"    Found cv indices pickle file")
            # Load the pickle if it exists
            with open(path_to_adsorb_indices_pickle, 'rb') as file:
                print(f"    Loading cv indices pickle file from {path_to_adsorb_indices_pickle}")
                cv_indices = pickle.load(file)
        else:
            print(f"    Could not find cv indices pickle file, generating new indices")
            # Generate indices and save to pickle if it does not exist
            gkf = GroupKFold(n_splits=n_splits)
            groups = self.csv_file[column_name]
            cv_indices = {}
            for index, (train_index, test_index) in enumerate(gkf.split(self.X, self.y, groups=groups)):
                fold_key = f"fold_{index}"  # Creating a unique key for each fold
                cv_indices[fold_key] = {
                    'train_index': train_index,
                    'test_index': test_index,
                    'train_names': self.names[train_index],
                    'test_names': self.names[test_index]
                }
            os.makedirs(os.path.dirname(path_to_adsorb_indices_pickle), exist_ok=True)
            with open(path_to_adsorb_indices_pickle, 'wb') as file:
                pickle.dump(cv_indices, file)
                print(f"    New indices saved to {path_to_adsorb_indices_pickle}")

        return cv_indices
    
    
    # Nested Cross Validation
    def cross_validation_SearchCV(self,
                                  algorithm,
                                  cv_indices,
                                  n_splits_inner = 5,
                                  ):
        
        # Load the parameter dict for the specified algorithm based on the search type
        if self.search_type == 'grid_search':
            param_grid = PARAMETER_DICT_GRID[algorithm]
        elif self.search_type == 'random_search':
            param_grid = PARAMETER_DICT_RANDOM[algorithm]
        elif self.search_type == 'bayesian_search':
            param_grid = PARAMETER_DICT_BAYESIAN[algorithm]
        elif self.search_type == 'best_model':
            param_grid = PARAMETER_BEST[algorithm]
        
        # Calculate the number of combinations, if not using Bayesian search
        if self.search_type != 'bayesian_search':
            num_combinations = reduce(mul, (len(v) for v in param_grid.values()), 1)
            print(f"--- Total number of hyperparameter combinations for {algorithm}: {num_combinations}")
        
        # Add the 'model__' prefix to the keys in the parameter grid
        param_grid = {'model__' + key: value for key, value in param_grid.items()}
        
        # Add parameter for the scaler
        param_grid['scaler'] = [
                                # StandardScaler(),
                                MinMaxScaler()
                                ]
        
        print(f"--- Hyperparameters Grid for {algorithm}: ")
        for key, value in param_grid.items():
            print(f"    '{key}': {value},")
        
        # Print the running message based on the cv_indices reference
        if cv_indices is self.cv_indices_dict_random:
            print('\n--- Running nested cross-validation on random data split ---')
        elif cv_indices is self.cv_indices_dict_adsorb:
            print('--- Running nested cross-validation on adsorbate-based data split ---')
        
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
            df_train = self.csv_file.iloc[fold['train_index']][['label', 'adsorbate', 'config', 'e_int_dft']]
            df_test = self.csv_file.iloc[fold['test_index']][['label', 'adsorbate', 'config', 'e_int_dft']]

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
                                        ('scaler', scaler), # Set up the scaler in the GridSearchCV later
                                        ('model', model)
                                        ])
            
            if self.search_type == 'grid_search' or self.search_type == 'best_model':
                # Use GridSearchCV to find the best hyperparameters
                grid = GridSearchCV(estimator = pipeline,
                                    param_grid = param_grid,
                                    cv = n_splits_inner,
                                    scoring = 'neg_root_mean_squared_error',
                                    verbose = self.verbose,
                                    n_jobs = -1,
                                    return_train_score = True,
                                    )
            
            elif self.search_type == 'random_search':
                # Use RandomizedSearchCV to find the best hyperparameters
                grid = RandomizedSearchCV(estimator = pipeline,
                                          param_distributions = param_grid,
                                          cv = n_splits_inner,
                                          scoring = 'neg_root_mean_squared_error',
                                          verbose = self.verbose,
                                          n_jobs = -1,
                                          return_train_score = True,
                                          n_iter = 100,
                                          )
            
            elif self.search_type == 'bayesian_search':
                # Use BayesSearchCV to find the best hyperparameters
                grid = BayesSearchCV(estimator = pipeline,
                                     search_spaces = param_grid,
                                     cv = n_splits_inner,
                                     scoring = 'neg_root_mean_squared_error',
                                     verbose = self.verbose,
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
            
            # Print best score for each fold
            print(f"--- Best score for fold {fold_key}: {grid.best_score_:.4f}")
            
            # Extract the best model from the grid search (pipeline object)
            best_model = grid.best_estimator_

            # Predict on the training data using the best model
            y_train_pred = best_model.predict(X_train)
            # Calculate the evaluation metrics for the training set
            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            print(f'    Train rMSE: {train_rmse:.4f}')
            train_r2 = r2_score(y_train, y_train_pred)
            print(f'    Train R2:   {train_r2:.4f}')
            train_mae = mean_absolute_error(y_train, y_train_pred)
            print(f'    Train MAE:  {train_mae:.4f}')

            # Predict using the pipeline (which includes the scaler)
            y_test_pred = best_model.predict(X_test)
            # Calculate the evaluation metrics for the test set
            test_rmse = root_mean_squared_error(y_test, y_test_pred)
            print(f'    Test rMSE: {test_rmse:.4f}')
            test_r2 = r2_score(y_test, y_test_pred)
            print(f'    Test R2:   {test_r2:.4f}')
            test_mae = mean_absolute_error(y_test, y_test_pred)
            print(f'    Test MAE:  {test_mae:.4f}')
            
            # Extract the best regressor from the pipeline
            best_regressor = grid.best_estimator_.named_steps['model']
            
            # Now you can access the feature_importances_ attribute if it exists
            if hasattr(best_regressor, 'feature_importances_'):
                feature_importances = best_regressor.feature_importances_
            else:
                feature_importances = None  # This handles models that do not have feature_importances_
            
            # Add predicted values to the DataFrame
            df_train['e_int_pred'] = y_train_pred
            df_test['e_int_pred'] = y_test_pred
            
            # Store the results in the model_storage dictionary
            model_storage[fold_key] = {
                                        'algorithm':                algorithm,
                                        'model':                    best_model,
                                        'feature_names':            self.molecular_descriptors,
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
                                        'feature_importance':       feature_importances,
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
        print("--- Best Weighted Hyperparameters Based on Performance ---")
        for param, value in best_weighted_params.items():
            print(f"    {param}: {value}")

        return model_storage



    
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
    def combine_model_storages(self, cv_type):
        # Determine which model storage dictionary to use
        if cv_type == 'random':
            model_storage_dict = self.model_storage_random
        elif cv_type == 'adsorb':
            model_storage_dict = self.model_storage_adsorb
        else:
            raise ValueError(f"Unknown cv_type: {cv_type}")

        # Initialize a dictionary to store the combined model information
        combined_model_storage = {}

        # Loop over each algorithm in the model storage dictionary
        for algorithm, folds in model_storage_dict.items():
            # Initialize a dictionary to store the aggregated information
            aggregated_info = {
                'algorithm':                    algorithm,
                'feature_names':                folds[next(iter(folds))]['feature_names'],

                'best_model':                   None,
                'best_params':                  {},

                'feature_importance_avg':       None,
                'feature_importance_best':      None,

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
                    aggregated_info['feature_importance_best'] = fold_data['feature_importance']
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
                df_train_list.append(fold_data['df_train'].assign(fold=fold_key))
                df_test_list.append(fold_data['df_test'].assign(fold=fold_key))

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
            
            # Only calculate the average feature importance if it exists
            if all(fold['feature_importance'] is not None for fold in folds.values()):
                aggregated_info['feature_importance_avg'] = np.mean([fold['feature_importance'] for fold in folds.values()], axis=0)

            # Store the aggregated information in the combined model storage
            combined_model_storage[algorithm] = aggregated_info

        # Assign combined model storage based on cv type
        if cv_type == 'random':
            self.combined_model_storage_random = combined_model_storage
        elif cv_type == 'adsorb':
            self.combined_model_storage_adsorb = combined_model_storage

        return combined_model_storage

    
    # Function to plot feature importance using sklearn feature_importances_ attribute
    def plot_feature_importance_sklearn(self,
                                        best_or_avg = 'best',):
        # Define the storages to loop over
        storages = {'adsorb': self.combined_model_storage_adsorb,
                    'random': self.combined_model_storage_random,}

        # Loop over each type of storage
        for storage_type, storage in storages.items():
            # Loop over each algorithm in the storage
            for algorithm, data in storage.items():
                # Determine which feature importance data to use
                if best_or_avg == 'best':
                    importances = data.get('feature_importance_best', None)
                    title_suffix = "Best Model"
                elif best_or_avg == 'avg':
                    importances = data.get('feature_importance_avg', None)
                    title_suffix = "Model Average"
                
                # Load the feature names from the data
                feature_names = data['feature_names']
                
                # Ensure importances are available
                if importances is not None:
                    # Combine importances and feature names into a list of tuples
                    features_with_importance = list(zip(feature_names, importances))

                    # Sort the list of tuples by the importances (second item in the tuples)
                    features_with_importance.sort(key=lambda x: x[1], reverse = False)

                    # Unzip the sorted list back into two lists
                    sorted_feature_names, sorted_importances = zip(*features_with_importance) if features_with_importance else ([], [])

                    # Plotting
                    plt.figure(figsize=(12, len(sorted_feature_names) * 0.4))  # Adjust the figure size
                    plt.barh(sorted_feature_names, sorted_importances)
                    plt.xlabel("Feature Importance")
                    plt.title(f"{algorithm} Model Feature Importance ({storage_type} {title_suffix})")
                    plt.tight_layout()  # Adjust layout to make room for long feature names

                    # Save the figure
                    file_name = f'{algorithm}_sklearn_{storage_type}_{title_suffix.replace(" ", "_")}.png'
                    full_path = os.path.join(self.path_output_figures, 'feature_importance', file_name)
                    plt.savefig(full_path, dpi = 1000)
                    print(f"--- Figure saved to {full_path} ---")
                    
                else:
                    print(f"--- No feature importances available for {algorithm} ---.")


    # https://scikit-learn.org/stable/modules/permutation_importance.html
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    # Function to plot feature importance using permutation test
    def plot_feature_importance_permutation(self, n_repeats=10, type='boxplot', top_n_features=20):
        # Create a dictionary to loop over the different types of storage
        storages = {'adsorb': self.model_storage_adsorb,
                    'random': self.model_storage_random}

        # Loop over each type of storage
        for storage_type, storage in storages.items():
            # Loop over each algorithm in the storage
            for algorithm, folds in storage.items():
                print(f'--- Permutation importance for {algorithm} ({storage_type} split) ---')
                
                # Assume all folds have the same feature names, take the first fold's feature names
                feature_names = next(iter(folds.values()))['feature_names']
                
                # Initialize a dictionary to store aggregated importances across all folds
                feature_importances_dict = {name: [] for name in feature_names}
                
                # Loop over each fold
                for fold_key, fold_data in folds.items():
                    # Get the best model from the stored results
                    model = fold_data['model']  # pipeline object
                    
                    # Extract the training data using the fold's train index
                    X_train = self.X[fold_data['train_index']].copy()
                    y_train = self.y[fold_data['train_index']].copy()
                    
                    # Perform permutation importance
                    result = permutation_importance(model, X_train, y_train,
                                                    scoring='neg_root_mean_squared_error',
                                                    n_repeats=n_repeats, random_state=42, n_jobs=-1)
                    print(f'    Permutation importance for {algorithm}, {fold_key}')
                    
                    # Append current fold's results to the respective lists
                    # Will have shape (n_features, n_repeats * 5) for 5 folds
                    for i, importances in enumerate(result.importances):
                        feature_importances_dict[feature_names[i]].extend(importances)
                
                # Calculate the mean importance
                importances_mean = np.array([np.mean(imp) for imp in feature_importances_dict.values()])
                
                if type == 'boxplot':
                    # Sort the importances and accordingly reorder the features
                    sorted_indices = np.argsort(importances_mean)
                    sorted_feature_names = np.array(feature_names)[sorted_indices]
                    
                    # Prepare data for boxplot
                    importances_df = pd.DataFrame(feature_importances_dict)
                    importances_df = importances_df[sorted_feature_names]

                    # Split data for top and bottom features
                    top_features_df = importances_df.iloc[:, -top_n_features:]
                    bottom_features_df = importances_df.iloc[:, :top_n_features]

                    # Plotting the boxplot
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, len(top_features_df.columns) * 0.4))
                    plt.rcParams.update({'font.size': 8})
                    
                    # Plot top features
                    ax = top_features_df.plot.box(ax=axes[0], vert=False, whis=10)
                    ax.set_title(f"Top {top_n_features} Permutation Importances ({storage_type}) - {algorithm}")
                    ax.axvline(x=0, color="k", linestyle="--")
                    ax.set_xlabel("Decrease in accuracy score")
                    ax.set_yticklabels(top_features_df.columns)
                    
                    # Plot bottom features
                    ax = bottom_features_df.plot.box(ax=axes[1], vert=False, whis=10)
                    ax.set_title(f"Bottom {top_n_features} Permutation Importances ({storage_type}) - {algorithm}")
                    ax.axvline(x=0, color="k", linestyle="--")
                    ax.set_xlabel("Decrease in accuracy score")
                    ax.set_yticklabels(bottom_features_df.columns)
                    
                    plt.tight_layout()

                    # Save the figure
                    file_name = f'{algorithm}_permutation_{type}_{storage_type}.png'
                    full_path = os.path.join(self.path_output_figures, 'feature_importance', file_name)
                    plt.savefig(full_path, dpi=300)
                    # plt.show()
                    print(f"--- Box Figure saved to {full_path} ---")
                    
                elif type == 'barplot':
                    # Calculate the average and std of importances over all folds
                    importances_mean = [np.mean(importances) for importances in feature_importances_dict.values()]
        
                    # Sorting indices based on mean importance
                    sorted_indices = np.argsort(importances_mean)
                    sorted_feature_names = [feature_names[idx] for idx in sorted_indices]
                    sorted_importances_mean = [importances_mean[idx] for idx in sorted_indices]
                    
                    # Split data for top and bottom features
                    top_feature_names = sorted_feature_names[-top_n_features:]
                    top_importances_mean = sorted_importances_mean[-top_n_features:]
                    bottom_feature_names = sorted_feature_names[:top_n_features]
                    bottom_importances_mean = sorted_importances_mean[:top_n_features]
                    
                    # Plotting
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, len(top_feature_names) * 0.4))
                    plt.rcParams.update({'font.size': 10})
                    
                    # Plot top features
                    axes[0].barh(top_feature_names, top_importances_mean)
                    axes[0].set_xlabel("Feature Importance")
                    axes[0].set_title(f"Top {top_n_features} Permutation Feature Importance ({storage_type}) - {algorithm}")
                    
                    # Plot bottom features
                    axes[1].barh(bottom_feature_names, bottom_importances_mean)
                    axes[1].set_xlabel("Feature Importance")
                    axes[1].set_title(f"Bottom {top_n_features} Permutation Feature Importance ({storage_type}) - {algorithm}")
                    
                    plt.tight_layout()
                    
                    # Save the figure
                    file_name = f'{algorithm}_permutation_{type}_{storage_type}.png'
                    full_path = os.path.join(self.path_output_figures, 'feature_importance', file_name)
                    plt.savefig(full_path, dpi = 1000)
                    # plt.show()
                    print(f"--- Bar Figure saved to {full_path} ---")
                    
                    
    
    # Function to plot feature importance using SHAP algorithm
    def plot_feature_importance_shap(self, plot_type):
        # Create a dictionary to loop over the different types of storage
        storages = {'adsorb': self.model_storage_adsorb,
                    'random': self.model_storage_random,}
        combined_storages = {'adsorb': self.combined_model_storage_adsorb,
                             'random': self.combined_model_storage_random,}
        
        if plot_type == 'barplot':
            # Loop over each type of storage
            for storage_type, storage in storages.items():
                # Loop over each algorithm in the storage
                for algorithm, folds in storage.items():
                    
                    # Create a list to store the SHAP values for each fold
                    all_shap_values = []

                    # Loop over each fold
                    for fold_key, fold_data in folds.items():
                        # Load the trained model from this fold, model was trained on training data
                        pipeline = fold_data['model']           # pipeline object
                        model = pipeline.named_steps['model']   # regressor object
                        
                        # Extract the training data using the fold's train index
                        X_fold = self.X[fold_data['train_index']].copy()

                        # Initialize a SHAP explainer object
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_fold)

                        # Add the SHAP values for the current fold to the list
                        all_shap_values.append(shap_values)

                    # Calculate the average SHAP values across all folds
                    average_shap_values = np.mean(all_shap_values, axis = 0)

                    # Use the SHAP values to calculate the average feature importance
                    X = self.X

                    # Visualize the average SHAP values
                    shap.summary_plot(average_shap_values, X, feature_names=self.molecular_descriptors, plot_type="bar")

                    # Save the figure
                    file_name = f'{algorithm}_SHAP_{storage_type}_{plot_type}.png'
                    full_path = os.path.join(self.path_output_figures, 'feature_importance', file_name)
                    plt.savefig(full_path, dpi = 1000)
                    plt.close()
                    print(f"--- Figure saved to {full_path} ---")
        
        elif plot_type == 'beeswarm_plot':
            # Loop over each type of storage
            for storage_type, storage in combined_storages.items():
                # Loop over each algorithm in the storage
                for algorithm, data in storage.items():
                    # Load the best model from the combined storage
                    model = data['best_model'].named_steps['model']
                    
                    # Retrieve the best train and test indices
                    train_index = data['train_index_best_model']

                    # Extract the training and testing data using the best model's indices
                    X_train = self.X[train_index]
                    
                    # Initialize SHAP explainer on the training data used by the best model
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_train)
                    
                    # Plot SHAP values using beeswarm plot
                    plt.figure(figsize = (12, 12))
                    shap.plots.beeswarm(shap_values,
                                        order=shap_values.abs.max(0),
                                        max_display = 14,
                                        show = False)
                    plt.title(f"SHAP Beeswarm Plot for {algorithm} ({storage_type})")
                    file_name = f"{algorithm}_SHAP_{storage_type}_beeswarm.png"
                    full_path = os.path.join(self.path_output_figures, 'feature_importance', file_name)
                    plt.savefig(full_path, dpi = 1000)
                    print(f"--- Bar Figure saved to {full_path} ---")
    
    
    # Function to save the predictions to CSV files
    def save_predictions_to_csv(self):
        # Create a dictionary to loop over the different types of storage
        storages = {'adsorb': self.combined_model_storage_adsorb,
                    'random': self.combined_model_storage_random
                    }

        # Loop over each type of storage
        for split_type, storage in storages.items():
            # Loop over each algorithm in the storage
            for algorithm, data in storage.items():
                # Define the filenames for the train and test predictions
                train_filename = f'Prediction_450-{algorithm}-{split_type}_split-{self.search_type}-train-{data["train_rmse_avg"]:.4f}.csv'
                test_filename = f'Prediction_450-{algorithm}-{split_type}_split-{self.search_type}-test-{data["test_rmse_avg"]:.4f}.csv'

                # Construct full paths for the CSV files
                train_file_path = os.path.join(self.path_output_csv, 'predictions_configuration', train_filename)
                test_file_path = os.path.join(self.path_output_csv, 'predictions_configuration', test_filename)

                # Extract the DataFrames from the combined model storage
                df_train = data['df_train']
                df_test = data['df_test']

                # Save the DataFrames to CSV files
                df_train.to_csv(train_file_path, index=False)
                df_test.to_csv(test_file_path, index=False)

                # Print a message to confirm the saving of the files
                print(f'--- Saved train predictions csv file to {train_file_path}')
                print(f'--- Saved test predictions csv file to {test_file_path}')
    
    
    
    
    
## Main Function
if __name__ == "__main__":
    
    # Defining Algorithm List
    algorithm_list = [
                    # 'SVR',
                    'LASSO',
                    # 'Ridge',
                    # 'SGD',
                    # 'Bayesian_Ridge',
                    # 'K_Neighbors',
                    # 'Decision_Tree',
                    # 'Random_Forest',  # checked
                    # 'Gradient_Boost', # checked
                    # 'XGBoost',          # checked
                    # 'LightGBM',
                    # 'MLP',              # checked
                    ]
    
    search_types = [
                    'grid_search',
                    'random_search',
                    'bayesian_search',
                    'best_model',
                    ]
    
    for search_type in search_types:
        
        ## Run Models, No Cross Validation, Train and Test on 450 Instances
        descriptor_model_config = Descriptor_Model_Config(
                                                          algorithm_list = algorithm_list,
                                                          search_type = search_type,
                                                          plot_learning_curve = False,
                                                          plot_feature_importance = False,
                                                          save_predictions_csv = True,
                                                          verbose = 2,
                                                          output_label = 'e_int_dft'
                                                          )