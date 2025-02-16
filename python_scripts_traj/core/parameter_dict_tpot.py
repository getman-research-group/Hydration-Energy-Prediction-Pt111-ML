
import numpy as np

CONFIG_DICT = {

    '''
    Selectors
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
    '''
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.00, 0.05),
        'estimator': {
            'sklearn.ensemble.RandomForestRegressor': {
                'n_estimators': [40, 60, 80, 100],
                'criterion': ['squared_error'],
            }
        }
    },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html
    'sklearn.feature_selection.SelectFwe': {
        'alpha': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 50),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },



    
    '''
    Transformers
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    '''
    
    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },
    
    
    
    
    '''
    Regressors
    '''
    
    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor
    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90, 100],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': range(1, 8),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 5),
        'subsample': np.arange(0.05, 1.01, 0.05),
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.linear_model.RidgeCV': {
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0],
        'objective': ['reg:squarederror']
    },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor
    'sklearn.linear_model.SGDRegressor': {
        'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': [None, 'l1', 'l2', 'elasticnet'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

}


def get_config_dict(config_dict, regressor_name):
    # Initialize three empty dictionaries to store selectors, transformers, and regressors
    selectors_dict = {}
    transformers_dict = {}
    regressor_dict = {}

    # Loop through the config_dict and separate the selectors, transformers, and regressors
    for key, value in config_dict.items():
        # Selectors
        if key.startswith('sklearn.feature_selection'):
            selectors_dict[key] = value
        # Transformers
        elif key.startswith('sklearn.preprocessing') or key.startswith('sklearn.decomposition'):
            transformers_dict[key] = value
        # Regressors
        elif key.endswith(regressor_name):
            regressor_dict[key] = value
   
    # Check if the regressor_dict is empty or contains more than one entry
    if len(regressor_dict) == 0:
        raise ValueError(f"No regressor found with the name ending in '{regressor_name}'. Please check the input name.")
    elif len(regressor_dict) > 1:
        raise ValueError("Multiple regressors found. Please ensure only one regressor is specified.")
    else:
        # Print the key of the single regressor found
        print(f"--- Regressor in config_dict: {list(regressor_dict.keys())[0]}")

    # Combine the selectors, transformers, and regressors into a single dictionary
    result_dict = {**selectors_dict, **transformers_dict, **regressor_dict}

    return result_dict


if __name__ == '__main__':
    test_dict = get_config_dict(config_dict = CONFIG_DICT,
                                regressor_name = 'RandomForestRegressor',
                                )