


from skopt.space import Real, Integer, Categorical




PARAMETER_DICT_BAYESIAN = {
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    'Multi_Linear': {
        'fit_intercept': Categorical([True, False]),
        },

    ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    'SVR': {
        'kernel'        : Categorical(['poly', 'rbf', 'sigmoid']),
        'C'             : Real(1e-2, 1e+2, prior = 'log-uniform'),  # default = 1
        # 'gamma'         : Real(1e-4, 1e+1, prior = 'log-uniform'),  # default = 'scale'
        'degree'        : Integer(1, 4),        # default = 3
        'epsilon'       : Real(0.002, 0.5),     # default = 0.1
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    'LASSO': {
        'alpha'         : Real(1e-3, 1e+3, prior = 'log-uniform'),   # default = 1.0
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    'Ridge': {
        'alpha'         : Real(1e-3, 1e+2, prior = 'log-uniform'),   # default = 1.0
        'fit_intercept' : Categorical([True, False]),
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
    'SGD':  {
        'max_iter'      : Integer(1000, 3000),         # default = 1000
        'loss'          : Categorical(['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),   # default = squared_error
        'alpha'         : Real(1e-06, 1e-01, prior = 'log-uniform'),                        # default = 0.0001
        'l1_ratio'      : Real(0, 1),                                                       # default = 0.15
        'penalty'       : Categorical([None, 'l1', 'l2', 'elasticnet']),                    # default = 'l2'
        'learning_rate' : Categorical(['invscaling', 'constant', 'optimal', 'adaptive']),   # default = 'invscaling'
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
    'Bayesian_Ridge': {
        'alpha_1'       : Real(1e-07, 1e+2, prior = 'log-uniform'),
        'alpha_2'       : Real(1e-07, 1e+2, prior = 'log-uniform'),
        'lambda_1'      : Real(1e-07, 1e+2, prior = 'log-uniform'),
        'lambda_2'      : Real(1e-07, 1e+2, prior = 'log-uniform'),
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    'K_Neighbors': {
        'n_neighbors'   : Integer(2, 12),       # default = 5
        'leaf_size'     : Integer(10, 50),      # default = 30
        'algorithm'     : Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),   # default = 'auto'
        'weights'       : Categorical(['uniform','distance']),                      # default = 'uniform'
        'metric'        : Categorical(['minkowski', 'euclidean', 'manhattan']),     # default = 'minkowski'
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    'Decision_Tree': {
        'criterion'         : Categorical(['squared_error', 'friedman_mse', 'absolute_error']), # default = 'squared_error'
        'splitter'          : Categorical(['best', 'random']),      # default = 'best'
        'max_features'      : Categorical([None, 'log2', 'sqrt']),  # Default = None
        'max_leaf_nodes'    : Categorical([None]),                  # default = None
        'min_samples_leaf'  : Integer(1, 4),                        # default = 1
        'min_samples_split' : Integer(2, 10),                       # default = 2, must be >= 2
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    'Random_Forest': {
        'criterion'         : Categorical(['squared_error', 'absolute_error']), # default = 'squared_error'
        'n_estimators'      : Integer(10, 200),                     # default = 100
        'min_samples_leaf'  : Integer(1, 8),                       # default = 1
        'min_samples_split' : Integer(2, 8),                       # default = 2, must be >= 2
        },
    
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    'Gradient_Boost': {
        'loss'              : Categorical(['squared_error', 'absolute_error','huber', 'quantile']), # default = 'squared_error'
        'learning_rate'     : Real(0.001, 0.2, prior = 'log-uniform'),  # default = 0.1
        'n_estimators'      : Integer(50, 300),                         # default = 100
        'subsample'         : Real(0.7, 1.0, prior = 'uniform'),        # default = 1.0
        'max_depth'         : Integer(2, 10),                           # default = 3
        'max_features'      : Categorical([None, 'log2', 'sqrt']),      # Default = None
        'min_samples_split' : Integer(2, 6),                            # default = 2
        'min_samples_leaf'  : Integer(1, 4),                            # default = 1
        },
    
    
    ## https://xgboost.readthedocs.io/en/stable/parameter.html
    'XGBoost': {
        'learning_rate'     : Real(0.01, 0.4, prior = 'uniform'),   # default = 0.3, range: [0,1], typically 0.01 - 0.2
        'gamma'             : Real(0, 0.2, prior = 'uniform'),      # default = 0
        'n_estimators'      : Integer(20, 300),                     # default = 100
        'max_depth'         : Integer(2, 8),                        # default = 6
        'min_child_weight'  : Integer(1, 5),                        # default = 1
        'subsample'         : Real(0.7, 1.0, prior = 'uniform'),    # default = 1
        'colsample_bytree'  : Real(0.7, 1.0, prior = 'uniform'),    # default = 1
        'scale_pos_weight'  : [1],                                  # default = 1
        'alpha'             : Real(0, 0.1, prior = 'uniform'),      # default = 0
        'lambda'            : [1],                                  # default = 1
        },
    
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    'MLP':  {
        'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50), (100, 100), (50, 50, 50), (100, 50, 25), (100, 100, 50), (100, 100, 100), (50, 50, 50, 50), (100, 50, 25, 12), (100, 100, 50, 25)]),
        # 'activation'        : Categorical(['identity', 'logistic', 'tanh', 'relu']),    # default = 'relu'
        # 'activation'        : Categorical(['relu']),    # default = 'relu'
        # 'solver'            : Categorical(['adam']),                        # default = 'adam'
        # 'alpha'             : Real(1e-05, 1e-02, prior = 'log-uniform'),    # default = 0.0001
        # 'learning_rate'     : Categorical(['constant']),                    # default = 'constant'
        # 'learning_rate_init': Real(1e-04, 1e-02, prior = 'log-uniform'),    # default = 0.001
        # 'max_iter'          : Integer(200, 500),                           # default = 200
        },
    
    }

