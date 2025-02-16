





PARAMETER_BEST = {
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    'Multi_Linear': {'fit_intercept': True},

    ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    'SVR': {
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10],  # default = 1
        'gamma': ['scale'], # default = 'scale'
        'epsilon': [0.05, 0.1, 0.2],  # default = 0.1
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    'LASSO':    {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],     # default = 1.0
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    'Ridge': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],     # default = 1.0
        'fit_intercept': [True, False],
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
    'SGD':  {
        'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],   # default = squared_error
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # default = 0.0001
        'penalty': [None, 'l1', 'l2', 'elasticnet'],    # default = 'l2'
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
    'Bayesian_Ridge': {
        'alpha_1':  [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10],
        'alpha_2':  [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10],
        'lambda_1': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10],
        'lambda_2': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10],
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    'K_Neighbors': {
        'n_neighbors'   : [2, 4, 6, 8, 10, 12],     # default = 5
        'algorithm'     : ['auto', 'ball_tree', 'kd_tree', 'brute'],     # default = 'auto'
        'weights'       : ['uniform','distance'],   # default = 'uniform'
        'metric'        : ['minkowski', 'euclidean', 'manhattan'],   # default = 'minkowski'
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    'Decision_Tree': {
        'criterion'         : ['squared_error'], # default = 'squared_error'
        'min_samples_leaf'  : [1, 2, 3],    # default = 1
        'min_samples_split' : [2, 3, 4],    # default = 2
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    'Random_Forest': {
        'criterion'         : ['squared_error'],
        'n_estimators'      : [20, 30], # default = 100
        'min_samples_leaf'  : [1, 2, 3],    # default = 1
        'min_samples_split' : [2, 3, 4],   # default = 2, must be >= 2
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    'Gradient_Boost': {
        'learning_rate': [0.1],      # default = 0.1
        'n_estimators' : [50, 100],   # default = 100
        'subsample'    : [0.8, 1.0],       # default = 1.0
        'max_depth'    : [2, 3, 4],          # default = 3
        # 'min_samples_split' : [2, 3, 4],     # default = 2
        # 'min_samples_leaf'  : [1, 2, 3],     # default = 1
        },
    
    
    ## https://xgboost.readthedocs.io/en/stable/parameter.html
    'XGBoost': {
        'eta': [0.1, 0.2],  # default = 0.3, range: [0,1], typically 0.01 - 0.2
        'max_depth': [5, 6],   # default = 6
        'min_child_weight': [1, 3, 5],  # default = 1
        'subsample': [1],               # default = 1
        'colsample_bytree': [1],        # default = 1
        'scale_pos_weight': [1],        # default = 1
        'alpha': [0],                   # default = 0
        'lambda': [1],                  # default = 1
        },
    
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    'MLP':  {
        'hidden_layer_sizes': [(10, 8, 6, 4),], # default = (100,)
        'activation'        : ['identity', 'logistic', 'tanh', 'relu'], # default = 'relu'
        'solver'            : ['adam'],         # default = 'adam'
        'learning_rate'     : ['constant'], # default = 'constant'
        'learning_rate_init': [0.001],      # default=0.001
        'max_iter'          : [200],  # default = 200
        },
 
    }

