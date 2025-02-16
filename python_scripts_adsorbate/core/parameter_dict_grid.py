



PARAMETER_DICT_GRID = {
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    'Multi_Linear': {
        # Whether to calculate the intercept for this model
        'fit_intercept': [True, False],
        
        },

    ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    'SVR': {
        ## 'rbf' is the most used kernel
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        
        # C determines allowable error in the model, controls the trade-off between model complexity and error.
        # Large C assigns higher penalties to errors, too large C causes overfitting and lower generalization.
        # Small C assigns fewer penalties to error, too small C causes underfitting.
        'C': [0.01, 0.1, 1, 10, 100],  # default = 1
        
        # Kernel coefficient
        # The larger the gamma, the fewer the support vectors, and the smaller the gamma, the more the support vectors.
        'gamma': ['scale', 'auto'], # default = 'scale'
        
        # Epsilon determines the width of the tube around the estimated function (hyperplane).
        # Points that fall inside this tube are considered as correct predictions.
        # The literature recommends an epsilon between 0.001 and 1.
        'epsilon': [0.02, 0.05, 0.1, 0.2, 0.5],  # default = 0.1
        
        },
    
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    'LASSO':    {
        # Constant that multiplies the L1 term, controlling regularization strength.
        'alpha': [0.01, 0.1, 1, 10, 100],     # default = 1.0
        #'lambda': [0.001, 0.01, 0.1, 1, 10, 100, 1000],    # default = 0.001
        },
    
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    'Ridge': {
        # Constant that multiplies the L2 term, controlling regularization strength.
        'alpha': [0.01, 0.1, 1, 10, 100],     # default = 1.0
        
        # Whether to calculate the intercept for this model
        'fit_intercept': [True, False],
        },
    
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
    'SGD':  {
        # The loss function to be used.
        'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],   # default = squared_error
        
        # Constant that multiplies the regularization term.
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],  # default = 0.0001
                
        # regularization term
        'penalty': [None, 'l1', 'l2', 'elasticnet'],    # default = 'l2'
        },
    
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
    'Bayesian_Ridge': {
        'alpha_1':  [1e-06, 1e-05, 1e-04, 1e-03],  # default=1e-6
        'alpha_2':  [1e-06, 1e-05, 1e-04, 1e-03],  # default=1e-6
        'lambda_1': [1e-06, 1e-05, 1e-04, 1e-03],  # default=1e-6
        'lambda_2': [1e-06, 1e-05, 1e-04, 1e-03],  # default=1e-6
        },
    
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    'K_Neighbors': {
        # Number of neighbors to use by default for kneighbors queries.
        'n_neighbors'   : [2, 4, 5, 6, 8, 10],     # default = 5
        
        # Algorithm used to compute the nearest neighbors
        'algorithm'     : ['auto', 'ball_tree', 'kd_tree', 'brute'],     # default = 'auto'
        
        # Weight function used in prediction.
        'weights'       : ['uniform','distance'],   # default = 'uniform'
        
        # Metric to use for distance computation.
        'metric'        : ['minkowski', 'euclidean', 'manhattan'],   # default = 'minkowski'
        
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    'Decision_Tree': {
        
        # The function to measure the quality of a split.
        'criterion'         : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], # default = 'squared_error'
        
        # ## The default "best" is good when the sample size is small, if the sample size is very large, "random" is recommended.
        # ## 'best' goes through all possible sets of splits on each feature in the dataset and selects the best split.
        # ## 'random' adds randomness to the model, the split has been made between random subsets of features.
        # 'splitter'          : ['best', 'random'],   # default = 'best'
        
        # more depth may result in overfitting.
        # Less depth may result in underfitting.
        # In general, this value can be ignored when there is little data or few features.
        # 'max_depth'         : [None, 2, 5, 10, 20, 50], # Default = None
        
        # the number of features to consider when looking for the best split.
        # In general, if the number of features less than 50, use the default 'None'.
        # 'max_features'      : [None, 'log2', 'sqrt'],   # Default = None
        
        # ## Limit the maximum number of leaf nodes, prevent pverfitting
        # ## If there are not many features, this can be ignored, but if the features are many, it can be set
        # 'max_leaf_nodes'    : [None],   # default = None
        
        # The minimum number of samples required to be at a leaf node.
        # If the sample size is not very large, this value is not needed.
        # If the sample size is very large, it is recommended to increase this value.
        'min_samples_leaf'  : [1, 2, 3],   # default = 1
        
        # Minimum number of samples required for node splitting:
        # If number of samples in a node is less than this value, no further attempts will be made to select the optimal features for splitting.
        # If the sample size is not very large, this value is not needed.
        # If the sample size is very large, it is recommended to increase this value.
        'min_samples_split' : [2, 3, 4], # default = 2
                    
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    'Random_Forest': {
        # The function to measure the quality of a split.
        'criterion'         :['squared_error'],
        
        # The number of decision trees. Too small cause underfitting, and too big cause overfitting
        'n_estimators'      :[20, 30, 40, 50, 60, 70, 80], # default = 100
        
        # more depth may result in overfitting.
        # Less depth may result in underfitting.
        # In general, this value can be ignored when there is little data or few features.
        # 'max_depth'         : [None, 2, 5, 10, 20, 50], # Default = None
        
        # the number of features to consider when looking for the best split.
        # In general, if the number of features less than 50, use the default 'None'.
        # 'max_features'      : [None, 1.0, 'log2', 'sqrt'],   # Default = None
        
        # ## Limit the maximum number of leaf nodes, prevent pverfitting
        # ## If there are not many features, this can be ignored, but if the features are many, it can be set
        # 'max_leaf_nodes'    : [None],   # default = None
        
        # The minimum number of samples required to be at a leaf node.
        # If the sample size is not very large, this value is not needed.
        # If the sample size is very large, it is recommended to increase this value.
        'min_samples_leaf'  : [1, 2, 3],    # default = 1
        
        # Minimum number of samples required for node splitting:
        # If number of samples in a node is less than this value, no further attempts will be made to select the optimal features for splitting.
        # If the sample size is not very large, this value is not needed.
        # If the sample size is very large, it is recommended to increase this value.
        'min_samples_split' : [2, 3],   # default = 2, must be >= 2
        
        },
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    'Gradient_Boost': {
        # Learning rate shrinks the contribution of each tree. There is a trade-off between learning_rate and n_estimators.
        'learning_rate': [0.05, 0.1, 0.2],      # default = 0.1
        
        # Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        'n_estimators' : [40, 60, 80, 100],   # default = 100
        
        # The fraction of samples to be used for fitting the individual base learners.
        'subsample'    : [0.8, 1.0],       # default = 1.0
        
        # define the maximum depth of the tree.
        # Too small cause underfitting and too big causes overfitting.
        # 'max_leaf_nodes' = 2 ** 'max_depth'
        'max_depth'    : [2, 3, 4, 5],          # default = 3
        
        # the number of features to consider when looking for the best split.
        # In general, if the number of features less than 50, use the default 'None'.
        # 'max_features'      : [None, 'log2', 'sqrt'],   # Default = None
        
        # Minimum number of samples required for node splitting:
        # If number of samples in a node is less than this value, no further attempts will be made to select the optimal features for splitting.
        # If the sample size is not very large, this value is not needed.
        # If the sample size is very large, it is recommended to increase this value.
        'min_samples_split' : [2, 3],     # default = 2
        
        # The minimum number of samples required to be at a leaf node.
        # If the sample size is not very large, this value is not needed.
        # If the sample size is very large, it is recommended to increase this value.
        'min_samples_leaf'  : [1, 2],     # default = 1

        },
    
    
    ## https://xgboost.readthedocs.io/en/stable/parameter.html
    'XGBoost': {
        ## Step size shrinkage used in update to prevents overfitting.
        ## Small number makes model more conservative
        'eta': [0.1, 0.2, 0.3],  # default = 0.3, range: [0,1], typically 0.01 - 0.2
        
        ## min_split_loss
        ## Minimum loss reduction required to make a further partition on a leaf node of the tree.
        ## Biggger value makes the model more conservative
        # 'gamma': [0, 0.1, 0.2],              # default = 0
        
        ## Define the maximum depth of the tree.
        ## Too small cause underfitting and too big causes overfitting.
        'max_depth': [4, 5, 6, 7, 8],   # default = 6
        
        ## Minimum sum of instance weight needed in a child.
        ## Minimum number of samples needed to build each model. The larger value, the more conservative it is.
        'min_child_weight': [1, 3, 5],  # default = 1
        
        ## Small number of portion makes the model conservative and aviod overfitting, but too small may casue underfitting
        ## Typical value: 0.5 - 1
        'subsample': [0.8, 0.9, 1],     # default = 1
        
        ## subsample ratio of columns when constructing each tree.
        'colsample_bytree': [0.8, 0.9, 1],  # default = 1
                
        ## Control the balance of positive and negative weights, useful for unbalanced classes.
        'scale_pos_weight': [1],        # default = 1
        
        ## L1 regularization term on weights. Increasing this value will make model more conservative.
        'alpha': [0, 0.1],                   # default = 0
        
        ## L2 regularization term on weights. Increasing this value will make model more conservative.
        'lambda': [1],                  # default = 1

        },
    
    
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    'LightGBM': {
        
            'num_leaves': [10, 20, 31, 40, 50],             # default = 31
            'min_data_in_leaf': [5, 10, 20, 50, 100, 200],  # default = 20
            # 'max_depth': [-1],                              # default = -1
            'learning_rate': [0.01, 0.05, 0.1],             # default = 0.1
            'n_estimators': [50, 100, 150],                 # Number of trees
    
    },

    
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    'MLP':  {
        
        # The ith element represents the number of neurons in the ith hidden layer.
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25), (10, 8, 6, 4),],
        
        # Activation function for the hidden layer.
        'activation'        : ['relu'], # default = 'relu'
        
        # The solver for weight optimization, aka optimizer
        # 'solver': ['lbfgs', 'sgd', 'adam'],   # default = 'adam'
        'solver'            : ['adam'],         # default = 'adam'
        
        # Strength of the L2 regularization term.
        'alpha'             : [0.0001, 0.001, 0.01, 0.1], # default=0.0001
        
        # Learning rate schedule for weight updates.
        'learning_rate': ['constant', 'invscaling', 'adaptive'], # default = 'constant'
        # 'learning_rate'     : ['constant'], # default = 'constant'
        
        # The initial learning rate used. Only used when solver = ’sgd’ or ‘adam’.
        'learning_rate_init': [0.001],      # default=0.001
        
        # Maximum number of iterations. The solver iterates until convergence or this number of iterations.
        'max_iter'          : [200, 500],  # default = 200
        },
    
    }


