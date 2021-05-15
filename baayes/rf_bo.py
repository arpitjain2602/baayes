import csv
import random
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
# optimization algorithm
tpe_algorithm = tpe.suggest
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import numpy as np
# from imblearn.over_sampling import SMOTE

# def metric_am(y_true, y_pred):
#     return 0.8*recall_score(y_true, y_pred) + 0.2*precision_score(y_true, y_pred)

from helpers import *
help_ = helpers()

class rf_bo():

	def __init__(self, X, y, MAX_EVALS, N_FOLDS, test_size, apply_smote, cross_val, metric):

		self.MAX_EVALS = MAX_EVALS
		self.N_FOLDS = N_FOLDS
		self.test_size = test_size
		self.apply_smote = apply_smote
		self.cross_val=cross_val
		self.metric=metric

		self.X = X
		self.y = y

		# # Hyperparameter grid
		# self.param_grid = {
		#     'class_weight': [None, 'balanced'],
		#     'boosting_type': ['gbdt', 'goss', 'dart'],
		#     'num_leaves': list(range(30, 150)),
		#     'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
		#     'subsample_for_bin': list(range(20000, 300000, 20000)),
		#     'min_child_samples': list(range(20, 500, 5)),
		#     'reg_alpha': list(np.linspace(0, 1)),
		#     'reg_lambda': list(np.linspace(0, 1)),
		#     'colsample_bytree': list(np.linspace(0.6, 1, 10))
		# }
		# # Subsampling (only applicable with 'goss')
		# self.subsample_dist = list(np.linspace(0.5, 1, 100))
		# self.params = {key: random.sample(value, 1)[0] for key, value in self.param_grid.items()}
		# self.params['subsample'] = random.sample(self.subsample_dist, 1)[0] if self.params['boosting_type'] != 'goss' else 1.0



		# Define the search space - for lightGBM
		self.space = {
			'n_estimators' : hp.quniform('n_estimators', 5, 100, 2),
			'max_features' : hp.choice('max_features', ['auto', 'sqrt', 'log2']),
			'criterion' : hp.choice('criterion', ['gini', 'entropy']),
			'max_depth': hp.quniform("max_depth", 3, 18, 1),
			'min_samples_split': hp.quniform("min_samples_split", 0.01, 1, 0.01), # min_samples_split must be an integer greater than 1 or a float in (0.0, 1.0]
			'min_samples_leaf': hp.quniform("min_samples_leaf", 0.01, 0.5, 0.01), #min_samples_leaf must be at least 1 or in (0, 0.5]
			'bootstrap' : hp.choice('bootstrap', [True, False]),
			'min_weight_fraction_leaf': hp.quniform("min_weight_fraction_leaf", 0.00001, 0.5, 0.01), #min_weight_fraction_leaf must in [0, 0.5]
			'max_leaf_nodes': hp.quniform("max_leaf_nodes", 3, 30, 1),
			'min_impurity_decrease': hp.quniform("min_impurity_decrease", 0.0, 30.0, 1.0),
			'class_weight': hp.choice('class_weight', [None, 'balanced']),
			'random_state': hp.choice('random_state', [0]),
		    }


	def objective(self, params):
	    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
	    
	    # Retrieve the subsample if present otherwise set to 1.0
	    # subsample = params['booster'].get('subsample', 1.0)
	    
	    # # Extract the boosting type
	    # params['booster'] = params['booster']['booster']
	    # params['subsample'] = subsample
	    
	    # Make sure parameters that need to be integers are integers
	    for parameter_name in ['max_depth', 'max_leaf_nodes','n_estimators','random_state']: # 'subsample_for_bin', 'min_child_samples']:
	        params[parameter_name] = int(params[parameter_name])
	    
	    start = timer()
	    
	    # Perform n_folds cross validation
	#     cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
	#                         early_stopping_rounds = 100, metrics = 'auc', seed = 50)
	    
	    self.model_temp = RandomForestClassifier( class_weight = params['class_weight'], \
	        max_depth = params['max_depth'], \
	        n_estimators = params['n_estimators'], \
	        max_features = params['max_features'], \
	        criterion = params['criterion'], \
	        min_samples_split = params['min_samples_split'], \
	        min_samples_leaf = params['min_samples_leaf'], \
	        bootstrap = params['bootstrap'], \
	        min_weight_fraction_leaf = params['min_weight_fraction_leaf'], \
	        max_leaf_nodes = params['max_leaf_nodes'], \
	        min_impurity_decrease = params['min_impurity_decrease'], \
	        random_state = params['random_state'], \
	        n_jobs=-1)
	    
	    # self.model_temp.fit(X_train, y_train)
	    
	    run_time = timer() - start
	    
	    # predictions = model_temp.predict(X_test)
	    # score = mape(y_test, predictions)

	    # self.score = self.model_temp.fit(X, y, self.N_FOLDS)
	    # self.score_am, self.score_recall, self.score_precision = self.cross_val_fit(self.model_temp, self.X, self.y, self.N_FOLDS)

	    self.results_dict = help_.cross_val_fit(self.model_temp, self.X, self.y, self.N_FOLDS, self.cross_val, self.apply_smote, self.test_size, self.metric)
	    # Extract the best score
	#     best_score = np.max(cv_results['auc-mean'])
	    
	    # Loss must be minimized
	#     loss = 1 - best_score
	    loss = 1-self.results_dict['metric_am_test']
	    
	    # Boosting rounds that returned the highest cv score
	#     n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

	    # Write to the csv file ('a' means append)
	    of_connection = open(self.out_file, 'a')
	    writer = csv.writer(of_connection)
	    writer.writerow([loss, params, run_time, self.results_dict['metric_am_test'], self.results_dict['metric_am_train']])
	#     writer.writerow([loss, params, ITERATION, n_estimators, run_time])
	    
	    # Dictionary with information for evaluation
	    return {'loss': loss, 'params': params, 'train_time': run_time, 'status': STATUS_OK,
    			'score_am_test':self.results_dict['metric_am_test'],'score_am_train':self.results_dict['metric_am_train']}
	#     return {'loss': loss, 'params': params, 'iteration': ITERATION,
	#             'estimators': n_estimators, 
	#             'train_time': run_time, 'status': STATUS_OK}

	def cross_val_fit(self, model, X, y, N_FOLDS, cross_val, apply_smote, test_size, metric):
		if metric=='accuracy':
			metric_am = accuracy_score
		elif metric=='f1_score':
			metric_am = f1_score
		elif metric=='precision':
			metric_am = precision_score
		scores_am, scores_am_train = [], []

		if cross_val=='k_fold':
			cv = KFold(n_splits=N_FOLDS,shuffle=True, random_state=0)
			for train_index, test_index in cv.split(X):
				X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
				# if apply_smote:
				# 	sm = SMOTE(random_state=42)
				# 	X_train, y_train = sm.fit_resample(X_train, y_train)

				model.fit(X_train, y_train)
				y_pred = model.predict(X_test)
				y_pred_train = model.predict(X_train)

				scores_am.append(metric_am(y_test, y_pred))
				scores_am_train.append(metric_am(y_train, y_pred_train))

			return { 
			'metric_am_train':sum(scores_am_train)/N_FOLDS,
			'metric_am_test':sum(scores_am)/N_FOLDS,
			}

		if cross_val=='train_test':
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
			y_pred_train = model.predict(X_train)

			scores_am.append(metric_am(y_test, y_pred))
			scores_am_train.append(metric_am(y_train, y_pred_train))
			return { 
			'metric_am_train':scores_am_train[0],
			'metric_am_test':scores_am[0],
			}

	def run_model(self, path):
		# File to save first results
		self.out_file = path
		of_connection = open(self.out_file, 'w')
		writer = csv.writer(of_connection)
		writer.writerow(['loss', 'params', 'run_time', 'score_test', 'score_train'])
		of_connection.close()
		# Keep track of results
		bayes_trials = Trials()
		best = fmin(fn = self.objective, space = self.space, algo = tpe.suggest, 
			max_evals = self.MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))


