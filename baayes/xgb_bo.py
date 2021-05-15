import csv
import random
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
# optimization algorithm
tpe_algorithm = tpe.suggest
from xgboost import XGBClassifier
import lightgbm as lgb
import numpy as np
# from imblearn.over_sampling import SMOTE
from helpers import *
# from PARAMETERS import *

class xgb_bo():

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
			'max_depth': hp.quniform("max_depth", 3, 18, 1),
			'gamma': hp.uniform ('gamma', 1,9),
			'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
			'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
			'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
			'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),

		    'class_weight': hp.choice('class_weight', [None, 'balanced']),
		    'booster': hp.choice('booster', [{'booster': 'gbtree', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
		                                                 {'booster': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
		                                                 {'booster': 'gblinear', 'subsample': 1.0}]),
		    'max_leaves': hp.quniform('max_leaves', 30, 150, 1),
		    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
		    # 'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
		    # 'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
		    }




	def objective(self, params):
	    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
	    
	    # Keep track of evals
	    # global ITERATION
	    
	    # ITERATION += 1
	    
	    # Retrieve the subsample if present otherwise set to 1.0
	    subsample = params['booster'].get('subsample', 1.0)
	    
	    # Extract the boosting type
	    params['booster'] = params['booster']['booster']
	    params['subsample'] = subsample
	    
	    # Make sure parameters that need to be integers are integers
	    for parameter_name in ['max_leaves','max_depth']: # 'subsample_for_bin', 'min_child_samples']:
	        params[parameter_name] = int(params[parameter_name])
	    
	    start = timer()
	    
	    # Perform n_folds cross validation
	#     cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
	#                         early_stopping_rounds = 100, metrics = 'auc', seed = 50)
	    
	    self.model_temp = XGBClassifier( class_weight = params['class_weight'], \
	        max_depth = params['max_depth'], \
	        gamma = params['gamma'], \
	        booster = params['booster'], \
	        max_leaves = params['max_leaves'], \
	        min_child_weight = params['min_child_weight'], \
	        learning_rate = params['learning_rate'], \
	        # subsample_for_bin = params['subsample_for_bin'], \
	        # min_child_samples = params['min_child_samples'], \
	        reg_alpha = params['reg_alpha'], \
	        reg_lambda = params['reg_lambda'], \
	        colsample_bytree = params['colsample_bytree'], \
	        subsample = params['subsample'],\
	        n_jobs=-1)
	    
	    # self.model_temp.fit(X_train, y_train)
	    
	    run_time = timer() - start
	    
	    # predictions = model_temp.predict(X_test)
	    # score = mape(y_test, predictions)

	    # self.score = self.model_temp.fit(X, y, self.N_FOLDS)
	    # self.score_am, self.score_recall, self.score_precision = self.cross_val_fit(self.model_temp, self.X, self.y, self.N_FOLDS, self.apply_smote)
	    self.results_dict = cross_val_fit(self.model_temp, self.X, self.y, self.N_FOLDS, self.cross_val, self.apply_smote, self.test_size, self.metric)
	    
	    # Loss must be minimized
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


	# def cross_val_fit(self, model, X, y, N_FOLDS, apply_smote=False):

	# 	cv = KFold(n_splits=N_FOLDS,shuffle=True, random_state=0)

	# 	scores_am = []
	# 	scores_recall = []
	# 	scores_precision = []
	# 	for train_index, test_index in cv.split(X):
	# 	    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
	# 	    if apply_smote:
	# 	    	sm = SMOTE(random_state=42)
	# 	    	X_train, y_train = sm.fit_resample(X_train, y_train)

	# 	    model.fit(X_train, y_train)
	# 	    y_pred = model.predict(X_test)
	# 	    scores_recall.append(recall_score(y_test, y_pred))
	# 	    scores_precision.append(precision_score(y_test, y_pred))
	# 	    scores_am.append(metric_am(y_test, y_pred))

	# 	return sum(scores_am)/N_FOLDS, sum(scores_recall)/N_FOLDS, sum(scores_precision)/N_FOLDS

	def run_model(self, path):
		# File to save first results
		self.out_file = path
		of_connection = open(self.out_file, 'w')
		writer = csv.writer(of_connection)
		writer.writerow(['loss', 'params', 'run_time', 'score_am_test', 'score_am_train'])
		of_connection.close()
		# Keep track of results
		bayes_trials = Trials()
		best = fmin(fn = self.objective, space = self.space, algo = tpe.suggest, 
			max_evals = self.MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

		# # File to save first results
		# self.out_file = path
		# self.apply_smote = apply_smote_
		# of_connection = open(self.out_file, 'w')
		# writer = csv.writer(of_connection)
		# writer.writerow(['loss', 'params', 'run_time', 'score_am', 'score_recall', 'score_precision'])
		# of_connection.close()
		# best = fmin(fn = self.objective, space = self.space, algo = tpe.suggest, 
		# 	max_evals = self.MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))





