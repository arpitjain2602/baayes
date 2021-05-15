from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from rf_bo import rf_bo
from xgb_bo import xgb_bo
from lgb_bo import lgb_bo
import pickle
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer
import pandas as pd
import ast

# def metric_am(y_true, y_pred):
#     return 0.8*recall_score(y_true, y_pred) + 0.2*precision_score(y_true, y_pred)

# my_scorer = make_scorer(metric_am, greater_is_better=True)
# cv = KFold(n_splits=5,shuffle=True, random_state=0)
# def cross_val_fit(model, X, y, N_FOLDS, apply_smote):

#     cv = KFold(n_splits=N_FOLDS,shuffle=True, random_state=0)

#     scores_am, scores_am_train = [],[]
#     scores_recall, scores_recall_train = [],[]
#     scores_precision, scores_precision_train = [],[]
#     scores_f1, scores_f1_train = [],[]
    
#     for train_index, test_index in cv.split(X):
#         X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#         if apply_smote:
#         	sm = SMOTE(random_state=42)
#         	X_train, y_train = sm.fit_resample(X_train, y_train)

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         y_pred_train = model.predict(X_train)
        
#         scores_am.append(metric_am(y_test, y_pred))
#         scores_recall.append(recall_score(y_test, y_pred))
#         scores_precision.append(precision_score(y_test, y_pred))
#         scores_f1.append(f1_score(y_test, y_pred, average='weighted'))
#         scores_am_train.append(metric_am(y_train, y_pred_train))
#         scores_recall_train.append(recall_score(y_train, y_pred_train))
#         scores_precision_train.append(precision_score(y_train, y_pred_train))
#         scores_f1_train.append(f1_score(y_test, y_pred, average='weighted'))

#     return { 
#             'metric_am_train':sum(scores_am_train)/N_FOLDS,
#             'metric_am':sum(scores_am)/N_FOLDS,
#             'recall_train':sum(scores_recall_train)/N_FOLDS,
#             'recall':sum(scores_recall)/N_FOLDS, 
#             'precision_train':sum(scores_precision_train)/N_FOLDS,
#             'precision':sum(scores_precision)/N_FOLDS,
#             'f1_train':sum(scores_f1_train)/N_FOLDS,
#             'f1':sum(scores_f1)/N_FOLDS,}


class train_models():

	def __init__(self):
		self.N_FOLDS = 5

		pass

	# def logistic(self, X, y, path, apply_smote):
	# 	model_logistic = LogisticRegression()
	# 	parameters = {'C': np.logspace(0, 4, 10),'penalty': ['l1', 'l2']}
	# 	clf = GridSearchCV(model_logistic, parameters, cv=cv, n_jobs=-1, scoring=my_scorer, verbose=0)
	# 	clf.fit(X, y)

	# 	print("tuned hpyerparameters :(best parameters) ",clf.best_params_)
	# 	print("score :",clf.best_score_)

	# 	best_logistic = LogisticRegression(penalty = clf.best_params_['penalty'], C = clf.best_params_['C'])
	# 	self.results_logistic = cross_val_fit(best_logistic, X, y, N_FOLDS=self.N_FOLDS, apply_smote=apply_smote)

	# 	output = {'model': best_logistic, 'result': self.results_logistic}

	# 	pickle.dump(output, open(path, 'wb'))


	# def support_vector(self, X, y, path, apply_smote):

	# 	model_2 = SVC()
	# 	parameters = [{'kernel': ['rbf'], 'gamma': [0.1,0.001,0.0001],
	# 	                     'C': [1, 10, 100, 1000]},
	# 	                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	# 	model_svc = GridSearchCV(model_2, parameters, cv=cv, n_jobs=-1, scoring=my_scorer, verbose=0)
	# 	model_svc.fit(X, y)

	# 	print("tuned hpyerparameters :(best parameters) ",model_svc.best_params_)
	# 	print("score :",model_svc.best_score_)

	# 	best_svc = SVC(gamma = model_svc.best_params_['gamma'], 
	# 	               C = model_svc.best_params_['C'], 
	# 	               kernel = model_svc.best_params_['kernel'])
	# 	# best_svc = SVC(model_svc.best_params_)
	# 	self.results_svc = cross_val_fit(best_svc, X, y, N_FOLDS=self.N_FOLDS, apply_smote=apply_smote)

	# 	output = {'model': best_svc, 'result': self.results_svc}

	# 	pickle.dump(output, open(path, 'wb'))

	def random_forest(self, X, y, path, apply_smote=False, cross_val='train_test', N_FOLDS=5, test_size=0.2, MAX_EVALS=500, metric='accuracy'):
		'''
		- cross_val: 'k_fold', 'train_test_split'
		'''
		rfm = rf_bo(X, y, MAX_EVALS=MAX_EVALS, N_FOLDS=N_FOLDS, test_size=test_size, apply_smote=apply_smote, cross_val=cross_val, metric=metric)
		rfm.run_model(path)

	def xgboost_model(self, X, y, path, apply_smote=False, cross_val='train_test', N_FOLDS=5, test_size=0.2, MAX_EVALS=500, metric='accuracy'):
		
		xgbm = xgb_bo(X, y, MAX_EVALS=MAX_EVALS, N_FOLDS=N_FOLDS, test_size=test_size, apply_smote=apply_smote, cross_val=cross_val, metric=metric)
		xgbm.run_model(path)

	def light_gbm(self, X, y, path, apply_smote=False, cross_val='train_test', N_FOLDS=5, test_size=0.2, MAX_EVALS=500, metric='accuracy'):
		
		lgbm = lgb_bo(X, y, MAX_EVALS=MAX_EVALS, N_FOLDS=N_FOLDS, test_size=test_size, apply_smote=apply_smote, cross_val=cross_val, metric=metric)
		lgbm.run_model(path)

	
	def run_all(self, X, y, path_xgboost, path_lightgbm, path_rf, path_support_vector, path_logistic, apply_smote=False):

		self.logistic(X, y, path_logistic, apply_smote)
		# print('Done logistic')

		self.light_gbm_model(X,y, path_lightgbm, apply_smote)
		# print('Done LightGBM')

		self.xgboost_model(X,y,path_xgboost, apply_smote)
		# print('Done XGBoost')

		self.random_forest(X,y, path_rf, apply_smote)
		# print('Done RF')

		self.support_vector(X,y, path_support_vector, apply_smote)
		# print('Done Support Vector')

	def get_best_model(self, path):
		results = pd.read_csv(path)
		results.sort_values('loss', ascending = True, inplace = True)
		results.reset_index(inplace = True, drop = True)
		# print(results.head())
		best_params_lgb = ast.literal_eval(results.loc[0, 'params'])
		return results, best_params_lgb



