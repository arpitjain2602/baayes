from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

def cross_val_fit(model, X, y, N_FOLDS, cross_val, apply_smote, test_size, metric):
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