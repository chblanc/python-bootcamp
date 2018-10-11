# import stuff
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn import datasets, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

# data-prep -------------------------------------------------------------------

# unpack into X (feature) matrix and y (target vector)
X,y = datasets.load_breast_cancer(return_X_y=True)

# pca -------------------------------------------------------------------------

# as is customary, instatiate the model first w/only 2 components; might be
# nice to standardize the dataset first, but haven't figured out how to do
# that yet :)
pca = PCA(n_components=2)

# fit the model
pca.fit(X)

# apply the transformation
X_pca = pca.transform(X)
X_pca.shape

# data pre-process ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
  X, y,
  test_size=0.25,
  random_state=0, stratify=y
)

# did the stratifcation work correctly? 
print("y_train proportion:", str(y_train.sum()/len(y_train)))
print("y_test proportion:", str(y_test.sum()/len(y_test)))

print("X_train shape: %s" % repr(X_train.shape))
print("y_train shape: %s" % repr(y_train.shape))
print("X_test shape: %s" % repr(X_test.shape))
print("y_test shape: %s" % repr(y_test.shape))

# prelim models ---------------------------------------------------------------

# fit a couple of models (linear SVM, RandomForrest) and assess in-train
# performance (we'll save the TEST data for a cross-validated model)

# svm: same as before, insatiate model first, then fit
svm = LinearSVC(max_iter=10000, verbose=True)
svm.fit(X_train, y_train)
svm.predict(X_train)
svm.score(X_train, y_train)

# random forest:
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf.predict_proba(X_train)
rf.score(X_train, y_train)
metrics.log_loss(y_true=y_train, y_pred=rf.predict_proba(X_train))

# cross-val models ------------------------------------------------------------

# insatiate 10-fold cv: what about stratification?
k_fold = KFold(n_splits=10)

# splitting on the `k_fold` object returns a set of `train_indices`
# and `test_indices` which can be used to fit models
for train_indices, test_indices in k_fold.split(X):
     print('Train: %s | test: %s' % (len(train_indices), len(test_indices)))

# fit the training indices in each fold and score on the testing indices
cv_accuracy = [
  rf.fit(X_train[train], y_train[train]).score(X_train[test], y_train[test])
  for train, test in k_fold.split(X_train)
]

print("cv mean accuary: %s" % np.mean(cv_accuracy))

# alternatively, you can use a helper function, note:
# n_jobs=-1 = computation will be dispatched on all the CPUs of the computer
cross_val_score(rf, X_train, y_train, cv=k_fold, n_jobs=-1)
cross_val_score(rf, X_train, y_train, cv=k_fold, n_jobs=-1, scoring="precision")

# stratified K-fold
st_k_fold = StratifiedKFold(n_splits=5, random_state=1337)
stratified_cv_precision = cross_val_score(
  rf,
  X_train,
  y_train,
  cv=st_k_fold,
  scoring = "precision"
)

# grid-search cross-val models ------------------------------------------------

# example using svc (support vector machine w/C support)
svc = svm.SVC(C=1, kernel='linear')

# if we're going to do a grid search we best find out which parameters are
# tuneable
help(svm.SVC)

# or ... 
pprint(svc.get_params())

# looks like we can tune `C` and `degree` among other things
Cs = np.logspace(-6, -1, 10)
degree = np.linspace(start=2, stop=5, num=4)

# let's set up our grid
tune_grid = {'C': Cs, 'degree': degree}
pprint(tune_grid)

# now, here we specify the tune grid and then fit all the models
svm_gs = GridSearchCV(svc, param_grid=tune_grid, scoring="accuracy", cv=10)
svm_gs.fit(X_train, y_train)

# best tunes and scores
svm_gs.best_score_
svm_gs.best_estimator_.C
svm_gs.best_estimator_.degree

# On the diabetes dataset, find the optimal regularization parameter alpha.
# Bonus: How much can you trust the selection of alpha?
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# split out the data, create train/validation
db_X, db_y = datasets.load_diabetes(return_X_y=True)

db_Xtrain, db_Xval, db_ytrain, dbyval = train_test_split(
  db_X, db_y,
  test_size=0.25,
  random_state=0
)

# lasso params
pprint(Lasso().get_params)

# alpha needs to be tuned
alphas = np.linspace(start=0, stop=1, num=100)
lasso_grid = {"alpha": alpha}
lasso = Lasso(random_state=1337)

# create a lasso obj; get the tune-able parms
lasso_cv = GridSearchCV(lasso, lasso_grid, cv=10, refit=False, scoring="neg_mean_squared_error")
lasso_cv.fit(db_Xtrain, db_ytrain)
lasso_cv.best_params_
scores = lasso_cv.cv_results_['mean_test_score']
scores_std = lasso_cv.cv_results_['std_test_score']
