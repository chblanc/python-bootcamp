# import stuff
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# data-prep -------------------------------------------------------------------

# unpack into X (feature) matrix and y (target vector)
X,y = load_breast_cancer(return_X_y=True)

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
sklearn.svm.libsvm.predict_proba(svm.predict(X_train))
svm.score(X_train, y_train)

# randome forrest:
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
