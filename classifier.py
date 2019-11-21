from __future__ import print_function
import numpy as np
from sklearn import svm, metrics, datasets
import tensorflow as tf
import datetime as dt
import pickle
import os

full_path = os.path.realpath(__file__)
path = os.path.dirname(full_path)

# load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
nsamples1, nx1, ny1 = X_train.shape
nsamples2, nx2, ny2 = X_test.shape
X_train = X_train.reshape((nsamples1, nx1*ny1))
X_test = X_test.reshape((nsamples2, nx2*ny2))
# normalize training data from [0,255] to [0,1]
X_train = X_train/255.0
X_test = X_test/255.0
# create param for classifier
param_C = 5
param_gamma = 0.05
classifier = svm.SVC(C=param_C, gamma=param_gamma)

# start to train the classifier
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(X_train, y_train)
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time = end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))


# predict the value of the test
predicted = classifier.predict(X_test)

# calculate accurracy of this classifier
cm = metrics.confusion_matrix(y_test, predicted)

print("Accuracy={}".format(metrics.accuracy_score(y_test, predicted)))

filename = path+"/output/finalized_model.sav"
pickle.dump(classifier, open(filename, 'wb'))
