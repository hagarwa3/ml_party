from __future__ import print_function
import numpy as np
from scipy.io import loadmat


def load_data():
  print('Loading matrix data...')
  data = loadmat('../datasets/digits_coursera.mat') # training data stored in arrays X, y

  Xy_pair = np.hstack((data['X'],data['y']))
  print('Shuffling data...')
  np.random.shuffle(Xy_pair)
  
  X = Xy_pair[:4000,:-1]
  y = Xy_pair[:4000,-1].reshape(-1)

  X_test = Xy_pair[4000:,:-1]
  y_test = Xy_pair[4000:,-1].reshape(-1)

  print('X: ', X.shape, '| Y: ', y.shape, '\nX_test: ', \
      X_test.shape, '| y_test: ', y_test.shape)

  print('y orig: ', np.unique(y))
  y[y == 10] = 0
  y_test[y_test == 10] = 0
  print('y fixed: ', np.unique(y))

  return X, y, X_test, y_test


def train_data(X, y):
  from sklearn import linear_model

  log_clf = linear_model.LogisticRegression()
  log_clf.fit(X, y)

  return log_clf


def predict_data(clf, X_test, y_test):
  prediction = clf.predict(X_test)

  # Compute accuracy on our training set
  p = zip(prediction, y_test)
  results = [1 if a == b else 0 for (a, b) in p]
  print('Train Accuracy: ', float(sum(results)) / float(len(results)))


def main():
  X, y, X_test, y_test = load_data()
  clf = train_data(X, y)
  predict_data(clf, X_test, y_test)


if __name__ == '__main__':
  main()