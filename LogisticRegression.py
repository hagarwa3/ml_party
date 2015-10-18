'''
Logistic Regression (Multi-class Linear Classifier)
'''

from __future__ import print_function
import numpy as np


def sigmoid(z):
  ones = np.ones_like(z)
  g = ones / (ones + np.exp(-z))
  return g


def computeLogisticParts(theta, X, y):
  h = sigmoid(np.dot(X, theta))
  pos = np.vdot(y, np.log(h)) # computes summation also
  neg = np.vdot((1-y), np.log(1-h)) # computes summation also
  
  return h, pos, neg


def computeCost(theta, X, y):
  m, n = X.shape
  h, pos, neg = computeLogisticParts(theta, X, y)

  J = (pos + neg) / (-m)
  return J


def computeCostReg(theta, X, y, reg_lambda):
  m, n = X.shape
  temp_theta = theta
  temp_theta[0] = 0
  J = computeCost(theta, X, y)
  J = J + (reg_lambda / (2 * m)) * (np.vdot(temp_theta, temp_theta))
  return J


def computeGrad(theta, X, y):
  m, n = X.shape
  h, pos, neg = computeLogisticParts(theta, X, y)

  grad = np.dot((h - y), X) / m
  return grad


def computeGradReg(theta, X, y, reg_lambda):
  m, n = X.shape
  temp_theta = theta
  temp_theta[0] = 0
  grad = computeGrad(theta, X, y)
  grad = grad + (reg_lambda / m) * temp_theta
  return grad


def mapFeature(X1, X2, degree):
  m = X1.shape[0]
  out = np.ones_like(X1)
  for i in range(1, degree+1):
      for j in range(i+1):
          out = np.vstack(( out, (X1**(i-j)) * (X2**j) ))
          
  return out.T


class LogisticRegression:

  def __init__(self, degree):
    self.degree = degree;
    self.theta_reg = []
    self.X_reg = []


  def fit(self, X, y):
    self.X_reg = mapFeature(X[:,0], X[:,1], self.degree)
    #X_reg = X
    print('X.shape =', X.shape, ', y.shape =', y.shape)
    print('X_reg.shape =', self.X_reg.shape)

    # Initialize fitting parameters
    initial_theta = np.ones(self.X_reg.shape[1])*0.5

    # Set regularization parameter lambda to 1
    reg_lambda = 1

    cost = computeCostReg(initial_theta, self.X_reg, y, reg_lambda)
    grad = computeGradReg(initial_theta, self.X_reg, y, reg_lambda)

    print('Cost at initial theta (zeros): ', cost)

    import scipy.optimize as optim
    res = optim.fmin_tnc(func=computeCostReg, x0=initial_theta, fprime=computeGradReg, args=(self.X_reg, y, reg_lambda))
    #res = optim.fmin_cg(f=computeCostReg, x0=initial_theta, fprime=computeGradReg, args=(self.X_reg, y, reg_lambda))
    self.theta_reg = res[0]
    cost_reg = computeCostReg(self.theta_reg, self.X_reg, y, reg_lambda)

    # Print theta to screen
    print('Cost at theta found by fminunc: ', cost_reg)
    print('theta shape: ', self.theta_reg.shape)


  def predict(self, X_test, y_test):
    probabilities = sigmoid(np.dot(mapFeature(X_test[:,0], X_test[:,1], self.degree), self.theta_reg))
    predictions = [1 if p >= 0.5 else 0 for p in probabilities]

    # Compute accuracy on training set
    p = zip(predictions, y_test)
    results = [1 if a == b else 0 for (a, b) in p]
    print('Train Accuracy: ', float(sum(results)) / float(len(results)))


def main():
  from scipy.io import loadmat
  print('Loading matrix data...')
  #data = loadmat('datasets/digits_coursera.mat') # training data stored in arrays X, y
  dataReg = np.loadtxt('datasets/ex2data2.txt', delimiter=',')
  np.random.shuffle(dataReg)
  X = dataReg[:100, 0:2]
  y = dataReg[:100, 2]
  X_test = dataReg[100:, 0:2]
  y_test = dataReg[100:, 2]

  '''Xy_pair = np.hstack((data['X'],data['y']))
  print('Shuffling data...')
  np.random.shuffle(Xy_pair)
  
  X = Xy_pair[:4000,:-1]
  y = Xy_pair[:4000,-1].reshape(-1)

  X_test = Xy_pair[4000:,:-1]
  y_test = Xy_pair[4000:,-1].reshape(-1)'''

  print('X: ', X.shape, '| Y: ', y.shape, '\nX_test: ', \
      X_test.shape, '| y_test: ', y_test.shape)

  #print('y orig: ', np.unique(y))
  #y[y == 10] = 0
  #y_test[y_test == 10] = 0
  #print('y fixed: ', np.unique(y))

  log_clf = LogisticRegression(degree=6)
  log_clf.fit(X, y)

  log_clf.predict(X_test, y_test)


if __name__ == '__main__':
  main()
