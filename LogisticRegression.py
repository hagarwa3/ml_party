'''
Logistic Regression (Multi-class Linear Classifier)
'''

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


  def fit(self, X, y):
    X_reg = mapFeature(X[:,0], X[:,1], 6)
    print 'X.shape =', X.shape, ', y.shape =', y.shape
    print 'X_reg.shape =', X_reg.shape

    # Initialize fitting parameters
    initial_theta = np.zeros(X_reg.shape[1])

    # Set regularization parameter lambda to 1
    reg_lambda = 1

    cost = computeCostReg(initial_theta, X_reg, y, reg_lambda)
    grad = computeGradReg(initial_theta, X_reg, y, reg_lambda)

    print 'Cost at initial theta (zeros): ', cost

    res = optim.fmin_tnc(func=computeCostReg, x0=initial_theta, fprime=computeGradReg, args=(X_reg, y, reg_lambda))
    theta_reg = res[0]
    cost_reg = computeCostReg(theta_reg, X_reg, y, reg_lambda)

    # Print theta to screen
    print 'Cost at theta found by fminunc: ', cost_reg
    print 'theta: ', theta_reg


  def predict(self, X):
    # Compute accuracy on our training set
    p = zip(predict(theta_reg, X_reg), y)
    results = [1 if a == b else 0 for (a, b) in p]
    print 'Train Accuracy: ', float(sum(results)) / float(len(results))
