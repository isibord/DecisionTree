import numpy as np


class LogisticRegressionModel_NumPy(object):

   def __init__(self):
       self.weight0 = 0.0
       self.weights = np.ndarray(1)

   def loss(self, x, y):
       y_probs = self.calculate_probabilities(x)
       loss_values = np.multiply(-y, np.log(y_probs)) - np.multiply(np.subtract(1, y), (np.log(np.subtract(1.0, y_probs))))

       return np.sum(loss_values)

   def fit(self, x, y, iterations, step):
       numWeights = x.shape[1]
       self.weight0 = 0.0
       self.weights = np.zeros(numWeights)

       yDimension = y.shape[0]
       for i in range(iterations):
           y_probs = self.calculate_probabilities(x)

           y_diffs = np.subtract(y_probs, y)

           self.weight0 -= np.divide(np.multiply(step, np.sum(y_diffs)), yDimension)
           dlosses = np.divide(np.matmul(x.T, y_diffs), yDimension)
           self.weights = np.subtract(self.weights, np.multiply(dlosses, step))


   def predict(self, x, threshold_value):
       y_probs = self.calculate_probabilities(x)
       return self.vectorThreshold(y_probs, threshold_value)

   def calculate_probabilities(self, x):
       z = np.add(self.weight0, np.matmul(x, self.weights))
       probabilities = self.vectorSigmoid(z)
       return probabilities

   @staticmethod
   def vectorThreshold(values, threshold):
       return (values > threshold).astype(int)

   @staticmethod
   def vectorSigmoid(z):
       return np.divide(1.0, np.add(1.0, np.exp(-z)))