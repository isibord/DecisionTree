import math

class LogisticRegressionModel(object):
    """A logistic regression spam filter"""


    def __init__(self):
        self.weights = [0.03, 0.05, -0.05, 0.04, 0.03]
        self.w0 = 0
        
    def fit(self, x, y, iterations=1, step=0.01, threshold=0.5):
        self.threshold = threshold
        for k in range(iterations):
            yPredicted = self.hypothesis(x)
            gradient = 0
            for i in range(len(y)):
                gradient += (yPredicted[i] - y[i])

            gradient = gradient / float(len(y))

            self.w0 = self.w0 - (step * gradient)

            for j in range(len(self.weights)):
                gradient = 0
                for i in range(len(y)):
                    gradient += (yPredicted[i] - y[i]) * x[i][j]

                gradient = gradient / float(len(y))

                self.weights[j] = self.weights[j] - (step * gradient)

            if k % 1000 == 0 and iterations == 50000:
                print("%d, %f" % (k, self.loss(x, y)))


    def predict(self, x):
        predictions = []

        for example in x:
            z = self.w0
            for i in range(len(example)):
                z += example[i] * self.weights[i]
            
            predictions.append(1 if self.sigmoid(z) > self.threshold else 0)
        
        return predictions
        
    def hypothesis(self, x):
        predictions = []

        for example in x:
            z = self.w0
            for i in range(len(example)):
                z += example[i] * self.weights[i]
            
            predictions.append(self.sigmoid(z))
        
        return predictions

    def loss(self, x, y):
        sumOverSample = 0
        yPredicted = self.hypothesis(x)
        
        for i in range(len(y)):
            loss_i = (-y[i] * math.log(yPredicted[i])) - ((1 - y[i]) * (math.log(1.0-yPredicted[i])))
            sumOverSample += loss_i
            
        return sumOverSample

    def sigmoid(self, z):
        sig = 1.0 / float(1.0 + math.exp(-z))
        return sig