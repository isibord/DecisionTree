class SpamHeuristicModel(object):
    """A heuristic spam filter"""

    def __init__(self):
        pass

    def fit(self, x, y):
        #self.weights = [.75, .75, .75, .25, .25]
        self.weights = [.75, .75, .75, .25, .25, 0, 0,0,0,0,0,0,0,0,0,0]
        pass

    def predict(self, x):
        predictions = []

        for example in x:
            scores = [ example[i] * self.weights[i] for i in range(len(example)) ]

            if sum(scores) > 1.5:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions