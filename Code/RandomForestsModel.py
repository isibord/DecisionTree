import DecisionTreeModel
import Assignment1Support
import random

class RandomForestsModel(object):
    """A random forests spam filter"""
    
    def __init__(self):
        self.dtModel = DecisionTreeModel.DecisionTreeModel()
        pass
        
    def fit(self, x, y, numTrees, minSplit, useBagging=False, featureRestriction=0, seed=100):
        random.seed(seed)
        self.trees = []
        for i in range(numTrees):
            if useBagging:
                (x, y) = self.bootstrapSample(x, y)
            featuresToUse = self.randomlySelect(len(x[0]), featureRestriction)
            self.trees.append(self.dtModel.growTree(x, y, minSplit, featuresToUse))


    def bootstrapSample(self, x, y):
        xchunks = Assignment1Support.partition(x, 5)
        ychunks = Assignment1Support.partition(y, 5)

        xBootstrap = []
        yBootstrap = []

        for i in range(5):
            index = random.randint(0, 4)
            xBootstrap.extend(xchunks[index])
            yBootstrap.extend(ychunks[index])

        return (xBootstrap, yBootstrap) 


    def randomlySelect(self, numFeatures, featuresToUse):
        if featuresToUse <= 0 or featuresToUse > numFeatures:
            return list(range(numFeatures))

        return random.sample(range(0, numFeatures), featuresToUse)
        

    def predictByMajorityVote(self, treePredictions, x ):
        yPredictions = []
        yProbabilityEstimates = []
        for i in range(len(x)):
            count1s = 0
            for treePrediction in treePredictions:
                if treePrediction[i] == 1:
                    count1s += 1
            yPredictions.append(1 if (count1s > (len(treePredictions) / 2.0)) else 0)

            countProb = count1s / len(treePredictions)
            yProbabilityEstimates.append(countProb)

        return (yPredictions, yProbabilityEstimates)


    def predictTree(self, x, tree):
        yPrediction = []
        for currval in x:
            resultval = self.dtModel.test_example(currval, tree)
            yPrediction.append(resultval)
        return yPrediction

    def predict(self, x):
        treePredictions = []
        for eachTree in self.trees:
            treePredictions.append(self.predictTree(x, eachTree))

        (yPredictions, yProbabilityEstimates) = self.predictByMajorityVote(treePredictions, x)
        return (yPredictions, yProbabilityEstimates, treePredictions)

    def predictThres(self, x, threshold=0.5):
        (yPredictions, yProbabilityEstimates, treePredictions) = self.predict(x)
        predictions = []

        for probVal in yProbabilityEstimates:
            predictions.append(1 if probVal > threshold else 0)

        return predictions
