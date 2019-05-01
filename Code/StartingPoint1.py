
import Assignment1Support
import EvaluationsStub
import BagOfWords
import collections
import operator
import numpy as np

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

print("Train is %f percent spam." % (sum(yTrainRaw)/len(yTrainRaw) * 100.0))
print("Test is %f percent spam." % (sum(yTestRaw)/len(yTestRaw) * 100.0 ))

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

############################
import MostCommonModel

model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

#print("### 'Most Common' model")

#EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

############################
import SpamHeuristicModel
model = SpamHeuristicModel.SpamHeuristicModel()
model.fit(xTrain, yTrain)
yTestPredicted = model.predict(xTest)

#print("### Heuristic model")

#EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

############################
import LogisticRegressionModel_NumPy
model = LogisticRegressionModel_NumPy.LogisticRegressionModel_NumPy()

print("### Logistic regression model")


for i in [50000]:

    thresholdval = 0.12135
    for j in range(101):
        thresholdval += 0.000005
        xTrain_np = np.asarray(xTrain)
        yTrain_np = np.asarray(yTrain)
        xTest_np = np.asarray(xTest)
        yTest_np = np.asarray(yTest)
        model.fit(xTrain_np, yTrain_np, iterations=i, step=0.01)
        yTestPredicted = model.predict(xTest_np, thresholdval)

        fpr = EvaluationsStub.FalsePositiveRate(yTest_np, yTestPredicted)

        print(thresholdval, fpr, EvaluationsStub.FalseNegativeRate(yTest_np, yTestPredicted))

        if fpr <= 0.101 and fpr >= 0.095:
            break

        #yTestPredProb = model.calculate_probabilities(xTest_np)

        #yPredDiff = EvaluationsStub.PredictionDiff(xTestRaw, yTest, yTestPredProb)

    #sorted_x = sorted(yPredDiff.items(), key=operator.itemgetter(1))

    #sorted_x = sorted(yPredDiff.items(), key=lambda kv: kv[1])

    #for i in range(20):
     #   print(sorted_x[i])

    #print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(xTest_np, yTest_np), EvaluationsStub.Accuracy(yTest_np, yTestPredicted)))

#EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

#print([k for (k, v) in collections.Counter(BagOfWords.FeaturizeBagOfWords(xTrainRaw)).most_common(10)])
#print([k for (k, v) in collections.Counter(BagOfWords.MutualInformation(xTrainRaw, yTrainRaw)).most_common(100)])

#print(collections.Counter(BagOfWords.MutualInformation(xTrainRaw, yTrainRaw)).most_common(100))
