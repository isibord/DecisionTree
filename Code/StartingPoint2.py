
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
import DecisionTreeModel
model = DecisionTreeModel.DecisionTreeModel()

print("### Decision Tree model")

model.fit(xTrain, yTrain)
model.visualize()
yTestPredicted = model.predict(xTest)

EvaluationsStub.ExecuteAll(yTest, yTestPredicted)


#thresholdval = -0.01
#for j in range(101):
#    thresholdval += 0.01
#    model.fit(xTrain, yTrain, 100, thresholdval)
#    #model.visualize()
#    yTestPredicted = model.predict(xTest)

#    fpr = EvaluationsStub.FalsePositiveRate(yTest, yTestPredicted)

#    print(thresholdval, fpr, EvaluationsStub.FalseNegativeRate(yTest, yTestPredicted))