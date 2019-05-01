import Assignment1Support
import EvaluationsStub
import collections
import math
import numpy as np

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

############################
import DecisionTreeModel
model = DecisionTreeModel.DecisionTreeModel()

print("### Decision Tree model")

numFolds = 5

avgAccr = 0
for i in range(numFolds):
    (foldTrainX, foldTrainY)  = Assignment1Support.GetAllDataExceptFold(xTrain, yTrain, i, numFolds)
    (foldValidationX, foldValidationY) = Assignment1Support.GetDataInFold(xTrain, yTrain, i, numFolds)

    # do feature engineering/selection on foldTrainX, foldTrainY
    
    xTrain_np = np.asarray(foldTrainX)
    yTrain_np = np.asarray(foldTrainY)
    xTest_np = np.asarray(foldValidationX)
    yTest_np = np.asarray(foldValidationY)

    model.fit(xTrain_np, yTrain_np, minSplit)

    yTestPredicted = model.predict(xTest_np)

    avgAccr += EvaluationsStub.Accuracy(yTest_np, yTestPredicted)
         
 

        #EvaluationsStub.ExecuteAll(yTest_np, yTestPredicted)