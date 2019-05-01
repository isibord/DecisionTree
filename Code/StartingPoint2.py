
import Assignment1Support
import EvaluationsStub
import BagOfWords
import AddNoise
import collections
import operator
import numpy as np

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRawOriginal, yTrainRawOriginal, xTestRawOriginal, yTestRawOriginal) = Assignment1Support.TrainTestSplit(xRaw, yRaw)
(xTrainRaw, yTrainRaw) = AddNoise.MakeProblemHarder(xTrainRawOriginal, yTrainRawOriginal)
(xTestRaw, yTestRaw) = AddNoise.MakeProblemHarder(xTestRawOriginal, yTestRawOriginal)

#(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

(xTrain, xTest) = Assignment1Support.FeaturizeExtenstion(xTrainRaw, yTrainRaw, xTestRaw, numFrequentWords=40, numMutualInformationWords=100, includeHandCraftedFeatures=True)

yTrain = yTrainRaw
yTest = yTestRaw

############################
import RandomForestsModel
rfmodel = RandomForestsModel.RandomForestsModel()

print("### BEST SMS model")

import LogisticRegressionModel_NumPy
logmodel = LogisticRegressionModel_NumPy.LogisticRegressionModel_NumPy()

#CROSS VALIDATION
#numFolds = 5
#rfavgAccr = 0
#logavgAccr = 0
#for i in range(numFolds):
#    (foldTrainX, foldTrainY)  = Assignment1Support.GetAllDataExceptFold(xTrain, yTrain, i, numFolds)
#    (foldValidationX, foldValidationY) = Assignment1Support.GetDataInFold(xTrain, yTrain, i, numFolds)

#    # do feature engineering/selection on foldTrainX, foldTrainY
    
#    xTrain_np = np.asarray(foldTrainX)
#    yTrain_np = np.asarray(foldTrainY)
#    xTest_np = np.asarray(foldValidationX)
#    yTest_np = np.asarray(foldValidationY)

#    rfmodel.fit(xTrain_np, yTrain_np, numTrees=2, minSplit=100, useBagging=False, featureRestriction=0, seed=300)
#    rfyPredicted = rfmodel.predictThres(xTest_np, 0.5)

#    logmodel.fit(xTrain_np, yTrain_np, iterations=30000, step=0.01)
#    logyPredicted = logmodel.predict(xTest_np, 0.5)

#    rfavgAccr += EvaluationsStub.Accuracy(yTest_np, rfyPredicted)
#    logavgAccr += EvaluationsStub.Accuracy(yTest_np, logyPredicted)

    
#print("RF CV Accuracy: %f" % (rfavgAccr / 5.0))
#print("LOG CV Accuracy: %f" % (logavgAccr / 5.0))


xTrain_np = np.asarray(xTrain)
yTrain_np = np.asarray(yTrain)
xTest_np = np.asarray(xTest)
yTest_np = np.asarray(yTest)

rfmodel.fit(xTrain_np, yTrain_np, numTrees=2, minSplit=100, useBagging=False, featureRestriction=0, seed=200)
logmodel.fit(xTrain_np, yTrain_np, iterations=30000, step=0.01)

#rfyPredicted = rfmodel.predictThres(xTest_np, 0.5)
#logyPredicted = logmodel.predict(xTest_np, 0.5)

#yPredictFinal = []
#for i in range(len(rfyPredicted)):
#    yPredictFinal.append(1 if rfyPredicted[i] == 1 and logyPredicted[i] == 1 else 0 )

    
#print(EvaluationsStub.ExecuteAll(yTest, rfyPredicted))
#print(EvaluationsStub.ExecuteAll(yTest, logyPredicted))
#print(EvaluationsStub.ExecuteAll(yTest, yPredictFinal))

thresholdval = -0.01
for j in range(101):
    thresholdval += 0.01

    rfyPredicted = rfmodel.predictThres(xTest_np, thresholdval)
    logyPredicted = logmodel.predict(xTest_np, thresholdval)

    yPredictFinal = []
    for i in range(len(rfyPredicted)):
        yPredictFinal.append(1 if rfyPredicted[i] == 1 and logyPredicted[i] == 1 else 0 )

    print(thresholdval, EvaluationsStub.FalsePositiveRate(yTest_np, rfyPredicted), EvaluationsStub.FalseNegativeRate(yTest_np, rfyPredicted))

   

