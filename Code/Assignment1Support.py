import collections

def LoadRawData(path):
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 5574

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:])
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)

def TrainTestSplit(x, y, percentTest = .25):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xTrain = x[numTest:]
    yTest = y[:numTest]
    yTrain = y[numTest:]

    return (xTrain, yTrain, xTest, yTest)

def Featurize(xTrainRaw, xTestRaw):
    words = ['call', 'to', 'your'] #original
    #words = ['to','you','I','a','the','and','is','in','i','u'] #frequency
    #words = ['Call','call','to','or','FREE','claim','To','mobile','Txt','&'] #mutual information
    #words = ['Call','call','to','or','FREE','claim','To','mobile','Txt','&', 'call', 'to', 'your']
    #words = ['Call', 'call', 'to', 'or', 'FREE', 'claim', 'To', 'mobile', 'Txt', '&', 'Your', 'I', 'now!', 'txt', 'a', 'won', 'contact', 'prize', 'STOP', 'Nokia', 'reply', 'our', 'from', 'your', 'Text', 'service', 'per', 'cash', 'i', 'awarded', 'Reply', 'URGENT!', '-', 'Free', 'PO', 'Â£1000', 'Claim', '2', 'Box', 'draw', 'Mobile', 'shows', 'text', 'win', '150ppm', 'Holiday', '4*', 'free', 'Â£100', '16+', 'selected', 'latest', 'This', 'my', 'Get', 'weekly', 'guaranteed', 'Â£2000', '150p', 'Â£5000', 'tone', 'customer', 'receive', 'attempt', 'Valid', 'prize.', 'await', 'for', 'ringtone', '500', 'T&Cs', 'land', 'WON', 'NOW!', 'landline.', 'collection.', '4', 'new', 'stop', 'NOW', '86688', '18', 'entry', 'GUARANTEED.', 'line.', '8007', 'Â£1.50', '16', 'network', 'have', '1st', 'Expires', '12hrs', 'Todays', '18+', '750', '10p', 'WIN', 'Orange', 'CALL']

    # featurize the training data, may want to do multiple passes to count things.
    xTrain = []
    for x in xTrainRaw:
        features = []

        # Have a feature for longer texts
        #if(len(x)>40):
        #    features.append(1)
        #else:
        #    features.append(0)

        # Have a feature for the length
        features.append(len(x))

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        for word in words:
            if word in x:
                features.append(1)
            else:
                features.append(0)

        xTrain.append(features)

    # now featurize test using any features discovered on the training set. Don't use the test set to influence which features to use.
    xTest = []
    for x in xTestRaw:
        features = []
       
        # Have a feature for longer texts
        #if(len(x)>40):
        #    features.append(1)
        #else:
        #    features.append(0)

        # Have a feature for the length
        features.append(len(x))

        # Have a feature for texts with numbers in them
        if(any(i.isdigit() for i in x)):
            features.append(1)
        else:
            features.append(0)

        # Have features for a few words
        for word in words:
            if word in x:
                features.append(1)
            else:
                features.append(0)

        xTest.append(features)

    return (xTrain, xTest)

def InspectFeatures(xRaw, x):
    for i in range(len(xRaw)):
        print(x[i], xRaw[i])

def GetAllDataExceptFold(xTrain, yTrain, i, numFolds):
    xTrainSplit = partition(xTrain, numFolds)
    yTrainSplit = partition(yTrain, numFolds)

    xTrainExceptFold = []
    yTrainExceptFold = []
    for num in range(numFolds):
        if num != i:
            xTrainExceptFold.extend(xTrainSplit[num])
            yTrainExceptFold.extend(yTrainSplit[num])
   
    return (xTrainExceptFold, yTrainExceptFold)

def GetDataInFold(xTrain, yTrain, i, numFolds):
    return (partition(xTrain, numFolds)[i], partition(yTrain, numFolds)[i])


def partition(seq, chunks):
    """Splits the sequence into equal sized chunks and them as a list"""
    result = []
    for i in range(chunks):
        chunk = []
        for element in seq[i:len(seq):chunks]:
            chunk.append(element)
        result.append(chunk)
    return result