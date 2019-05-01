import collections
import BagOfWords
import re

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


def FeaturizeExtenstion(xTrainRaw, yTrainRaw, xTestRaw, numFrequentWords=0, numMutualInformationWords=295, includeHandCraftedFeatures=True):
    freqWords = []
    miWords = []
    
    if numFrequentWords > 0:
        freqWords = [k for (k, v) in collections.Counter(BagOfWords.FeaturizeBagOfWords(xTrainRaw)).most_common(numFrequentWords)]
        #freqWords = ['i', 'you', 'a', 'u', 'in', 'my', 'your', 'for', 'me', 'of', 'have', 'on', 'it', 'are', 'that']
        #freqWords = ['to', 'i', 'you', 'a', 'the', 'u', 'and', 'is', 'in', 'my', 'your', 'for', 'me', 'of', 'have', 'call', 'on', 'it', 'are', 'that']
        #print(freqWords)
    
    if numMutualInformationWords > 0:
        #miWords = [k for (k, v) in collections.Counter(BagOfWords.MutualInformation(xTrainRaw, yTrainRaw)).most_common(numMutualInformationWords)]
        #print(miWords)
        miWords = ['call', 'txt', 'free', 'claim', 'now!', 'mobile', 'to', 'your', 'or', '&', 'reply', 'our', 'stop', 'text', 'cash', 'won', 'i', 'service', 'from', 'prize', 'tone', 'win', 'contact', 'per', 'nokia', 'urgent!', '-', 'for', 'chat', 'latest', 'box', 'po', 'a', '4*', '16+', 'video', 'ringtone', 'holiday', 'awarded', 'new', 'weekly', 'selected', 'await', 'camera', '2', 'tones', '500', 't&cs', 'customer', 'free!', '4', 'â£1000', '86688', 'bonus', 'landline.', 'â£100', '150p', 'send', 'ur', '18', 'dating', '10p', 'only', 'collection.', 'attempt', '150ppm', 'receive', 'draw', 'prize.', 'guaranteed', 'â£5000', 'chance', 'â£250', 'code:', 'sae']
        #print(len(miWords))
        
        #miWords = ['call', 'txt', 'free', 'claim', 'now!', 'mobile', 'to', 'your', 'or', '&', 'reply', 'our', 'stop', 'text', 'cash', 'won', 'i', 'service', 'from', 'prize', 'tone', 'win', 'contact', 'per', 'nokia', 'urgent!', '-', 'for', 'chat', 'latest', 'box', 'po', 'a', '4*', '16+', 'video', 'ringtone', 'holiday', 'awarded', 'new', 'weekly', 'selected', 'await', 'camera', '2', 'tones', '500', 't&cs', 'customer', 'free!', '4', 'â£1000', '86688', 'bonus', 'landline.', 'â£100', '150p', 'send', 'ur', '18', 'dating', '10p', 'only', 'collection.', 'attempt', '150ppm', 'receive', 'draw', 'prize.', 'guaranteed', 'â£5000', 'chance', 'â£250', 'code:', 'sae', 'pobox', 'vouchers', '8007', 'mob', 'orange', '16', '1st', 'cost', 'pounds', 'expires', '18+', '08000839402', 'delivery', 'live', 'â£2000', 'colour', 'entry', 'on', 'now', 'valid', 'â£2,000', 'national', 'numbers', 'complimentary', 't&c']
        #miWords = ['Call', 'FREE', 'or', 'call', 'mobile', 'claim', 'to', '&', 'Txt', 'To', 'now!', 'Your', 'STOP', 'txt', 'Text', 'service', 'our', 'from', 'per', 'contact', 'Nokia', 'cash', '-', 'your', 'a', 'PO', 'won', 'prize', 'reply', 'I', 'Holiday', '4*', '16+', 'Box', 'latest', 'Reply', 'Free', 'awarded', 'selected', 'await', 'tone', 'for', 'Mobile', 'URGENT!', '2', 'ringtone', '500', 'T&Cs', 'video', 'NOW!', 'win', 'weekly', '150p', 'camera', '4', 'Â£1000', 'Get', '86688', 'This', 'Â£100', 'receive', '18', 'YOU!', '10p', 'landline.', 'collection.', 'customer', 'text', 'attempt', 'Claim', 'guaranteed', 'Â£5000', 'Â£250', 'dating', 'SAE', 'CHAT', 'entry', '8007', '150ppm', 'prize.', '16', '1st', 'new', 'tones', 'Code:', 'TONE', 'WIN', 'pounds', 'Expires', '18+', 'Orange', '08000839402', 'Â£2000', 'WON', 'ur', 'NOW', 'Â£2,000', 'rate', 'numbers', 'Urgent!', 'complimentary', 'draw', 'me', 'every', 'line', 'Â£500', 'No:', 'Bonus', 'NOKIA', 'PRIVATE!', 'Statement', 'Identifier', 'FREE!', 'Cost', 'Â£1.50', 'colour', 'chance', 'shows', 'on', 'i', 'Chat', 'NTT', 'T&C', 'land', 'free', 'vouchers', 'Account', 'un-redeemed', 'mob', '750', 'MobileUpd8', 'stop', 'Please', 'You', 'Valid', 'send', 'Had', 'network', '1', '2nd', 'GUARANTEED.', 'line.', 'LIVE', '1327', '5WB', 'www.getzed.co.uk', '08712300220', 'Costa', 'Del', 'YES', 'charged', 'delivery', 'easy,', 'Prize', 'Todays', '87066', '800', 'operator', 'collection', 'phones', 'week!', 'Update', 'Only', 'been', 'chat', 'tried', 'Send', 'CALL', 'For', '12hrs', 'live', 'custcare', 'No1', '20p', 'Ltd,', 'Croydon', 'CR9', '0870', 'national', '86021', 'S.', 'I.', 'eg', 'ending', 'Â£350', 'Choose', 'BT-national-rate', '09050090044', 'toClaim.', 'SAE,', 'POBox334,', 'SK38xh,', 'CostÂ£1.50/pm,', 'Max10mins', 'POBox', 'entitled', 'yr', 'points.', 'arrive', '2003', 'award.', '0800', 'mobile!', 'rates', 'Mins', 'apply.', 'my', 'number', "I'm", 'have', 'only', 'FOR', 'Our', '08000930705', 'today!', 'Sol', 'worth', 'wk', 'pics', 'apply', 'price', 'unsubscribe', 'Double', 'valued', 'about', '40GB', 'iPod', 'replying', '08707509020', 'M.', 'matches', 'UK', 'Press', 'FREEPHONE', 'mobiles', 'standard', 'Now!', 'onto', 'Calls', 'only!', 'winner', 'specially', '36504', 'TC', 'order,', 'contacted', 'Tones', 'Gr8', 'POBOX', 'BOX', 'Welcome', 'inviting', 'STOP?', 'FRND', '62468', 'week', 'final', 'Congratulations', 'voucher', 'prize!', 'HG/Suite342/2Lands', 'CASH', 'code', 'representative', 'Stockport,', 'Caller', 'txting', 'each', 'Latest', 'Half', 'anytime', 'collect', 'with', 'gift', 'NEW', 'UR', 'C', 'currently', 'services', '100', 'now', '&lt;#&gt;', 'SMS', 'Â£800', 'More', 'STOP.', 'offers', 'Xmas', 'ON', '3', 'in', 'MUSIC', 'word:', '3030.', 'award']

    xTrain = FeaturizeSingle(xTrainRaw, freqWords, miWords, includeHandCraftedFeatures)
    xTest = FeaturizeSingle(xTestRaw, freqWords, miWords, includeHandCraftedFeatures)

    return (xTrain, xTest)

def FeaturizeSingle(xRaw, freqWords, miWords, includeHandCraftedFeatures):
    origwords = ['call', 'to', 'your'] #original
    xFeaturized = []
    for x in xRaw:
        x = x.lower()
        features = []

        if includeHandCraftedFeatures:

            # Have a feature for the length
            #features.append(len(x))

            #features.append(len(re.findall('([A-Z])', x)))

            if '!' in x:
                features.append(1)
            else:
                features.append(0)

            if '$' in x or '£' in x:
                features.append(1)
            else:
                features.append(0)

            if any(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in x):
                features.append(1)
            else:
                features.append(0)


            # Have a feature for longer texts
            if(len(x)>40):
                features.append(1)
            else:
                features.append(0)

            # Have a feature for texts with numbers in them
            if(any(i.isdigit() for i in x)):
                features.append(1)
            else:
                features.append(0)

            # Have features for a few words
            for word in origwords:
                if word in x.split():
                    features.append(1)
                else:
                    features.append(0)

        for word in freqWords:
            if word in x.split():
                features.append(1)
            else:
                features.append(0)
            
        for word in miWords:
            if word in x.split():
                features.append(1)
            else:
                features.append(0)
                          
        xFeaturized.append(features)

    return xFeaturized


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