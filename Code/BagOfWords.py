import collections
import math

def FeaturizeBagOfWords(xTrainRaw):
    wordFreq = {}
    for x in xTrainRaw:
        for word in x.split():
            if word in wordFreq:
                wordFreq[word] += 1
            else:
                wordFreq[word] = 1

    return wordFreq

def MutualInformation(xTrainRaw, yTrainRaw):

    bagOfWords = FeaturizeBagOfWords(xTrainRaw)
    distinctWords = bagOfWords.keys()
    lenSamples = len(yTrainRaw)
    
    wordWithCounts = {key: [0, 0, 0, 0] for key in distinctWords} #add count when matches for [{X:1, Y:0}, {X:1, Y:1}, {X:0, Y:0}, {X:0, Y:1}]
    wordWithMI = {key: 0.0 for key in distinctWords}

    for i in range(lenSamples):
        for word in distinctWords:
            probs = []
            if word in xTrainRaw[i].split():
                if yTrainRaw[i] == 0:
                    wordWithCounts[word][0] += 1
                else:
                    wordWithCounts[word][1] += 1
            else:
                if yTrainRaw[i] == 0:
                    wordWithCounts[word][2] += 1
                else:
                    wordWithCounts[word][3] += 1



    for word in distinctWords:
        p_x0 = (wordWithCounts[word][2] + wordWithCounts[word][3] + 1) / (lenSamples + 2)
        p_x1 = (wordWithCounts[word][0] + wordWithCounts[word][1] + 1) / (lenSamples + 2)
        
        p_y0 = (wordWithCounts[word][0] + wordWithCounts[word][2] + 1) / (lenSamples + 2)
        p_y1 = (wordWithCounts[word][1] + wordWithCounts[word][3] + 1) / (lenSamples + 2)
        
        p_x1y0 = (wordWithCounts[word][0] + 1) / (lenSamples + 2)
        p_x1y1 = (wordWithCounts[word][1] + 1) / (lenSamples + 2)
        p_x0y0 = (wordWithCounts[word][2] + 1) / (lenSamples + 2)
        p_x0y1 = (wordWithCounts[word][3] + 1) / (lenSamples + 2)

        mi_x0y0 = p_x0y0 * math.log2(p_x0y0 / (p_x0 * p_y0))
        mi_x0y1 = p_x0y1 * math.log2(p_x0y1 / (p_x0 * p_y1))
        mi_x1y0 = p_x1y0 * math.log2(p_x1y0 / (p_x1 * p_y0))
        mi_x1y1 = p_x1y1 * math.log2(p_x1y1 / (p_x1 * p_y1))

        wordWithMI[word] = mi_x0y0 + mi_x0y1 + mi_x1y0 + mi_x1y1

    return wordWithMI

