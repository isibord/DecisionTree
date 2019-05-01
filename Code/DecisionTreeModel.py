import math
import collections
import operator

class TreeNode(object):
    def __init__(self, isLeaf, valueOrIdx, leftChild, rightChild, countWithLabel0, countWithLabel1, midVal):
        self.isLeaf = isLeaf
        self.valueOrIdx = valueOrIdx
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.attrSplitValue = midVal 
        self.countWithLabel0 = countWithLabel0
        self.countWithLabel1 = countWithLabel1


class DecisionTreeModel(object):
    """A decision tree spam filter"""


    def __init__(self):
        pass
        
    def fit(self, x, y, minToSplit=100, threshold = 0.5):
        self.threshold = threshold
        self.featuresLeft = list(range(len(x[0])))
        self.minToSplit = minToSplit
        self.Tree = self.growTree(x, y)

    def growTree(self, x, y):        
        if all(yi == 0 for yi in y):
            return TreeNode(True, 0, None, None, len(y), 0, None)
        elif all(yi == 1 for yi in y):
            return TreeNode(True, 1, None, None, 0, len(y), None)
        elif len(y) < self.minToSplit:
            (mostcommon, count0, count1) = self.countMostCommon0and1(y)
            return TreeNode(True, mostcommon, None, None, count0, count1, None)  # Majority class
        else:
            bestAttr = self.bestSplitAttr(x, y)

            if bestAttr is None:
                (mostcommon, count0, count1) = self.countMostCommon0and1(y)
                return TreeNode(True, mostcommon, None, None, count0, count1, None)  # Majority class
            #else:
            #    if bestAttr in self.featuresLeft:
            #        self.featuresLeft.remove(bestAttr)

            datasplit, midVal = self.splitByFeatureValues(x, y, bestAttr)
            (leftSplitx, leftSplity) = datasplit[0]
            (rightSplitx, rightSplity) = datasplit[1]

            return TreeNode(False, bestAttr, self.growTree(leftSplitx, leftSplity), self.growTree(rightSplitx, rightSplity), None, None, midVal)


    def predict(self, x):
        yPrediction = []
        for currval in x:
            resultval = self.test_example(currval, self.Tree)
            yPrediction.append(resultval)
        return yPrediction

    def test_example(self, example, node):
        if node.isLeaf:
            #return node.valueOrIdx
            return 1 if ((node.countWithLabel1 + 1) / (node.countWithLabel0 + node.countWithLabel1 + 2)) >= self.threshold else 0
        else:
            if example[node.valueOrIdx] < node.attrSplitValue:
                return self.test_example(example, node.leftChild)
            else:
                return self.test_example(example, node.rightChild)
        
    def bestSplitAttr(self, x, y):
        informationGains = {}

        for i in self.featuresLeft:
            informationGains[i] = self.informationGain(x, y, i)

        if self.allEqualZero(informationGains):
            return None
        
        return self.findIndexWithHighestVaue(informationGains)

    def splitByFeatureValues(self, x, y, bestAttr):
        allValuesForAttr = [ x[i][bestAttr] for i in range(len(x)) ] 
        
        midVal = (min(allValuesForAttr) + max(allValuesForAttr)) / 2.0

        leftSplitx = [] #less than midval
        leftSplity = []
        rightSplitx = [] #greater than or equals midval
        rightSplity = []

        for i in range(len(x)):
            if x[i][bestAttr] < midVal:
                leftSplitx.append(x[i])
                leftSplity.append(y[i])
            else:
                rightSplitx.append(x[i])
                rightSplity.append(y[i])

        return [(leftSplitx, leftSplity), (rightSplitx, rightSplity)], midVal


    def loss(self, x, y, i):
        lossSum = 0
        splitResult, midVal = self.splitByFeatureValues(x, y, i)
        for (xval, yval) in splitResult:
            lossSum += self.entropy(xval, yval) * len(xval)
        return lossSum / len(x)

    def informationGain(self, x, y, i):
        return self.entropy(x, y) - self.loss(x, y, i)

    def entropy(self, x, y):
        if len(y) == 0:
            return 0

        entropyval = 0

        county = collections.Counter(y)

        py0 = county[0] / len(y)
        py1 = county[1] / len(y)

        #py0 = (county[0] + 1) / (len(y) + 2)
        #py1 = (county[1] + 1) / (len(y) + 2)

        if py0 != 0:
            entropyval += (py0 * math.log2(py0))
        if py1 != 0:
            entropyval += (py1 * math.log2(py1))

        return -entropyval

    def allEqualZero(self, infoGains):
        for key in infoGains.keys():
            #if infoGains[key] != 0:
            if infoGains[key] > 0.0001:
                return False
        return True

    def findIndexWithHighestVaue(self, infoGains):
        return max(infoGains.items(), key=operator.itemgetter(1))[0]

    def countMostCommon0and1(self, y):
        count = collections.Counter()

        for label in y:
            count[label] += 1

        return (count.most_common(1)[0][0], count[0], count[1])  
   
    def visualize(self):
        self.printTree(self.Tree, 0)

    def printTree(self, node, level):
        blanks = ""
        for i in range(level):
            blanks += "\t"
        if node.isLeaf:
            print(blanks, "Leaf: ", node.countWithLabel1, " with label 1, ", node.countWithLabel0, " with label 0")
        else:
            print(blanks, "Feature ", node.valueOrIdx,": (mid: ", node.attrSplitValue, ")")
            print(blanks, "\t", ">= ", self.threshold)
            self.printTree(node.rightChild, level + 1)
            print(blanks, "\t", "< ", self.threshold)
            self.printTree(node.leftChild, level + 1)
