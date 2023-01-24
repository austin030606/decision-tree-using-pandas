import pandas as pd
import math

""" 
This function is used for getting the potential splitting point of continuous features.
"""
def getContFeatures(df: pd.DataFrame) -> set:
    res = set()
    mylist = []
    for d in df.columns:
        if d != 'class':
            #print("column:", d)
            tmplist = []
            tmpdf = df[[d, 'class']]
            tmpdf = tmpdf.sort_values(by=d)
            i = 0
            prev = (0, 0)
            cur = (0, 0)
            for index, row in tmpdf.iterrows():
                cur = (row[d], row['class'])
                if i != 0 and cur[1] != prev[1] and cur[0] != prev[0]:
                    res.add((d, (cur[0] + prev[0]) / 2))
                    tmplist.append((cur[0] + prev[0]) / 2)
                prev = cur
                i += 1
            mylist.append(tmplist)
    return res, mylist


""" 
This function is used for getting a set of categorical features.
"""
def getFeatures(df: pd.DataFrame) -> set:
    res = set()
    mylist = []
    for d in df.columns:
        if d != 'class':
            res.add((d, 'categorical'))
    return res, mylist

"""
Features are in the form of a tuple
where for continuous features, it's ([feature_name], [split_value])
and for categorical features, it's ([feature_name], 'categorical')
"""

def Hi():
    print("hello")

class Node:
    def __init__(self, D: pd.DataFrame, d: set, depth):
        self.D = D.copy() # training instances (dataframe)
        self.d = set() # current set of usable features for splitting
        self.d.update(d)
        self.depth = depth 
        self.isLeaf = False
        self.lable = '' 
        self.dBest = ('', 0) # best feature for splitting
        self.children = []

    """ 
    This method finds the best feature for splitting, and it's based on Information Gain.
    """
    def finddBest(self):
        maxIG = 0
        bestFeature = (0, 0)
        #calculating H(t, D)
        cntOfTarget = self.D['class'].value_counts().to_dict()
        totalCnt = self.D.shape[0]
        originalEntropy = 0
        for target in cntOfTarget:
            originalEntropy -= (cntOfTarget[target] / totalCnt) * math.log2(cntOfTarget[target] / totalCnt)

        #calculating rem and information gain
        cnt = 0
        for feature in self.d:
            curRem = 0
            if type(feature[1]) != str:
                childD1 = self.D[self.D[feature[0]] < feature[1]]
                childD2 = self.D[self.D[feature[0]] >= feature[1]]
                
                cntOfTarget = childD1['class'].value_counts().to_dict()
                child1TotalCnt = childD1.shape[0]
                child1Entropy = 0
                for target in cntOfTarget:
                    child1Entropy -= (cntOfTarget[target] / child1TotalCnt) * math.log2(cntOfTarget[target] / child1TotalCnt)
                curRem += (child1TotalCnt / totalCnt) * child1Entropy

                cntOfTarget = childD2['class'].value_counts().to_dict()
                child2TotalCnt = childD2.shape[0]
                child2Entropy = 0
                for target in cntOfTarget:
                    child2Entropy -= (cntOfTarget[target] / child2TotalCnt) * math.log2(cntOfTarget[target] / child2TotalCnt)
                curRem += (child2TotalCnt / totalCnt) * child2Entropy
            else:
                #print('here1', cnt)
                cnt += 1
                tmpChildren = []
                unique = self.D[feature[0]].unique()
                #print('length is', len(unique))
                for value in unique:
                    tmpChild = self.D[self.D[feature[0]] == value]
                    tmpChildren.append(tmpChild)
                
                for child in tmpChildren:
                    #print('here2')
                    cntOfTarget = child['class'].value_counts().to_dict()
                    childTotalCnt = child.shape[0]
                    childEntropy = 0
                    for target in cntOfTarget:
                        childEntropy -= (cntOfTarget[target] / childTotalCnt) * math.log2(cntOfTarget[target] / childTotalCnt)
                    curRem += (childTotalCnt / totalCnt) * childEntropy
            curIG = originalEntropy - curRem
            if curIG > maxIG:
                maxIG = curIG
                bestFeature = feature
        self.dBest = bestFeature
        #print(self.dBest)

    """
    An implementation of the ID3 algorithm with depth checking and minimum node count checking.
    (current feature set size and dataframe row count is printed to give a hint on the progress)
    """
    def expand(self, stack: list, parentMajorityLable: str, maxDepth, minCount):
        print(len(self.d), self.D.shape[0])
        if self.D['class'].nunique() == 1:
            self.isLeaf = True
            self.lable = self.D['class'].iloc[0]
            return
        elif len(self.d) == 0 or self.depth >= maxDepth or self.D.shape[0] <= minCount:
            self.isLeaf = True
            self.lable = self.D['class'].mode().to_numpy()[0]
            return
        elif self.D.shape[0] == 0:
            self.isLeaf = True
            self.lable = parentMajorityLable
            return
        else:
            self.finddBest()
        if self.dBest == (0, 0):
            self.dBest = list(self.d).pop()
            self.isLeaf = True
            self.lable = self.D['class'].mode().to_numpy()[0]
            return
        if self.dBest[1] != 'categorical':
            self.d.remove(self.dBest)
            childD1 = self.D[self.D[self.dBest[0]] < self.dBest[1]]
            childD2 = self.D[self.D[self.dBest[0]] >= self.dBest[1]]
            child1 = Node(childD1, self.d, self.depth + 1)
            child2 = Node(childD2, self.d, self.depth + 1)
            self.children.append(child1)
            self.children.append(child2)
        else:
            self.d.remove(self.dBest)

            unique = self.D[self.dBest[0]].unique()
            accumulate = []
            for value in unique:
                childD = self.D[self.D[self.dBest[0]] == value]
                if childD.shape[0] < minCount:
                    accumulate.append(childD)
                else:
                    child = Node(childD, self.d, self.depth + 1)
                    self.children.append(child)
            if len(accumulate) > 0:
                accDf = accumulate[0]
                for i in range(len(accumulate)):
                    if i != 0:
                        accDf.append(accumulate[i])
                child = Node(accDf, self.d, self.depth + 1)
                self.children.append(child)

"""
The tree I built works with dataframes so it should be initialized with it.
Note that the column name of the target feature must be "class"
"""
class Tree:
    def __init__(self, d: set, dataset: pd.DataFrame):
        self.dataset = dataset.copy()
        self.root = Node(dataset, d, 0)
        self.d = d
        self.cnt = 0
    
    """
    Building the tree.
    """
    def build(self, maxDepth=10, minCount=1):
        cnt = 0
        stack = []
        stack.append(self.root)
        while len(stack) != 0:
            #print(cnt)
            cnt += 1
            curNode = stack.pop()
            curNode.expand(stack, str(curNode.D['class'].mode().to_numpy()[0]), maxDepth, minCount)
            for node in curNode.children:
                stack.append(node)

    """
    Make prediction on a single query.
    """
    def predictRow(self, row) -> str:
        print(self.cnt)
        self.cnt += 1
        queue = [self.root]
        while len(queue) != 0:
            curNode = queue.pop()
            if curNode.isLeaf:
                return curNode.lable
            else:
                if curNode.dBest[1] != 'categorical':
                    if row[curNode.dBest[0]] < curNode.dBest[1]:
                        queue.append(curNode.children[0])
                    else:
                        queue.append(curNode.children[1])
                else:
                    for child in curNode.children:
                        if row[curNode.dBest[0]] == child.D[curNode.dBest[0]].iloc[0]:
                            queue.append(child)
                            break
        return '0'
        

    """
    Make predictions.
    """
    def predict(self, query: pd.DataFrame) -> pd.DataFrame:
        self.cnt = 0
        qcopy = query.copy()
        res = query.copy()
        res['class'] = qcopy.apply(lambda row: self.predictRow(row), axis=1)
        return res