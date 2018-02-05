import numpy as np
from  functools import reduce

def loadDataSet():  #创建样例数据
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1] #1代表坏话
    return postingList, classVec

def createVocabList(dataSet):  #创建词库 这里就是直接把所有词去重后，当作词库
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet): #文本词向量，词库中每个词当作一个特征，文本中有该词，该词的特征是1，没有是0
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!', word)
    print('returnVec=', returnVec)
    return returnVec

def typeProportion(listClass):

    classLenght = len(listClass)
    one_number = listClass.count(1)
    two_number = listClass.count(0)

    one_value = one_number/classLenght #坏话的概率
    two_value = two_number/classLenght #好话的概率

    return one_value, two_value

def conditionProportion(trainMat, listClass): #求每个特征坏话／好话的条件概率

    trainMat = np.array(trainMat)
    listClass = np.array(listClass)

    one_index = np.where(listClass == 1) #类别为1的索引
    two_index = np.where(listClass == 0) #类别为0的索引

    one_class = np.array(trainMat[one_index]) #所有坏话的样本
    two_class = np.array(trainMat[two_index]) #所有好话的样本

    #坏话中单个词的概率，即坏话某个词的条件概率
    one_sum = np.sum(one_class ,axis=0)  #所有坏话样本中，每个特征为1时，各特征的总个数
    one_Proportion = one_sum / one_class.shape[0]  # 每个特征各自总个数／坏话样本总数
    one_Proportion[one_Proportion == 0] = 1   #求得的条件概率为0时，设置为1，这样在多个特征的概率相乘时，不会使结果为0

    #好话中单个词的概率，即好话某个词的条件概率
    two_sum = np.sum(two_class, axis= 0)      #所有好话样本中，每个特征为1时，各特征的总个数
    two_Proportion = two_sum / two_class.shape[0]   # 每个特征各自总个数／好话样本总数
    two_Proportion[two_Proportion == 0] = 1

    #print('one_Proportion=', one_Proportion, 'two_Proportion=', two_Proportion)

    return one_Proportion, two_Proportion

def featureProportion(trainMat, listClass): #求每个特征占总样本数的概率

    #求每个特征为1时，在总样本数中的概率
    classSum = len(listClass)
    trainMat_sum = np.sum(trainMat, axis= 0) #矩阵的列求和运算，每个特征为1时的总数
    trainMat_Proportion = trainMat_sum / classSum
    #print('trainMat_Proportion=', trainMat_Proportion)

    return trainMat_Proportion

def modelWork(testArray, myVocabList, one_Proportion, two_Proportion, trainMat_Proportion, one_value, two_value):

    # myVocabList:词库，testArray：测试数据
    # one_Proportion:特征的坏话条件概率，two_Proportion特征坏话的条件概率， trainMat_Proportion：特征的概率
    # one_value:坏话的概率   two_value：好话的概率

    indexeList = list()
    for data in testArray:
        indexeList.append(myVocabList.index(data))
    print('indexeList=', indexeList)

    dataOne_Proportion = one_Proportion[indexeList]
    dataTwo_Proportion = two_Proportion[indexeList]
    dataTrainMat_Proportion = trainMat_Proportion[indexeList]

    print('one_Proportion=', one_Proportion)
    print('two_Proportion=', two_Proportion)
    print('trainMat_Proportion=', trainMat_Proportion)

    def prod(L):
        return reduce(lambda x, y: x*y, L)
    dataOne = prod(dataOne_Proportion)
    dataTwo = prod(dataTwo_Proportion)
    dataTrainMat = prod(dataTrainMat_Proportion)

    print('dataOne=', dataOne, 'dataTwo=', dataTwo, 'dataTrainMat=', dataTrainMat)

    # 求测试数据假设为坏话时的概率
    oneProportion = (dataOne * one_value)/dataTrainMat
    print('oneProportion=', oneProportion)

    # 求测试数据假设为好话时的概率
    twoProportion = (dataTwo * two_value)/dataTrainMat
    print('twoProportion=', twoProportion)

    # 比较测试数据，好话与坏话的概率值，返回判断结果 result: 0 无法判断， 1 坏话， 2 好话
    result = 2
    if oneProportion > twoProportion:
        result = 0
    elif oneProportion < twoProportion:
        result = 1
    else:
        result = 2
    return result

def workSpace():
    listPosts, listClass = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = list()
    for postClass in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postClass))

    one_value, two_value = typeProportion(listClass)
    one_Proportion, two_Proportion = conditionProportion(trainMat, listClass)
    trainMat_Proportion = featureProportion(trainMat, listClass)

    #print('myVocabList=', myVocabList)

    #训练所有现有数据，看准确率

    trainResult = list()
    for oneData in listPosts:
        #print('oneData=', oneData)
        oneClass = modelWork(oneData, myVocabList, one_Proportion, two_Proportion, trainMat_Proportion, one_value, two_value)
        trainResult.append(oneClass)

    print('trainResult=', trainResult)

    '''
    #创建一条新数据，检验下模型的输出
    testData = ['I', 'love', 'my', 'food']
    testData = setOfWords2Vec(myVocabList, testData)
    testClass = modelWork(testData, myVocabList, one_Proportion, two_Proportion, trainMat_Proportion, one_value,
                         two_value)
    print('testClass=', testClass)
    '''

    return

workSpace()
