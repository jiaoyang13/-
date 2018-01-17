import numpy as np
from scipy.sparse.csr import csr_matrix

# CSR是Compressed Sparse Row的缩写，稀疏矩阵，是一种能够使数据压缩的存储原理
def works():

    oneWork()
    twoWork()

    return

def oneWork():

    indptr = np.array([0, 2, 3, 6])
    #最后一个元素是总共有多少个数据，这里为6，因为data的数据为1到6，6个数字
    #前三个元素，分别为每一行第一个有数据的元素在data数据中的索引，比如3，指的是第3行的第一个数字在data中的index为3，值为4
    indices = np.array([0, 2, 2, 0, 1, 2])
    #元素意义： 每个数在最终生成的数组数据中，位于每一行中的索引值
    data = np.array([1, 2, 3, 4, 5, 6])
    #在矩阵中的所有元素
    data_array = csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()

    print('data_array=', data_array)  # [[1 0 2] [0 0 3] [4 5 6]]

    return

def twoWork():
    docs = [['hello', 'world', 'hello'], ['goodbye', 'cruel', 'world']]
    indptr = [0]  # 存放行偏移量
    indices = []  # 存放的是data中元素对应的列编号（列编号可重复）
    data = []  # 存放非0的数据元素
    vocabulary = {}  # key是word词汇，value是列编号

    for d in docs:  # 遍历每个文档
        for term in d:  # 遍历文档中的每个词汇term
            # setdefault如果term不存在，则将新term和他的列编号len(vocabulary)加入到词典中，返回他的编号
            # 如果term存在，则不添加，返回已存在的编号
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)  #这里把数据都填为了1
        indptr.append(len(indices))
        #遍历结束后：indices= [0, 1, 0, 2, 3, 1]，indptr= [0, 3, 6]，data= [1, 1, 1, 1, 1, 1]
    # csr_matrix可以将同一个词汇次数求和
    data_array = csr_matrix((data, indices, indptr), dtype=int)
    '''
    data_array =
    (0, 0)	1   结构：（所在行， 对应在data中的索引（但这里data都设置了1）） data中对应的值
    (0, 1)	1
    (0, 0)	1
    (1, 2)	1
    (1, 3)	1
    (1, 1)	1
    
    '''

    data_toArray = csr_matrix((data, indices, indptr), dtype=int).toarray()

    #结果输出为：toArray= [[2 1 0 0] [0 1 1 1]]

    '''
    结果说明：因为indices中最大的数为 3，因此结果数组中每一行的元素个数为4个
             indptr中一共有三个元素，最后一个元素表示的是总数据个数，还有两个，因此一共有2行
             到这里推算出，结果是个（2，4）的矩阵
             
             结果中每个元素的说明，没有值的元素补0
             2---hello出现了 2次
             1---world出现了 1次
             0---goodbye出现了 0次
             0---cruel出现了 0次
             
             0---hello出现了 0次
             1---world出现了 1次
             1---goodbye出现了 1次
             1---cruel出现了 1次
    '''

    return

works()
