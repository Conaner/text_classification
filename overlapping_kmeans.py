import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import random
import itertools
import linecache
import re
import matplotlib
import matplotlib.pyplot as plt

import datetime

random.seed(int(datetime.datetime.now().timestamp()))


class KmeansClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None):
        """
        加载停用词
        :param stopwords:
        :return:
        """
        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []

    def cut_words(self):
        jieba.add_word("物联网", freq=None, tag=None)
        jieba.add_word("新冠肺炎", freq=None, tag=None)
        jieba.add_word("西游记", freq=None, tag=None)

    def preprocess_data(self, corpus_path):
        """
        文本预处理，每行一个文本
        :param corpus_path:
        :return:
        """
        corpus = []
        self.cut_words()
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(' '.join([word for word in jieba.lcut(line.strip()) if word not in self.stopwords]))
        return corpus

    def get_text_tfidf_matrix(self, corpus):
        """
        获取tfidf矩阵
        :param corpus:
        :return:
        """
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))

        # 获取词袋中所有词语
        words = self.vectorizer.get_feature_names()

        # 获取tfidf矩阵中权重
        weights = tfidf.toarray()
        return weights, words

    # 计算欧几里得距离
    def distance_Euclid(self, A, B):
        return np.sqrt(np.sum(np.power(A - B, 2)))

    # 构建K个聚类中心点（随机质心）
    def make_centers(self, data, k):
        m = np.shape(data)[0]  # 样本个数
        indexs = random.sample(range(m), k)  # 从data中随机取k个样本作为质心
        centers = []  # 存储质心坐标，为了统一数据格式，将值取出至列表在转化为numpy格式
        for i in indexs:
            centers.append(data[i])
        return np.array(centers)

    def average_cluster(self, A, P, centers, weights):
        n, k = np.shape(A)
        clusters = []
        for i in range(k):
            s = []
            for m in range(len(A[:, i])):
                if A[:, i][m] == 1:
                    s.append(m)
            clusters.append(s)
        x_aver = []

    def get_distance(self, weight, centers):
        s = []
        for m in range(np.shape(centers)[0]):
            s.append(self.distance_Euclid(weight, centers[m]))

        return s

    def j_okm(self, A, P, x):
        """
        目标函数，计算点与簇均值之间聚类
        A:元素x的隶属情况
        P：中心点
        x：元素x的向量值
        :return: a分配数组
        """
        sum = 0
        pc = np.zeros(np.shape(P)[1], dtype=float)
        for i in range(len(A)):
            if A[i] == 1:
                sum += 1
                pc += P[i]
        x_aver = pc / sum
        j_okm = self.distance_Euclid(x, x_aver)

        return j_okm

    def J_okm(self, A, P, weights):
        """
        目标函数，计算所有点与其各簇均值之间聚类
        A:元素x的隶属情况
        P：中心点
        weights：tf-idf矩阵
        :return: a分配数组
        """
        J_okm = 0
        for i in range(np.shape(A)[0]):
            J_okm += self.j_okm(A[i], P, weights[i])

        return J_okm

    def Assign(self, weight, centers, k):
        """
        分配函数，为每个元素分配一个隶属矩阵（1 * m）
        weight:tf-idf矩阵中的每行，也即一个元素
        centers：簇的中心点
        :return: a分配数组
        """
        a = np.zeros(np.shape(centers)[0], dtype=int)
        b = np.zeros(np.shape(centers)[0], dtype=int)
        j_okm = np.inf
        itr = 0
        stop = False
        distance = self.get_distance(weight, centers)
        while (itr < k) and (not stop):
            c = distance.index(min(distance))  # 查找下一个近的簇中心
            b[c] = 1
            distance[c] = 1000000000
            temp_jokm = self.j_okm(b, centers, weight)
            if temp_jokm < j_okm:
                j_okm = temp_jokm
                a = b.copy()
                itr += 1
            else:
                stop = True

        return a

    def kmeans(self, corpus_path, n_clusters=5, itr_number=10000):
        """
        KMeans文本聚类
        :param corpus_path: 语料路径（每行一篇）,文章id从0开始
        :param n_clusters: ：聚类类别数目
        :return:
        """
        corpus = self.preprocess_data(corpus_path)
        weights, words = self.get_text_tfidf_matrix(corpus)
        m = np.shape(weights)[0]
        A_weight = np.zeros((m, n_clusters), dtype=float)  # 初始化隶属度矩阵
        B_weight = np.zeros((m, n_clusters), dtype=float)  # 初始化隶属度矩阵
        centers = self.make_centers(weights, n_clusters)  # 初始化簇中心
        # print(np.shape(centers))
        itrMAX = itr_number  # 最大迭代次数
        converge = 0  # 收敛标志
        itr = 0  # 迭代标志
        j_okm = np.inf  # 设置初始簇距离

        while (itr < itrMAX) and (not converge):
            # 为每个样本分配簇中心
            for i in range(np.shape(weights)[0]):
                a = self.Assign(weights[i], centers, n_clusters)
                B_weight[i] = a

            # 更新簇中心
            for i in range(n_clusters):
                sum = 0
                x = np.zeros(np.shape(weights)[1], dtype=float)
                for m in range(len(B_weight[:, i])):
                    if B_weight[:, i][m] == 1:
                        sum += 1
                        x += weights[m]
                centers[i] = x / sum

            temp_Jokm = self.J_okm(B_weight, centers, weights)
            if temp_Jokm < j_okm:
                A_weight = B_weight.copy()
                j_okm = temp_Jokm
            else:
                converge = 1

            itr += 1
            # print(itr)

        # print(centers)
        # print(A_weight)

        return centers, A_weight, words, weights


def set_cluster(clusters_number, itr):
    #print('参数为：',datetime.datetime.now())
    Kmeans = KmeansClustering(stopwords_path='./stopwords/stopwords.txt')
    centers, A_weight, words, weights = Kmeans.kmeans('./sum.txt', n_clusters=clusters_number, itr_number=itr)
    pca = PCA(n_components=2)
    pca.fit(weights)
    change_x = pca.transform(weights)
    txt_number = np.shape(weights)[0]
    c_number = []
    for i in range(np.shape(A_weight)[1]):
        temp = 0
        for m in range(len(A_weight[:, i])):
            if A_weight[:, i][m] == 1:
                temp += 1
        c_number.append(temp)

    return change_x, txt_number, c_number, centers, words, A_weight, weights


"""
获得一个聚类簇的主题
:param center: 聚类中心（1个）
:param words: ：词袋
:return:该聚类中心文本向量各词权重，本聚类中心的主题（排序前五的）
"""


def get_Center_topics(center, words):
    dic = dict(map(lambda x, y: [x, y], words, center))
    s = []
    for m in range(5):
        ss = ''
        for key, value in dic.items():
            if (value == max(dic.values())):
                s.append(key)
                ss = key
        del dic[ss]

    return dic, s


"""
中间变量获取
:param A_weight: 隶属矩阵
:param centers: ：所有聚类中心
:param words: ：词袋
:return:
"""


def get_sumtopics_contents(A_weight, centers, words):
    sum_topics = []
    for i in range(len(centers)):
        dic = dict(map(lambda x, y: [x, y], words, list(centers[i])))
        for key, value in dic.items():
            if (value == max(dic.values())):
                sum_topics.append(key)

    content = []
    for i in range(np.shape(A_weight)[1]):
        s = []
        for m in range(len(A_weight[:, i])):
            if A_weight[:, i][m] == 1:
                s.append(m)
        content.append(s)

    return sum_topics, content


"""
获取该聚类簇下文本
:param sum_topics:
:param content: ：
:param topic: ：想要查询的簇的主题
:return:
"""


def get_text(sum_topics, content, topic):
    n = sum_topics.index(topic)
    f = './data/sum.txt'
    context = []
    for i in range(len(content[n])):
        current_context = linecache.getline(f, content[n][i]).strip()
        context.append(current_context)

    return context


"""
对指定类簇进行分析
:param sum_topics:
:param content: ：
:param topic: ：想要查询的簇的主题
:return:
"""


def overlappingShow(kind, weights, content):
    Kmeans = KmeansClustering(stopwords_path='./stopwords/stopwords.txt')
    length = len(kind)
    distribute = [[] for i in range(length)]  # 存放重叠部分聚类后类簇的各个情况
    f = open("./sum.txt", "r", encoding="utf-8")
    f.seek(0)
    lines = f.readlines()
    s = list(itertools.combinations(list(range(len(content))), length))
    key = []  # 存放重叠的中心词
    sum = 0  # 计算总个数
    text = []  # 存放重叠文本
    cutText = []  # 存放重叠切词文本
    vector = []  # 存放重叠文本的向量
    temp0 = []  # 存放重叠部分的编号
    newList = []  # 将s中的元素转换为列表
    dict = {}  # 字典存放词频
    nClustersNum = 0  # 重叠部分聚类个数
    for m in range(len(s)):
        temp = []
        newList.append(list(s[m]))
    kind = sorted(kind)
    index = newList.index(kind)
    for n in s[index]:
        temp.append(content[n])
        sum = sum + len(content[n])
        # content包含了各个类簇的数据点
    temp0.append(list(set(temp[0]).intersection(*temp[1:])))
    if len(temp0[0]) == 0:
        print("无重叠")
        return 0, [0], [0]
    percent = len(temp0[0]) / sum
    print("重叠度为" + str(percent))
    # print(temp0)
    for l in temp0[0]:
        cutText.append(Kmeans.preprocess_data("./data/sum.txt")[l])
        text.append(lines[l].strip())
        vector.append(weights[l])
    print("重叠文本为" + str(text))
    if len(temp0[0]) < length:
        nClustersNum = len(temp0)
    else:
        nClustersNum = length
    model = KMeans(n_clusters=nClustersNum)
    model.fit(vector)
    # 计算每个类别的中心点
    centers = model.cluster_centers_
    # print("centers"+str(centers))
    # 预测值
    result = model.predict(vector)
    # print("resule如下："+str(result))
    for i in result:
        distribute[i].append(cutText[list(result).index(i)])
    for p in range(nClustersNum):
        dict = {}
        for words in distribute[p]:
            words = re.split(" |,", words)  # 以\t分割字符串为列表
            words = [x.strip() for x in words if x.strip() != '']  # !!!!去掉\n,\t,空元素
            for word in words:
                # print(word)
                # 值的自加
                dict.setdefault(word, 0)
                dict[word] = dict[word] + 1
        # 排序
        d_order = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        key.append(d_order[0][0])
    print("中心词为" + str(key))
    return percent, key, text



change_x, txt_number, c_number, centers, words, A_weight, weights = set_cluster(3, 1000)
sum_topics, content = get_sumtopics_contents(A_weight, centers, words)
dic, topic = get_Center_topics(centers[0], words)
context = get_text(sum_topics, content, topic[0])
if context[0] == '':
    context = context[1:]

print("topic: ", end="")
print(topic)
print("sum_topics: ", end="")
print(sum_topics)
print("context: ", end="")
print(context)

percent, key, text = overlappingShow([0, 2], weights, content)

print(percent)
print(key)
print(text)

