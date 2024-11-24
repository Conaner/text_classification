# -*- coding: utf-8 -*-

import os

import time

import numpy as np

from math import log

import joblib


def trainNaiveBayesClassifier():
    """
    训练朴素贝叶斯分类器
    """
    # 训练集特征矩阵保存路径
    train_matrix_path = "matrix/train/matrix.pkl"
    # 后验概率保存路径
    prob_clf_path = "classifier/baseline.pkl"

    # 加载数据
    matrix = joblib.load(train_matrix_path)
    # print("Train Matrix Shape:", matrix.shape) 
    num_classes = 8
    num_samples = matrix.shape[0]  # 行数即样本数
    samples_per_class = num_samples // num_classes
    target = np.array([i for i in range(num_classes) for _ in range(samples_per_class)])
    # 计算每一类中每个特征的后验概率
    prob_clf = []
    for clf in range(0, num_classes):
        start_row = clf * samples_per_class 
        end_row = (clf + 1) * samples_per_class 

        vector = matrix[start_row:end_row].sum(axis=0)
        total_word = matrix[start_row:end_row].sum()

        prob = np.log((vector + 1) / float(total_word))
        prob_clf.append(prob)

    # 保存后验概率
    joblib.dump(prob_clf, prob_clf_path)


def testNaiveBayesClassifier():
    """
    测试朴素贝叶斯分类器
    """
    # 测试集特征矩阵保存路径
    test_matrix_path = "matrix/test/matrix.pkl"
    # 后验概率保存路径
    prob_clf_path = "classifier/baseline.pkl"
    Bayes_confusion_matrix_path = "results/Bayes_confusion_matrix.pkl"
    os.makedirs(os.path.dirname(Bayes_confusion_matrix_path), exist_ok=True)
    # 加载数据
    matrix = joblib.load(test_matrix_path)
    # target = np.array([x for x in range(10) for i in range(50000)])
    num_classes = 8  # 有 8 个类别
    num_samples = matrix.shape[0]  # 行数即样本数
    samples_per_class = num_samples // num_classes
    target = np.array([i for i in range(num_classes) for _ in range(samples_per_class)])


    # 加载贝叶斯每一类的后验概率
    prob_clf = joblib.load(prob_clf_path)

    # 预测
    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=int)
    for i in range(len(target)):
        max_value, predicted = -float("inf"), 0
        a = np.array(matrix[i].sum(axis=0))[0]

        for clf in range(0, num_classes):
            b = np.array(prob_clf[clf])[0]
            value = np.dot(a, b)
            if value > max_value:
                max_value = value
                predicted = clf

        confusion_matrix[target[i]][predicted] += 1
    
    joblib.dump(confusion_matrix, Bayes_confusion_matrix_path)

    # 统计
    recall_list, precision_list, f_list = [], [], []
    correct = 0
    r = confusion_matrix.sum(axis=1)
    p = confusion_matrix.sum(axis=0)
    for clf in range(num_classes):
        recall = confusion_matrix[clf][clf] / float(r[clf]) 
        precision = confusion_matrix[clf][clf] / float(p[clf]) 
        f = 2 * recall * precision / (recall + precision)
        recall_list.append(recall)
        precision_list.append(precision)
        f_list.append(f)
        correct += confusion_matrix[clf][clf]
    correct /= float(matrix.shape[0])

    # 打印测试报告
    print(confusion_matrix, "\n")

    print(
        "{0:>14}\t{1:<10}\t{2:<10}\t{3:<10}".format(
            "classification", "Recall", "Precision", "F1-Score"
        )
    )
    for i, target_name in enumerate(os.listdir("data/test/raw/")):
        print(
            "{0:>14}\t{1:<10.4f}\t{2:<10.4f}\t{3:<10.4f}".format(
                target_name, recall_list[i], precision_list[i], f_list[i]
            )
        )
    print("")
    # avg_r, avg_p, avg_f = 0.0, 0.0, 0.0
    # for a, b, c in zip(recall_list, precision_list, f_list):
    #     avg_r += a
    #     avg_p += b
    #     avg_f += c
    avg_r, avg_p, avg_f = np.mean(recall_list), np.mean(precision_list), np.mean(f_list)
    print(
        "{0:>14}\t{1:<10.4f}\t{2:<10.4f}\t{3:<10.4f}".format(
            "avg / total", avg_r, avg_p, avg_f
        )
    )

    print("\n", "Correct Rate:", correct)


if __name__ == "__main__":
    # 训练
    time_start = time.time()
    trainNaiveBayesClassifier()
    print("Training time:", time.time() - time_start, "s")

    # 测试
    time_start = time.time()
    testNaiveBayesClassifier()
    print("Testing time:", time.time() - time_start, "s")
