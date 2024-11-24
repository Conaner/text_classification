# -*- coding: utf-8 -*-

import sys
import os

import time

import pickle

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib


def readTerm(term_file_folder_path):
    """
    读取Term文件,返回Term字符串生成器和类别生产器
    """

    def getTerm():
        classification = os.listdir(term_file_folder_path)
        for num, clsf in enumerate(classification):
            print(num, "/", len(classification))
            clsf_folder = os.path.join(term_file_folder_path, clsf)
            for term_filename in os.listdir(clsf_folder)[:50000]:
                path = os.path.join(clsf_folder, term_filename)
                with open(path, "rb") as f:
                    term_list = pickle.load(f)
                term = " ".join(term_list)
                yield term

    def getTarget():
        classification = os.listdir(term_file_folder_path)
        clsf_folder = os.path.join(term_file_folder_path, clsf)
        for num, clsf in enumerate(classification):
            for term_filename in os.listdir(clsf_folder)[:50000]:
                yield num

    # Term字符串生成器
    term_generator = getTerm()
    # Term的类别生成器
    target_generator = getTarget()

    return term_generator, target_generator


def generateMatrix():
    """
    生成特征矩阵
    """
    vectorizer = CountVectorizer(min_df=0.001)

    for x in ["train", "test"]:
        # 分词数据的文件夹路径
        term_file_folder_path = os.path.join("data", x, "term")
        # 特征矩阵保存路径
        matrix_folder = os.path.join("matrix", x)
        matrix_path = os.path.join("matrix", x, "matrix.pkl")

        # 确保保存路径的目录存在
        os.makedirs(matrix_folder, exist_ok=True)

        # 读取数据
        term_generator, target_generator = readTerm(term_file_folder_path)

        # 训练集拟合后转换为矩阵，测试集根据拟合好的矢量器直接转换为矩阵
        if x == "train":
            matrix = vectorizer.fit_transform(term_generator)
            vocab_path = os.path.join("matrix", "vocabulary.pkl")
            joblib.dump(vectorizer.vocabulary_, vocab_path)

        elif x == "test":
            matrix = vectorizer.transform(term_generator)

        # 保存特征矩阵
        joblib.dump(matrix, matrix_path)


if __name__ == "__main__":
    time_start = time.time()
    generateMatrix()
    print("Transform time:", time.time() - time_start, "s")
