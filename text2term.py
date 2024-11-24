# -*- coding: utf-8 -*-

import os
import jieba
import pickle
import re
import multiprocessing
# from sklearn.feature_extraction.text import TfidfVectorizer
# from overlapping_kmeans import OverlappingKMeans

def convertText2Term(*args):
    """
    将Text文本文件转为Term分词文件
    此函数作为一个进程
    """

    text_path = args[0]
    term_path = args[1]
    stopwords_list = args[2]

    # 读取Text文件
    with open(text_path, 'r', encoding='utf-8') as f: 
        text = f.read()
    
    # 分词
    term_list = [x.strip() for x in jieba.cut(text) if x.strip()]  # 去掉空字符的分词

    # 打印分词结果以便调试
    # print(f"Processed file: {text_path}, Total words: {len(term_list)}")
    # print(f"First 10 terms: {term_list[:10]}")  # 输出前10个分词，帮助调试

    # 过滤分词
    # filter_pattern = re.compile(r'[-+]?[\w\d]+|零|一|二|三|四|五|六|七|八|九|十|百|千|万|亿')
    filter_pattern = re.compile(r'^[\d]+$|^[\W_]+$')  # 过滤纯数字和纯符号
    filtered_term_list = []
    for term in term_list:
        # 被过滤的分词：长度小于2, 包含数字或字母或中文数词, 停用词
        if len(term) < 2 or filter_pattern.search(term) or term in stopwords_list:
            continue  
        
        filtered_term_list.append(term)
    # for term in term_list:
    #     if len(term) < 2:
    #         print(f"Skipped term (too short): {term}")
    #         continue
    #     elif filter_pattern.search(term):
    #         print(f"Skipped term (matches pattern): {term}")
    #         continue
    #     elif term in stopwords_list:
    #         print(f"Skipped term (stopword): {term}")
    #         continue
    #     else:
    #         filtered_term_list.append(term)

    # # 打印过滤后的分词以便调试
    # print(f"Filtered terms count: {len(filtered_term_list)}")
    # print(f"First 10 filtered terms: {filtered_term_list[:10]}")  # 输出过滤后前10个分词

    # 存储分词
    if len(filtered_term_list) >= 10:  # 仅保存至少10个有效分词的文件
        with open(term_path, 'wb') as f:  
            pickle.dump(filtered_term_list, f)
        print(f"Saved term file: {term_path}")
    else:
        print(f"Skipped file (too few terms): {text_path}")

# def apply_overlapping_kmeans(text_folder_path, term_folder_path, num_clusters=8):
#     """
#     应用重叠聚类
#     """
#     all_texts = []
#     all_term_paths = []

#     # 获取所有分词文件路径
#     for clsf in os.listdir(text_folder_path):
#         cls_folder_path = os.path.join(text_folder_path, clsf)
#         if os.path.isdir(cls_folder_path):  # 确保是文件夹
#             for text_filename in os.listdir(cls_folder_path):
#                 text_path = os.path.join(cls_folder_path, text_filename)
#                 term_path = os.path.join(term_folder_path, clsf, text_filename.split('.')[0] + '.pkl')
#                 # 加载已分词的文本
#                 with open(term_path, 'rb') as f:
#                     terms = pickle.load(f)
#                     all_texts.append(" ".join(terms))  # 将分词后的词汇合并成一个字符串
#                     all_term_paths.append(term_path)

#     # 使用TfidfVectorizer将分词结果转换为TF-IDF特征矩阵
#     # KMC = KmeansClustering()
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(all_texts)

#     # 应用OverlappingKMeans聚类
#     clustering = OverlappingKMeans(n_clusters=num_clusters)  # 设置聚类的个数
#     clustering.fit(X)

#     # 将聚类结果保存
#     for idx, term_path in enumerate(all_term_paths):
#         cluster_label = clustering.labels_[idx]
#         print(f"Text file {term_path} is in cluster {cluster_label}")

def processText(text_file_folder_path, term_file_folder_path):
    """
    处理指定路径下的所有Text文件
    """
    # 创建进程池,参数为池中进程数
    pool = multiprocessing.Pool(6)

    # 获取停用词表
    with open('stopwords/stopwords.txt', 'r', encoding="utf-8") as f:
        stopwords_list = [line.strip() for line in f.readlines()]

    for clsf in os.listdir(text_file_folder_path):
        print(clsf)
        cls_folder_path = os.path.join(text_file_folder_path, clsf)
        if os.path.isdir(cls_folder_path):  # 确保是文件夹
            for text_filename in os.listdir(cls_folder_path):
                text_path = os.path.join(cls_folder_path, text_filename)
                term_path = os.path.join(term_file_folder_path, clsf, text_filename.split('.')[0] + '.pkl')

                # 创建保存term文件的文件夹
                os.makedirs(os.path.dirname(term_path), exist_ok=True)

                args = (text_path, term_path, stopwords_list)

                # 调用文本转分词的进程
                pool.apply_async(convertText2Term, args=args)
            
    pool.close()
    pool.join()


if __name__ == '__main__':
    # 训练集文本数据的文件夹路径
    text_file_folder_path = 'data/train/raw/'
    # 训练集分词数据的文件夹路径
    term_file_folder_path = 'data/train/term/'
    # 处理训练集文本
    processText(text_file_folder_path, term_file_folder_path)


    # 测试集文本数据的文件夹路径
    text_file_folder_path = 'data/test/raw/'
    # 测试集分词数据的文件夹路径
    term_file_folder_path = 'data/test/term/'
    # 处理测试集文本
    processText(text_file_folder_path, term_file_folder_path)
