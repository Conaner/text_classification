# -*- coding: utf-8 -*-

# 从给定的文件路径 filepath 中读取停用词，并去除重复的停用词，返回一个去重后的停用词列表
def deduplicateStopWords(filepath):
    deduplicated = []
    i = 0
    with open(filepath, "rb") as f:
        for line in f.readlines():
            i += 1
            word = line.strip()
            if word not in deduplicated:
                deduplicated.append(word)
            else:
                print(i)

    return deduplicated


def writeNewStopWordsList(filepath, stopwords):
    with open(filepath, "w", encoding="utf-8") as f:
        for word in stopwords:
            if isinstance(word, bytes):
                word = word.decode("utf-8")
            f.write(word + "\n")


if __name__ == "__main__":
    stopwords = deduplicateStopWords("stopwords/cn_stopwords.txt")
    print("Completion Detection")
    # writeNewStopWordsList('stopwords.txt', stopwords)
    # print ('Completion Deduplicate')
