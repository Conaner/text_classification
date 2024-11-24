import jieba
import joblib
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar, Text, END
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

svm_model_path = r"classifier\classifier.pkl"
bayes_model_path = r"classifier\baseline.pkl"  # 后验概率
vocab_path = r"matrix\vocabulary.pkl"

svm_model = joblib.load(svm_model_path)
bayes_model = joblib.load(bayes_model_path)
with open(vocab_path, "rb") as f:
    vocabulary = joblib.load(f)
# print (vocabulary)
count_vectorizer = CountVectorizer(vocabulary=vocabulary)  # 疑惑点：换成TfidfVectorizer会初始化失败


labels = [
        "education",
        "entertainment",
        "fashion",
        "finance",
        "games",
        "military",
        "sports",
        "technology",
    ]
def preprocess_text(text):
    with open('stopwords/stopwords.txt', 'r', encoding="utf-8") as f:
        stopwords_list = [line.strip() for line in f.readlines()]
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords_list and len(word) > 1]
    return " ".join(filtered_words)

# 贝叶斯预测
def bayes_predict(text_vector, bayes_model):
    posterior_scores = []
    for class_probs in bayes_model:
        posterior_score = np.sum(text_vector.toarray() * class_probs)
        posterior_scores.append(posterior_score)
    return np.argmax(posterior_scores)

def predict_category(text):
    preprocessed_text = preprocess_text(text)
    text_vector = count_vectorizer.transform([preprocessed_text])
    # SVM 预测
    svm_predicted_label = svm_model.predict(text_vector)[0]
    svm_result = labels[svm_predicted_label] if 0 <= svm_predicted_label < len(labels) else "未知类别"
    # 朴素贝叶斯预测
    bayes_predicted_label = bayes_predict(text_vector, bayes_model)
    bayes_result = labels[bayes_predicted_label] if 0 <= bayes_predicted_label < len(labels) else "未知类别"

    return {
        "SVM预测": svm_result,
        "朴素贝叶斯预测": bayes_result
    }
# 创建主界面
def create_gui():
    # 设置窗口宽高
    window_width = 900
    window_height = 700

    # 初始化应用程序
    app = ttk.Window(title="文本分类器", themename="superhero", size=(window_width, window_height))

    # 获取屏幕宽高
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    
    # 计算居中的位置
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # 设置窗口的初始位置
    app.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # 创建标题
    title_label = ttk.Label(app, text="文本分类器", font=("SimSun", 20), bootstyle="primary")
    title_label.pack(pady=20)

    # 创建文本输入框
    input_text = Text(app, height=10, wrap=WORD, font=("SimSun", 12))
    input_text.pack(fill=X, padx=20, pady=10)

    # 默认提示文本
    placeholder = "请输入新闻内容"
    input_text.insert("1.0", placeholder)

    # 绑定事件
    def clear_placeholder(event):
        # 如果内容是占位符，清空内容，并更改字体颜色
        if input_text.get("1.0", END).strip() == placeholder:
            input_text.delete("1.0", END)
            input_text.config(fg="white")  # 改为正常字体颜色

    def add_placeholder(event):
        # 如果内容为空，添加占位符，并更改字体颜色
        if not input_text.get("1.0", END).strip():
            input_text.insert("1.0", placeholder)
            input_text.config(fg="gray")  # 改为灰色字体表示提示

    # 绑定输入框的点击和失焦事件
    input_text.bind("<FocusIn>", clear_placeholder)
    input_text.bind("<FocusOut>", add_placeholder)

    # 创建结果显示框
    result_var = StringVar()
    result_label = ttk.Label(app, textvariable=result_var, font=("SimSun", 14), bootstyle="info")
    result_label.pack(pady=10)

    # 按钮点击事件
    def classify_text():
        text = input_text.get("1.0", END).strip()
        if text:
            category = predict_category(text)
            result_var.set(f"预测类别：{category}")
        else:
            result_var.set("请输入有效的文本！")

    # 创建按钮
    classify_button = ttk.Button(app, text="分类", command=classify_text, bootstyle="success")
    classify_button.pack(pady=20)

    # 运行主循环
    app.mainloop()

# 启动GUI
if __name__ == "__main__":
    create_gui()
