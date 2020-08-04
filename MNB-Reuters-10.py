import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report,f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
train_data = [] #训练数据
train_label = []#训练标签

test_data = []#测试数据
test_label = []#测试标签
for root, dirs, files in os.walk("D:\\data\\reuters\\train\\low5-move"):
    # 遍历文件
    for f in files:
        data = open("D:\\data\\reuters\\train\\low5-move\\"+f)
        while True:
            text = data.readline()  # 只读取一行内容
            if text != "":
                train_label.append(f[:-4])
                train_data.append(text)
            # 判断是否读取到内容
            if not text:
                break


for root, dirs, files in os.walk("D:\\data\\reuters\\test\\low5-move"):
    # 遍历文件
    for f in files:
        data = open("D:\\data\\reuters\\test\\low5-move\\"+f)
        while True:
            text = data.readline()  # 只读取一行内容
            if text != "":
                test_label.append(f[:-4])
                test_data.append(text)
            # 判断是否读取到内容
            if not text:
                break

x_train = train_data
x_test = test_data

y_train = train_label
y_test = test_label



# 贝叶斯 
# max_features 当没有用外部词表的时候，会对词进行排序，取 max_features 个
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',MultinomialNB(alpha=0.01))])
text_clf=text_clf.fit(x_train,y_train)
predicted=text_clf.predict(x_test)
print(classification_report(y_test, predicted))
score = f1_score(y_test, predicted,average='macro')
print(score)

'''
    precision    recall  f1-score   support
         acq       0.95      1.00      0.97       620
      coffee       1.00      0.95      0.98        21
       crude       0.99      0.99      0.99        98
        earn       1.00      0.97      0.98      1040
        gold       1.00      1.00      1.00        20
    interest       0.97      1.00      0.98        57
    money-fx       0.97      0.97      0.97        69
        ship       0.97      1.00      0.99        35
       sugar       1.00      1.00      1.00        24
       trade       0.97      0.99      0.98        73
   micro avg       0.98      0.98      0.98      2057
   macro avg       0.98      0.99      0.98      2057
weighted avg       0.98      0.98      0.98      2057

macro avg f1:0.9840790734302163

'''
