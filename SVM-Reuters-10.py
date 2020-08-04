from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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

from sklearn.metrics import classification_report,f1_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',SVC(kernel='linear'))])
text_clf=text_clf.fit(x_train,y_train)
predicted=text_clf.predict(x_test)
print(classification_report(y_test, predicted))
score = f1_score(y_test, predicted,average='macro')
print(score)
'''
                precision    recall  f1-score   support
         acq       1.00      1.00      1.00       620
      coffee       1.00      0.95      0.98        21
       crude       1.00      1.00      1.00        98
        earn       1.00      1.00      1.00      1040
        gold       1.00      1.00      1.00        20
    interest       1.00      1.00      1.00        57
    money-fx       1.00      1.00      1.00        69
        ship       1.00      1.00      1.00        35
       sugar       1.00      1.00      1.00        24
       trade       1.00      1.00      1.00        73
   micro avg       1.00      1.00      1.00      2057
   macro avg       1.00      1.00      1.00      2057
weighted avg       1.00      1.00      1.00      2057

macro avg f1:0.9973516314668966
'''