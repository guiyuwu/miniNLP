# from sklearn import preprocessing
# from sklearn import  svm,datasets
import jieba
# clf=svm.SVC()
# iris=datasets.load_iris()
# x=iris.data
# y=iris.target
# print(x)
# print(y)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  svm
from sklearn.model_selection import train_test_split
import numpy as np
import sys

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def readDoc2List(filepath):
    doc = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return doc

#分词、去停
def tokenizer(docList):
    stopwords = stopwordslist('停用词表.txt')  # 这里加载停用词的路径
    docKeywords = []
    num = 1
    print('now tokenizer doc...')

    for doc in docList:
        num += 1
        print('tokenizerdoc', num)
        seg_list = jieba.cut(doc)
        outstr = ''
        for word in seg_list:
            if word not in stopwords:
                if word != '\t' and word != '\r\n' and word != '\n' and word != '　' and word != '\r':
                    outstr += word
                    outstr += " "
        docKeywords.append(outstr[:-1])
    f = open('trainData', 'w', encoding='utf-8')
    for i in range(len(docKeywords)):
        f.write(docKeywords[i].strip() + '\n')
    f.close()
    print('Done!')
    return docKeywords


f=open(r'data/all.txt','r',encoding="utf-8",errors='ignore')
x=[]
y=[]
for line in f:
    sentence=line.replace('\n','')
    x_y=sentence.split('\t')
    if(len(x_y)==2):
        x.append(x_y[0])
        y.append(x_y[1])
vectorizer=TfidfVectorizer(min_df=2)
train_vectors=vectorizer.fit_transform(x)
# print(type(train_vectors))
matrixs=train_vectors.toarray()
data=np.array(matrixs)
max=0
max_score=0
train_x,test_x,train_y,test_y=train_test_split(data,y,test_size=0.2,random_state=5)
# print(train_vectors)
# print(train_y)
clf=svm.SVC()
clf.fit(train_x,train_y)
# pre_y=clf.predict(test_x)
score=clf.score(test_x,test_y)
print(score)


# sen=input()
# x.append(sen)
# vectorizer1=TfidfVectorizer(min_df=2)
# train_vectors1=vectorizer1.fit_transform(x)
# matrixs1=train_vectors1.toarray()
# data1=np.array(matrixs1)
# train1_x,test1_x,train1_y,test1_y=train_test_split(data1[:-1],y,test_size=0.2,random_state=59)
# # print(train_x)
# # print(train_y)
# clf1=svm.SVC()
# clf1.fit(train1_x,train1_y)
# # pre_y=clf.predict(test_x)
# # print(data1[-1:])
# pre_y=clf1.predict(data1[-1:])
# print(pre_y)



# corpus=['This is the first document.',
#         'This is the second second document.',
#         'And the third one.',
#         'Is this the first document?',]
# vectorizer=TfidfVectorizer(min_df=2)
# vectors=vectorizer.fit_transform(corpus)
# matrixs=vectors.toarray()
# print(matrixs[1:3])
# print(matrixs)