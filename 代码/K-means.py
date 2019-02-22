# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import gensim
import sklearn
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
import multiprocessing
import jieba

# 创建停用词list
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

def cluster(x_train):
    infered_vectors_list = []
    print("load doc2vec model...")
    model_dm = Doc2Vec.load("model/模型.model")
    print("load train vectors...")
    i = 0
    for text, label in x_train:
        vector = model_dm.infer_vector(text)
        infered_vectors_list.append(vector)
        i += 1

    print("train kmean model...")
    kmean_model = KMeans(n_clusters=6)
    kmean_model.fit(infered_vectors_list)
    labels = kmean_model.predict(infered_vectors_list[0:4999])
    cluster_centers = kmean_model.cluster_centers_

    label_pred = kmean_model.labels_  # 获取聚类标签
    # 绘制k-means结果
    x0 = np.array(infered_vectors_list)[label_pred == 0]
    x1 = np.array(infered_vectors_list)[label_pred ==  1]
    x2 = np.array(infered_vectors_list)[label_pred ==  2]
    x3= np.array(infered_vectors_list)[label_pred ==  3]
    x4= np.array(infered_vectors_list)[label_pred ==  4]
    x5= np.array(infered_vectors_list)[label_pred ==  5]
    # x6= np.array(infered_vectors_list)[label_pred ==  6]
    # x7= np.array(infered_vectors_list)[label_pred ==  7]
    # x8= np.array(infered_vectors_list)[label_pred == 8]
    # x9= np.array(infered_vectors_list)[label_pred == 9]
    # x10= np.array(infered_vectors_list)[label_pred == 10]
    # x11= np.array(infered_vectors_list)[label_pred == 11]
    # x12= np.array(infered_vectors_list)[label_pred == 12]
    # x13= np.array(infered_vectors_list)[label_pred ==13]
    # x14= np.array(infered_vectors_list)[label_pred ==14]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='+', label='label3')
    plt.scatter(x4[:, 0], x4[:, 1], c="m", marker='', label='label4')
    plt.scatter(x5[:, 0], x5[:, 1], c="k", marker='+', label='label5')
    # plt.scatter(x6[:, 0], x6[:, 1], c="c", marker='+', label='label6')
    # plt.scatter(x7[:, 0], x7[:, 1], c="m", marker='+', label='label7')
    # plt.scatter(x8[:, 0], x8[:, 1], c="blue", marker='+', label='label2')
    # plt.scatter(x9[:, 0], x9[:, 1], c="blue", marker='+', label='label2')
    # plt.scatter(x10[:, 0], x10[:, 1], c="blue", marker='+', label='label2')
    # plt.scatter(x11[:, 0], x11[:, 1], c="blue", marker='+', label='label2')
    # plt.scatter(x12[:, 0], x12[:, 1], c="blue", marker='+', label='label2')
    # plt.scatter(x13[:, 0], x13[:, 1], c="blue", marker='+', label='label2')
    # plt.scatter(x14[:, 0], x14[:, 1], c="blue", marker='+', label='label2')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()




    with open("own_claasify.txt", 'w', encoding='utf-8') as wf:
        for i in range(4998):
            string = ""
            text = x_train[i][0]
            for word in text:
                string = string + word
            string = string + '\t'
            string = string + str(labels[i])
            string = string + '\n'
            wf.write(string)

    return cluster_centers

def cun(num):
    # text_decode = text.decode('GBK')
    text = open("own_claasify.txt", 'r', errors='ignore')
    f2=open('data/' + str(num) + '.txt', 'w')
    for i in text:
        k = i.replace('\n', '')
        if k[-1:] == str(num):
            print(k[:len(k) - 1])
            f2.write(k[:len(k) - 1] + '\n')
    print(num)
    f2.close()
    # with open('./nlp_test1.txt') as f3:
    #     res1 = f3.read()



def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=100)
    model_dm.save('model/model_dm')

    return model_dm

def get_datasest():
    filepath = 'data/trainData'
    doc = [line.strip() for line in open(filepath, 'r',encoding="UTF-8").readlines()]
    x_train = []
    for i, text in enumerate(doc):
        document = gensim.models.doc2vec.TaggedDocument(text, tags=[i])
        x_train.append(document)
    return x_train

# filepath = 'data/neg.txt'
# tokenizer(readDoc2List(filepath))


# filepath = 'data/trainData'
# doc = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
# model = train(doc)

# docslist = doc2vec.TaggedLineDocument('data/trainData') # path为txt的路径
# x_train = train(docslist)
# cluster_centers = cluster(x_train)

if __name__ == '__main__':
    x_train = get_datasest()
    # model_dm = train(x_train)
    cluster_centers = cluster(x_train)
#
# def quchu(res):
#     res = res.replace('\n', ' ')
#     res = res.replace('  ', ' ')
#     return res
#
#
# # for i in range(0,6):
# #     cun(i)
# with open('data/0.txt') as f0:
#     res0_1 = f0.read()
#     res0=quchu(res0_1)
#     print(res0)
# with open('data/1.txt') as f1:
#     res1_1 = f1.read()
#     res1=quchu(res1_1)
#     print(res1)
# with open('data/2.txt') as f2:
#     res2_1 = f2.read()
#     res2=quchu(res2_1)
#     print(res2)
# with open('data/3.txt') as f3:
#     res3_1 = f3.read()
#     res3=quchu(res3_1)
#     print(res3)
# with open('data/4.txt') as f4:
#     res4_1 = f4.read()
#     res4=quchu(res4_1)
#     print(res4)
# with open('data/5.txt') as f5:
#     res5_1 = f5.read()
#     res5=quchu(res5_1)
#     print(res5)
#
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# corpus = [res1,res2,res3]
# cntVector = CountVectorizer(stop_words=stpwrdlst)
# cntTf = cntVector.fit_transform(corpus)
# print cntTf

