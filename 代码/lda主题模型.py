from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib  #也可以选择pickle等保存模型，请随意
import collections

def quchu(res):
    res = res.replace('\n', ' ')
    res = res.replace('  ', ' ')
    return res

with open('data/0.txt') as f0:
    res0_1 = f0.read()
    res0=quchu(res0_1)
    print(res0)
with open('data/1.txt') as f1:
    res1_1 = f1.read()
    res1=quchu(res1_1)
    print(res1)
with open('data/2.txt') as f2:
    res2_1 = f2.read()
    res2=quchu(res2_1)
    print(res2)
with open('data/3.txt') as f3:
    res3_1 = f3.read()
    res3=quchu(res3_1)
    print(res3)
with open('data/4.txt') as f4:
    res4_1 = f4.read()
    res4=quchu(res4_1)
    print(res4)
with open('data/5.txt') as f5:
    res5_1 = f5.read()
    res5=quchu(res5_1)
    print(res5)

res=[res0,res1,res2,res3,res4,res5]

def get_datasest():
    filepath = 'data/trainData'
    doc = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return doc

#构建词汇统计向量并保存，仅运行首次
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=2500,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(res)
joblib.dump(tf_vectorizer,'lda.model')
#得到存储的tf_vectorizer,节省预处理时间
# tf_vectorizer = joblib.load('lda.model')
# tf = tf_vectorizer.fit_transform(get_datasest())

n_topics = 5
lda = LatentDirichletAllocation(n_topics=n_topics,
                                max_iter=50,
                                learning_method='batch')
lda.fit(tf) #tf即为Document_word Sparse Matrix

docres = lda.fit_transform(tf)
for i in range(len(docres)):
    print(docres)

def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

n_top_words=20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# f = open('data/trainData', 'r', encoding='utf-8')
# file = f.read().replace('\n', '').split(' ')
# ci = collections.Counter(file)
# print(ci)
