NUMBER_CLUSTER=30
def Vec_Kmeans(wv):
    """
    输出Kmeans聚类的可视化效果
    """
    from sklearn.cluster import KMeans
    import numpy as np
    from numpy import unique
    from numpy import where
    from matplotlib import pyplot

    # with open(CB_VEC_SAVE_DATA,'r',encoding='utf-8') as r:
    vec_list=wv.vectors
    # with open(CB_CUT_WORD_DATA, 'r', encoding='utf-8') as r:
    data_list = wv.index_to_key
    word_list=[i.strip() for i in data_list]
    encoder_list=vec_list
    # for i in vec_list:
    #     cache=[]
    #     for j in i.strip().split('\t'):
    #         cache.append(float(j))
    #     encoder_list.append(cache)

    X = np.array(encoder_list) #向量矩阵X
    kmeans = KMeans(n_clusters=NUMBER_CLUSTER, random_state=0).fit(X)
    kmeans.fit(X)
    yhat = kmeans.predict(X)
    # 检索唯一群集

    result=[]
    for i in range(NUMBER_CLUSTER):
        result.append([])
    for i,cls in enumerate(yhat):
        result[cls].append(word_list[i])
    for cls in result:
        print(cls)

    #可视化
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # 创建这些样本的散布
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
    pyplot.show()

import gensim
import jieba
if __name__ == '__main__':
    jieba.load_userdict("word.txt")
    with open("sentence.txt",'r',encoding='utf8') as r:
        sentence=r.readlines()
    cut_sentence=[jieba.lcut(line) for line in sentence]
    model =gensim.models.Word2Vec(cut_sentence,min_count=2,window=5)
    wv = model.wv
    Vec_Kmeans(wv)