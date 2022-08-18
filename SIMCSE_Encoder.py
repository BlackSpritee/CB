# -*- encoding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random
from typing import  List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

# 基本参数
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-5
DROPOUT = 0.1
MAXLEN = 30  #sentence的最大截断长度
TRAIN_MODE=True
TEST_MODE=True
#向量输出方式
POOLING = 'cls'  # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 预训练模型目录
BERT_PATH = './pretrained_model/bert-base-chinese'
# 微调后参数存放位置
SAVE_PATH = './saved_model/cb_{}_simcse_unsup.pt'.format(POOLING)
# 数据目录
CB_SENTENCE_DATA = 'sentence.txt'
CB_CUT_WORD_DATA = 'word.txt'
CB_VEC_SAVE_DATA = 'vec.txt'
#聚类的类个数
NUMBER_CLUSTER=12

def load_data(path: str) -> List:
    """根据名字加载不同的数据集"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        # return [tokenizer(text),tokenizer(text)]

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])

class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = DROPOUT  # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss

def train(model, train_dl, optimizer) -> None:
    """模型训练函数"""
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)

        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            model.train()
    return model

def Encode():
    """
    输出词向量的txt文件
    """
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = SimcseModel(pretrained_model=BERT_PATH, pooling=POOLING).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH))
    encoder_list=[]
    with open(CB_CUT_WORD_DATA,'r',encoding='utf-8') as r:
        data_list=r.readlines()
    for data in data_list:
        source=tokenizer(data, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
        source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
        source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
        source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
        source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
        encoder_list.append(source_pred.cpu().detach().numpy().tolist())
    with open(CB_VEC_SAVE_DATA,'w',encoding='utf8') as w:
        for encode in encoder_list:
            i=0
            for v in encode[0][0:-1] :
                w.write(str(v)+'\t')
                i+=1
            w.write(str(encode[0][i])+'\n')

def Vec_Kmeans():
    """
    输出Kmeans聚类的可视化效果
    """
    from sklearn.cluster import KMeans
    import numpy as np
    from numpy import unique
    from numpy import where
    from matplotlib import pyplot

    with open(CB_VEC_SAVE_DATA,'r',encoding='utf-8') as r:
        vec_list=r.readlines()
    with open(CB_CUT_WORD_DATA, 'r', encoding='utf-8') as r:
        data_list = r.readlines()
    word_list=[i.strip() for i in data_list]
    encoder_list=[]
    for i in vec_list:
        cache=[]
        for j in i.strip().split('\t'):
            cache.append(float(j))
        encoder_list.append(cache)

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

if __name__ == '__main__':

    if TRAIN_MODE:
        #SIMCSE训练过程
        logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {BERT_PATH}')
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        cb_data = load_data(CB_SENTENCE_DATA)
        train_data = random.sample(cb_data, len(cb_data))  # 随机采样
        train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
        model = SimcseModel(pretrained_model=BERT_PATH, pooling=POOLING).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        for epoch in range(EPOCHS):
            logger.info(f'epoch: {epoch}')
            model = train(model, train_dataloader, optimizer)
        torch.save(model.state_dict(), SAVE_PATH)
        logger.info(f'train is finished, best model is saved at {SAVE_PATH}')

    if TEST_MODE:
        #用训练好的模型对词编码
        Encode()

        #对编码好的向量文件读取并K-means聚类
        Vec_Kmeans()