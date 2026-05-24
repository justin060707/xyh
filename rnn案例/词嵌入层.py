"""

RNN:
    循环神经网络，主要处理序列数据
    序列数据：后边数据对前边数据有依赖，例如：天气预测，股市分析，文本生成

    组成：
        词嵌入层
        循环网络层
        输出层
词嵌入层介绍：
    把词转成词向量


"""


import torch
import torch.nn as nn
import jieba

# 词嵌入层的api 如何把词->词向量
def dm01():
    text='北京冬奥会的进度条已经过半，不少外国运动员在完成自己的表赛后踏上归途。'
    words = jieba.lcut(text)
    print(words)

    # 创建词嵌入层
    # 词的个数   词向量的维度
    embed = nn.Embedding(len(words),4)

    # enumerate()：返回雷彪中每个值 机器对应的索引
    for i,word in enumerate(words):
        # 把词索引（张量形式）转化成词向量
        word_vector = embed(torch.tensor(i))

if __name__ == '__main__':
    dm01()