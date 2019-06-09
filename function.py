import re
import jieba
from gensim.models import KeyedVectors

def clean(content, model):
    tokens = []
    for text in content:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        # 结巴分词
        cut = jieba.cut(text)
        # 结巴分词的输出结果为一个生成器
        # 把生成器转换为list
        cut_list = [ i for i in cut ]
        for i, word in enumerate(cut_list):
            try:
                # 将词转换为索引index
                cut_list[i] = model.vocab[word].index
            except KeyError:
                # 如果词不在字典中，则输出0
                cut_list[i] = 0
        tokens.append(cut_list)
    return tokens

