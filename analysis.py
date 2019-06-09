import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import warnings
from function import *

warnings.filterwarnings("ignore")

train_text = []

cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)

with open('1.txt', 'r', encoding='utf-16') as f:
    for sentence in f.readlines():
        train_text.append(sentence.replace('\n', ''))
with open('0.txt', 'r', encoding='utf-16') as f:
    for sentence in f.readlines():
        train_text.append(sentence.replace('\n', ''))


train_text = clean(train_text, cn_model)
print(len(train_text),len(train_text[0]))
num_tokens = np.array([len(token) for token in train_text])
max_token = int(np.mean(num_tokens) + 2*np.std(num_tokens))

num_words = 50000
embedding = np.zeros((num_words,300))
for i in range(num_words):
    embedding[i,:] = cn_model[cn_model.index2word[i]]
embedding = embedding.astype(np.float32)

train_pad = pad_sequences(train_text, maxlen=max_token, padding='pre', truncating='pre')
train_pad[train_pad>=num_words] = 0

from sklearn.model_selection import train_test_split
train_target = np.concatenate((np.ones(2000), np.zeros(2000)))
X_train, X_test, y_train, y_test = train_test_split(train_pad, train_target, 
                                                test_size=0.1, random_state=12)

#model
model = Sequential()
model.add(Embedding(num_words, 300, weights=[embedding],
                        input_length=max_token, trainable=False))
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
# 我们使用adam以0.001的learning rate进行优化
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

path_checkpoint = 'sentiment_checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                      verbose=1, save_weights_only=True,
                                      save_best_only=True)
try:
    model.load_weights(path_checkpoint)
except Exception as e:
    print(e)

earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1, min_lr=1e-5, patience=0,
                                       verbose=1)
callbacks = [earlystopping, checkpoint, lr_reduction]

model.fit(X_train, y_train,
          validation_split=0.1, 
          epochs=20,
          batch_size=128,
          callbacks=callbacks)

result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))

def predict_sentiment(text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_token,
                           padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)

test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '房间很冷，不给开暖气',
    '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好',
    '房间隔音不到位' ,
    '晚上回来发现没有打扫卫生',
    '因为过节所以要我临时加钱，比团购的价格贵',
    '烤肉',
    '美国',
    '中国',
]
for text in test_list:
    predict_sentiment(text)