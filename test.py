
import os
pos_txts = os.listdir('pos')
neg_txts = os.listdir('neg')

train_texts_orig = []

neg = []

for i in range(len(pos_txts)):
    with open('pos/'+pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        text = text.replace('\n', ' ')
        train_texts_orig.append(text)
        f.close()
for i in range(len(neg_txts)):
    with open('neg/'+neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        text = text.replace('\n', ' ')
        neg.append(text)
        f.close()
print(train_texts_orig[:10])

with open('1.txt', 'w', encoding='utf-16') as f:
    for a in train_texts_orig:
        f.write(a + '\n')
with open('0.txt', 'w', encoding='utf-16') as f:
    for a in neg:
        f.write(a + '\n')
