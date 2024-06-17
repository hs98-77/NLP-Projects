from nltk.corpus import treebank
import numpy as np
from tqdm import tqdm
dataset = list(treebank.tagged_sents())
uni_dataset = list(treebank.tagged_sents(tagset='universal'))
#%%
tags = list()
for sen in dataset:
    for word in sen:
        tags.append(word[1])
tags = list(set(tags))
uni_tags = list()
for sen in uni_dataset:
    for word in sen:
        uni_tags.append(word[1])
uni_tags = list(set(uni_tags+ ['START']))
del sen
del word
words = list()
for sen in uni_dataset:
    for w,t in sen:
        words.append(w)
words = list(set(words)) + ["[UNK]"]
del t
del w
del sen
#%%
def CreateTagsMatrix(ds, tags):
    Tags = np.ones((len(tags),len(tags)))
    #Count number of each tag followed by another one
    for sen in ds:
        prev_tag = 'START'
        for w,t in sen:
            Tags[tags.index(prev_tag), tags.index(t)] += 1
            prev_tag = t
    #Calculate probability of each tag followed by another one(smoothed by laplace method(+1))
    for i in range(len(tags)):
        Tags[i,:] = Tags[i,:]/np.sum(Tags[i,:])
    return Tags
#%%
def CreateWordsTagsMatrix(ds, tags, words):
    emission = np.ones((len(tags),len(words)))
    #Count Number of Each Words with a particular tag
    for sen in ds:
        for w,t in sen:
            emission[tags.index(t), words.index(w)] += 1
    #Calculate probability of tags emitting each word
    for i in range(len(tags)):
        emission[i,:] = emission[i,:]/np.sum(emission[i,:])
    return emission
    #%%
from random import gauss
def split(ds, train_size):
    train = list()
    test = list()
    for sen in uni_dataset:
        if gauss(0,1) > train_size:
            test.append(sen)
        else:
            train.append(sen)
    return train,test
#%%
train_set, test_set = split(uni_dataset, 0.8)
transition = CreateTagsMatrix(train_set, uni_tags)
emission = CreateWordsTagsMatrix(train_set, uni_tags, words)
#%%
def RepTagedSen(path, tags, words, seq):
    p_shape = path.shape
    tagged_sen = list()
    prev_tag_ind =  p_shape[0]-1
    for i in range(p_shape[1]-1,0,-1):
        tagged_sen.append(tags[path[prev_tag_ind,i]])
        prev_tag_ind = path[prev_tag_ind,i]
    tagged_sen = list(reversed(tagged_sen))
    """
    output = list()
    for i in range(len(seq)):
        output.append(tuple([seq[i],tagged_sen[i]]))
    """
    return tagged_sen
    
#%%
def Viterbi(em, tr, seq, tags, words):
    v = np.zeros((len(tags),len(seq)+1))
    p = np.full((len(tags),len(seq)+1),-1)
    for i in range(len(tags)):
        if seq[0] in words:
            v[i,0] = tr[tags.index('START'),i] * em[i,words.index(seq[0])]
        else:
            v[i,0] = tr[tags.index('START'),i] * em[i,words.index("[UNK]")]
    if len(seq)>1:
        for i in range(1,len(seq)):
            for curr_t in tags:
                l = list()
                for prev_t in tags:
                    if seq[i] in words:
                        l.append( v[tags.index(prev_t),(i-1)] * (tr[tags.index(prev_t),tags.index(curr_t)] * em[tags.index(curr_t),words.index(seq[i])]))
                    else:
                        l.append( v[tags.index(prev_t),(i-1)] * (tr[tags.index(prev_t),tags.index(curr_t)] * em[tags.index(curr_t),words.index("[UNK]")]))
                    m = max(l)
                    m_ind = l.index(m)
                    v[tags.index(curr_t),i] = m
                    p[tags.index(curr_t),i] = m_ind
        m = max(v[:,i])
        m_ind = list(v[:,i]).index(m)
        v[:,i+1] = m
        p[:,i+1] = m_ind
    else:
        m = max(v[:,0])
        m_ind = list(v[:,0]).index(m)
        v[:,1] = m
        p[:,1] = m_ind
    return RepTagedSen(p, tags, words, seq)
#%%
test_sen = list()
test_tags = list()
for sen in test_set:
    test_sen.append([w for w,t in sen])
    test_tags.append([t for w,t in sen])
#%%
pred = list()
for sen in tqdm(test_sen):
    pred.append(Viterbi(emission, transition, sen, uni_tags, words))
del sen
#%%
def EvalViterbi(gold, pred):
    count = 0
    correct_pred = 0
    for i in range(len(gold)):
        for j in range(len(gold[i])):
            count += 1
            if gold[i][j] == pred[i][j]:
                correct_pred += 1
    return round(correct_pred/count,4)
#%%
acc = EvalViterbi(test_tags, pred)
#%%
