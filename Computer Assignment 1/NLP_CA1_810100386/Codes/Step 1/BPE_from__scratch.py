#Hosein Seifi حسین سیفی
#810100386
import numpy as np

#Create Vocab out of corpus characters
def CreateVocab(corpus):
    v = list(set(corpus)) +['_']
    v.remove(' ')
    return v
#convert string corpus to list of words
#each word is a list of characters
def CleanCorpus(corpus):
    corpus = corpus.split(' ')
    for i in range(len(corpus)):
        corpus[i] = list(corpus[i]) + ['_']
    return corpus
#claculate frequency of the given pair of symbols
def CalcFreq(corpus, str1, str2):
    f = 0
    for w in corpus:
        for l in range(len(w)-1):
            if w[l] == str1 and w[l+1] == str2:
                f += 1
    return f
#create a |v|*|v| matrix
#Initializes the matrix with frequency of each pair of symbols
def InitialSetupMatrix(corpus, FreqMatrix, v):
    for i in range(len(v)):
        for j in range(len(v)):
            FreqMatrix[i,j] = CalcFreq(corpus, v[i], v[j])
#merge two symbols in the corpus
def MergeLetters(corpus, str1, str2):
    for w in corpus:
        for l in range(len(w)-1):
            if w[l] == str1 and w[l+1] == str2:
                del w[l]
                del w[l]
                w.insert(l, (str1+str2))
    return corpus
#add new pair of symbols to vocabulary and extend the frequency matrix
def ExtendMatrix(FreqMatrix, v, str1, str2, corpus):
    NewStr = str1+str2
    if NewStr in v:
        return FreqMatrix
    print(NewStr)
    MergeLetters(corpus, str1, str2)
    v.append(NewStr)
    n = FreqMatrix.shape[0]
    temp = np.full((n+1,n+1),-1)
    temp[:n,:n] = FreqMatrix
    s1Index = v.index(str1)
    s2Index = v.index(str2)
    for i in range(n+1):
        temp[i,n] = CalcFreq(corpus, v[i], NewStr)
        temp[n,i] = CalcFreq(corpus, NewStr, v[i])
        temp[i,s1Index] = CalcFreq(corpus, v[i], str1)
        temp[s1Index,i] = CalcFreq(corpus, str1, v[i])
        temp[i,s2Index] = CalcFreq(corpus, v[i], str2)
        temp[s2Index,i] = CalcFreq(corpus, str2, v[i])
    return temp
#Byte Pair Encoding Algorithm
def BytePairEncoding(Input_text):
    v = CreateVocab(Input_text)
    corpus = CleanCorpus(Input_text)
    FreqMatrix = np.full((len(v),len(v)),-1)
    InitialSetupMatrix(corpus, FreqMatrix, v)
    MaxIndex = np.unravel_index(FreqMatrix.argmax(), FreqMatrix.shape)
    while not np.max(FreqMatrix) == 0:
        temp = ExtendMatrix(FreqMatrix, v, v[MaxIndex[0]], v[MaxIndex[1]], corpus)
        MaxIndex = np.unravel_index(FreqMatrix.argmax(), FreqMatrix.shape)
        del FreqMatrix
        FreqMatrix = temp
    return v
    
 #%%   
corpus = 'low lower newest low lower newest low widest newest low widest newest low widest newest'
vocab = BytePairEncoding(corpus)