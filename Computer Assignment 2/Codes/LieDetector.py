import  pandas as pd

train_final = pd.read_csv('SentimentalLIAR-master/train_final.csv')
test_final = pd.read_csv('SentimentalLIAR-master/test_final.csv')
#%%
train_label = list()
for l in train_final['label']:
    if l.lower() in ['half-true', 'mostly-true', 'true']:
        train_label.append('truth')
    else:
        train_label.append('lie')
test_label = list()
for l in test_final['label']:
    if l.lower() in ['half-true', 'mostly-true', 'true']:
        test_label.append('truth')
    else:
        test_label.append('lie')
del l
#%%
"""
Features:
    Length
    
"""
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(train_final[['fear','anger','joy','disgust','sad']], train_label)
#%%
y_pred = NB.predict(test_final[['fear','anger','joy','disgust','sad']])
#%%
from sklearn.metrics import accuracy_score 
acc = accuracy_score(test_label,y_pred)