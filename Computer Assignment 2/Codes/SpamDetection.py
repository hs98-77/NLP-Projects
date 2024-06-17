import numpy as np
from nltk.tokenize import word_tokenize
import re
import pandas as pd
mails = pd.read_excel("spam.xlsx")
text = np.array([[w for w in word_tokenize(str(s).lower()) if w.isalpha() or w in ['!', '£', '$', '€']] for s in list(mails['text'])])
label = np.array(mails['label'])
del mails
#%%
def Find(string):
    # findall() has been used 
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)      
    return [x[0] for x in url]
#%%
currency = list()
exclam = list()
url = list()
free = list()
urgent = list()
length = list()
for i in range(np.shape(text)[0]):
    if '£' in str(text[i]) or '$' in str(text[i]) or '€' in str(text[i]):
        currency.append(1)
    else:
        currency.append(0)
    if '!' in str(text[i]).lower():
        exclam.append(1)
    else:
        exclam.append(0)
    if Find(str(text[i])):
        url.append(1)
    else:
        url.append(0)
    if 'free' in str(text[i]).lower():
        free.append(1)
    else:
        free.append(0)
    if 'urgent' in str(text[i]).lower():
        urgent.append(1)
    else:
        urgent.append(0)
    length.append(len(str(text[i])))
del i      
features = np.column_stack([currency,exclam,url,free,length,urgent])
#%%
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, shuffle=True)
NB = GaussianNB()
NB.fit(X_train,y_train)
y_pred = NB.predict(X_test)
#%%
from sklearn.metrics import precision_score , recall_score , f1_score , accuracy_score , confusion_matrix
p_micro = precision_score(y_test, y_pred, average='micro')
r_micro = recall_score(y_test, y_pred, average='micro')
acc = accuracy_score(y_test, y_pred)
f1_micro = f1_score(y_test, y_pred, average='micro')
cm = confusion_matrix(y_test, y_pred)
p_class = precision_score(y_test, y_pred, average=None)
r_class = recall_score(y_test, y_pred, average=None)
f1_class = f1_score(y_test, y_pred, average=None)
p_macro = precision_score(y_test, y_pred, average='macro')
r_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
#%%