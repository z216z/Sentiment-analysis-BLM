#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import string
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import multidict as multidict
import re
import wordcloud
from os import path
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN,AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords as sw
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.svm import SVC
import spacy
from textblob import TextBlob


# In[2]:


documents = pd.read_json('development.jsonl', lines=True)
documents_res=pd.read_json("evaluation.jsonl",lines=True)


# In[3]:


class LemmaTokenizer(object):
    def __init__(self): 
        self.lemmatizer = WordNetLemmatizer()
        
    def __call__(self, documents): 
        lemmas = [] 
        for t in word_tokenize(documents):
            t = t.strip() 
            lemma = self.lemmatizer.lemmatize(t) 
            lemmas.append(lemma) 
        return lemmas


# In[26]:


lemmaTokenizer = LemmaTokenizer() 
stopwords=sw.words('english')+['ha','le','u','wa']+["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would']
vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer,stop_words=stopwords,use_idf=True,ngram_range=(1,3))
X=documents["full_text"]
Y=documents["class"]
X=vectorizer.fit_transform(X)


# In[27]:


#Generate train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[28]:


#model=RandomForestClassifier()
model  = LinearSVC(random_state=42,tol=1e-10,loss="hinge",C=4,dual=True,max_iter=100000)
#model = SVC(random_state=0,tol=1e-1,C=0.9)
#model= MultinomialNB(alpha=0.11,fit_prior='False')
#model=BernoulliNB(alpha=0.11)
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
y_pred


# In[29]:


print(accuracy_score(y_pred,y_test))


# In[8]:


X_res=documents_res
X_res=vectorizer.transform(X_res["full_text"])


# In[9]:


model.fit(X,Y)
res_test=model.predict(X_res)


# In[111]:


#Data Exploration

positive=documents[documents["class"]==1]["class"].count()
negative=documents[documents["class"]==0]["class"].count()
perc_pos=positive/(positive+negative)*100;
perc_neg=(100-perc_pos)
print((perc_pos),(perc_neg))

#Histogram rapresentation




x = np.arange(2)  
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, [perc_pos,perc_neg], width,color=["orange","blue"])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage')

ax.set_title('Percentage of positive and negative posts')
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)

fig.tight_layout()

plt.show()

#WordCloud
text_pos=" ".join(pos_post for pos_post in documents[documents["class"]==1]["full_text"])
text_neg=" ".join(neg_post for neg_post in documents[documents["class"]==0]["full_text"])
wordcloud_pos=WordCloud(background_color="white").generate(text_pos)
wordcloud_neg=WordCloud(background_color="white").generate(text_neg)
plt.imshow(wordcloud_pos)
plt.figure()
plt.imshow(wordcloud_neg)


# In[10]:


with open('result.csv', 'w') as file:
    file.write('Id,Predicted\n')
    for i in range(len(res_test)):
        file.write(f"{i},{res_test[i]}\n")

