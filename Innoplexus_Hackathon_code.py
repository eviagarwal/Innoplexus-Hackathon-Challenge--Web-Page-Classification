import numpy as np
import random
import urllib2
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from bs4 import Beautifulsoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


#### Created by Evi Agarwal ####

df=pd.read_csv("train/train.csv")

fixed_size=50
data_size=len(df)
categories={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
for idx, r in df.iterrows():
    if(r['Tag']=="news"):
        categories["news"].append(r['Webpage_id'])
   
    if(r['Tag']=="publication"):
        categories["publication"].append(r['Webpage_id'])
        
    if(r['Tag']=="others"):
        categories["others"].append(r['Webpage_id'])  
        
    if(r['Tag']=="guidelines"):
        categories["guidelines"].append(r['Webpage_id'])
        
    if(r['Tag']=="clinicalTrials"):
        categories["clinicalTrials"].append(r['Webpage_id'])
    
    if(r['Tag']=="profile"):
        categories["profile"].append(r['Webpage_id'])
        
    if(r['Tag']=="conferences"):
        categories["conferences"].append(r['Webpage_id'])
        
    if(r['Tag']=="forum"):
        categories["forum"].append(r['Webpage_id'])
        
    if(r['Tag']=="thesis"):
        categories["thesis"].append(r['Webpage_id'])
         
for key, value in categories.iteritems():
    k=(len(value)*fixed_size)/data_size
    value=random.sample(value,k)
    categories[key]=value

cat_train_data={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
cat_test_data={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
          
# function webScraper uses Beautifulsoup to parse html into text and take all relevant data from the p tag in HTML.
def webScraper(html):
    html_data = Beautifulsoup(html)
    req_data=html_data.find_all(['p'])
    res=""
    for i in req_data:
        res=res+str(i)
    return res
for key, val in categories.iteritems():
    for idx in range(len(val)):
        if(idx%3==0):
            cat_test_data[key].append(val[idx])
        else:
            cat_train_data[key].append(val[idx])


train_data=[]
test_data=[]
for key, val in cat_train_data.iteritems():
    for idx in range(len(val)):
        train_data.append(val[idx])
        
for key, val in cat_test_data.iteritems():
    for idx in range(len(val)):
        test_data.append(val[idx])
        
output_tar=[]
for idx,r in df.iterrows():
    if r['Webpage_id'] in train_data:
        output_tar.append(r['Tag'] )
        
test_targets=[]

for idx,r in df.iterrows():
    if r['Webpage_id'] in test_data:
        test_targets.append(r['Tag'])


min_train_list=[]
for i in range(79345):
    if i+1 not in train_data:
        min_train_list.append(i+1)
min_test_list=[]
for i in range(79345):
    if i+1 not in test_data:
        min_test_list.append(i+1)
s1=sorted(min_train_list)
skip2=sorted(min_test_list)
data=[]
testData=[]

def fun1(d):
    for html in d['Html']:
        data.append(webScraper(html))


def pro2(d):
    for html in d['Html']:
        testData.append(webScraper(html))

batch_size=5000
for batch in pd.read_csv("train/html_data.csv",chunksize=batch_size,skiprows=s1):
    fun1(batch)


for batch in pd.read_csv("train/html_data.csv",chunksize=batch_size,skiprows=s2):
    fun2(batch)

cntVector = CountVectorizer()
train_cnt = cntVector.fit_transform(data)
trans = TfidfTransformer()
train_trans = trans.fit_transform(train_cnt)

mNB_clf = MultinomialNB().fit(train_trans, output_tar)

classifier_txt = Pipeline([('v', CountVectorizer()), ('trans_tf_idf', TfidfTransformer()), ('mNB_clf', MultinomialNB())])

classifier_txt = classifier_txt.fit(data, output_tar)


pred = classifier_txt.predict(testData)


#building a svm classifier pipeline.
svm_classifier = Pipeline([('v', CountVectorizer()), ('trans_tf_idf', TfidfTransformer()),
                         ('classifier-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])

                         
# Training using SVM
svm_classifier = svm_classifier.fit(data, output_tar)
# Testing using SVM
pred_svm = svm_classifier.predict(testData)



predicted=pd.read_csv("train/test_nvPHrOx.csv")
predicted_wbpg=predicted['Webpage_id'].tolist()

predictions_min=[]
for i in range(79345):
    if i+1 not in preddoc:
        predictions_min.append(i+1)

s=sorted(predictions_min)
final_pred=pd.read_csv("train/html_data.csv",skiprows=s)
predictions=[]

def append_pred(d):
    for html in d['Html']:
        predictions.append(webScraper(html))


for batch in pd.read_csv("train/html_data.csv",chunksize=batch_size,skiprows=s):
    append_pred(batch)
final = svm_classifier.predict(predictions)

output=pd.DataFrame({'Webpage_id':predicted_wbpg})
output['Tag']=final

# Generating the submission.csv file
output.to_csv("submission.csv",idx=False)

