import pandas as pd 
import numpy as np
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocessing(userQuery):      
    letters_only = re.sub("[^a-zA-Z\\d]", " ", userQuery)
    words = letters_only.lower().split()                   
    return( " ".join(words ))


#read data
train = pd.read_csv('Sheet1.csv')

#create tfidf
tfidf_vectorizer = TfidfVectorizer(min_df=1, norm='l2',ngram_range=(1, 1), stop_words='english')

query_features = train['QUERY']
new_query = [preprocessing(query) for query in query_features]	

features = tfidf_vectorizer.fit_transform(new_query).toarray()

model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(features, train['INTENT'])

print("score: ", model.oob_score)

userQuery = "type "
userQueryList=[]
userQueryList.append(preprocessing(userQuery))
utfidf = tfidf_vectorizer.transform(userQueryList)

print(" prediction: ", model.predict(utfidf))
print(" prediction scores: ", model.predict_proba(utfidf))



