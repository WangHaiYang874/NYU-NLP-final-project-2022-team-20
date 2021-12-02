import csv
from numpy.core.fromnumeric import prod
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
import numpy as np
from functools import cmp_to_key
import math

# Using guide: mannully change the input


input = ["I like this doll so much"] #right here


def clean(df):
    for x in range(len(df)):
        df["body"][x] = df["body"][x].lower().encode('ascii', 'ignore').decode()
        df["body"][x] = ' '.join([word for word in df["body"][x].split(' ') if word not in stop_words])
        df["body"][x] = re.sub("@\S+", " ", df["body"][x])
        df["body"][x] = re.sub("https*\S+", " ", df["body"][x])
        df["body"][x] = re.sub("#\S+", " ", df["body"][x])
        df["body"][x] = re.sub("\'\w+", '', df["body"][x])
        df["body"][x] = re.sub('[%s]' % re.escape(string.punctuation), ' ', df["body"][x])
        df["body"][x] = re.sub(r'\w*\d+\w*', '', df["body"][x])
        df["body"][x] = re.sub('\s{2,}', " ", df["body"][x])
    return df


def IDF(queries):
    dict = {}
    for sentence in queries:
        sentence = sentence.split()
        words = set()
        for word in sentence:
            dict[word] = dict.get(word,0)+1
    return dict

def TF(queries):
    TF = [dict() for x in range(len(queries))]
    i = 0
    while i < len(queries):
        words = queries[i].split()
        for word in words:
            TF[i][word] = TF[i].get(word,0)+1
        i+=1
    return TF    

def comparator(a,b):
    if a[1] < b[1]:
        return 1
    if a[1] > b[1]:
        return -1
    return 0    


stop_words = stopwords.words("english")
df= pd.read_csv("data/development.csv")
df = clean(df)
bodies = df.loc[:,"body"]
queryIDF = IDF(input)
docIDF = IDF(bodies)
queryTF = TF(input)
docTF = TF(bodies)


i = 0
j = 0
while i < len(queryTF):
    j = 1
    solution = []
    while j < len(docTF):
        product = 0
        querySqaure = 0
        docSqaure = 0
        for key in queryTF[i].keys():
            queryScore = queryTF[i][key]*np.log(225/queryIDF[key])
            idf = 0
            if docIDF.get(key,0) != 0:
                idf = np.log(1400/docIDF[key])
            docScore = docTF[j].get(key,0)*idf
            product += queryScore*docScore
            querySqaure+= queryScore**2
            docSqaure+=docScore**2
        
        similarity = 0
        if querySqaure*docSqaure != 0:
            similarity = product/math.sqrt(querySqaure*docSqaure)
        solution.append([j,similarity])
        j+=1
    solution = sorted(solution,key=cmp_to_key(comparator))
    for data in solution:
        if data[1] != 0:
            print(str(data[0])+" "+str(data[1])+" "+df["type"][data[0]])
    i+=1
