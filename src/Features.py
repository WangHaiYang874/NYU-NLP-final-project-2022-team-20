import re
import string

import nltk
from nltk.stem import WordNetLemmatizer

import gensim
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess

import pickle
import datetime
import numpy as np

class Features:
    '''
    initialize this object with an array of corpus, and a file storing a list of stopwords.
    after initialization, run self.build() to build the model
    Then, for any array of corpus, use self.get_features(corpus) to get the features calculated from this corpus. 
    '''
    # class attr
    
    wnl = WordNetLemmatizer()
    # reFilter = re.compile(
    #     "(http.*?([ ]|\|\|\||$))|" + r"((:|;).)|" + 
    #     '([' + re.escape(string.punctuation) + '])|' + 
    #     '((\[|\()*\d+(\]|\))*)|' + '''([’‘“\.”'"`…–])|''' +
    #     '([^(\w|\s)])|' + '(gt|lt)|'
    # )
    
    
    def __init__(self, series, stopwords):
        
        # data
        self.raw_series = series
        self.stopwords =  open(stopwords,'r').read().split()
        with open('../data/Emoticon_Dict.p', 'rb') as fp:
            self.emoticons = sorted(list(pickle.load(fp).keys()))
        self.cleaned_series = None
        
        # models
        
        self.vectorizer = CountVectorizer(max_features=4000)
        self.tfidf = TfidfTransformer()
        self.lda_model = None
        self.dictionary = None
    
    
    # BUILDING
    def clean_sentence(self,sentence):
        '''This function cleans a sentence
        - lower every char, remain only ascii characters
        - lemmatize every words
        - filter some stopwords
        - remove email, http links, punctuations, etc
        '''

        words = nltk.tokenize.word_tokenize(sentence.lower().encode('ascii', 'ignore').decode())
        
        for i in range(len(words)):
            
            word = words[i]
            word = Features.wnl.lemmatize(word,'n')
            word = Features.wnl.lemmatize(word,'v')
            word = Features.wnl.lemmatize(word,'a')
            
            words[i] = word
        
        words = [word for word in words if not word in self.stopwords]
        
        sentence = ' '.join(words)
        sentence = re.sub("@\S+", " ", sentence)
        sentence = re.sub("https*\S+", " ", sentence)
        sentence = re.sub("#\S+", " ", sentence)
        sentence = re.sub("\'\w+", '', sentence)
        sentence = re.sub('[%s]' % re.escape(string.punctuation), ' ', sentence)
        sentence = re.sub(r'\w*\d+\w*', '', sentence)
        sentence = re.sub('\s{2,}', " ", sentence)
                
        return sentence
            
    def build_tfidf(self):
        '''build the tfidf score with the cleaned series'''
        self.tfidf.fit(self.vectorizer.fit_transform(
            self.cleaned_series))
        
    def build_topics(self):
        
        processed_doc = [simple_preprocess(doc) for doc in self.cleaned_series]
        self.dictionary = gensim.corpora.Dictionary(processed_doc)
        self.dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in processed_doc]
        self.lda_model = gensim.models.LdaMulticore(
            bow_corpus, 
            num_topics = 300, 
            id2word = self.dictionary,                                    
            passes = 10,
            workers = 4)
            
    def save_self(self):

        suffix = str(datetime.date.today())
        prefix = '../models/features'
        
        self.raw_series = None
        self.cleaned_series = None
        
        with open(prefix + suffix + '.model', 'wb') as f:
            pickle.dump(self ,f)
            
        return prefix + suffix + '.debug', 'wb'

    def build_model(self):
        print('cleaning the copora')
        self.cleaned_series = self.raw_series.apply(self.clean_sentence)
        print('building tfidf model')
        self.build_tfidf()
        print('building lda topic model')
        self.build_topics()
        print('model built, saving it')
        path = self.save_self()
        print('model saved at:')
        return path

    # GETTING FEATURES
    def get_tfidf(self, cleaned_series):
        return self.tfidf.transform(
            self.vectorizer.transform(cleaned_series))

    def extract_emoticons(self,s):
        '''
        extract all the emoticons from a string s
        should return an counter or an array? 
        '''
        ret = [len(re.findall(emoticon, s)) for emoticon in self.emoticons]
        ret.append(sum(ret))
        return ret
    
    def get_emoticons(self, raw_series):
        return np.array(list(map(self.extract_emoticons, raw_series.values)))
        
    def get_topics(self,series):
        
        processed_doc = [simple_preprocess(doc) for doc in series]
        bows = [self.dictionary.doc2bow(doc) for doc in processed_doc]
        ret = []
        for doc in bows:
            prob = self.lda_model.get_document_topics(doc, minimum_probability=0)
            prob.sort(key=lambda x:x[0])
            prob = [i[1] for i in prob]
            ret.append(prob)
        return np.array(ret)
        
    def get_features(self,series):
        print('this could take some time, please be patient')
        cleaned_series = series.apply(self.clean_sentence)
        print('getting tfidfs')
        tfidfs = self.get_tfidf(cleaned_series)
        print('getting emoticons')
        emoticons = self.get_emoticons(series)
        print('getting topics')
        topics = self.get_topics(cleaned_series)
        
        print('aggregating')
        return [np.concatenate(a,axis=None) for a in zip(tfidfs,emoticons,topics)]