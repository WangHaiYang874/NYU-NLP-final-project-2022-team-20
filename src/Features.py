import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import datetime
import gensim
import numpy as np
from gensim.utils import simple_preprocess
from sklearn import random_projection

class Features:
    # class attr
    
    wnl = WordNetLemmatizer()
    # reFilter = re.compile(
    #     "(http.*?([ ]|\|\|\||$))|" + r"((:|;).)|" + 
    #     '([' + re.escape(string.punctuation) + '])|' + 
    #     '((\[|\()*\d+(\]|\))*)|' + '''([’‘“\.”'"`…–])|''' +
    #     '([^(\w|\s)])|' + '(gt|lt)|'
    # )
    
    
    def __init__(self, series, stopwords):
        
        # datas
        self.raw_series = series
        self.stopwords =  open(stopwords,'r').read().split()
        with open('../data/Emoticon_Dict.p', 'rb') as fp:
            self.emoticons = sorted(list(pickle.load(fp).keys()))
        self.cleaned_series = None
        
        # models
        
        self.vectorizer = CountVectorizer(min_df=0.0005)
        self.tfidf = TfidfTransformer()
        self.dimen_red = TfidfTransformer()
        self.lda_model = None
        self.dictionary = None
        
    
    
    # BUILDING
    def clean_sentence(self,sentence):
        
        # lower and only ascii
        words = sentence.lower().encode('ascii', 'ignore').decode().split()
        
        # lemmatizing
        for i in range(len(words)):
            
            word = words[i]
            word = Features.wnl.lemmatize(word,'n')
            word = Features.wnl.lemmatize(word,'v')
            word = Features.wnl.lemmatize(word,'a')
            
            words[i] = word
        
        words = [word for word in words if not word in self.stopwords]
        
        
        # todo, replace all the regexp with a complied re as self.reFilter.
        # I tried with the reFilter before and it seems to be buggy. 
        
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
        assert(self.cleaned_series.any())
                
        X = self.vectorizer.fit_transform(
            self.cleaned_series)
        
        self.tfidf.fit(X)
        
    def build_topics(self):
        
        processed_doc = [simple_preprocess(doc) for doc in self.cleaned_series]
        self.dictionary = gensim.corpora.Dictionary(processed_doc)
        self.dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in processed_doc]
        self.lda_model = gensim.models.LdaMulticore(
            bow_corpus, 
            num_topics = 200, 
            id2word = self.dictionary,                                    
            passes = 10,
            workers = 4)

    def build_dimension_reduct(self):
        # todo
        self.dimen_red = random_projection.SparseRandomProjection(n_components=5000)
        
        # 你先把 cleaned 的 tfidf算出来, 然后对这个数据进行dimension reduce.        
            
    def save_self(self):
        # this will not save the raw_series and cleaned series
        
        suffix = str(datetime.date.today())
        prefix = '../models/features'
        
        # setting the data to none, avoiding to save a large model file with all the datas. 
        
        # this is the debug version of the model, where all datas are saved
        with open(prefix + suffix + '.debug', 'wb') as f:
            pickle.dump(self ,f)
        
        
        # setting all the datas to non to save only the models. 
        self.raw_series = None
        self.cleaned_series = None
        
        with open(prefix + suffix + '.model', 'wb') as f:
            pickle.dump(self ,f)
            
        return prefix + suffix + '.model', 'wb'
        
        
    def build_model(self):
        print('cleaning the copora')
        self.cleaned_series = self.raw_series.apply(self.clean_sentence)
        print('building tfidf model')
        self.build_tfidf()
        print('building lda topic model')
        self.build_topics()
        print('building dimension reduct')
        self.build_dimension_reduct()
        print('model built, saving it')
        path = self.save_self()
        print('model saved at:')
        return path
        
    

    # GETTING FEATURES
    def get_tfidf(self, cleaned_series):
        return self.tfidf.transform(
            self.vectorizer.transform(cleaned_series))
            
    def get_reduct_tfidf(self, cleaned_series):
        return self.dimen_red.fit_transform(self.get_tfidf(cleaned_series))
        

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
        
    def get_features(self,series, if_compress=False):
        
        cleaned_series = series.apply(self.clean_sentence)
        if if_compress:
            tfidfs = self.get_reduct_tfidf(cleaned_series)
        else:            
            tfidfs = self.get_tfidf(cleaned_series)
        
        emoticons = self.get_emoticons(series)
        topics = self.get_topics(cleaned_series)
        return list(zip(tfidfs,emoticons,topics))
        # TODO: I need to reshaping this