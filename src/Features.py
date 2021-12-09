import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import datetime
import gensim
import emoji


class Features:
    # class attr
    
    wnl = WordNetLemmatizer()
    reFilter = re.compile(
        "(http.*?([ ]|\|\|\||$))|" + r"((:|;).)|" + 
        '([' + re.escape(string.punctuation) + '])|' + 
        '((\[|\()*\d+(\]|\))*)|' + '''([’‘“\.”'"`…–])|''' +
        '([^(\w|\s)])|' + '(gt|lt)|'
    )
    
    def __init__(self, series, stopwords):
        
        # inputs
        self.raw_series = series
        self.stop_words = stopwords
        
        # cleaned
        self.cleaned_series = None
        
        # tfidf
        self.vectorizer = CountVectorizer(min_df=0.0005)
        self.tfidf = TfidfTransformer()
        
        # topics 
        self.lda_model = None
        # todo
        
        # emoticons
        # todo
    
    # CLEANING
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
        
        words = [word for word in words if not word in self.stop_words]
        
        
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
        
    
    def clean(self):
        self.cleaned_series = self.raw_series.apply(self.clean_sentence)
        
        
    # TFIDF 
    def build_tfidf(self):
        assert(feat.cleaned_series.any())
                
        X = self.vectorizer.fit_transform(
            self.cleaned_series)
        
        self.tfidf.fit(X)
    
    def get_tfidf(self, series):
        return self.tfidf.transform(
            self.vectorizer(series.apply(self.clean_sentence)))
    
    # EMOTICONS
    def extract_emoticons(self,s):
        '''
        extract all the emoticons from a string s
        should return an counter or an array? 
        '''
        return (c for c in s if c in emoji.UNICODE_EMOJI['en']) # extract emojis
    
    def build_emoticons(self):
        self.emoticon_series = self.raw_series.apply(self.extract_emoticons)
        # todo: transform this into a numpy array? that could be read into other dataset. 
    
    def get_emoticons(self, series):
        return self.raw_series.apply(self.extract_emoticons)
        # todo: transform this into a numpy array? that could be read into other dataset. 


    # TOPICS

    def build_topics(self):
        processed_doc = self.cleaned_series.apply(lambda row: nltk.word_tokenize(row['sentences']), axis=1)
        dictionary = gensim.corpora.Dictionary(processed_doc)
        dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_doc]
        self.lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 200, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)
        
    def get_topics(self,series):
        return self.lda_model
        
    
    
    

    # Aggregate
    # TODO: update after the features are completed.         
    def get_features(self,series):
        # TODO, once topics and emoticons are finished, add them here. 
        return self.get_tfidf(self,series)
    
    def save_self(self):
        # this will not save the raw_series and cleaned series
        
        suffix = str(datetime.date.today())
        prefix = '../models/'
        
        d = (self.raw_series, self.cleaned_series)
        self.raw_series = None
        self.cleaned_series = None
        self.emoticon_series = None

        with open('../models/features' +str(datetime.date.today()) + '.model', 'wb') as f:
            pickle.dump(self ,f)
        
        with open(prefix +  "data_series" + suffix + ".model",'wb') as f:
            pickle.dump(d,f)