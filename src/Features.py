import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import datetime


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
        self.topics_list = None
        self.ngrams_vectorizer = CountVectorizer(ngram_range=(2,5),max_df=0.001)
            # todo: finish the topic selection
        
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
    
    def build_topics(self):
        # TODO
        # select the n-grams according to the PMI index, 
        # see p5 of Personality, Gender, and Age in the Language of Social Media: The Open-Vocabulary Approach 
        pass
    
    def build_emoticons(self):
        # TODO
        # the cleaned series has no emoticons, therefore one should use the raw sequence. 
        
        pass
    
    def get_tfidf(self, series):
        return self.tfidf.transform(
            self.vectorizer(series.apply(self.clean_sentence)))
    
    def get_topics(self,series):
        # TODO
        pass
    
    def get_emoticons(self,series):
        # TODO
        pass
        
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

        with open('../models/features' +str(datetime.date.today()) + '.model', 'wb') as f:
            pickle.dump(self ,f)
        
        with open(prefix +  "data_series" + suffix + ".model",'wb') as f:
            pickle.dump(d,f)