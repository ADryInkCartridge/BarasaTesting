from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader import WordNetError
from nltk.corpus import wordnet as wn
import nltk
from nlp_id.tokenizer import Tokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from nlp_id.stopword import StopWord 
from nltk.corpus import stopwords
import words as w
import numpy as np
import pandas as pd
import spacy
import re


class SentiSynset:
    def __init__(self, pos_score, neg_score, synset):
        self._pos_score = pos_score
        self._neg_score = neg_score
        self._obj_score = 1.0 - (self._pos_score + self._neg_score)
        self.synset = synset


    def pos_score(self):
        return self._pos_score


    def neg_score(self):
        return self._neg_score


    def obj_score(self):
        return self._obj_score


    def __str__(self):
        """Prints just the Pos/Neg scores for now."""
        s = "<"
        s += self.synset.name() + ": "
        s += "PosScore=%s " % self._pos_score
        s += "NegScore=%s" % self._neg_score
        s += ">"
        return s

    def __repr__(self):
        return "Senti" + repr(self.synset)




class CustomSentiWordNet(object):
    def __init__(self):
        with open("barasa.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        # create empty 2d dict
        synsets = {}
        id_dict = {}
        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            synset_id = parts[0]

            if synset_id not in synsets:
                synsets[synset_id] = {}
            
            synset = {}
            id, lang, goodness, lemma, pos, neg = parts
            pos = float(pos)
            neg = float(neg)
            synsets[synset_id][lemma] = (pos, neg, 1 - (pos + neg))
            id_dict[lemma] = synset_id

        self.lemma_dict = id_dict
        self.synsets = synsets
        self.not_found = {}
    
    def _get_synset(self, synset_id):
        # helper function to map synset_id to synset
        synsets = self.synsets[synset_id]
        return synsets
        
        
    
    def _get_pos_file(self, pos):
        # helper function to map WordNet POS tags to file names
        if pos == 'n':
            return 'noun'
        elif pos == 'v':
            return 'verb'
        elif pos == 'a' or pos == 's':
            return 'adj'
        elif pos == 'r':
            return 'adv'
        else:
            raise WordNetError('Unknown POS tag: {}'.format(pos))
    
    
    def senti_synset(self, synset_id):
        pos_score,neg_score,obj_score = self.synsets[synset_id]
        synset = self._get_synset(synset_id)
        return SentiSynset(synset, pos_score, neg_score)
    
    def calculate_sentiment(self,tokens):
        pos = []
        neg = []
        for token in tokens:
            if token not in self.lemma_dict:
                self.not_found[token] = self.not_found.get(token, 0) + 1
                continue
            synsets = self.synsets[self.lemma_dict[token]][token]
            pos_score, neg_score, obj_score = synsets
            pos.append(pos_score)
            neg.append(neg_score)
        return pos, neg
    
    def get_not_found(self):
        return self.not_found


class normalizer():
    def __init__(self):
        nltk.download('stopwords')
        stopwords_sastrawi = StopWordRemoverFactory()
        stopwords_nlpid = StopWord() 
        stopwords_nltk = stopwords.words('indonesian')
        stopwords_github = list(np.array(pd.read_csv("stopwords.txt", header=None).values).squeeze())
        more_stopword = w.custom_stopwords
        data_stopword = stopwords_sastrawi.get_stop_words() + stopwords_nlpid.get_stopword() + stopwords_github + stopwords_nltk + more_stopword 
        data_stopword = list(set(data_stopword))

        # Only use 'rt' as stopwords
        data_stopword = list(set(data_stopword))

        # Combine slang dictionary
        import json
        with open('slang.txt') as f:
            data = f.read()
        data_slang = json.loads(data) 

        with open('sinonim.txt') as f:
            data = f.readlines()
        for line in data:
            word = line.split('=')
            data_slang[word[0].strip()] = word[1].strip()

        # print(data_slang)
        more_dict = w.custom_dict
        data_slang.update(more_dict)

        self.stopwords, self.slang = data_stopword, data_slang
        self.tokenizer = Tokenizer()


    def normalize(self,text):
        text = text.lower()
  
        # Change HTML entities
        text = text.replace('&amp;', 'dan')
        text = text.replace('&gt;', 'lebih dari')
        text = text.replace('&lt;', 'kurang dari')
        
        # Remove url
        text = re.sub(r'http\S+', 'httpurl', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', ' ', text)
        
        # Replace @mentions with 'user'
        text = re.sub(r'@\w+', 'user', text)

        # Remove non-letter characters
        text = re.sub('[^a-zA-z]', ' ', text)

        # Remove excess space
        text = re.sub(' +', ' ', text)
        text = text.strip()

        result = []
        word_token = self.tokenizer.tokenize(text) # Tokenize words
        for word in word_token:
            word = word.strip().lower() # Case Folding to Lower Case
            if word in self.slang:
                word = self.slang[word]
            if word not in self.stopwords: # Stopwords removal
                result.append(word)
            else:
                continue
        return result

        
    



# swn = CustomSentiWordNet()
# normalized = normalizer()



# text = "Pada hari minggu ku turut ayah ke kota."


# pos, neg = swn.calculate_sentiment(normalized.normalize(text))
# print(sum(pos)/len(pos), sum(neg)/len(neg))

# # print positive or negative score using ternal operator
# print("Positive" if sum(pos)/len(pos) > sum(neg)/len(neg) else "Negative")


# print(synsets["00001740-a"])
# swn = CustomSentiWordNet(synsets)
# swn.senti_synset("00001740-a")

# text = "Pada hari minggu ku turut ayah ke kota."
# tokenizer = Tokenizer()
# tokens = tokenizer.tokenize(text)

# pos_scores = []
# neg_scores = []
# for token in tokens:
#     synsets = custom_swn.synsets_for_word(token)
#     for synset in synsets:
#         senti_synset = custom_swn.senti_synset(synset.name())
#         pos_scores.append(senti_synset.pos_score())
#         neg_scores.append(senti_synset.neg_score())

# if len(pos_scores) > 0 or len(neg_scores) > 0:
#     avg_pos_score = sum(pos_scores) / len(pos_scores)
#     avg_neg_score = sum(neg_scores) / len(neg_scores)
#     if avg_pos_score > avg_neg_score:
#         print("Positive sentiment")
#     elif avg_pos_score < avg_neg_score:
#         print("Negative sentiment")
#     else:
#         print("Neutral sentiment")
# else:
#     print("No sentiment detected")