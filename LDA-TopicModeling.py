# from ctypes.wintypes import WORD
# from tracemalloc import stop
import sys
import os
import spacy
import numpy as np
import math

from spacy.tokenizer import Tokenizer
from spacy import displacy
from spacy import tokens
from spacy.tokens import DocBin
import random
import re
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS


from time import time
import re  
import nltk  
from sklearn.datasets import load_files  
import pickle
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import gensim
from gensim import corpora, models
import shutil

def preprocessing(doc1):
	doc=nlp(doc1)
	tokens=[token.text for token in doc]
	s1 = [token.lemma_.lower() for token in doc if token.is_stop == False and token.text.isalpha() == True]
	return s1
path = sys.argv[1]
os.chdir(path)
path = os.getcwd()
doc_list = []
document_list_dict={}
document_list=[]
for file in os.listdir():
	if file.endswith(".story"):
		doc_list.append(file)
for file1 in doc_list:
	f1 = open(f"{path}/{file1}")
	fr1 = f1.read()
	t1=preprocessing(fr1)
	document_list.append(t1)  
	document_list_dict[file1]=t1
	
save_path = sys.argv[2]
os.chdir(save_path)
save_path = os.getcwd()	


dictionary=corpora.Dictionary(document_list)
corpus = [dictionary.doc2bow(text) for text in document_list]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=10)
# print(ldamodel.print_topics(num_topics=3, num_words=40))
words_list=[]
for i in ldamodel.top_topics(corpus,topn=40):
	temp_list=[]
	for j in i[0]:
		temp_list.append(j[1])
	words_list.append(temp_list)

count = 1
for i in words_list:
	folder ="class"+str(count)
	os.mkdir(os.path.join(save_path,folder))
	for j,k in document_list_dict.items():
		common=set(i).intersection(set(k))
		common_num=len(common)
		print(count , common_num, j)
		if (common_num >= 16):
			source = os.path.join(path, j)
			if os.path.isfile(source):
				shutil.copy(source, folder)
	count += 1
	print()
					

			



