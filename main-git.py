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


nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS



def jaccard_similarity(list1, list2):
	s1 = set(list1)
	s2 = set(list2)	
	if len(s1.union(s2)):
		return float(len(s1.intersection(s2)) / len(s1.union(s2)))
	else:
		return 0

def cosine_similarity(s1, s2, d1, d2):
	tf_s1 = {}
	tf_s2 = {}
	for w in s1:
		if w in tf_s1:
			tf_s1[w] += 1
		else:
			tf_s1[w] = 1
			
	for w in s2:
	    if w in tf_s2:
	        tf_s2[w] += 1
	    else:
	        tf_s2[w] = 1
	tf_idf_s1 = {}
	tf_idf_s2 = {}
	for w in tf_s2:
		tf_idf_s2[w] = tf_s2[w]/float(d2[w])
	for w in tf_s1:
		tf_idf_s1[w] = tf_s1[w]/float(d1[w])
	zigma = 0
	for i in tf_s1:
		if i in tf_s2:
			zigma += tf_idf_s1[i]*tf_idf_s2[i]
	r1 = 0
	r2 = 0
	for w in tf_s1:
		r1 += (tf_s1[w]/float(d1[w])) ** 2
	for w in tf_s2:
		r2 += (tf_s2[w]/float(d2[w])) ** 2
	if r1 != 0 and r2 !=0:
		cosine_sim = zigma/float(math.sqrt(r1) * math.sqrt(r2))
	else:
		cosine_sim = 0
	return cosine_sim

def identitysimilarity(s1, s2):
	tfs1={}
	tfs2={}
	for i in s1:
		if i in tfs1:
			tfs1[i]+=1
		else:
			tfs1[i]=1
	for i in s2:
		if i in tfs2:
			tfs2[i]+=1
		else:
			tfs2[i]=1
	l_s1=len(s1)
	l_s2=len(s2)
	absolute_value=abs(l_s1-l_s2)
	idf={}
	for k in tfs1:
		if k in tfs2:
			idf[k]=tfs1[k]+tfs2[k]
		else:
			idf[k]=tfs1[k]
	for k in tfs2:
		if k not in idf:
			idf[k]=tfs2[k]
	zigma=0
	for i in tfs1:
		if i in tfs2:
			zigma+=idf[i]+absolute_value
	if zigma==0:
		fraction_1=0
	else:
		fraction_1=1/float(zigma)
	sec_fraction=0
	for i in tfs1:
		if i in tfs2:
			sec_fraction+=idf[i]/float(1+abs(tfs1[i]-tfs2[i]))
	prod_ident=fraction_1*sec_fraction
	return prod_ident




def similarity(doc1, doc2):
	doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
	d1 = nlp(doc1)
	d2 = nlp(doc2)
	idf_d1 = {}
	idf_d2 = {}
	cosine_score = []
	jaccard_score = []
	identity_score=[]
	for tok in d1:
		if tok.is_stop == False and tok.text.isalpha() == True :
			t = tok.lemma_.lower()
			if t in idf_d1:
				idf_d1 [t] += 1
			else:
				idf_d1 [t] = 1
	for tok in d2:
		if tok.is_stop == False and tok.text.isalpha() == True :
			t = tok.lemma_.lower()
			if t in idf_d2:
				idf_d2 [t] += 1
			else:
				idf_d2 [t] = 1
	# print(idf_d1)
	for sentence1 in d1.sents:
		score_list1 = []
		score_list2 = []
		score_list3 = []
		s1 = [token.lemma_.lower() for token in sentence1 if token.is_stop == False and token.text.isalpha() == True]
		for sentence2 in d2.sents:
			s2 = [token.lemma_.lower() for token in sentence2 if token.is_stop == False and token.text.isalpha() == True]
			score_list1.append(jaccard_similarity(s1, s2))
			score_list2.append(cosine_similarity(s1, s2, idf_d1, idf_d2))
			score_list3.append(identitysimilarity(s1, s2))
		jaccard_score.append(score_list1)
		cosine_score.append(score_list2)
		identity_score.append(score_list3)
	return jaccard_score, cosine_score, identity_score



path = sys.argv[1]
os.chdir(path)
path = os.getcwd()


doc_list = []
for file in os.listdir():
	if file.endswith(".story"):
		doc_list.append(file)
x=0
sumup=0
summ_doc={}
summ_dict={}
harmunic_dict={}
for file1 in doc_list:
	for file2 in doc_list:
		f1 = open(f"{path}/{file1}").read()
		f2 = open(f"{path}/{file2}").read()
		l1, l2, l3 = similarity(f1,f2)
		list_array=[]
		list_h=[]
		for i in range(0 , len(l1)-1):
			harmunic_list=[]
			summ_list=[]
			for j in range(0, len(l1[i])-1):
				x=(l1[i][j]+l2[i][j]+l3[i][j])/3.0
				sumup=(1/l1[i][j])+(1/l2[i][j])+(1/l3[i][j])
				H=3/sumup
				harmunic_list.append(H)
				summ_list.append(x)
			list_h.append(harmunic_list)
			list_array.append(summ_list)
		name= file1+"_"+file2
		harmunic_dict[name]=list_h
		summ_dict[name]=list_array
			# x=len(doc_list)
			# summ_doc.fromkeys(x[summ_dict[i]])
print(summ_dict)
print(harmunic_dict)
