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
Page_Rank_vertex_initialization={}
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
				vertex=file2+'_S'+str(len(summ_list))
				if vertex not in Page_Rank_vertex_initialization:
					Page_Rank_vertex_initialization[vertex]=random.uniform(0,1)
			list_h.append(harmunic_list)
			list_array.append(summ_list)
		name= file1+"_"+file2
		harmunic_dict[name]=list_h
		summ_dict[name]=list_array
# print(Page_Rank_vertex_initialization)
# print(summ_dict)
# PageRank
neighbor_dict={}
for k1 in Page_Rank_vertex_initialization.keys():
	doc1_name=k1[0:int(k1.find("_"))]
	s_num=k1[int(k1.find("_"))+2:len(k1)]
	list_neighbor=[]
	for k,d in summ_dict.items():
		doc2_name=k[0:int(k.find("_"))]
		doc3_name=k[int(k.find("_"))+1:len(k)+1]
		if (doc1_name==doc2_name): # and doc1_name != k[int(k.find("_"))+1:len(k)+1]
			for index,date in enumerate(d[int(s_num)-1]):
				tmp_dict={}
				tmp_dict[doc3_name+"_S"+str(index+1)]= date
				list_neighbor.append(tmp_dict)
	neighbor_dict[k1]=list_neighbor

# print(neighbor_dict)
# print(dictionary_value)
# print(summ_dict)
# print(harmunic_dict)


n=0
sum_list_PR_values={}
for i , j in neighbor_dict.items():
	n=0
	for k in j:
		for l , m in k.items():
			n+=m
			sum_list_PR_values[i]=n
# print(sum_list_PR_values)
whole_zigma_vertex_dictionary={}
g=0
for i , j in neighbor_dict.items():
	for k in j:
		for m , n in k.items():
			for p , q in sum_list_PR_values.items():
				for u , v in Page_Rank_vertex_initialization.items():
					if (m==p):
						if (p==u):
							g=(n/float(q))*v
							whole_zigma_vertex_dictionary[i]=g
# print(whole_zigma_vertex_dictionary)
p=1
zigma_second_fraction={}
for m , n in whole_zigma_vertex_dictionary.items():
	p=n*0.85
	zigma_second_fraction[m]=p
# print(zigma_second_fraction)
PR_final_list=[]
for i , j in neighbor_dict.items():
	for k,l in zigma_second_fraction.items():
		if(i==k):
			N=len(j)
			PR_formula=(0.15/float(N))*l
			PR_final_list.append(PR_formula)
# print(PR_final_list)
# End_Of_PR


#HITS Algorithm
old_HITS={}
for i , j in sum_list_PR_values.items():
	old_HITS[i]=1

k=10
for w in range(0, k):
	sum_new_HITS=0
	new_HITS={}
	for k, l in neighbor_dict.items():
		sum_new_HITS=0
		for m in l :
			for i,j in m. items():
				for p,q in old_HITS.items():
					if (p==i):
						sum_new_HITS+=j*q
						new_HITS[k]=sum_new_HITS



	new_HITS_normalize={}
	new_sum_value=0
	for i, j in new_HITS.items():
		new_sum_value=sum(new_HITS.values())
		new_HITS_normalize[i]=j/float(new_sum_value)
		old_HITS=new_HITS_normalize.copy()

					
	#HITS_Convergence
	Epsilon=0.5
	zigma_convergence={}
	for i, j in new_HITS_normalize.items():
		for p,q in old_HITS.items():
			if (i==p)and ((j-q)<Epsilon) :
				zigma_convergence[i]=j
			
				





# print(new_HITS)
print(zigma_convergence)
# print(old_HITS)
# print(new_HITS_normalize)
# print(HITS_inial)
# print(old_HITS)
# print(HITS_normalize)


