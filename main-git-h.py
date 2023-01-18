# from ctypes.wintypes import WORD
# from tracemalloc import stop
import sys
import os
import spacy
import numpy as np
from numpy.linalg import norm
import math
import pandas as pd
import collections
from spacy.tokenizer import Tokenizer
from spacy import displacy
from spacy import tokens
from spacy.tokens import DocBin
import random
import re
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS
from time import time
import nltk  
from sklearn.datasets import load_files  
# nltk.download('stopwords')  
import pickle  
# from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency

def get_sentens(doc, sent_number):
	d = nlp(doc)
	index=1
	for s in d.sents:
		if index == int(sent_numb) + 1:
			return s
		else:
			index += 1

def bow_cosine(d1,d2):
	bow_list=[]
	s1=nlp(d1)
	for w in s1.sents:
		for t in w:
			if t.lemma_.lower() not in bow_list:
				bow_list.append(t.lemma_.lower())
	for k in d2:
		x=nlp(k)
		for y in x.sents:
			for z in y:
				if z.lemma_.lower() not in bow_list:
					bow_list.append(z.lemma_.lower())
	vector_s1=[]
	vector_s2=[]
	for word in bow_list:
		for wd in s1.sents:
			if word==wd.lemma_.lower():
				vector_s1.append(1)
			else:
				vector_s1.append(0)
		for item in d2:
			for wr in item.sents:
				if word==wr.lemma_.lower():
					vector_s2.append(1)
				else:
					vector_s2.append(0)
	#nominator
	cosine_numinator=0
	for m in vector_s1:
		for n in vector_s2:
			cosine_numinator+=m*n
	#denominator
	power1_r1=0
	r1_denom=0
	for p in vector_s1:
		power1_r1+=p**2
		r1_denom=math.sqrt(power1_r1)
	power2_r2=0
	r2_denom=0
	for q in vector_s2:
		power2_r2+=p**2
		r2_denom=math.sqrt(power2_r2)
	cosine_denominator=r1_denom*r2_denom
	#cosine_fraction
	if cosine_denominator != 0:
		cosine_fraction= cosine_numinator/float(cosine_denominator)
	else:
		cosine_fraction=0
	return cosine_fraction

	

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

def topic_signature (releventPath, nonreleventPath):
	r_doc_list = []
	r_tf = {}
	all_tf = {}
	nonr_doc_list = []
	docs_entropy = {}
	terms_entropy = {}
	os.chdir(releventPath)
	for file in os.listdir():
		if os.path.isfile(file):
			doc_terms_entropy = {}
			doc_tf = {}
			r_doc_list.append(file)
			f1 = open(f"{file}").read()
			d1 = nlp(f1)
			for tok in d1:
				if tok.is_stop == False and tok.text.isalpha() == True :
					t = tok.lemma_.lower()
					if t in doc_tf:
						doc_tf[t] += 1
						all_tf[t] += 1
						r_tf[t] += 1
					else:
						doc_tf[t] = 1
						if t in r_tf:
							r_tf[t] += 1
							all_tf[t] += 1
						else :
							r_tf[t] = 1
							all_tf[t] = 1
							# if t in all_tf:
							# 	all_tf[t] += 1
							# else :
							# 	all_tf[t] = 1
			doc_total_terms = sum(doc_tf.values())
			for k,v in doc_tf.items():
				pi = v/doc_total_terms
				doc_terms_entropy[k] = -pi * math.log2(pi) 
			docs_entropy [file] = sum(doc_terms_entropy.values())
	os.chdir(nonreleventPath)
	for file in os.listdir():
		for file2 in os.listdir(file):
			if os.path.isfile(os.path.join(file,file2)):
				doc_terms_entropy = {}
				doc_tf = {}
				nonr_doc_list.append(file2)
				f1 = open(f"{file}/{file2}").read()
				d1 = nlp(f1)
				for tok in d1:
					if tok.is_stop == False and tok.text.isalpha() == True :
						t = tok.lemma_.lower()
						if t in doc_tf:
							doc_tf[t] += 1
							all_tf[t] += 1
						else:
							doc_tf[t] = 1
							if t in all_tf:
								all_tf[t] += 1
							else:
								all_tf[t] = 1
				doc_total_terms = sum(doc_tf.values())
				for k,v in doc_tf.items():
					pi = v/doc_total_terms
					doc_terms_entropy[k] = -pi * math.log2(pi) 
				docs_entropy [file] = sum(doc_terms_entropy.values())
	
	all_terms = sum(all_tf.values())
	all_entropy = 0
	for k,v in all_tf.items():
		pi = v/all_terms
		entropy = -pi * math.log2(pi)
		all_entropy += entropy 
		if k in r_tf:
			terms_entropy [k] = entropy
	likelihood = {}
	for k,v in terms_entropy.items():
		likelihood[k] = 2 * all_terms * (all_entropy - v )
	
	sorted_likelihood=dict(sorted(likelihood.items(),key=lambda item : item[1]))
	# print (sorted_likelihood)
	# print(len(sorted_likelihood), len(doc_terms_entropy) , len(terms_entropy))

	lh = np.array(list(sorted_likelihood.values()))
	mean = lh.mean()
	std = lh.std()
	size = len(lh)
	lowerband = mean - 10.83 * ( std / np.sqrt(size) )
	upperband = mean + 10.83 * ( std / np.sqrt(size) )
	# print(mean , lowerband, upperband)
	topicSig=[]
	for k,v in sorted_likelihood.items():
		if v > lowerband and v < upperband :
			topicSig.append(k)
	# print (topicSig)
	# print ( len (topicSig))
	return topicSig

def topic_signature_similarity (topics, s1, s2, d1, d2):
	tf_s1 = {}
	tf_s2 = {}
	for w in s1:
		if w in topics:
			if w in tf_s1:
				tf_s1[w] += 1
			else:
				tf_s1[w] = 1
			
	for w in s2:
		if w in topics:
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
		topic_sim = zigma/float(math.sqrt(r1) * math.sqrt(r2))
	else:
		topic_sim = 0
	return topic_sim

def similarity(doc1, doc2, topics):
	doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
	d1 = nlp(doc1)
	d2 = nlp(doc2)
	idf_d1 = {}
	idf_d2 = {}
	cosine_score = []
	jaccard_score = []
	identity_score=[]
	topic_score = []
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
	jacard_time = 0.0
	cosine_time = 0.0
	identity_time = 0.0
	for sentence1 in d1.sents:
		score_list1 = []
		score_list2 = []
		score_list3 = []
		score_list4 = []
		s1 = [token.lemma_.lower() for token in sentence1 if token.is_stop == False and token.text.isalpha() == True]
		for sentence2 in d2.sents:
			s2 = [token.lemma_.lower() for token in sentence2 if token.is_stop == False and token.text.isalpha() == True]
			ts1 = time()
			score_list1.append(jaccard_similarity(s1, s2))
			jacard_time += time()-ts1
			ts2 = time()
			score_list2.append(cosine_similarity(s1, s2, idf_d1, idf_d2))
			cosine_time += time()-ts2
			ts3 = time()
			score_list3.append(identitysimilarity(s1, s2))
			identity_time += time()-ts3
			score_list4.append(topic_signature_similarity(topics, s1, s2, idf_d1, idf_d2))
		jaccard_score.append(score_list1)
		cosine_score.append(score_list2)
		identity_score.append(score_list3)
		topic_score.append(score_list4)
	# if jacard_time == 0.0 : 
	# 	print (jaccard_score)
	# print ("jaccard time : {}".format(jacard_time))
	# print ("cosine time : {}".format(cosine_time))
	# print ("identity time : {}".format(identity_time))
	return jaccard_score, cosine_score, identity_score, topic_score

def hits (summ_dict):
	neighbors = {}
	hits = {}
	for k, d in summ_dict.items():
		doc1 = k[0:int(k.find("_"))]
		doc2 = k[int(k.find("_"))+1:len(k)+1]
		for index,data in enumerate(d):
			vertex = doc1 + '_S' + str(index)
			neighbor_list = []
			if vertex not in hits:
				hits[vertex] = 1
			else:
				neighbor_list = neighbors[vertex]
			for ind , data2 in enumerate(data):
				tmp = {}
				if data2 != 0 :
					tmp[doc2 + '_S' + str(ind)] = data2
					neighbor_list.append(tmp)
			neighbors[vertex] = neighbor_list
	replit = 10
	epsilon = 0.0000008
	new_hits = {}
	for r in range(replit):
		# print('hits repets : %d' % r , end='\r')
		convergence = 0
		for vi , hits_vi in hits.items():
			sum_vi = 0
			for n_vi in neighbors[vi]:
				vj = list(n_vi.keys())[0]
				dj = list(n_vi.values())[0]
				sum_vi += dj * hits[vj]
			new_hits[vi] = sum_vi
		#normalize and convergence
		norm = sum (new_hits.values())
		for vi , hits_vi in new_hits.items():
			new_hits[vi] = hits_vi / norm
			convergence += ( new_hits[vi] - hits[vi] ) **2 
		if convergence <= epsilon:
			break
		else:
			hits = new_hits.copy()
	# print()
	return new_hits

def pageRank (summ_dict):
	page_rank = {}
	neighbors = {}
	#initialaze PageRank and find neighbors
	for k, d in summ_dict.items():
		doc1 = k[0:int(k.find("_"))]
		doc2 = k[int(k.find("_"))+1:len(k)+1]
		for index,data in enumerate(d):
			vertex = doc1 + '_S' + str(index)
			neighbor_list = []
			if vertex not in page_rank:
				page_rank[vertex] = random.uniform(0,0.5)
			else:
				neighbor_list = neighbors[vertex]
			for ind , data2 in enumerate(data):
				tmp = {}
				if data2 != 0 :
					tmp[doc2 + '_S' + str(ind)] = data2
					neighbor_list.append(tmp)
			neighbors[vertex] = neighbor_list
	#calculate Page Rank
	
	n = len(neighbors.keys())
	# print(n)
	d = 0.85
	epsilon = 0.0000008
	replit = 10
	new_page_rank = {}
	for r in range(replit):
		# print('Page rank repets : %d' % r , end='\r')
		convergence = 0
		for vi , pr_vi in page_rank.items():
			# print("Vi", vi, pr_vi)
			sum_vi = 0
			for n_vi in neighbors[vi]:
				# print("n_vi", n_vi)
				if bool(n_vi):
					vj = list(n_vi.keys())[0]
					dj = list(n_vi.values())[0]
					sim_vj_vz = 0.0
					for n_vj in neighbors[vj]:
						# print("n_vj", n_vj)
						if bool(n_vj):
							vz = list(n_vj.keys())[0]
							dz = list(n_vj.values())[0]
							sim_vj_vz += dz
							# print(vi,vj,dj,vz,dz, sim_vj_vz)
					sum_vi += (dj / sim_vj_vz) * page_rank[vj]
					# print(sum_vi)
			# print("*******************************")
			new_page_rank[vi] = (1 - d) / n + d * sum_vi
		norm = sum(new_page_rank.values())
		for vi , PR_vi in new_page_rank.items():
			new_page_rank[vi] = PR_vi / norm
			convergence += (page_rank[vi] - new_page_rank[vi]) **2 
		if convergence <= epsilon:
			break
		else:
			page_rank = new_page_rank.copy()
	print()
	return page_rank


basePath = os.getcwd()
path = sys.argv[1]
# nonRePath = os.path.join(basePath, sys.argv[2])
nonRePath = sys.argv[2]
os.chdir(path)
path = os.getcwd()
doc_list = []

topics = topic_signature(path, nonRePath)

os.chdir(path)
for file in os.listdir():
	if os.path.isfile(file):
		doc_list.append(file)
x=0
sumup=0
summ_doc={}
summ_dict={}
harmunic_dict={}
doc1_count = 1
doc_list_size = len(doc_list)
# print(doc_list)
# Page_Rank_vertex_initialization={}
# page_rank={}
for file1 in doc_list:
	doc2_count = 1
	f1 = open(f"{path}/{file1}")
	fr1 = f1.read()
	for file2 in doc_list:
		ts1 = time()
		print('Document : %d / %d ( %d )' % (doc1_count, doc2_count, doc_list_size))
		doc2_count += 1
		f2 = open(f"{path}/{file2}")
		fr2 = f2.read()
		ts2 = time()
		l1, l2, l3, l4 = similarity(fr1,fr2, topics)
		# print("similarity time : {}".format(time() - ts2))
		list_array=[]
		list_h=[]
		for i in range(0 , len(l1)-1):
			harmunic_list=[]
			summ_list=[]
			a=0
			b=0
			c=0
			d=0
			h=0
			for j in range(0, len(l1[i])-1):
				x=(l1[i][j]+l2[i][j]+l3[i][j]+l4[i][j])/4.0
				summ_list.append(x)
				if (l1[i][j]==0):
					a=0
				else:
					a=1/l1[i][j]
				if (l2[i][j]==0):
					b=0
				else:
					b=1/l2[i][j]
				if (l3[i][j]==0):
					c=0
				else:
					c=1/l3[i][j]
				if (l4[i][j]==0):
					d=0
				else:
					d=1/l4[i][j]
				if (a+b+c+d==0):
					h=0
				else:
					h=4/(a+b+c+d)
				harmunic_list.append(h)
			list_h.append(harmunic_list)
			list_array.append(summ_list)
		name= file1+"_"+file2
		harmunic_dict[name]=list_h
		summ_dict[name]=list_array
		# f2.close()
		# print("loop time : {}".format(time() - ts1))
	# f1.close()
	doc1_count += 1
# print()
# print("finish similarity")


# print(Page_Rank_vertex_initialization)
# print(summ_dict)
# print(neighbor_dict)
# print(dictionary_value)
# print(summ_dict)
# print(harmunic_dict)

# Harmonic between PageRank and HITS
Harmonic_between_2_algo={}
# print("start page rank ....")
PR_final_list=pageRank(summ_dict)
# print(PR_final_list)
# print("start hits ....")
new_HITS_normalize=hits(summ_dict)
# print(new_HITS_normalize)
for i, j in PR_final_list.items():
	l = new_HITS_normalize[i]
	if j == 0 :
		n = 0.0 + (1/float(l))
		# doc_name = i[0:int(i.find("_"))]
		# sent_numb = i[int(i.find("_"))+2:len(i)]
		# l = new_HITS_normalize[i]
		# print(i , j , l)
		# f1 = open(f"{path}/{doc_name}").read()
		# print(get_sentens(f1 , sent_numb))
	elif l == 0 :
		n=(1/float(j)) + 0.0
		# doc_name = i[0:int(i.find("_"))]
		# sent_numb = i[int(i.find("_"))+2:len(i)]
		# l = new_HITS_normalize[i]
		# print(i , j , l)
		# f1 = open(f"{path}/{doc_name}").read()
		# print(get_sentens(f1 , sent_numb))
	else :
		n=(1/float(j))+(1/float(l))
	Harmonic_between_2_algo[i]=2/n


n_sentns = 5
harmonic_sorted = sorted(Harmonic_between_2_algo.items(), key=lambda x: x[1], reverse=True)
# print(harmonic_sorted)

for i in range(0, n_sentns):
	k,d = harmonic_sorted[i]
	doc_name = k[0:int(k.find("_"))]
	sent_numb = k[int(k.find("_"))+2:len(k)]
	f1 = open(f"{path}/{doc_name}").read()
	print (doc_name , sent_numb , d)
	s=get_sentens(f1,sent_numb)
	list_summary=[]
	if (len(list_summary)==0):
		list_summary.append(s)
	elif (bow_cosine(s,list_summary) < 0.7):
		list_summary.append(s)
	print(list_summary)







# print(new_HITS)
# print(Harmonic_between_2_algo)
# print(w)
# print(old_HITS)
# print(new_HITS_normalize)
# print(HITS_inial)
# print(old_HITS)
# print(HITS_normalize)