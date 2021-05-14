import random
from preprocess import preprocess_raw_sent
from preprocess import sim_with_title
from preprocess import sim_with_doc
from preprocess import sim_2_sent
from preprocess import count_noun
from copy import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import nltk
import os.path
import statistics as sta
from rouge import Rouge
import re
import time
import os
import glob
from shutil import copyfile
import pandas as pd
import math
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


     
def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
	docs = list()  
	for name in os.listdir(directory):
		filename = directory + '/' + name
		doc = load_a_doc(filename)
		docs.append((doc, name))
	return docs

def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",", "'", "(", ")")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 4:
        return 'None'
    return text

def evaluate_rouge(raw_sentences, abstract):
    rouge_scores = []
    for index, sent in enumerate(raw_sentences):
        try:
            rouge = Rouge()
            scores = rouge.get_scores(sent, abstract, avg=True)
            rouge1f = scores["rouge-1"]["f"]
        except Exception:
            rouge1f = 0 
        rouge_scores.append((sent, rouge1f, index))
    return rouge_scores
  

def start_run(processID, sub_stories, save_path, order_params):
   
    for example in sub_stories:
        start_time = time.time()
        raw_sents = re.split("\n\n", example[0])[1].split(' . ')
        title = re.split("\n\n", example[0])[0] 
        abstract = re.split("\n\n", example[0])[2]

        #remove too short sentences
        df = pd.DataFrame(raw_sents, columns =['raw'])
        df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
        newdf = df.loc[(df['preprocess_raw'] != 'None')]
        raw_sentences = newdf['preprocess_raw'].values.tolist()
        if len(raw_sentences) == 0:
            continue

        preprocessed_sentences = []
        for raw_sent in raw_sentences:
            preprocessed_sent = preprocess_raw_sent(raw_sent)
            preprocessed_sentences.append(preprocessed_sent)

        preprocessed_abs_sentences_list = []
        raw_abs_sent_list = abstract.split(' . ')
        for abs_sent in raw_abs_sent_list:
            preprocessed_abs_sent = preprocess_raw_sent(abs_sent)
            preprocessed_abs_sentences_list.append(preprocessed_abs_sent)    
        preprocessed_abs_sentences = (" ").join(preprocessed_abs_sentences_list)  

        if len(preprocessed_sentences) < 7 or len(preprocessed_abs_sentences_list) < 3:
            continue
        # Extract word vectors
        word_embeddings = {}
        f = open('glove.6B.50d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()

        sentences = preprocessed_sentences.copy()

        #create vectors for sentences.
        sentence_vectors = []
        for i in sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((50,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((50,))
            sentence_vectors.append(v)

        # similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0,0]
        
                
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph) # score of all sentences in article
        scores_with_sentences = []
        for i in range(len(raw_sentences)):
            tmp = (raw_sentences[i], scores[i], i)
            scores_with_sentences.append(tmp)

        rank_scores_with_sentences = sorted(scores_with_sentences, key=lambda x: x[1], reverse=True)
        length_of_summary = int(0.2*len(raw_sentences))
        rank_text = rank_scores_with_sentences[ : length_of_summary]
        rank_text = sorted(rank_text, key=lambda x: x[2], reverse=False)

        print("Done preprocessing!")
        
        print('time for processing', time.time() - start_time)

        
        file_name = os.path.join(save_path, example[1] )    
        f = open(file_name,'w', encoding='utf-8')
        for sent in rank_text:
            f.write(sent[0] + ' ')
        f.close()

    
def multiprocess(num_process, stories, save_path):
    processes = []
    n = math.floor(len(stories)/5)
    set_of_docs = [stories[i:i + n] for i in range(0, len(stories), n)] 
    for index, sub_stories in enumerate(set_of_docs):
        p = multiprocessing.Process(target=start_run, args=(
            index,sub_stories, save_path[index], 0))
        processes.append(p)
        p.start()      
    for p in processes:
        p.join()



def main():
    # Setting Variables
    directory = 'full_text_data'
    save_path=['hyp1', 'hyp2', 'hyp3', 'hyp4', 'hyp5']

    if not os.path.exists('hyp1'):
        os.makedirs('hyp1')
    if not os.path.exists('hyp2'):
        os.makedirs('hyp2')
    if not os.path.exists('hyp3'):
        os.makedirs('hyp3')
    if not os.path.exists('hyp4'):
        os.makedirs('hyp4')
    if not os.path.exists('hyp5'):
        os.makedirs('hyp5')

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()
    
    multiprocess(5, stories, save_path)
    # start_run(1, stories, save_path[0], 0)

    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))

if __name__ == '__main__':
    main()  
        
        
     
    


    
    
    
    
        
            
            
         
