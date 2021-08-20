import json as js
import jieba as jb
from tqdm import tqdm
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import json as js
def triple_to_graph(triple_file_path,wether_procedure = True):
	'''
	input:   三元组集合
	output:  由三元组组成的大图, 节点集合
	'''
	if wether_procedure:
		triple = pd.read_csv(triple_file_path,sep='\t')
	else:
		triple = pd.read_csv(triple_file_path,sep=',')
	triple = triple.dropna(axis=0,how='any')
	if wether_procedure:
		nodes = list(filter(lambda x:str(x)!='nan' or x!='操作者',set(list(triple['object'])+list(triple['subject'])+list(triple['verb']))))
	else:
		nodes = list(filter(lambda x:str(x)!='nan' or x!='操作者',set(list(triple['Entity1'])+list(triple['Entiy2']))))
	edges = []
	for i in range(len(triple)):
		node1 = nodes.index(triple.iloc[i,0])
		node2 = nodes.index(triple.iloc[i,1])
		if wether_procedure:
			verb = nodes.index(triple.iloc[i,2])
			if node1!='操作者' and (node1,verb) not in edges:
				edges.append((node1,verb))
			elif node2!='操作者' and (verb,node2) not in edges:
				edges.append((verb,node2))
		else:
			edges.append((node1,node2))
	G = nx.Graph()
	G.add_nodes_from(list(range(len(nodes))))
	G.add_edges_from(edges)
	return G,nodes
def cut_senntence_list(sentence_list,vocab):
    for string in tqdm(sentence_list,ncols=80):
        vocab+=jb.cut(string)
    return vocab
def node2sentence_list(node_list):
    sentence_list = []
    for node in node_list:
        for _,item in node.items():
            sentence_list.append(item[0]+item[2]+item[1])
    return sentence_list
def answer2sentence_list(answer_list):
    sentence_list = []
    for instance in answer_list:
        for _,item in instance.items():
            sentence_list+=list(map(lambda x:x[0]+x[2]+x[1],item))
    return sentence_list
def get_vocab(decision_file_path, sequence_file_path):
    """
    docstring
    """
    vocab = []
    decision_meta_data = js.load(open(decision_file_path,'r',encoding = 'utf8'))
    sequence_meta_data = js.load(open(sequence_file_path,'r',encoding = 'utf8'))
    text = sequence_meta_data[0]+decision_meta_data[0]
    node = node2sentence_list(sequence_meta_data[1]+decision_meta_data[1])
    error_node = node2sentence_list(sequence_meta_data[3]+decision_meta_data[3])
    answer_choice = answer2sentence_list(sequence_meta_data[5]+decision_meta_data[5])
    vocab = cut_senntence_list(text,vocab)
    vocab+= cut_senntence_list(node,vocab)
    vocab+= cut_senntence_list(error_node,vocab)
    vocab+= cut_senntence_list(answer_choice,vocab)
    vocab = list(set(vocab))
    return vocab


parser = argparse.ArgumentParser()
parser.add_argument("-s_path", "--SEQUENCE_DATA_PATH", type=str, default='/remote-home/my/operation_detection/new_data/origin_data/old_sequence_created_data.json', help="Sequence data path")
parser.add_argument("-d_path", "--DECISION_DATA_PATH", type=str, default='/remote-home/my/operation_detection/new_data/origin_data/old_decision_created_data.json', help="Sequence data path")
parser.add_argument("-p_path", "--PROCEDURE_KG_PATH", type=str, default='/remote-home/my/operation_detection/new_data/procedure_graph/procedure_graph.csv', help="procedure KG path")
args = parser.parse_args()
# _,procedure_nodes = triple_to_graph(args.PROCEDURE_KG_PATH)
# vocab = get_vocab(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH)
# vocab = cut_senntence_list(procedure_nodes,vocab)
# vocablist = list(set(vocab))
# js.dump(vocablist,open('code/model_code_v2/vocablist.json','w',encoding='utf8'))
vocablist = js.load(open('code/model_code_v2/vocablist.json','r',encoding='utf8'))
vocab = {item:i for i,item in enumerate(vocablist)}
js.dump(vocab,open('code/model_code_v2/vocab.json','w',encoding='utf8'),ensure_ascii=False)
#读入 embedding
embedding_file = open('/remote-home/my/operation_detection/embeddings/merge_sgns_bigram_char300.txt','r',encoding = 'utf8')
vocab_embedding = np.random.rand(len(vocablist),300)
for i,line in tqdm(enumerate(embedding_file)):
    if i == 0:
        vocab_len,embedding_dim = map(lambda x:int(x),line.strip().split(' '))
    else:
        line_list = line.strip().split(' ')
        word,embedding = line_list[0],line_list[1:]
        embedding = list(map(lambda x:float(x),embedding))
        if word in vocablist:
            index = vocab[word]
            vocab_embedding[index] = np.array(embedding)
        else:
            pass
# vocablist 还剩下一些没有embedding的词
np.save('/remote-home/my/operation_detection/code/model_code_v2/vocab_embedding.npy',vocab_embedding)