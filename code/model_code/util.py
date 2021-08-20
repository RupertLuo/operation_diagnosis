import pandas as pd
import networkx as nx
from fuzzywuzzy import process
import torch
import numpy as np
import random
import torch
def setup_seed(seed):
	"""设置随机数

	Args:
		seed ([type]): [description]
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
def collate_fn(batch_data):
	for i in range(len(batch_data)):
		if i ==0:
			original_text = batch_data[i][0]
			original_text['attention_mask'] = original_text['attention_mask']
			node = batch_data[i][1]
			label = torch.unsqueeze(batch_data[i][3],dim=0)
			edge = torch.unsqueeze(batch_data[i][2],dim=0)
		return original_text,node,edge,label
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
def nodeAlign(graphNodes,node):
	if node in graphNodes:
		return graphNodes.index(node)
	else:
		bestnode = sorted(process.extract(node, graphNodes,limit=3),key=lambda x:(int(x[1]),len(x[0])),reverse = True)
		return graphNodes.index(bestnode[0][0])
def get_sub_graph(G,nodes,sequence,hop_number):
	#首先先对序列中的与外部图中的节点进行对齐
	sequence_node = []
	for seq in sequence:
		sequence_node+=seq
	aligned_node = []
	for each_node in sequence_node:
		if each_node!='操作者' and each_node!='':
			aligned_node.append(nodeAlign(nodes,each_node))
	#对齐后开始寻找对齐后节点在大图中的子图
	subgraph_edges = []
	subgraph_nodes = set()
	# for each_node in aligned_node:
	# 	subgraph_nodes = subgraph_nodes|findKhopNeighbor(G,each_node,hop_number)
	subgraph = G.subgraph(aligned_node)
	# for each_node in aligned_node:
	# 	for neighbor in G[each_node]:
	# 		# subgraph_nodes.add(each_node)
	# 		if neighbor in aligned_node:
	# 			subgraph_nodes.add(each_node)
	# 			subgraph_nodes.add(neighbor)
	# 			if((each_node,neighbor) not in subgraph_edges and (neighbor,each_node) not in subgraph_edges):
	# 				subgraph_edges.append((each_node,neighbor))
	return subgraph
def change_bert_encoding(encoding_dict):
	"""将bert encoding 转成一行的向量

	Args:
		encoding_dict ([type]):

	Returns:
		bert_encoding [type]:
	"""
	input_ids = encoding_dict['input_ids']
	attention_mask = encoding_dict['attention_mask']
	token_type_ids = encoding_dict['token_type_ids']
	bert_encoding = torch.cat([input_ids,attention_mask,token_type_ids],dim=1)
	return bert_encoding

def adjust_learning_rate(optimizer, epoch, stepsize, gamma, set_lr):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = set_lr * (gamma ** (epoch // stepsize))
	for param_group in optimizer.param_groups:
		if param_group['bert']!=1:
			param_group['lr'] = lr
	return optimizer

def Noam(optimizer,d_model,step_number,warm_up_number):
	lr = d_model**(-0.5)*min(step_number**(-0.5),step_number*warm_up_number**(-1.5))
	for param_group in optimizer.param_groups:
		if param_group['bert']!=1:
			param_group['lr'] = lr
	return optimizer

