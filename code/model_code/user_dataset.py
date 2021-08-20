import random
import torch
from torch.utils.data import Dataset, DataLoader
import json as js
from util import triple_to_graph,get_sub_graph, change_bert_encoding
from transformers import BertTokenizer
import os
import networkx as nx
from torch_geometric.data import Data, Batch
from dgl import DGLGraph
import dgl
from tqdm import tqdm
class Task1Dataset(Dataset):
	"""用于任务1的数据集
	"""
	def __init__(self,decision_file_path, sequence_file_path, tokenizer = None, test = False, dev = False, procedure_triple_path = None, dgl_form = True):
		self.dgl_form = dgl_form
		self.tokenizer = tokenizer
		self.max_text_length = 0
		self.max_sequence_length = 0
		self.max_operation_length = 91
		self.procedure_triple_path = procedure_triple_path
		# 读取数据
		decision_meta_data = js.load(open(decision_file_path,'r',encoding = 'utf8'))
		sequence_meta_data = js.load(open(sequence_file_path,'r',encoding = 'utf8'))
		self.sequence_meta_data = sequence_meta_data
		#序列数据的长度
		self.sequence_data_len = len(sequence_meta_data[0])
		self.decision_data_len = len(decision_meta_data[0])
		#划分训练集数据集
		self.sequence_train_index = int(self.sequence_data_len*0.7)#633
		self.sequence_test_index = int(self.sequence_data_len*0.9)#814
		self.decision_train_index = int(self.decision_data_len*0.7)#112
		self.decision_test_index = int(self.decision_data_len*0.9)#144
		if test:
			sequence_start = self.sequence_test_index
			sequence_end = self.sequence_data_len
			decision_start = self.decision_test_index
			decision_end = self.decision_data_len
		elif dev:
			sequence_start = self.sequence_train_index
			sequence_end = self.sequence_test_index
			decision_start = self.decision_train_index
			decision_end = self.decision_test_index
		else:
			sequence_start = 0
			sequence_end = self.sequence_train_index
			decision_start = 0
			decision_end = self.decision_train_index
		#存储数据
		self.text = sequence_meta_data[0][sequence_start:sequence_end]+\
					decision_meta_data[0][decision_start:decision_end]
		self.node = sequence_meta_data[1][sequence_start:sequence_end]+\
					decision_meta_data[1][decision_start:decision_end]
		self.edge = sequence_meta_data[2][sequence_start:sequence_end]+\
					decision_meta_data[2][decision_start:decision_end]
		self.error_node = sequence_meta_data[3][sequence_start:sequence_end]+\
						  decision_meta_data[3][decision_start:decision_end]
		self.node_label = sequence_meta_data[4][sequence_start:sequence_end]+\
						  decision_meta_data[4][decision_start:decision_end]
		self.answer_choice = sequence_meta_data[5][sequence_start:sequence_end]+\
						     decision_meta_data[5][decision_start:decision_end]
		self.answer_label = sequence_meta_data[6][sequence_start:sequence_end]+\
						    decision_meta_data[6][decision_start:decision_end]
		if procedure_triple_path:
			#获取过程类图谱
			self.procedure_graph,self.procedure_nodes = triple_to_graph(procedure_triple_path)
			# 对每个节点先分词，利用glove编码每个节点
			self.procedure_node_length = max(map(lambda x:len(x),self.procedure_nodes))
	def __len__(self):
		return len(self.text)
	def __getitem__(self, index):
		text = self.text[index]
		text_ids = change_bert_encoding(self.tokenizer(
				text.replace('\n','[SEP]'),
				max_length = 512,
				padding = 'max_length',
				truncation = True,
				return_tensors = 'pt'
			))
		node = self.error_node[index]
		edge_node =list(set([edge[0] for edge in self.edge[index]]+[edge[1] for edge in self.edge[index]]))# 把edge中没出现的节点删除
		node = {key:element for key,element in node.items() if key in edge_node}
		#过程图谱子图寻找和编码
		if self.procedure_triple_path:
			procedure_subgraph = get_sub_graph(self.procedure_graph,self.procedure_nodes,node.values(),hop_number=1)
			subgraph_node_map = list(procedure_subgraph.nodes)
			procedure_subgraph_edge = list(map(lambda x:[subgraph_node_map.index(x[0]),subgraph_node_map.index(x[1])],list(procedure_subgraph.edges)))

			procedure_subgraph_node = [self.procedure_nodes[index] for index in list(procedure_subgraph.nodes)]
			procedure_node_feature = []
			for each_node in procedure_subgraph_node:
				node_feature = self.tokenizer.encode_plus(
							each_node,                     # 输入文本
							add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
							max_length = self.procedure_node_length,           # 填充 & 截断长度
							pad_to_max_length = True,
							return_attention_mask = True,   # 返回 attn. masks.
							return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
							truncation='longest_first'
						)
				procedure_node_feature.append(change_bert_encoding(node_feature))
			procedure_node_feature = torch.cat(procedure_node_feature,dim=0)
			if procedure_subgraph_edge ==[]:
				procedure_subgraph_edge = torch.tensor(procedure_subgraph_edge,dtype = torch.long)
			else:
				procedure_subgraph_edge = torch.tensor(procedure_subgraph_edge,dtype = torch.long)
				procedure_subgraph_edge_inverse = torch.stack([procedure_subgraph_edge[:,1],procedure_subgraph_edge[:,0]]).t()
				procedure_subgraph_edge = torch.cat([procedure_subgraph_edge,procedure_subgraph_edge_inverse])
			procedure_subgraph =  Data(x=procedure_node_feature, edge_index=procedure_subgraph_edge.t().contiguous())
		# 序列图编码
		node_label = {key:element for key,element in self.node_label[index].items() if key in node.keys()}
		for key,this_node in node.items():
			if this_node[0]!='[]':
				node[key] = self.tokenizer.encode_plus(
							this_node[0]+this_node[2]+this_node[1],                     # 输入文本
							add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
							max_length = self.max_operation_length,           # 填充 & 截断长度
							pad_to_max_length = True,
							return_attention_mask = True,   # 返回 attn. masks.
							return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
							truncation='longest_first'
					)
		node = list(node.items())
		node_index = list(map(lambda x:x[0],node))
		node_label = torch.tensor([node_label[key] for key in node_index])
		x = torch.cat(list(map(lambda x:change_bert_encoding(x[1]),node)),dim=0)
		edge_index = torch.tensor(list(map(lambda x:[node_index.index(x[0]),node_index.index(x[1])],self.edge[index])),dtype = torch.long)
		if not self.dgl_form:
			data = Data(x=x, edge_index=edge_index.t().contiguous(),y=node_label)
		else:
			data = dgl.graph((edge_index.t()[0],edge_index.t()[1]))
			data.ndata['x']=x
			data.ndata['y']=node_label
		if self.procedure_triple_path:
			return (text_ids,
					data,
					procedure_subgraph,
					dgl.topological_nodes_generator(data))
		else:
			return (text_ids,data)
class Task2Dataset(Task1Dataset):
	def __init__(self,decision_file_path, sequence_file_path, tokenizer = None, test = False, dev = False, procedure_triple_path = None, dgl_form = True):
		super(Task2Dataset, self).__init__(decision_file_path, sequence_file_path, tokenizer, test, dev, procedure_triple_path, dgl_form)
	def __getitem__(self, index):
		text = self.text[index]
		text_ids = change_bert_encoding(self.tokenizer(
				text.replace('\n','[SEP]'),
				max_length = 512,
				padding = 'max_length',
				truncation = True,
				return_tensors = 'pt'
			))
		node = self.error_node[index]
		edge_node =list(set([edge[0] for edge in self.edge[index]]+[edge[1] for edge in self.edge[index]]))# 把edge中没出现的节点删除
		node = {key:element for key,element in node.items() if key in edge_node}
		#过程图谱子图寻找和编码
		if self.procedure_triple_path:
			procedure_subgraph = get_sub_graph(self.procedure_graph,self.procedure_nodes,node.values(),hop_number=1)
			subgraph_node_map = list(procedure_subgraph.nodes)
			procedure_subgraph_edge = list(map(lambda x:[subgraph_node_map.index(x[0]),subgraph_node_map.index(x[1])],list(procedure_subgraph.edges)))

			procedure_subgraph_node = [self.procedure_nodes[index] for index in list(procedure_subgraph.nodes)]
			procedure_node_feature = []
			for each_node in procedure_subgraph_node:
				node_feature = self.tokenizer.encode_plus(
							each_node,                     # 输入文本
							add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
							max_length = self.procedure_node_length,           # 填充 & 截断长度
							pad_to_max_length = True,
							return_attention_mask = True,   # 返回 attn. masks.
							return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
							truncation='longest_first'
						)
				procedure_node_feature.append(change_bert_encoding(node_feature))
			procedure_node_feature = torch.cat(procedure_node_feature,dim=0)
			if procedure_subgraph_edge ==[]:
				procedure_subgraph_edge = torch.tensor(procedure_subgraph_edge,dtype = torch.long)
			else:
				procedure_subgraph_edge = torch.tensor(procedure_subgraph_edge,dtype = torch.long)
				procedure_subgraph_edge_inverse = torch.stack([procedure_subgraph_edge[:,1],procedure_subgraph_edge[:,0]]).t()
				procedure_subgraph_edge = torch.cat([procedure_subgraph_edge,procedure_subgraph_edge_inverse])
			procedure_subgraph =  Data(x=procedure_node_feature, edge_index=procedure_subgraph_edge.t().contiguous())
		#------------------------------
		#
		#answer
		choose_answer = self.answer_choice[index]
		answer_label = self.answer_label[index]
		assert len(choose_answer)==1
		error_key = list(choose_answer.keys())[0]
		assert answer_label[error_key]==0
		answer_list = choose_answer[error_key]
		for i in range(len(answer_list)):
			answer_encoding = self.tokenizer.encode_plus(
							answer_list[i][0]+answer_list[i][2]+answer_list[i][1],                     # 输入文本
							add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
							max_length = self.max_operation_length,           # 填充 & 截断长度
							pad_to_max_length = True,
							return_attention_mask = True,   # 返回 attn. masks.
							return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
							truncation='longest_first'
					)
			answer_list[i] = change_bert_encoding(answer_encoding)
		#---------------------------------------------------------
		#
		# 打乱选项的label
		answer_label = random.randint(0,2)
		true_answer = answer_list[0]
		false_answer = answer_list[1:]
		random.shuffle(false_answer)
		false_answer.insert(answer_label,true_answer)
		answer_choice = torch.cat(false_answer)
		#______________________________________
		# 序列图编码
		node[error_key] = ['[MASK]','[MASK]','[MASK]']
		node_label_dict = {key:element for key,element in self.node_label[index].items() if key in node.keys()}
		for key,this_node in node.items():
			if this_node[0]!='[]':
				node[key] = self.tokenizer.encode_plus(
							this_node[0]+this_node[2]+this_node[1],                     # 输入文本
							add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
							max_length = self.max_operation_length,           # 填充 & 截断长度
							pad_to_max_length = True,
							return_attention_mask = True,   # 返回 attn. masks.
							return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
							truncation='longest_first'
					)
		# 制造label
		node = list(node.items())
		node_index = list(map(lambda x:x[0],node))
		node_label = torch.tensor([node_label_dict[key] for key in node_index])
		x = torch.cat(list(map(lambda x:change_bert_encoding(x[1]),node)),dim=0)
		edge_index = torch.tensor(list(map(lambda x:[node_index.index(x[0]),node_index.index(x[1])],self.edge[index])),dtype = torch.long)
		if not self.dgl_form:
			data = Data(x=x, edge_index=edge_index.t().contiguous(),y=node_label)
		else:
			data = dgl.graph((edge_index.t()[0],edge_index.t()[1]))
			data.ndata['x']=x
			data.ndata['y']=node_label
		
		if self.procedure_triple_path:
			if self.dgl_form:
				return (text_ids,
					data,
					procedure_subgraph,
					dgl.topological_nodes_generator(data),
					answer_choice,
					answer_label
					)
			else:
				return (text_ids,
					data,
					procedure_subgraph,
					answer_choice,
					answer_label
					)
		else:
			if self.dgl_form:
				return (text_ids,
					data,
					dgl.topological_nodes_generator(data),
					answer_choice,
					answer_label
					)
			else:
				return (text_ids,
					data,
					answer_choice,
					answer_label
					)

class Collater(object):
	def __init__(self, follow_batch):
		self.follow_batch = follow_batch

	def collate(self, batch):
		elem = batch[0][1]
		if isinstance(elem, DGLGraph):
			batch_graph = [bat[1]for bat in batch]
			text = torch.cat([bat[0]for bat in batch],dim=0)
			procedure_graph = [bat[2]for bat in batch]
			return dgl.batch(batch_graph),text,Batch.from_data_list(procedure_graph, self.follow_batch)
		if isinstance(elem, Data):
			batch_graph = [bat[1]for bat in batch]
			text = torch.cat([bat[0]for bat in batch],dim=0)
			procedure_graph = [bat[2]for bat in batch]
			return Batch.from_data_list(batch_graph, self.follow_batch),text,Batch.from_data_list(procedure_graph, self.follow_batch)
		elif isinstance(elem, torch.Tensor):
			return default_collate(batch)
		elif isinstance(elem, float):
			return torch.tensor(batch, dtype=torch.float)
		elif isinstance(elem, int_classes):
			return torch.tensor(batch)
		elif isinstance(elem, string_classes):
			return batch
		elif isinstance(elem, container_abcs.Mapping):
			return {key: self.collate([d[key] for d in batch]) for key in elem}
		elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
			return type(elem)(*(self.collate(s) for s in zip(*batch)))
		elif isinstance(elem, container_abcs.Sequence):
			return [self.collate(s) for s in zip(*batch)]

		raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

	def __call__(self, batch):
		return self.collate(batch)
class GraphDataLoader(torch.utils.data.DataLoader):
	def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
				**kwargs):
		super(GraphDataLoader,
			self).__init__(dataset, batch_size, shuffle,
							collate_fn=Collater(follow_batch),**kwargs)

class Collater_tassk2(object):
	def __init__(self, follow_batch):
		self.follow_batch = follow_batch

	def collate(self, batch):
		elem = batch[0][1]
		if isinstance(elem, DGLGraph):
			batch_graph = [bat[1]for bat in batch]
			text = torch.cat([bat[0]for bat in batch],dim=0)
			procedure_graph = [bat[2]for bat in batch]
			answer_choice = torch.stack([bat[4] for bat in batch])
			label = torch.tensor([bat[5]for bat in batch])
			return dgl.batch(batch_graph),text,Batch.from_data_list(procedure_graph, self.follow_batch),answer_choice,label
		if isinstance(elem, Data):
			batch_graph = [bat[1]for bat in batch]
			text = torch.cat([bat[0]for bat in batch],dim=0)
			procedure_graph = [bat[2]for bat in batch]
			answer_choice = torch.stack([bat[3] for bat in batch])
			label = torch.tensor([bat[4]for bat in batch])
			return Batch.from_data_list(batch_graph, self.follow_batch),text,Batch.from_data_list(procedure_graph, self.follow_batch),answer_choice,label
		elif isinstance(elem, torch.Tensor):
			return default_collate(batch)
		elif isinstance(elem, float):
			return torch.tensor(batch, dtype=torch.float)
		elif isinstance(elem, int_classes):
			return torch.tensor(batch)
		elif isinstance(elem, string_classes):
			return batch
		elif isinstance(elem, container_abcs.Mapping):
			return {key: self.collate([d[key] for d in batch]) for key in elem}
		elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
			return type(elem)(*(self.collate(s) for s in zip(*batch)))
		elif isinstance(elem, container_abcs.Sequence):
			return [self.collate(s) for s in zip(*batch)]

		raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

	def __call__(self, batch):
		return self.collate(batch)
class GraphDataLoader_task2(torch.utils.data.DataLoader):
	def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
				**kwargs):
		super(GraphDataLoader_task2,
			self).__init__(dataset, batch_size, shuffle,
							collate_fn=Collater_tassk2(follow_batch),**kwargs)


class TrmCollater(object):
	def __init__(self, follow_batch):
		self.follow_batch = follow_batch

	def collate(self, batch):
		elem = batch[0][1]
		if isinstance(elem, DGLGraph):
			batch_graph = [bat[1]for bat in batch]
			text = torch.cat([bat[0]for bat in batch],dim=0)
			procedure_graph = [bat[2]for bat in batch]
			topological_list = [bat[3] for bat in batch]

			return batch_graph,text,Batch.from_data_list(procedure_graph, self.follow_batch),topological_list
		elif isinstance(elem, torch.Tensor):
			return default_collate(batch)
		elif isinstance(elem, float):
			return torch.tensor(batch, dtype=torch.float)
		elif isinstance(elem, int_classes):
			return torch.tensor(batch)
		elif isinstance(elem, string_classes):
			return batch
		elif isinstance(elem, container_abcs.Mapping):
			return {key: self.collate([d[key] for d in batch]) for key in elem}
		elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
			return type(elem)(*(self.collate(s) for s in zip(*batch)))
		elif isinstance(elem, container_abcs.Sequence):
			return [self.collate(s) for s in zip(*batch)]

		raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))
	def __call__(self, batch):
		return self.collate(batch)

class TrmDataloader(torch.utils.data.DataLoader):
	def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
				**kwargs):
		super(TrmDataloader,
			self).__init__(dataset, batch_size, shuffle,
							collate_fn=TrmCollater(follow_batch),**kwargs)

class TrmCollater_task2(object):
	def __init__(self, follow_batch):
		self.follow_batch = follow_batch

	def collate(self, batch):
		elem = batch[0][1]
		if isinstance(elem, DGLGraph):
			batch_graph = [bat[1]for bat in batch]
			text = torch.cat([bat[0]for bat in batch],dim=0)
			procedure_graph = [bat[2]for bat in batch]
			topological_list = [bat[3] for bat in batch]
			answer_choice = torch.stack([bat[4] for bat in batch])
			label = torch.tensor([bat[5]for bat in batch])
			return batch_graph,text,Batch.from_data_list(procedure_graph, self.follow_batch),topological_list,answer_choice,label
		elif isinstance(elem, torch.Tensor):
			return default_collate(batch)
		elif isinstance(elem, float):
			return torch.tensor(batch, dtype=torch.float)
		elif isinstance(elem, int_classes):
			return torch.tensor(batch)
		elif isinstance(elem, string_classes):
			return batch
		elif isinstance(elem, container_abcs.Mapping):
			return {key: self.collate([d[key] for d in batch]) for key in elem}
		elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
			return type(elem)(*(self.collate(s) for s in zip(*batch)))
		elif isinstance(elem, container_abcs.Sequence):
			return [self.collate(s) for s in zip(*batch)]

		raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))
	def __call__(self, batch):
		return self.collate(batch)

class TrmDataloader_task2(torch.utils.data.DataLoader):
	def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
				**kwargs):
		super(TrmDataloader_task2,
			self).__init__(dataset, batch_size, shuffle,
							collate_fn=TrmCollater_task2(follow_batch),**kwargs)


if __name__ == "__main__":
	BERT_PATH = '/remote-home/my/operation_detection/bert-base-chinese'
	VOCAB = 'vocab.txt'
	tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_PATH, VOCAB))
	dataset = Task2Dataset("new_data/origin_data/old_decision_created_data.json",
							"new_data/origin_data/old_sequence_created_data.json",
							tokenizer=tokenizer,
							test=True,
							procedure_triple_path = 'new_data/procedure_graph/procedure_graph.csv')
	d = dataset[15]
