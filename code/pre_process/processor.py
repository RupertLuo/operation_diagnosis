
import os
import json as js
import random
import copy
from ltp import LTP
import numpy as np
import scipy.stats as t
from tqdm import tqdm
class Processor():
	"""
	用来处理流程图数据
	"""
	def __init__(self):
		self.sequence_files = []
		self.decision_files = []
		self.sequence_dir = ''
		self.decision_dir = ''
	def read_dir_data(self, sequence_dir,decision_dir):
		"""
		读取一个文件夹下的所有文件，并存进sequence_files 和 decision_files
		"""
		self.sequence_files = sorted(os.listdir(sequence_dir))
		self.decision_files = sorted(os.listdir(decision_dir))
		self.sequence_dir = sequence_dir
		self.decision_dir = decision_dir
	def edge_format_change(self, edge_str):
		"""这是一个edge字符串转换函数

		Args:
			edge_str ([str]): the string of edge in original daata
			'[start]-0\n0-1\n1-2\n2-[end]'
		Returns:
			[list]: edgelist [('start','0),('0','1'),('1','2'),('2','[end]')]
		"""
		edge_list = filter(lambda x:x!='', edge_str.split('\n'))
		edge_list = list(map(lambda x:tuple(x.split('-')),edge_list))
		return edge_list
	def data_process(self, mode=1):
		"""处理原数据集的函数

		Args:
			mode (int, optional): [description]. Defaults to 1.
			默认是1代表处理序列化数据

		Returns:
			[list]: 所有数据的三部分内容
		"""
		context_list = []
		node_list = []
		edge_list = []
		if mode==1:
			for file in tqdm(self.sequence_files):
				filepath = os.path.join(self.sequence_dir,file)
				context, node= open(filepath,'r',encoding = 'utf8').read().split('**********')
				#把node分开
				l = list(filter(lambda x:x!='',node.split('\n')))
				for i in range(len(l)):
					s = l[i].split('\t')
					if len(s)==6:
						l[i] = s[1::2]
					elif len(s)==7:
						l[i] = s[2::2]
				node = {str(i):x for i,x in enumerate(filter(lambda x: len(x)==3,list(l)))}

				if len(node.keys())==0:
					continue
				node['[start]']=['[pad]','[pad]','[pad]']
				node['[end]']=['[pad]','[pad]','[pad]']
				edge = ''.join(['[start]-0\n']+[str(i)+'-'+str(i+1)+'\n' for i in range(len(node)-3)]+ [str(len(node)-3)+'-[end]\n'])
				edge = self.edge_format_change(edge)
				context_list.append(context)
				node_list.append(node)
				edge_list.append(edge)
			self.sequence_data = (context_list,node_list,edge_list)
		else:
			for file in tqdm(self.decision_files):
				filepath = os.path.join(self.decision_dir,file)
				context, node, edge= open(filepath,'r',encoding = 'utf8').read().split('**********')
				edge = self.edge_format_change(edge)
				#把node分开
				l = list(filter(lambda x:x!='',node.split('\n')))
				for i in range(len(l)):
					s = l[i].split('\t')
					if len(s)==6:
						l[i] = s[1::2]
					elif len(s)==7:
						l[i] = s[2::2]
				node = {str(i):x for i,x in enumerate(filter(lambda x:len(x)==3,list(l)))}
				if len(node.keys())==0:
					continue
				node['[start]']=['[pad]','[pad]','[pad]']
				node['[end]']=['[pad]','[pad]','[pad]']
				context_list.append(context)
				node_list.append(node)
				edge_list.append(edge)
			self.decision_data = (context_list,node_list,edge_list)
	def save_data(self, data_list,save_path):
		"""这是一个用于保存已处理好的数据的函数

		Args:
			data_list (tuple): [三部分：context，node，edge]

		"""
		f = open(save_path,'w',encoding = 'utf8')
		js.dump(data_list,f,ensure_ascii=False,indent=2)
		f.close()
	def read_data(self, sequence_file,decision_file):
		"""
		docstring
		"""
		self.sequence_data = js.load(open(sequence_file,'r',encoding='utf8'))
		self.decision_data = js.load(open(decision_file,'r',encoding='utf8'))

class DatasetCreator(object):
	"""
	对原数据进行构建数据集，创建错误节点，procedural graph的存储
	方法如下：
		（1）读取数据
		（2）错误节点创建
		（3）错误节点答案创建
		（4）保存数据集
	"""
	def __init__(self) -> None:
		super().__init__()
		self.sequence_data = None
		self.decision_data = None
	def load_data(self, sequence_file, decision_file):
		"""这是从原数据加载数据的函数

		Args:
			sequence_file ([str]): [序列数据文件路径]
			decision_file ([str]): [分支数据文件路径]
		"""
		self.sequence_data = js.load(open(sequence_file,'r',encoding='utf8'))
		self.decision_data = js.load(open(decision_file,'r',encoding='utf8'))
	def make_error_node(self,change_node,verblist,entitylist):
		"""给定一个正确节点修改为错误节点

		Args:
			change_node ([list]): [节点三元组]
			verblist ([list]): [供选择的动词表]
			entitylist ([list]): [供选择的名词表]

		Returns:
			[list]: [修改后的节点三元组]
		"""
		change_node = copy.deepcopy(change_node)
		mode = random.randint(0,1)
		if mode==0:
			#修改动词
			verblist_filtered = list(filter(lambda x:x!=change_node[2] and str(x)!='nan',verblist))
			replaced_index = random.randint(0,len(verblist_filtered)-1)
			change_node[2]=verblist_filtered[replaced_index]
		else:
			#修改实体2
			entitylist_filtered = list(filter(lambda x:x!=change_node[1] and str(x)!='nan',entitylist))
			replaced_index = random.randint(0,len(entitylist_filtered)-1)
			change_node[1]=entitylist_filtered[replaced_index]
		return change_node
	def find_verb_noun_list(self, pos, seg, node):
		"""找出文中和节点中所有的动词list和名词list

		Args:
			pos (list): [词性标注结果]
			seg ([list]): [分词结果]
			node ([dict]]): [所有node三元组的字典]
		Return:
			entitylist, verblist
		"""
		entitylist = []
		verblist = []
		for key in node.keys():
			if key!='[start]' and key!='[end]':
				entitylist.append(node[key][1])
				verblist.append(node[key][2])
		for i in range(len(pos)):
			for j in range(len(pos[i])):
				if pos[i][j] in ['n','ws','nd','nh','nl','ns','nt','ni']:
					entitylist.append(seg[i][j])
				elif pos[i][j]=='v':
					verblist.append(seg[i][j])
		return entitylist, verblist
	def make_answer(self, change_node, verblist, entitylist, error_mode = 1):
		"""[summary]

		Args:
			change_node ([type]): [description]
		"""
		answer_list = []
		if error_mode==1:#错误节点只新建两个错误节点，加上正确的答案一共三个选项
			for i in range(2):
				answer_list.append(self.make_error_node(change_node, verblist, entitylist))
		else:
			for i in range(3):
				answer_list.append(self.make_error_node(change_node, verblist, entitylist))
		return answer_list
	def change_data_form(self):
		seqnence_data_len = len(self.sequence_data)
		decision_data_len = len(self.decision_data)
		# sequence_data 训练集划分
		sequence_train_data = list()
		train_data_index = list(range(seqnence_data_len))
		random.shuffle(train_data_index)
		for index in train_data_index:
			sequence_train_data.append(
				(
					self.sequence_data[0][index],
					self.sequence_data[1][index],
					self.sequence_data[2][index],
					self.sequence_data[3][index],
					self.sequence_data[4][index],
					self.sequence_data[5][index],
					self.sequence_data[6][index]
				)
			)
		self.sequence_data = sequence_train_data
		# decision_data 训练集划分
		decision_train_data = list()
		train_data_index = list(range(decision_data_len))
		random.shuffle(train_data_index)
		for index in train_data_index:
			decision_train_data.append(
				(
					self.decision_data[0][index],
					self.decision_data[1][index],
					self.decision_data[2][index],
					self.decision_data[3][index],
					self.decision_data[4][index],
					self.decision_data[5][index],
					self.decision_data[6][index]
				)
			)
		self.decision_data = decision_train_data
	def create_error(self, error_rate = 0.2, sequence = True):
		"""用于创建错误节点的函数

		Args:
			error_rate (float, optional): [该节点出错的概率]. Defaults to 0.2.
		"""
		ltp = LTP(path = "base")
		if sequence:
			n = len(self.sequence_data[0])
			data = self.sequence_data
		else:
			n = len(self.decision_data[0])
			data = self.decision_data
		error_node_label_list = []
		error_node_list = []
		answer_list = []
		answer_label_list = []
		for i in tqdm(range(n)):
			context =  list(filter(lambda x:x!='',data[0][i].split('\n')))
			node = data[1][i]
			#寻找替代的词
			seg, hidden = ltp.seg(context)
			pos = ltp.pos(hidden)
			entitylist, verblist = self.find_verb_noun_list(pos,seg,node)
			#制造随机数寻找出现错误的节点
			error_node_index = []
			while(1):
				node_rvs = np.random.normal(0, 1, len(node)-2)
				down = t.norm.ppf(error_rate/2)# 下分位点
				up = t.norm.ppf(1-error_rate/2)# 上分位点
				node_key = list(filter(lambda x:x not in ['[start]','[end]'], node.keys()))
				error_node_index = list(filter(lambda x: x[0],zip(((node_rvs<down) + (node_rvs>up)).tolist(),node_key)))
				if error_node_index!=[]:
					break
			#复制原node 进行替换
			error_node_dict = copy.deepcopy(node)
			error_node_label = dict(zip(node.keys(),[0]*len(node.keys())))
			answer_dict = dict()
			answer_label_dict = dict()

			for _,error_key in error_node_index:
				error_node = self.make_error_node(error_node_dict[error_key],verblist,entitylist)
				error_node_dict[error_key] = error_node
				error_node_label[error_key] = 1
				#为错误节点生成选项
				answer_option = self.make_answer(error_node_dict[error_key],verblist,entitylist)
				answer_label = random.randint(0,2)
				answer_option.insert(answer_label,node[error_key])
				answer_dict[error_key] = answer_option
				answer_label_dict[error_key] = answer_label
			#将这一条数据的错误节点内容和标签加入列表
			error_node_list.append(error_node_dict)
			error_node_label_list.append(error_node_label)
			answer_list.append(answer_dict)
			answer_label_list.append(answer_label_dict)
		data.append(error_node_list)
		data.append(error_node_label_list)
		data.append(answer_list)
		data.append(answer_label_list)
	def save_data(self, sequence_save_path, decision_save_path):
		"""保存构建好的数据集

		Args:
			sequence_save_path (str): [description]
			decision_save_path (str): [description]
		"""
		f = open(sequence_save_path,'w',encoding = 'utf8')
		js.dump(self.sequence_data,f,ensure_ascii=False,indent=2)
		f.close()
		f = open(decision_save_path,'w',encoding = 'utf8')
		js.dump(self.decision_data,f,ensure_ascii=False,indent=2)
		f.close()
if __name__ == "__main__":
	decision_dir = 'data/final_data/desicion_procedure'
	sequence_dir = 'data/final_data/sequencial'
	save_path_dir = 'new_data/origin_data'

	
	# 处理成原数据形式
	processor = Processor()
	processor.read_dir_data(sequence_dir,decision_dir)
	processor.data_process(mode=1)
	processor.data_process(mode=0)
	processor.save_data(processor.sequence_data,os.path.join(save_path_dir,'sequence_data.json'))
	processor.save_data(processor.decision_data,os.path.join(save_path_dir,'decision_data.json'))
	
	#对原数据进行创建数据集
	creator = DatasetCreator()
	creator.load_data(os.path.join(save_path_dir,'sequence_data.json'),os.path.join(save_path_dir,'decision_data.json'))
	creator.create_error()
	creator.create_error(sequence=False)
	# creator.change_data_form()
	creator.save_data(os.path.join(save_path_dir,'sequence_created_data.json'),os.path.join(save_path_dir,'decision_created_data.json'))

