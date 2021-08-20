import json as js
def change_sequence_data(squence_data):
	text = []
	node = []
	error = []
	edge = []
	node_label = []
	answer = []
	answer_choices = []
	#读取序列图
	for line in squence_data:
		data_dict = js.loads(line)
		# data_dict['origin_text'] = data_dict['origin_text'].split('\n
		text.append(data_dict['origin_text'])
		if len(list(data_dict['error_node'].items())[0][1].split('\t'))==6:
			this_node_dict = dict(map(lambda x:(x[0],x[1].split('\t')[1::2]),data_dict['error_node'].items()))
		if len(list(data_dict['error_node'].items())[0][1].split('\t'))==7:
			this_node_dict = dict(map(lambda x:(x[0],x[1].split('\t')[2::2]),data_dict['error_node'].items()))
		if '17' in this_node_dict.keys() and this_node_dict['17']==['[]','[]']:
			this_node_dict['17'] = ['操作者','线缆','布放']
		this_node_dict['[start]']=['[PAD]','[PAD]','[PAD]']
		this_node_dict['[end]']=['[PAD]','[PAD]','[PAD]']
		error.append(this_node_dict)

		if len(list(data_dict['correct_node'].items())[0][1].split('\t'))==6:
			correct_node_dict = dict(map(lambda x:(x[0],x[1].split('\t')[1::2]),data_dict['correct_node'].items()))
		if len(list(data_dict['correct_node'].items())[0][1].split('\t'))==7:
			correct_node_dict = dict(map(lambda x:(x[0],x[1].split('\t')[2::2]),data_dict['correct_node'].items()))
		correct_node_dict['[start]']=['[PAD]','[PAD]','[PAD]']
		correct_node_dict['[end]']=['[PAD]','[PAD]','[PAD]']
		node.append(correct_node_dict)

		this_edge = list(map(lambda x:[str(x[0]),str(x[1])],data_dict['edge']))
		if this_edge !=[]:
			this_edge = [['[start]',this_edge[0][0]]]+this_edge+[[this_edge[-1][-1],'[end]']]
		else:
			this_edge = [['[start]','0']+this_edge+[['0','[end]']]]
		edge.append(this_edge)

		data_dict['label'] = {key:(0 if val==-1 else 1) for key,val in data_dict['label'].items()}
		data_dict['label']['[start]']=0
		data_dict['label']['[end]']=0
		node_label.append(data_dict['label'])
		error_node_key = list (data_dict['label'].keys()) [list (data_dict['label'].values()).index (1)]
		answer.append({error_node_key:data_dict['choosed_answer']})
		answer_choices.append({error_node_key:0})
	return text,node,edge,error,node_label,answer,answer_choices
def change_decision_data(decision_data):
	text = []
	node = []
	error = []
	edge = []
	node_label = []
	answer = []
	answer_choices = []
	#读取决策图
	for line in decision_data:
		data_dict = js.loads(line)
		text.append(data_dict['origin_text'])
		data_dict['correct_node']['[start]']=['[PAD]','[PAD]','[PAD]']
		data_dict['correct_node']['[end]']=['[PAD]','[PAD]','[PAD]']
		node.append(data_dict['correct_node'])
		data_dict['error_node']['[start]']=['[PAD]','[PAD]','[PAD]']
		data_dict['error_node']['[end]']=['[PAD]','[PAD]','[PAD]']
		error.append(data_dict['error_node'])
		edge.append(list(map(lambda x:x.split('-'),data_dict['edge'])))
		data_dict['label'] = {key:(0 if val==-1 else 1) for key,val in data_dict['label'].items()}
		data_dict['label']['[start]']=0
		data_dict['label']['[end]']=0
		node_label.append(data_dict['label'])
		error_node_key = list (data_dict['label'].keys()) [list (data_dict['label'].values()).index (1)]
		answer.append({error_node_key:data_dict['choosed_answer']})
		answer_choices.append({error_node_key:0})
	return text,node,edge,error,node_label,answer,answer_choices

train_sequence_data_path = '/remote-home/my/operation_detection/data/dataset/trainset/less_sequence.txt'
test_sequence_data_path = '/remote-home/my/operation_detection/data/dataset/testset/less_sequence.txt'
train_decision_data_path = '/remote-home/my/operation_detection/data/dataset/trainset/less_decision.txt'
test_decision_data_path = '/remote-home/my/operation_detection/data/dataset/testset/less_decision.txt'
train_squence_data = open(train_sequence_data_path,'r',encoding='utf8').readlines()
test_squence_data = open(test_sequence_data_path,'r',encoding='utf8').readlines()
text1,node1,edge1,error1,node_label1,answer1,answer_choices1 = change_sequence_data(train_squence_data)
text2,node2,edge2,error2,node_label2,answer2,answer_choices2 = change_sequence_data(test_squence_data)
text = text1+text2
node = node1+node2
edge =edge1+edge2
error = error1+error2
node_label = node_label1+node_label2
answer = answer1+answer2
answer_choices = answer_choices1+answer_choices2
sequence_data = [text,node,edge,error,node_label,answer,answer_choices]
f = open('/remote-home/my/operation_detection/new_data/origin_data/old_sequence_created_data.json','w',encoding = 'utf8')
js.dump(sequence_data,f,ensure_ascii=False,indent=2)
f.close()

train_decision_data = open(train_decision_data_path,'r',encoding='utf8').readlines()
test_decision_data = open(test_decision_data_path,'r',encoding='utf8').readlines()
text1,node1,edge1,error1,node_label1,answer1,answer_choices1 = change_decision_data(train_decision_data)
text2,node2,edge2,error2,node_label2,answer2,answer_choices2 = change_decision_data(test_decision_data)
text = text1+text2
node = node1+node2
edge =edge1+edge2
error = error1+error2
node_label = node_label1+node_label2
answer = answer1+answer2
answer_choices = answer_choices1+answer_choices2
decision_data = [text,node,edge,error,node_label,answer,answer_choices]
f = open('/remote-home/my/operation_detection/new_data/origin_data/old_decision_created_data.json','w',encoding = 'utf8')
js.dump(decision_data,f,ensure_ascii=False,indent=2)
f.close()
	