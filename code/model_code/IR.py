import torch
from transformers import BertTokenizer, AdamW, BertConfig, BertModel
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import sampler,Dataset, DataLoader
import os
import random
import copy
import torch.nn as nn
import numpy as np
from user_dataset import Task1Dataset
from util import collate_fn
from tqdm import tqdm
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,accuracy_score
# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子
# setup_seed(20)

BERT_PATH = '/remote-home/my/operation_detection/bert-base-chinese'
VOCAB = 'vocab.txt'
sequence_data_PATH = '/remote-home/my/operation_detection/new_data/origin_data/sequence_created_data.json'
decision_data_PATH = '/remote-home/my/operation_detection/new_data/origin_data/decision_created_data.json'
procedure_triple_PATH = '/remote-home/my/operation_detection/new_data/procedure_graph/procedure_graph.csv'

torch.cuda.set_device(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_PATH, VOCAB))

# 加载
# TODO 新的数据需要更改这里
trainset = Task1Dataset(decision_data_PATH,sequence_data_PATH,tokenizer=tokenizer)
testset = Task1Dataset(decision_data_PATH,sequence_data_PATH,test = True,tokenizer=tokenizer)
trainloader = DataLoader(trainset, batch_size=1)
testloader = DataLoader(testset, batch_size=1)
model = BertModel.from_pretrained(BERT_PATH).to(device)
all_label = []
all_predict = []
thread = 0.82

with torch.no_grad():
    for batch in tqdm(testloader):
        text_ids,node,edge,node_label =  batch
        text_ids['input_ids'] = text_ids['input_ids'].squeeze(0).to(device)
        text_ids['token_type_ids'] = text_ids['token_type_ids'].squeeze(0).to(device)
        text_ids['attention_mask'] = text_ids['attention_mask'].squeeze(0).to(device)

        for key in node.keys():
            node[key]['input_ids'] = node[key]['input_ids'].squeeze(0).to(device)
            node[key]['token_type_ids'] = node[key]['token_type_ids'].squeeze(0).to(device)
            node[key]['attention_mask'] = node[key]['attention_mask'].squeeze(0).to(device)
        text_feature = model(input_ids = text_ids['input_ids'], attention_mask = text_ids['attention_mask'],token_type_ids = text_ids['token_type_ids'])[0][:,0,:]
        node_feature = dict()
        for key in node.keys():
            node_feature[key] =  model(input_ids = node[key]['input_ids'], attention_mask = node[key]['attention_mask'],token_type_ids = node[key]['token_type_ids'])[0][:,0,:]

        node_similarity = dict()
        for key in node.keys():
            if key not in ['[start]','[end]','[pad]']:
                node_similarity[key] = torch.cosine_similarity(node_feature[key],text_feature)
                if max(node_similarity[key])<thread:
                    node_similarity[key] = 1
                else:
                    node_similarity[key] = 0
                all_label.append(node_label[key].item())
                all_predict.append(node_similarity[key])

    print(classification_report(all_label, all_predict))
    print(precision_score(all_label, all_predict))
    print(recall_score(all_label, all_predict))
    print(f1_score(all_label, all_predict))
    print(accuracy_score(all_label, all_predict))



