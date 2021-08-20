import torch.nn as nn
from transformers import BertModel
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv
from dgl.nn import GraphConv
import torch_geometric.transforms as T
import os.path as osp
import warnings
import dgl
import math
from torch.nn import LayerNorm
warnings.filterwarnings("ignore")
# Task1 model
class GraphNet(torch.nn.Module):
    """利用图模型做主干做任务1的模型
    """
    def __init__(self,bert_path,compute_relative = True,add_external = False,add_context = False, add_main_feature = True):
        super(GraphNet, self).__init__()
        self.conv1 = GraphConv(768, 768, norm='both', weight=True, bias=True)
        self.KG_encoding_layer = GCNConv(768,768)
        self.bert = BertModel.from_pretrained(bert_path)
        self.compute_relative = compute_relative
        self.add_external = add_external
        self.add_context = add_context
        self.add_main_feature = add_main_feature
        # 冻结bert前10层
        unfreeze_layers = ['layer.10','layer.11','bert.pooler','out.']
        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        # 输出层
        if self.add_external and self.add_context:
            #加入上下文和全局流程子图
            self.out = nn.Linear(768*3,2)
        elif not self.add_external and self.add_context:
            self.out = nn.Linear(768*2,2)
        elif self.add_external and not self.add_context:
            self.out = nn.Linear(768*2,2)
        else:
            self.out = nn.Linear(768,2)
    def get_origintext_embedding(self,origintext):
        """获取上下文的bert embedding

        Args:
            origintext (tensor): the input context tensor

        Returns:
            bert output embedding
        """
        text_representation = []
        for i in range(origintext.shape[0]):
            text_representation.append(self.bert(input_ids = torch.unsqueeze(origintext[i,:512],dim=0),
                                                 attention_mask = torch.unsqueeze(origintext[i,512:1024],dim=0),
                                                 token_type_ids = torch.unsqueeze(origintext[i,1024:],dim=0))[0])
        text_representation = torch.cat(text_representation,dim=0)
        return text_representation
    def get_node_embedding(self,input_graph_x):
        '''
        input:  input_graph( input_graph.x is the node feature matrix ,shape as [ n , bert encoding max length ])
        output: the bert embedding matrix of input_graph
        '''
        node_embedding = []
        bert_encoding_len = int(input_graph_x.shape[1]/3)
        for i in range(input_graph_x.shape[0]):
            node_embedding.append(self.bert(
                                    input_ids = torch.unsqueeze(input_graph_x[i,:bert_encoding_len],dim=0),
                                    attention_mask = torch.unsqueeze(input_graph_x[i,bert_encoding_len:bert_encoding_len*2],dim=0),
                                    token_type_ids = torch.unsqueeze(input_graph_x[i,bert_encoding_len*2:bert_encoding_len*3],dim=0))[0][:,0,:])
        return torch.cat(node_embedding,dim=0)
    def external_knowledge_split(self,external_graph,external_node_out):
        '''
        input:  external_graph,external_node_out
        output: external_batch_representation
        '''
        batch_num = external_graph.batch_num_nodes().detach().cpu().numpy().tolist()
        external_batch_index = []
        last = 0
        for i in range(len(batch_num)):
            external_batch_index+=[last+batch_num[i]]
            last = external_batch_index[-1]

        external_batch_representation = []
        start = 0
        for i in range(len(external_batch_index)):
            end = external_batch_index[i]
            external_batch_representation.append(external_node_out[start:end,:])
            start = end
        return external_batch_representation
    def compute_node_attention_with(self,data,main_out,external_batch_representation):
        '''
        input: data (use to get node batch split)
               main_out(node feature through GAT)
               external_batch_representation ([b,n,f])
        output: x_external_representation
        '''
        batch_num = data.batch_num_nodes().detach().cpu().numpy().tolist()
        batch_list = []
        for i in range(len(batch_num)):
            batch_list+=[i]*batch_num[i]
        x_external_representation = []
        for i,index in enumerate(batch_list):
            #attention 加在这里
            attention_vector = F.softmax(torch.mm(torch.unsqueeze(main_out[i,:],dim=0),external_batch_representation[index].permute(1,0)),dim=1)
            x_external_representation.append(torch.mm(attention_vector,external_batch_representation[index]))
        x_external_representation = torch.cat(x_external_representation,dim=0)
        return x_external_representation
    def compute_most_relative_representation(self,data,main_out,external_batch_representation):
        '''
        input: data (use to get node batch split)
               main_out(node feature through GAT)
               external_batch_representation ([b,n,f])
        output: x_external_representation
        '''
        batch_list = data.batch.detach().cpu().numpy().tolist()
        x_external_representation = []
        for i,index in enumerate(batch_list):
            #attention 加在这里
            most_relative_index = torch.argmax(F.softmax(torch.mm(torch.unsqueeze(main_out[i,:],dim=0),external_batch_representation[index].permute(1,0)),dim=1))
            x_external_representation.append(external_batch_representation[index][most_relative_index.item()])
        x_external_representation = torch.stack(x_external_representation,dim=0)
        return x_external_representation
    def forward(self,data,text,external_graph):
        if self.add_context:
            text_representation = self.get_origintext_embedding(text)
        data.ndata['x'] = self.get_node_embedding(data.ndata['x'])
        if self.add_main_feature:
            new_data = dgl.add_self_loop(data)
            main_out= self.conv1(new_data, data.ndata['x'])
        else:
            main_out = data.ndata['x']
        if self.add_external:
            external_graph.x = self.get_node_embedding(external_graph.x)
            external_node_out = F.relu(self.KG_encoding_layer(external_graph.x, external_graph.edge_index))
            external_batch_representation = self.external_knowledge_split(external_graph,external_node_out)
            if self.compute_relative:
                x_external_representation = self.compute_most_relative_representation(data,main_out,external_batch_representation)
            else:
                x_external_representation = self.compute_node_attention_with(data,main_out,external_batch_representation)
        if self.add_context:
            x_text_representation = self.compute_node_attention_with(data,main_out,text_representation)
        if self.add_external and self.add_context:
            x = torch.cat([main_out,x_text_representation,x_external_representation],dim=1)
            output = self.out(x)
        elif not self.add_external and self.add_context:
            x = torch.cat([main_out,x_text_representation],dim=1)
            output = self.out(x)
        elif self.add_external and not self.add_context:
            x = torch.cat([main_out,x_external_representation],dim=1)
            output = self.out(x)
        else:
            output = self.out(main_out)
        return output
class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size,bias=False)
        self.W_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        batch_size = nodes.batch_size()
        c = []
        for i in range(batch_size):
            c_j = torch.sum(torch.sigmoid(self.U_f(nodes.mailbox['h'][i,:,:])+self.W_f(nodes.data['x'][i,:]))*nodes.mailbox['c'][i],0)
            c.append(c_j)
        h_sum = torch.sum(nodes.mailbox['h'],dim=1)
        # second term of equation (5)
        c = torch.stack(c,dim=0)
        return {'iou': self.U_iou(h_sum), 'c': c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * torch.tanh(c)
        return {'h' : h, 'c' : c}
class TreeLstmNet(GraphNet):
    def __init__(self,lstm_x_size,lstm_hidden_size,bert_path,compute_relative = True,add_external = False,add_context = False, add_main_feature = True):
        super(TreeLstmNet, self).__init__(bert_path,compute_relative,add_external,add_context, add_main_feature)
        self.init_h = None
        self.init_c = None
        self.lstm_hidden_size = lstm_hidden_size
        self.cell = TreeLSTMCell(lstm_x_size,lstm_hidden_size)
    def compute_node_attention_with(self,data,main_out,external_batch_representation):
        '''
        input: data (use to get node batch split)
               main_out(node feature through GAT)
               external_batch_representation ([b,n,f])
        output: x_external_representation
        '''
        batch_num = data.batch_num_nodes().detach().cpu().numpy().tolist()
        batch_list = []
        for i in range(len(batch_num)):
            batch_list+=[i]*batch_num[i]
        x_external_representation = []
        for i,index in enumerate(batch_list):
            #attention 加在这里
            attention_vector = F.softmax(torch.mm(torch.unsqueeze(main_out[i,:],dim=0),external_batch_representation[index].permute(1,0)),dim=1)
            x_external_representation.append(torch.mm(attention_vector,external_batch_representation[index]))
        x_external_representation = torch.cat(x_external_representation,dim=0)
        return x_external_representation
    def compute_most_relative_representation(self,data,main_out,external_batch_representation):
        '''
        input: data (use to get node batch split)
               main_out(node feature through GAT)
               external_batch_representation ([b,n,f])
        output: x_external_representation
        '''
        batch_num = data.batch_num_nodes().detach().cpu().numpy().tolist()
        batch_list = []
        for i in range(len(batch_num)):
            batch_list+=[i]*batch_num[i]
        x_external_representation = []
        for i,index in enumerate(batch_list):
            #attention 加在这里
            most_relative_index = torch.argmax(F.softmax(torch.mm(torch.unsqueeze(main_out[i,:],dim=0),external_batch_representation[index].permute(1,0)),dim=1))
            x_external_representation.append(external_batch_representation[index][most_relative_index.item()])
        x_external_representation = torch.stack(x_external_representation,dim=0)
        return x_external_representation
    def rand_init_hidden(self, node_number):
        """
        初始化 h 和 c 向量
        """
        return (Variable(torch.randn(node_number, self.lstm_hidden_size)).to(next(self.parameters()).device),
                Variable(torch.randn(node_number, self.lstm_hidden_size)).to(next(self.parameters()).device))
    
    def forward(self,data,text,external_graph):
        # text 编码
        if self.add_context:
            text_representation = self.get_origintext_embedding(text)
        # tree lstm主体
        data.ndata['x'] = self.get_node_embedding(data.ndata['x'])
        h,c = self.rand_init_hidden(data.number_of_nodes())
        data.ndata['iou'] = self.cell.W_iou(data.ndata['x'])
        data.ndata['h'] = h
        data.ndata['c'] = c
        traversal_order = tuple(map(lambda x:x.to(next(self.parameters()).device),dgl.topological_nodes_generator(data)))
        data.prop_nodes(traversal_order,message_func=self.cell.message_func,reduce_func = self.cell.reduce_func, apply_node_func = self.cell.apply_node_func)
        main_out = data.ndata['h']
        # attention 和 relative node计算
        if self.add_external:
            external_graph.x = self.get_node_embedding(external_graph.x)
            external_node_out = F.relu(self.KG_encoding_layer(external_graph.x, external_graph.edge_index))
            external_batch_representation = self.external_knowledge_split(external_graph,external_node_out)
            if self.compute_relative:
                x_external_representation = self.compute_most_relative_representation(data,main_out,external_batch_representation)
            else:
                x_external_representation = self.compute_node_attention_with(data,main_out,external_batch_representation)
        if self.add_context:
            x_text_representation = self.compute_node_attention_with(data,main_out,text_representation)
        if self.add_external and self.add_context:
            x = torch.cat([main_out,x_text_representation,x_external_representation],dim=1)
            output = self.out(x)
        elif not self.add_external and self.add_context:
            x = torch.cat([main_out,x_text_representation],dim=1)
            output = self.out(x)
        elif self.add_external and not self.add_context:
            x = torch.cat([main_out,x_external_representation],dim=1)
            output = self.out(x)
        else:
            output = self.out(main_out)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class TransformerNet(GraphNet):
    def __init__(self,bert_path,compute_relative = True,add_external = False,add_context = False, add_main_feature = True,nhead=8,d_model = 768,num_layers = 6,standard_transformer = True):
        super(TransformerNet, self).__init__(bert_path,compute_relative,add_external,add_context, add_main_feature)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.reverse_transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.position_encoder = PositionalEncoding(d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.standard_transformer = standard_transformer
        self.position_net = nn.Linear(768,768)
        self.bi_direction_net = nn.Linear(768,768)
        if self.add_context and self.add_external:
            self.out = nn.Linear(768*3,2)
        if self.add_context:
            self.context_mul_head_Att = nn.MultiheadAttention(d_model, nhead)
        if self.add_external:
            self.KG_mul_head_Att = nn.MultiheadAttention(d_model, nhead)

    def get_batch_data_xfeature(self,data):
        '''
        data 是一个list
        对list每个元素进行编码
        '''
        for d in data:
            d.ndata['x'] = self.get_node_embedding(d.ndata['x'])
        return data
    def get_batch_topological_indexlist(self,data,topological_list):
        '''
        获得每个graph的拓扑排序
        '''
        max_len = 0
        y_list=[]
        for i in range(len(topological_list)):
            max_len = max(len(topological_list[i]),max_len)
            y_list +=[data[i].ndata['y'][index].item() for index in topological_list[i]]

        for t in topological_list:
            t+=[-1]*(max_len-len(t))
        
        return topological_list,torch.tensor(y_list)
    def get_topological_sequence_feature(self,data,traversal_order):
        feature = []
        mask = []
        for i,d in enumerate(data):
            d_feature = []
            d_mask = []
            for j in traversal_order[i]:
                if j !=-1:
                    d_feature.append(d.ndata['x'][j])
                    d_mask.append(1)
                else:
                    d_feature.append(torch.zeros(768).to(next(self.parameters()).device))
                    d_mask.append(0)
            feature.append(torch.stack(d_feature))
            mask.append(torch.tensor(d_mask))
        feature = torch.stack(feature).permute(1,0,2)
        mask = torch.stack(mask)
        return feature,mask
    def compute_node_attention_with(self,main_out,external_batch_representation):
        '''
        input: data (use to get node batch split)
               main_out(node feature through GAT)
               external_batch_representation ([b,n,f])
        output: x_external_representation
        '''
        batch_num,node_num,_ = main_out.shape
        batch_list = []
        for i in range(batch_num):
            batch_list+=[i]*node_num
        x_external_representation = []
        main_out = main_out.reshape(-1,768)
        for i,index in enumerate(batch_list):
            #attention 加在这里
            attention_vector = F.softmax(torch.mm(torch.unsqueeze(main_out[i,:],dim=0),external_batch_representation[index].permute(1,0)),dim=1)
            x_external_representation.append(torch.mm(attention_vector,external_batch_representation[index]))
        x_external_representation = torch.cat(x_external_representation,dim=0)
        return x_external_representation
    def compute_most_relative_representation(self,main_out,external_batch_representation):
        '''
        input: data (use to get node batch split)
               main_out(node feature through GAT)
               external_batch_representation ([b,n,f])
        output: x_external_representation
        '''
        batch_num,node_num,_ = main_out.shape
        batch_list = []
        for i in range(batch_num):
            batch_list+=[i]*node_num
        x_external_representation = []
        main_out = main_out.reshape(-1,768)
        for i,index in enumerate(batch_list):
            #attention 加在这里
            most_relative_index = torch.argmax(F.softmax(torch.mm(torch.unsqueeze(main_out[i,:],dim=0),external_batch_representation[index].permute(1,0)),dim=1))
            x_external_representation.append(external_batch_representation[index][most_relative_index.item()])
        x_external_representation = torch.stack(x_external_representation,dim=0)
        return x_external_representation
    def create_edge_mask(self,data,topological_list,nhead,reverse = False):
        max_len = max([d.num_nodes() for d in data])
        edge_mask = torch.stack([1-abs(torch.diag(torch.ones(max_len),0)) for i in range(len(topological_list))]).bool()
        flag = False
        for i,t_list in enumerate(topological_list):
            already_attend = []
            if reverse:
                t_list = t_list.copy()
                t_list.reverse()
            for t in t_list:
                if t.shape[0]==1:
                    edge_mask[i,t.item(),t.item()]=flag
                    for index in already_attend:
                        edge_mask[i,t.item(),index]=flag
                    already_attend.append(t.item())
                else:
                    node_list = t.cpu().numpy().tolist()
                    for node in node_list:
                        edge_mask[i,node,node]=flag
                        for index in already_attend:
                            edge_mask[i,node,index]=flag
                    already_attend+=node_list
        edge_mask_final = []
        for i in range(edge_mask.shape[0]):
            for j in range(nhead):
                edge_mask_final.append(edge_mask[i].clone().detach())
        edge_mask = torch.stack(edge_mask_final).to(next(self.parameters()).device)
        return edge_mask
    def de_feature_mask(self,main_out,mask):
        feature = []
        N,S = mask.shape 
        for i in range(N):
            for j in range(S):
                if mask[i][j]==1:
                    feature.append(main_out[i,j])
        feature = torch.stack(feature)
        return feature
    def external_knowledge_split(self,external_graph,external_node_out):
        '''
        input:  external_graph,external_node_out
        output: external_batch_representation
        '''
        external_batch_list = external_graph.batch.detach().cpu().numpy().tolist()
        #获取subgraph batch分段
        external_batch_index = []
        for i in range(len(external_batch_list)-1):
            if external_batch_list[i]!=external_batch_list[i+1]:
                external_batch_index.append(i+1)
        external_batch_index.append(len(external_batch_list))
        external_batch_representation = []
        start = 0
        for i in range(len(external_batch_index)):
            end = external_batch_index[i]
            external_batch_representation.append(external_node_out[start:end,:])
            start = end
        return external_batch_representation
    def alian_KG_feature(self, external_batch_representation):
        """
        补齐KGfeature，用0代替
        """
        batch_num = len(external_batch_representation)
        max_len = max([external_batch_representation[i].shape[0] for i in range(batch_num)])
        for i,repersentation in enumerate(external_batch_representation):
            m = nn.ConstantPad2d((0,0,0,max_len-repersentation.shape[0]),0)
            external_batch_representation[i] = m(repersentation)
        external_batch_representation = torch.stack(external_batch_representation)
        return external_batch_representation
    def forward(self,data,text,external_graph,topological_list):
        data= self.get_batch_data_xfeature(data)
        # if not self.standard_transformer:
        #     order_mask = self.create_edge_mask(data,topological_list,self.nhead)
        #     reverse_mask = self.create_edge_mask(data,topological_list,self.nhead,reverse=True)

        topological_list = [torch.cat(topo).cpu().numpy().tolist() for topo in topological_list]# 将获得的拓扑排序转为序列
        traversal_order,new_y =self.get_batch_topological_indexlist(data,topological_list)
        src,data_mask = self.get_topological_sequence_feature(data,traversal_order)
        # 加入position encoding
        if self.standard_transformer:
            src = src+self.position_encoder(src)
            main_out = self.transformer_encoder(src)
            # main_out = F.relu(self.bi_direction_net(main_out))
            # main_out = src
            # main_out = torch.cat([d.ndata['x'] for d in data])
            # new_y = torch.cat([d.ndata['y'] for d in data])
        else:
            src = src+self.position_encoder(src)
            main_out = self.transformer_encoder(src)+self.position_net(self.position_encoder(src))
            # position_embedding = F.relu(self.position_net(self.position_encoder(src)))
            # src = src+self.position_encoder(src)
            # 使用 position feature
            # main_out = self.transformer_encoder(src,mask=order_mask)
            # reverse_main_out = self.reverse_transformer_encoder(src,mask=reverse_mask)
            # main_out += reverse_main_out+self.position_net(self.position_encoder(src))
            # main_out = F.relu(self.bi_direction_net(main_out))
                     
        if self.add_context:
            text_representation = self.get_origintext_embedding(text).permute(1,0,2)
            x_text_representation = self.context_mul_head_Att(main_out,text_representation,text_representation)[0]

        if self.add_external:
            external_graph.x = self.get_node_embedding(external_graph.x)
            external_node_out = F.relu(self.KG_encoding_layer(external_graph.x, external_graph.edge_index))
            external_batch_representation = self.external_knowledge_split(external_graph,external_node_out)
            external_batch_representation = self.alian_KG_feature(external_batch_representation).permute(1,0,2)
            x_external_representation = self.KG_mul_head_Att(main_out,external_batch_representation,external_batch_representation)[0]

        
        if self.add_external and self.add_context:
            main_out = torch.cat([main_out,x_text_representation,x_external_representation],dim=2).permute(1,0,2)
        elif not self.add_external and self.add_context:
            main_out = torch.cat([main_out,x_text_representation],dim=2).permute(1,0,2)
        elif self.add_external and not self.add_context:
            main_out = torch.cat([main_out,x_external_representation],dim=2).permute(1,0,2)
        else:
            main_out = main_out.permute(1,0,2)
        output = self.out(main_out)
        output = self.de_feature_mask(output,data_mask)
        return output,new_y

#Task2 model
class GraphNet_task2(GraphNet):
    def __init__(self,bert_path,compute_relative = True,add_external = False,add_context = False, add_main_feature = True,nhead=8,d_model = 768):
        super(GraphNet_task2, self).__init__(bert_path,compute_relative,add_external,add_context, add_main_feature)
        self.out = nn.Linear(768*2,1)
    def alian_KG_feature(self, external_batch_representation):
        """
        补齐KGfeature，用0代替
        """
        batch_num = len(external_batch_representation)
        max_len = max([external_batch_representation[i].shape[0] for i in range(batch_num)])
        for i,repersentation in enumerate(external_batch_representation):
            m = nn.ConstantPad2d((0,0,0,max_len-repersentation.shape[0]),0)
            external_batch_representation[i] = m(repersentation)
        external_batch_representation = torch.stack(external_batch_representation)
        return external_batch_representation
    def answer_embedding(self,answer_choice):
        '''
        choosed_answer
        '''
        answer_embedding = []
        bert_encoding_len = int(answer_choice.shape[2]/3)
        for i in range(answer_choice.shape[0]):
            onebatch_answer = []
            for j in range(answer_choice.shape[1]):
                onebatch_answer.append(self.bert(input_ids = torch.unsqueeze(answer_choice[i,j,:bert_encoding_len],dim=0),
                                        token_type_ids = torch.unsqueeze(answer_choice[i,j,bert_encoding_len:bert_encoding_len*2],dim=0),
                                        attention_mask = torch.unsqueeze(answer_choice[i,j,bert_encoding_len*2:bert_encoding_len*3],dim=0))[0][:,0,:])
            onebatch_answer = torch.unsqueeze(torch.cat(onebatch_answer,dim=0),dim=0)
            answer_embedding.append(onebatch_answer)
        return torch.cat(answer_embedding,dim=0)
    def external_knowledge_split(self,external_graph,external_node_out):
        '''
        input:  external_graph,external_node_out
        output: external_batch_representation
        '''
        batch_num = external_graph.batch_num_nodes().detach().cpu().numpy().tolist()
        external_batch_index = []
        last = 0
        for i in range(len(batch_num)):
            external_batch_index+=[last+batch_num[i]]
            last = external_batch_index[-1]

        external_batch_representation = []
        start = 0
        for i in range(len(external_batch_index)):
            end = external_batch_index[i]
            external_batch_representation.append(external_node_out[start:end,:])
            start = end
        return external_batch_representation
    def forward(self,data,text,external_graph,answer_choice):
        # answer_choice [N,3,273]

        data.ndata['x'] = self.get_node_embedding(data.ndata['x'])
        if self.add_main_feature:
            new_data = dgl.add_self_loop(data)
            main_out= self.conv1(new_data, data.ndata['x'])
        else:
            main_out = data.ndata['x']
        # 对main_out 恢复batch
        main_out = self.external_knowledge_split(data,main_out)
        main_out = self.alian_KG_feature(main_out)
        main_out = torch.mean(main_out,dim=1)
        QA_feature = self.answer_embedding(answer_choice).permute(1,0,2)
        feature = torch.cat([QA_feature,main_out.repeat(3,1,1)],dim=2)
        output = self.out(feature).reshape(3,-1).permute(1,0)
        return output
class TreeLstmNet_task2(TreeLstmNet):
    def __init__(self,lstm_x_size,lstm_hidden_size,bert_path,compute_relative = True,add_external = False,add_context = False, add_main_feature = True,d_model = 768,nhead=8):
        super(TreeLstmNet_task2, self).__init__(lstm_x_size,lstm_hidden_size,bert_path,compute_relative,add_external,add_context, add_main_feature)
        self.init_h = None
        self.init_c = None
        self.lstm_hidden_size = lstm_hidden_size
        self.cell = TreeLSTMCell(lstm_x_size,lstm_hidden_size)
        self.answer_mul_Attention = nn.MultiheadAttention(d_model, nhead)
        self.out = nn.Linear(768*2,1)
    def external_knowledge_split(self,external_graph,external_node_out):
        '''
        input:  external_graph,external_node_out
        output: external_batch_representation
        '''
        batch_num = external_graph.batch_num_nodes().detach().cpu().numpy().tolist()
        external_batch_index = []
        last = 0
        for i in range(len(batch_num)):
            external_batch_index+=[last+batch_num[i]]
            last = external_batch_index[-1]

        external_batch_representation = []
        start = 0
        for i in range(len(external_batch_index)):
            end = external_batch_index[i]
            external_batch_representation.append(external_node_out[start:end,:])
            start = end
        return external_batch_representation
    def alian_KG_feature(self, external_batch_representation):
        """
        补齐KGfeature，用0代替
        """
        batch_num = len(external_batch_representation)
        max_len = max([external_batch_representation[i].shape[0] for i in range(batch_num)])
        for i,repersentation in enumerate(external_batch_representation):
            m = nn.ConstantPad2d((0,0,0,max_len-repersentation.shape[0]),0)
            external_batch_representation[i] = m(repersentation)
        external_batch_representation = torch.stack(external_batch_representation)
        return external_batch_representation
    def answer_embedding(self,answer_choice):
        '''
        choosed_answer
        '''
        answer_embedding = []
        bert_encoding_len = int(answer_choice.shape[2]/3)
        for i in range(answer_choice.shape[0]):
            onebatch_answer = []
            for j in range(answer_choice.shape[1]):
                onebatch_answer.append(self.bert(input_ids = torch.unsqueeze(answer_choice[i,j,:bert_encoding_len],dim=0),
                                        token_type_ids = torch.unsqueeze(answer_choice[i,j,bert_encoding_len:bert_encoding_len*2],dim=0),
                                        attention_mask = torch.unsqueeze(answer_choice[i,j,bert_encoding_len*2:bert_encoding_len*3],dim=0))[0][:,0,:])
            onebatch_answer = torch.unsqueeze(torch.cat(onebatch_answer,dim=0),dim=0)
            answer_embedding.append(onebatch_answer)
        return torch.cat(answer_embedding,dim=0)
    
    def forward(self, data, text, external_graph, answer_choice):
        # tree lstm主体
        data.ndata['x'] = self.get_node_embedding(data.ndata['x'])
        h,c = self.rand_init_hidden(data.number_of_nodes())
        data.ndata['iou'] = self.cell.W_iou(data.ndata['x'])
        data.ndata['h'] = h
        data.ndata['c'] = c
        traversal_order = tuple(map(lambda x:x.to(next(self.parameters()).device),dgl.topological_nodes_generator(data)))
        data.prop_nodes(traversal_order,message_func=self.cell.message_func,reduce_func = self.cell.reduce_func, apply_node_func = self.cell.apply_node_func)
        main_out = data.ndata['h']

        # answer_embedding
        main_out = self.external_knowledge_split(data,main_out)
        main_out = self.alian_KG_feature(main_out)
        main_out = torch.mean(main_out,dim=1)
        QA_feature = self.answer_embedding(answer_choice).permute(1,0,2)
        
        feature = torch.cat([QA_feature,main_out.repeat(3,1,1)],dim=2)
        output = self.out(feature).reshape(3,-1).permute(1,0)
        #输出层
        return output
class TransformerNet_task2(TransformerNet):
    def __init__(self,bert_path,compute_relative = True,add_external = False,add_context = False, add_main_feature = True,nhead=8,d_model = 768,num_layers = 6,standard_transformer = True):
        super(TransformerNet_task2, self).__init__(bert_path,compute_relative,add_external,add_context, add_main_feature,nhead,d_model,num_layers,standard_transformer)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.reverse_transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.position_encoder = PositionalEncoding(d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.standard_transformer = standard_transformer
        self.position_net = nn.Linear(768,768)
        self.bi_direction_net = nn.Linear(768,768)
        if self.add_context and self.add_external:
            self.out = nn.Linear(768*2,2)
        if self.add_context:
            self.context_mul_head_Att = nn.MultiheadAttention(d_model, nhead)
        if self.add_external:
            self.KG_mul_head_Att = nn.MultiheadAttention(d_model, nhead)
        self.answer_mul_Attention = nn.MultiheadAttention(d_model, nhead)
        if self.add_external and self.add_context:
            #加入上下文和全局流程子图
            # self.feature_merge = nn.Linear(768*3,768)
            self.out = nn.Linear(768*4,1)
        elif self.add_external and not self.add_context:
            #加入上下文和全局流程子图
            # self.feature_merge = nn.Linear(768*3,768)
            self.out = nn.Linear(768*3,1)
        elif not self.add_external and self.add_context:
            #加入上下文和全局流程子图
            # self.feature_merge = nn.Linear(768*3,768)
            self.out = nn.Linear(768*3,1)
        else:
            self.out = nn.Linear(768*2,1)

    def answer_embedding(self,answer_choice):
        '''
        choosed_answer
        '''
        answer_embedding = []
        bert_encoding_len = int(answer_choice.shape[2]/3)
        for i in range(answer_choice.shape[0]):
            onebatch_answer = []
            for j in range(answer_choice.shape[1]):
                onebatch_answer.append(self.bert(input_ids = torch.unsqueeze(answer_choice[i,j,:bert_encoding_len],dim=0),
                                        token_type_ids = torch.unsqueeze(answer_choice[i,j,bert_encoding_len:bert_encoding_len*2],dim=0),
                                        attention_mask = torch.unsqueeze(answer_choice[i,j,bert_encoding_len*2:bert_encoding_len*3],dim=0))[0][:,0,:])
            onebatch_answer = torch.unsqueeze(torch.cat(onebatch_answer,dim=0),dim=0)
            answer_embedding.append(onebatch_answer)
        return torch.cat(answer_embedding,dim=0)
    def external_knowledge_split(self,external_graph,external_node_out):
        '''
        input:  external_graph,external_node_out
        output: external_batch_representation
        '''
        external_batch_list = external_graph.batch.detach().cpu().numpy().tolist()
        #获取subgraph batch分段
        external_batch_index = []
        for i in range(len(external_batch_list)-1):
            if external_batch_list[i]!=external_batch_list[i+1]:
                external_batch_index.append(i+1)
        external_batch_index.append(len(external_batch_list))
        external_batch_representation = []
        start = 0
        for i in range(len(external_batch_index)):
            end = external_batch_index[i]
            external_batch_representation.append(external_node_out[start:end,:])
            start = end
        return external_batch_representation
    def forward(self, data, text, external_graph, topological_list, answer_choice):
        data= self.get_batch_data_xfeature(data)
        topological_list = [torch.cat(topo).cpu().numpy().tolist() for topo in topological_list]# 将获得的拓扑排序转为序列
        traversal_order,new_y =self.get_batch_topological_indexlist(data,topological_list)
        src,_ = self.get_topological_sequence_feature(data,traversal_order)
        # 加入position encoding
        src = src+self.position_encoder(src)
        if self.standard_transformer:
            main_out = self.transformer_encoder(src)
        else:
            main_out = self.transformer_encoder(src)+self.position_net(self.position_encoder(src))
        if self.add_context:
            text_representation = self.get_origintext_embedding(text).permute(1,0,2)
            x_text_representation = self.context_mul_head_Att(main_out,text_representation,text_representation)[0]

        if self.add_external:
            external_graph.x = self.get_node_embedding(external_graph.x)
            external_node_out = F.relu(self.KG_encoding_layer(external_graph.x, external_graph.edge_index))
            external_batch_representation = self.external_knowledge_split(external_graph,external_node_out)
            external_batch_representation = self.alian_KG_feature(external_batch_representation).permute(1,0,2)
            x_external_representation,_ = self.KG_mul_head_Att(main_out,external_batch_representation,external_batch_representation)

        
        if self.add_external and self.add_context:
            main_out = torch.cat([main_out,x_text_representation,x_external_representation],dim=2)
        elif not self.add_external and self.add_context:
            main_out = torch.cat([main_out,x_text_representation],dim=2)
        elif self.add_external and not self.add_context:
            main_out = torch.cat([main_out,x_external_representation],dim=2)
        
        main_out = torch.mean(main_out,dim=0)
        QA_feature = self.answer_embedding(answer_choice).permute(1,0,2)
        feature = torch.cat([QA_feature,main_out.repeat(3,1,1)],dim=2)
        output = self.out(feature).reshape(3,-1).permute(1,0)
        return output,_
        #输出层