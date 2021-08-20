import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import dgl
import argparse
import torch
from transformers import BertTokenizer, AdamW, BertConfig, BertModel
from torch_geometric.data import Data
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import sampler
import random
import copy
import torch.nn as nn
from torch_geometric.nn.data_parallel import DataParallel
import numpy as np
import logging
import coloredlogs
from user_dataset import Task1Dataset,GraphDataLoader,TrmDataloader,Task2Dataset,TrmDataloader_task2,GraphDataLoader_task2
from model import GraphNet,TreeLstmNet,TransformerNet,GraphNet_task2,TreeLstmNet_task2,TransformerNet_task2
from util import setup_seed,adjust_learning_rate,Noam
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

#配置parser

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--GPUid", type=int,choices=[0, 1, 2, 3], default=2, help="Choose which GPU")
parser.add_argument("-mul_gpu", "--mul_gpu", type=int, default=0, help="Choose wether mul GPU")
parser.add_argument("-b_path", "--BERT_PATH", type=str, default='bert-base-chinese', help="pretraining_BERT_model_path")
parser.add_argument("-p_path", "--PROCEDURE_KG_PATH", type=str, default='new_data/procedure_graph/procedure_graph.csv', help="procedure KG path")
parser.add_argument("-s_path", "--SEQUENCE_DATA_PATH", type=str, default='new_data/origin_data/old_sequence_created_data.json', help="Sequence data path")
parser.add_argument("-d_path", "--DECISION_DATA_PATH", type=str, default='new_data/origin_data/old_decision_created_data.json', help="Sequence data path")
parser.add_argument("-train_path", "--train_list_path", type=str, default='new_data/dataset_list/dgl_old_train_list.pkl', help="train list data path")
parser.add_argument("-test_path", "--test_list_path", type=str, default='new_data/dataset_list/dgl_old_test_list.pkl', help="test list data path")
parser.add_argument("-val_path", "--val_list_path", type=str, default='new_data/dataset_list/dgl_old_val_list.pkl', help="val list data path")
parser.add_argument("-t2_train_path", "--task2_train_list_path", type=str, default='new_data/dataset_list/task2_dgl_old_train_list.pkl', help="train list data path")
parser.add_argument("-t2_test_path", "--task2_test_list_path", type=str, default='new_data/dataset_list/task2_dgl_old_test_list.pkl', help="test list data path")
parser.add_argument("-t2_val_path", "--task2_val_list_path", type=str, default='new_data/dataset_list/task2_dgl_old_val_list.pkl', help="val list data path")
parser.add_argument('--local_rank', type=int, default=0,help='node rank for distributed training')
parser.add_argument("-seed", "--random_seed", type=int, default=20, help="random seed")
parser.add_argument("-decay", "--weight_decay", type=float, default=5e-4, help="l2 weight decay")
parser.add_argument("-w", "--number_workers", type=int, default=0, help="load data number of workers")
parser.add_argument("-pin", "--pin_mem", type=bool, default=False, help="pin memory")
parser.add_argument("-bs", "--batch_size", type=int, default=2, help="batch_size")
parser.add_argument("-es", "--epoch_size", type=int, default=50, help="epoch_size")
parser.add_argument("-lr", "--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-b_lr", "--bert_lr", type=float, default=2e-5, help="learning rate")
parser.add_argument("-mom", "--momentum", type=float, default=0.9, help="momentum rate")
parser.add_argument("-drop", "--dropout", type=float, default=0.5, help="dropout rate")
parser.add_argument("-s_gamma", "--scheduler_gamma", type=float, default=0.1, help="dropout rate")
parser.add_argument("-s_stepsize", "--scheduler_stepsize", type=float, default=30, help="dropout rate")
parser.add_argument("-net", "--network", default='TransformerNet', choices=['GraphNet','SequenceNet','TreeLstmNet','TransformerNet'], help="choose wich net")
parser.add_argument("-loss", "--loss", default='CE', help="choose loss")
parser.add_argument("-opt", "--optimizer", default='Adam', help="choose optimizer")
parser.add_argument("-load_dataset", "--load_dataset", type = int ,default=1, help="whether load saved dataset")
parser.add_argument("-load_check", "--load_checkpoint", type = int ,default=0, help="whether load saved checkpoint")
parser.add_argument("-p", "--procedure_KG", type = int,default=1, help="whether procedure KG subgraph")
parser.add_argument("-c", "--context", type = int,default=1, help="whether context")
parser.add_argument("-r", "--relative", type = int,default=1, help="compute the subgraph using relative node")
parser.add_argument("-strm", "--standard_trm", type = int,default=0, help="wether_standard_transformer")
parser.add_argument("-main_feature", "--add_main_feature", type = int,default=1, help="whether add main feature")
parser.add_argument('-task','--task',type=int,default=1,help='whether task1 or task2')
parser.add_argument('-m_path','--model_path',type=str,default='best_model/')
parser.add_argument('-log_path','--log_path',type=str,default='log/task2_result.log')
args = parser.parse_args()
if args.mul_gpu:
    torch.distributed.init_process_group(
            'nccl',
            init_method='env://'
        )
#--------
#
#配置 tensorboard
writer = SummaryWriter('tensorboard_file')
# 配置logger
logging.basicConfig(level=logging.DEBUG,
                    filename=args.log_path,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='[%(levelname)s] %(asctime)s %(filename)s %(message)s',
                    filemode='a'
                    )
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG',fmt = '[%(levelname)s] %(asctime)s \033[1;35m %(filename)s \033[0m %(message)s')
#-----------------------------------------------------------------------------------------------------------------
#
# train 主体
try:
    #设置随机数种子
    setup_seed(args.random_seed)
    #定义哪块gpu
    logger.warn("Expriment: net---%s  procedure KG:%s  context:  %s main_feature: %s"%(args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature)))
    #定义分词器
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.BERT_PATH, 'vocab.txt'))
    #定义数据集
    if args.task==1:
        logger.info('Task1')
        if not args.load_dataset:
            trainset = Task1Dataset(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH,tokenizer=tokenizer,procedure_triple_path=args.PROCEDURE_KG_PATH)
            testset = Task1Dataset(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH,test = True,tokenizer=tokenizer,procedure_triple_path=args.PROCEDURE_KG_PATH)
            valset = Task1Dataset(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH,dev=True,tokenizer=tokenizer,procedure_triple_path=args.PROCEDURE_KG_PATH)
            logger.info('create train list')
            train_list = [trainset[i] for i in tqdm(range(len(trainset)),ncols = 80)]
            logger.info('saving train list...')
            torch.save(train_list,args.train_list_path)

            logger.info('create test list')
            test_list = [testset[i] for i in tqdm(range(len(testset)),ncols = 80)]
            logger.info('saving test list...')
            torch.save(test_list,args.test_list_path)

            logger.info('create val list')
            val_list = [valset[i] for i in tqdm(range(len(valset)),ncols = 80)]
            logger.info('saving val list...')
            torch.save(val_list,args.val_list_path)
            raise NotImplementedError
        else:
            logger.info('loading train list')
            train_list = torch.load(args.train_list_path)
            logger.info('loading test list')
            test_list = torch.load(args.test_list_path)
            logger.info('loading val list')
            val_list = torch.load(args.val_list_path)
        #--------------------------------------------
        #
        # 多pgu下对训练集进行分发 dataloader
        if args.network == "TransformerNet":
            loader = TrmDataloader
        else:
            loader = GraphDataLoader
        
        if args.mul_gpu:
            trainloader = loader(train_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=True,sampler = DistributedSampler(train_list))
        else:
            trainloader = loader(train_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=True)
        valloader = loader(val_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=True)
        testloader = loader(test_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=True)

        logger.info('dataset is builded successfully!')
        # device = torch.device("cuda")
        #------------------------------
        #
        # 定义模型
        if args.load_checkpoint:
            raise NotImplementedError
        else:
            if args.network == "GraphNet":
                model = GraphNet(
                    args.BERT_PATH,
                    compute_relative = args.relative,
                    add_external = args.procedure_KG,
                    add_context = args.context,
                    add_main_feature= args.add_main_feature
                )
            elif args.network == "TransformerNet":
                model = TransformerNet(
                    args.BERT_PATH,
                    compute_relative = args.relative,
                    add_external = args.procedure_KG,
                    add_context = args.context,
                    add_main_feature= args.add_main_feature,
                    num_layers=3,
                    standard_transformer=args.standard_trm)
            else:
                model = TreeLstmNet(
                    768,
                    768,
                    bert_path = args.BERT_PATH,
                    add_external = args.procedure_KG,
                    add_context = args.context,
                    add_main_feature = args.add_main_feature
                )
        #---------------------------------------------------------------------------------------------------------------------
        #
    else:
        logger.info('Task2')
        # Task2的数据 和 模型输入
        if not args.load_dataset:
            trainset = Task2Dataset(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH,tokenizer=tokenizer,procedure_triple_path=args.PROCEDURE_KG_PATH)
            testset = Task2Dataset(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH,test = True,tokenizer=tokenizer,procedure_triple_path=args.PROCEDURE_KG_PATH)
            valset = Task2Dataset(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH,dev=True,tokenizer=tokenizer,procedure_triple_path=args.PROCEDURE_KG_PATH)
            logger.info('create train list')
            train_list = [trainset[i] for i in tqdm(range(len(trainset)),ncols = 80)]
            logger.info('saving train list...')
            torch.save(train_list,args.task2_train_list_path)

            logger.info('create test list')
            test_list = [testset[i] for i in tqdm(range(len(testset)),ncols = 80)]
            logger.info('saving test list...')
            torch.save(test_list,args.task2_test_list_path)

            logger.info('create val list')
            val_list = [valset[i] for i in tqdm(range(len(valset)),ncols = 80)]
            logger.info('saving val list...')
            torch.save(val_list,args.task2_val_list_path)
            raise NotImplementedError
        else:
            logger.info('loading train list')
            train_list = torch.load(args.task2_train_list_path)
            logger.info('loading test list')
            test_list = torch.load(args.task2_test_list_path)
            logger.info('loading val list')
            val_list = torch.load(args.task2_val_list_path)
        #--------------------------------------------
        #
        # 多pgu下对训练集进行分发 dataloader
        if args.network == "TransformerNet":
            loader = TrmDataloader_task2
        else:
            loader = GraphDataLoader_task2
        if args.mul_gpu:
            trainloader = loader(train_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=args.pin_mem,sampler = DistributedSampler(train_list))
        else:
            trainloader = loader(train_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=args.pin_mem)
        valloader = loader(val_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=args.pin_mem)
        testloader = loader(test_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=args.pin_mem)
        logger.info('dataset is builded successfully!')
        #---------------------------------------------------------------------------------------------------------------------
        #
        # 定义task2模型
        if args.network == "GraphNet":
            model = GraphNet_task2(
                args.BERT_PATH,
                compute_relative = args.relative,
                add_external = args.procedure_KG,
                add_context = args.context,
                add_main_feature= args.add_main_feature
            )
        elif args.network == "TransformerNet":
            model = TransformerNet_task2(
                args.BERT_PATH,
                compute_relative = args.relative,
                add_external = args.procedure_KG,
                add_context = args.context,
                add_main_feature= args.add_main_feature,
                num_layers=3,
                standard_transformer=args.standard_trm)
        else:
            model = TreeLstmNet_task2(
                768,
                768,
                bert_path = args.BERT_PATH,
                add_external = args.procedure_KG,
                add_context = args.context,
                add_main_feature = args.add_main_feature
            )
        if args.load_checkpoint:
            model.load_state_dict(torch.load(args.model_path+'task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+\
                                    'best_parameter.pkl'))
        logger.info('model is loaded successfully!')

    # 并行模型
    if not args.mul_gpu:
        # torch.cuda.set_device()
        device = torch.device("cuda:"+str(args.GPUid) if torch.cuda.is_available() else "cpu")
        logger.info('device is set on GPU-%d'%(args.GPUid))
        model = model.to(device)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=[args.local_rank],find_unused_parameters=True)
        logger.info("Let's use "+ str(torch.cuda.device_count())+" GPUs!")
    logger.info('model is created successfully!!')
    #---------------------------------------------------------------------------------------------------------------------
    #
    # 为bert 和下游任务设置不同的学习率
    if not args.mul_gpu:
        bert_params = list(map(id, model.bert.parameters()))
        downstrain_params = filter(lambda p: id(p) not in bert_params,model.parameters())
        params = [
            {'params':model.bert.parameters(),'lr':args.bert_lr,'bert':1},{'params':downstrain_params,'bert':0}
        ]
    else:
        bert_params = list(map(id, model.module.bert.parameters()))
        downstrain_params = filter(lambda p: id(p) not in bert_params,model.parameters())
        params = [
            {'params':model.module.bert.parameters(),'lr':args.bert_lr,'bert':1},{'params':downstrain_params,'bert':0}
        ]
    #---------------------------------------------------------------------------------------------------------------------
    #
    # 优化器
    if args.optimizer == "Adam":
        optimizer = optim.Adam(params,lr = args.lr)
    else:
        optimizer = optim.SGD(params,lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #loss
    if args.loss == "CE":
        loss_f = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    #---------------------------------------------------------------------------------------------------------------------
    #
    # 训练开始
    if args.task==1:
        current_i = 0
        best_f1 = 0
        for epoch in range(args.epoch_size):
            mean_loss = []
            if epoch%5 ==4:
                with torch.no_grad():
                    model.eval()
                    predict = []
                    label = []
                    for batch in tqdm(valloader,ncols = 80):
                        if len(batch)==4:
                            data,text,subgraph,topological_list = batch
                        else:
                            data,text,subgraph = batch
                        if isinstance(data, list):
                            data,text,subgraph = list(map(lambda x:x.to(device),data)),text.to(device),subgraph.to(device)
                            output,y = model(data,text,subgraph,topological_list)
                        else:
                            data,text,subgraph = data.to(device),text.to(device),subgraph.to(device)
                            y = data.ndata['y']
                            output = model(data,text,subgraph)
                        label += y.detach().cpu().numpy().tolist()
                        predict+= torch.max(output,dim=1)[1].detach().cpu().numpy().tolist()
                    acc = accuracy_score(label,predict)
                    pre = precision_score(label, predict, average='macro')
                    recall = recall_score(label, predict, average='macro')
                    f1 = f1_score(label, predict, average='macro')
                    
                    writer.add_scalar('task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+'/dev_f1', f1, epoch)
                    logger.error('---------------------------------------------------------------')
                    logger.info('Dev --- Acc:%f Pre:%f Recall:%f f1:%f best_f1:%f'%(acc,pre,recall,f1,best_f1))
                    logger.error('---------------------------------------------------------------')
                    predict = []
                    label = []
                    for batch in tqdm(testloader,ncols = 80):
                        if len(batch)==4:
                            data,text,subgraph,topological_list = batch
                        else:
                            data,text,subgraph = batch
                        if isinstance(data, list):
                            data,text,subgraph = list(map(lambda x:x.to(device),data)),text.to(device),subgraph.to(device)
                            output,y = model(data,text,subgraph,topological_list)
                        else:
                            data,text,subgraph = data.to(device),text.to(device),subgraph.to(device)
                            y = data.ndata['y']
                            output = model(data,text,subgraph)
                        label += y.detach().cpu().numpy().tolist()
                        predict+= torch.max(output,dim=1)[1].detach().cpu().numpy().tolist()
                    acc = accuracy_score(label,predict)
                    pre = precision_score(label, predict, average='macro')
                    recall = recall_score(label, predict, average='macro')
                    f1 = f1_score(label, predict, average='macro')
                    if f1>best_f1:
                        if args.mul_gpu:
                            torch.save(model.module.state_dict(), args.model_path+'task1_'+'best_parameter.pkl')
                            logger.info('best model saved!!!')
                        else:
                            torch.save(model.state_dict(), args.model_path+'task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+\
                                    'best_parameter.pkl')
                            logger.info('best model saved!!!')
                        best_f1 = f1
                    writer.add_scalar('task%s-%s-%s-%s-%s_'%\
                            (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+'/test_f1', f1, epoch)
                    logger.error('---------------------------------------------------------------')
                    logger.info('test --- Acc:%f Pre:%f Recall:%f f1:%f best_f1:%f'%(acc,pre,recall,f1,best_f1))
                    logger.error('---------------------------------------------------------------')
                    model.train()
                #----------------
                #
                # training
            # if args.network!='TransformerNet':
            optimizer = adjust_learning_rate(optimizer,epoch,args.scheduler_stepsize,args.scheduler_gamma,args.lr)
            for batch in tqdm(trainloader,ncols = 80):
                # if args.network=='TransformerNet':
                #     optimizer = Noam(optimizer,768,current_i+1,200)
                if len(batch)==4:
                    data,text,subgraph,topological_list = batch
                else:
                    data,text,subgraph = batch
                if isinstance(data, list):
                    data,text,subgraph = list(map(lambda x:x.to(device),data)),text.to(device),subgraph.to(device)
                    output,y = model(data,text,subgraph,topological_list)
                    y = y.to(device)
                else:
                    data,text,subgraph = data.to(device),text.to(device),subgraph.to(device)
                    y = data.ndata['y']
                    output = model(data,text,subgraph)
                loss = loss_f(output,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss.append(loss.item())
                writer.add_scalar('task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+'/loss', loss.item(), current_i)
                current_i+=1
            logger.info("eopch:%d mean_loss:%f"%(epoch,sum(mean_loss)/len(mean_loss)))
    else:
        current_i = 0
        best_f1 = 0
        for epoch in range(args.epoch_size):
            mean_loss = []
            if epoch%5 ==4:
                with torch.no_grad():
                    model.eval()
                    predict = []
                    label = []
                    for batch in tqdm(valloader,ncols = 80):
                        if len(batch)==6:
                            data,text,subgraph,topological_list,answer_choice,y = batch
                        else:
                            data,text,subgraph,answer_choice,y = batch
                        if isinstance(data, list):
                            data,text,subgraph,answer_choice,y = list(map(lambda x:x.to(device),data)),text.to(device),subgraph.to(device),answer_choice.to(device),y.to(device)
                            output= model(data,text,subgraph,topological_list,answer_choice)
                        else:
                            data,text,subgraph,answer_choice,y = data.to(device),text.to(device),subgraph.to(device),answer_choice.to(device),y.to(device)
                            output = model(data,text,subgraph,answer_choice)
                        label += y.detach().cpu().numpy().tolist()
                        predict+= torch.max(output,dim=1)[1].detach().cpu().numpy().tolist()
                    acc = accuracy_score(label,predict)
                    pre = precision_score(label, predict, average='macro')
                    recall = recall_score(label, predict, average='macro')
                    f1 = f1_score(label, predict, average='macro')
                    if f1>best_f1:
                        if args.mul_gpu:
                            torch.save(model.module.state_dict(), args.model_path+'task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+\
                                    'best_parameter.pkl')
                            logger.info('best model saved!!!')
                        else:
                            torch.save(model.state_dict(), args.model_path+'task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+\
                                    'best_parameter.pkl')
                            logger.info('best model saved!!!')
                        best_f1 = f1
                    writer.add_scalar('task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+'/dev_f1', f1, epoch)
                    logger.error('---------------------------------------------------------------')
                    logger.info('Dev --- Acc:%f Pre:%f Recall:%f f1:%f best_f1:%f'%(acc,pre,recall,f1,best_f1))
                    logger.error('---------------------------------------------------------------')
                    predict = []
                    label = []
                    for batch in tqdm(testloader,ncols = 80):
                        if len(batch)==6:
                            data,text,subgraph,topological_list,answer_choice,y = batch
                        else:
                            data,text,subgraph,answer_choice,y = batch
                        if isinstance(data, list):
                            data,text,subgraph,answer_choice,y = list(map(lambda x:x.to(device),data)),text.to(device),subgraph.to(device),answer_choice.to(device),y.to(device)
                            output= model(data,text,subgraph,topological_list,answer_choice)
                        else:
                            data,text,subgraph,answer_choice,y = data.to(device),text.to(device),subgraph.to(device),answer_choice.to(device),y.to(device)
                            output = model(data,text,subgraph,answer_choice)
                        label += y.detach().cpu().numpy().tolist()
                        predict+= torch.max(output,dim=1)[1].detach().cpu().numpy().tolist()
                    acc = accuracy_score(label,predict)
                    pre = precision_score(label, predict, average='macro')
                    recall = recall_score(label, predict, average='macro')
                    f1 = f1_score(label, predict, average='macro')
                    writer.add_scalar('task%s-%s-%s-%s-%s_'%\
                            (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+'/test_f1', f1, epoch)
                    logger.error('---------------------------------------------------------------')
                    logger.info('test --- Acc:%f Pre:%f Recall:%f f1:%f best_f1:%f'%(acc,pre,recall,f1,best_f1))
                    logger.error('---------------------------------------------------------------')
                    model.train()
                #----------------
                #
                # training
            # if args.network!='TransformerNet':
            optimizer = adjust_learning_rate(optimizer,epoch,args.scheduler_stepsize,args.scheduler_gamma,args.lr)
            for batch in tqdm(trainloader,ncols = 80):
                # if args.network=='TransformerNet':
                #     optimizer = Noam(optimizer,768,current_i+1,200)
                if len(batch)==6:
                    data,text,subgraph,topological_list,answer_choice,y = batch
                else:
                    data,text,subgraph,answer_choice,y = batch

                if isinstance(data, list):
                    data,text,subgraph,answer_choice,y = list(map(lambda x:x.to(device),data)),text.to(device),subgraph.to(device),answer_choice.to(device),y.to(device)
                    output= model(data,text,subgraph,topological_list,answer_choice)
                else:
                    data,text,subgraph,answer_choice,y = data.to(device),text.to(device),subgraph.to(device),answer_choice.to(device),y.to(device)
                    output = model(data,text,subgraph,answer_choice)
                loss = loss_f(output,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss.append(loss.item())
                writer.add_scalar('task%s-%s-%s-%s-%s_'%\
                                (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+'/loss', loss.item(), current_i)
                current_i+=1
            logger.info("eopch:%d mean_loss:%f"%(epoch,sum(mean_loss)/len(mean_loss)))
except Exception:
    logger.error('Faild to get result', exc_info=True)




