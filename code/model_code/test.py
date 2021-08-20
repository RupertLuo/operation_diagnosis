import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import argparse
from util import setup_seed
import logging
import coloredlogs
from transformers import BertTokenizer
from user_dataset import Task1Dataset,GraphDataLoader,TrmDataloader
from model import GraphNet,TreeLstmNet,TransformerNet
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--GPUid", type=int,choices=[0, 1, 2,3], default=0, help="Choose which GPU")
parser.add_argument("-b_path", "--BERT_PATH", type=str, default='/remote-home/my/operation_detection/bert-base-chinese', help="pretraining_BERT_model_path")
parser.add_argument("-p_path", "--PROCEDURE_KG_PATH", type=str, default='/remote-home/my/operation_detection/new_data/procedure_graph/procedure_graph.csv', help="procedure KG path")
parser.add_argument("-s_path", "--SEQUENCE_DATA_PATH", type=str, default='/remote-home/my/operation_detection/new_data/origin_data/sequence_created_data.json', help="Sequence data path")
parser.add_argument("-d_path", "--DECISION_DATA_PATH", type=str, default='/remote-home/my/operation_detection/new_data/origin_data/decision_created_data.json', help="Sequence data path")
parser.add_argument("-test_path", "--test_list_path", type=str, default='new_data/dataset_list/dgl_old_test_list.pkl', help="test list data path")
parser.add_argument("-w", "--number_workers", type=int, default=16, help="load data number of workers")
parser.add_argument("-bs", "--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("-seed", "--random_seed", type=int, default=20, help="random seed")
parser.add_argument("-net", "--network", default='TransformerNet', choices=['GraphNet','SequenceNet','TransformerNet'], help="choose wich net")
parser.add_argument("-load_dataset", "--load_dataset", type = int ,default=1, help="whether load saved dataset")
parser.add_argument("-p", "--procedure_KG", type = int,default=0, help="whether procedure KG subgraph")
parser.add_argument("-c", "--context", type = int,default=0 , help="whether context")
parser.add_argument("-r", "--relative", type = int,default=1, help="compute the subgraph using relative node")
parser.add_argument("-main_feature", "--add_main_feature", type = int,default=0, help="whether add main feature")
parser.add_argument('-task','--task',type=int,default=1,help='whether task1 or task2')
parser.add_argument('-m_path','--model_path',type=str,default='best_model/')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG,
                    filename='log/test_result.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='[%(levelname)s] %(asctime)s %(filename)s %(message)s',
                    filemode='a'
                    )
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG',fmt = '[%(levelname)s] %(asctime)s \033[1;35m %(filename)s \033[0m %(message)s')
setup_seed(args.random_seed)

logger.warn("Expriment: net---%s  procedure KG:%s  context:  %s"%(args.network,str(args.procedure_KG),str(args.context)))
    #定义分词器
tokenizer = BertTokenizer.from_pretrained(os.path.join(args.BERT_PATH, 'vocab.txt'))
torch.cuda.set_device(args.GPUid)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info('device is set on GPU-%d'%(args.GPUid))
if args.task==1:
        if not args.load_dataset:
            testset = Task1Dataset(args.DECISION_DATA_PATH,args.SEQUENCE_DATA_PATH,test = True,tokenizer=tokenizer,procedure_triple_path=args.PROCEDURE_KG_PATH)
            logger.info('create test list')
            test_list = [testset[i] for i in tqdm(range(len(testset)),ncols = 80)]
            logger.info('saving test list...')
            torch.save(test_list,args.test_list_path)
        else:
            logger.info('loading test list')
            test_list = torch.load(args.test_list_path)

        if args.network == "TransformerNet":
            loader = TrmDataloader
        else:
            loader = GraphDataLoader
        testloader = loader(test_list, batch_size=args.batch_size,num_workers=args.number_workers,pin_memory=True)
        if args.network == "GraphNet":
            model = GraphNet(args.BERT_PATH,compute_relative = args.relative,add_external = args.procedure_KG,add_context = args.context, add_main_feature= args.add_main_feature)
        elif args.network == "TransformerNet":
            model = TransformerNet(args.BERT_PATH,compute_relative = args.relative,add_external = args.procedure_KG,add_context = args.context, add_main_feature= args.add_main_feature,num_layers=3)
        else:
            model = TreeLstmNet(768,768,bert_path = args.BERT_PATH,add_external = args.procedure_KG,add_context = args.context,add_main_feature = args.add_main_feature)
        model.load_state_dict(torch.load(args.model_path+'task%s-%s-%s-%s-%s_'%\
                            (args.task,args.network,str(args.procedure_KG),str(args.context),str(args.add_main_feature))+\
                                'best_parameter.pkl'))
        model = model.to(device)
        with torch.no_grad():
            model.eval()
            predict = []
            label = []
            for batch in tqdm(testloader,ncols = 80):
                data,text,subgraph,topological_list = batch
                if isinstance(data, list):
                    data,text,subgraph = list(map(lambda x:x.to(device),data)),text.to(device),subgraph.to(device)
                    output,y = model(data,text,subgraph,topological_list)
                elif args.network=='GraphNet':
                    data,text,subgraph = data.to(device),text.to(device),subgraph.to(device)
                    y = data.y
                    output = model(data,text,subgraph)
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
            logger.error('---------------------------------------------------------------')
            logger.info('Dev --- Acc:%f Pre:%f Recall:%f f1:%f'%(acc,pre,recall,f1))
            logger.error('---------------------------------------------------------------')
else:
    raise NotImplementedError
