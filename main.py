import numpy as np
import pickle
import torch
import time
from utils import scipy_sparse_mat_to_torch_sparse_tensor, metrics
from parser import args

device = 'cpu'
if args.device == 'cuda':
    device = 'cuda:' + args.cuda
k = args.k

# load and process data
f = open('data/'+args.data+'/train_mat.pkl','rb')
train = pickle.load(f)
f = open('data/'+args.data+'/test_mat.pkl','rb')
test = pickle.load(f)

test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Data loaded and processed.')

start_time = time.time()

# define initial representations
n_u, n_i = train.shape
item_rep = torch.eye(n_i).to(device)
user_rep = torch.zeros(n_u,n_i).to(device)

# process the adjacency matrix
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)

# iterative representation propagation on graph
for i in range(k):
    print("Running layer", i)
    user_rep_temp = torch.sparse.mm(adj,item_rep) + user_rep
    item_rep_temp = torch.sparse.mm(adj.transpose(0,1),user_rep) + item_rep
    user_rep = user_rep_temp
    item_rep = item_rep_temp

# evaluation
pred = user_rep.cpu().numpy()

train_csr = (train!=0).astype(np.float32)

batch_user = 256
test_uids = np.array([i for i in range(test.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    preds = pred[start:end]
    mask = train_csr[start:end].toarray()
    preds = preds * (1-mask)
    predictions = (-preds).argsort()
    
    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
print('-------------------------------------------')
print('recall@20',all_recall_20/batch_no,'ndcg@20',all_ndcg_20/batch_no,'recall@40',all_recall_40/batch_no,'ndcg@40',all_ndcg_40/batch_no)

end_time = time.time()
print("Total running time (seconds):", end_time-start_time)