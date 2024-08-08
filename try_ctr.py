
# def f():
#     global x
#     global y
#     y = 20
#     x=10

# def g():
#     print(x)
# f()
# g()

import torch
from ctr_base.models import BaseModel
from ctr_base.config import Config
from torch.utils.data import Dataset, DataLoader
from dataset_sr import OurDataset
import argparse
import numpy as np
from utils import get_ctr_config

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="BookCrossing")
parser.add_argument("--ctr_K", type=int, default=30)
parser.add_argument("--ctr_dropout", type=float, default=0.1)
parser.add_argument("--lora_assemble_num", type=int, default=8)
parser.add_argument("--ctr_out_layer", type=str, default='final')
args = parser.parse_args()

# ctr_model_path = f"/home/ma-user/work/llm/models/ctr_model/din/din_BookCrossing.pt"
ctr_model_path = f"/data/XXXX/llm/models/ctr_model/{args.dataset}/din/hist_len_{args.ctr_K}/model.pt"

ctr_config = get_ctr_config(args)
ctr_config = Config.from_dict(ctr_config)

from dataset import load_ctr_dataset

if args.dataset == 'ml-25m':
    sample_num,train_type,test_range = 90000,'sequential',[0,30000]
elif args.dataset == 'ml-1m':
    sample_num,train_type,test_range = 70000,'sequential',[0,30000]
elif args.dataset == 'GoodReads':
    sample_num,train_type,test_range = 70000,'sequential',[0,30000]
elif args.dataset == 'BookCrossing':
    sample_num,train_type,test_range = 10000,'sequential',[0,1772]
data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": args.dataset, "sample_num": sample_num, "hist_len": args.ctr_K}

ctr_data = load_ctr_dataset(f"/data/XXXX/Datasets/{args.dataset}/benchmark_proc_data",256,train_type,test_range,data_config)



print('loaded')
X,Y,hist_ids,hist_ratings, hist_mask = torch.tensor(ctr_data['test']['X']).long(),torch.tensor(ctr_data['test']['Y']).long(),torch.tensor(ctr_data['test']['hist_ids']).long(),torch.tensor(ctr_data['test']['hist_ratings']).long(),torch.tensor(ctr_data['test']['hist_mask']).long()

dataset = OurDataset(X,Y,hist_ids=hist_ids,hist_ratings=hist_ratings,hist_mask=hist_mask)
dataloader = DataLoader(dataset, batch_size = 1024, shuffle = False)


model = BaseModel.from_config(ctr_config)
state_dict = torch.load(ctr_model_path)
model.load_state_dict(state_dict)

print(X.shape)
model.eval()
logits = []
loss=[]
with torch.no_grad():
    for batch in dataloader:
        batch_X,batch_Y,batch_hist_ids,batch_hist_ratings,batch_hist_mask = batch
        result_batch = model(batch_X,batch_Y,batch_hist_ids,batch_hist_ratings,batch_hist_mask)
        logits.append(result_batch['logits'].view(-1).detach().numpy())
        loss.append(result_batch['loss'].detach().numpy())
logits = np.concatenate(logits)
loss = np.array(loss)
print(np.mean(loss))

print(logits)
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

print(roc_auc_score(Y.detach().numpy(),logits))
print(log_loss(Y.detach().numpy(),logits))

print(logits)