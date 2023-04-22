import torch
import os
import pandas as pd
import numpy as np


def metrics(model_dir, result_folder, top_k, val_turns):
    """Evaluate the model with metrics. Validate with a post-filter."""
    val_result_folder = os.path.join(model_dir, result_folder)

    dialog_turns = val_turns
    turn={}
    for k in range(dialog_turns):
        turn[k+1] = pd.read_csv(val_result_folder + '/turn_'+str(k+1)+'.csv')

    rank_df={}
    rank_df["customer_id"] = turn[1]["customer_id"]
    rank_df["target_img_idx"] = turn[1]["target_img_idx"]
    rank_df["target_img_name"] = turn[1]["target_img_name"]
    for i in range(dialog_turns):
        rank_df["turn_"+str(i+1)] = turn[i+1]["rank"]
    rank_df = pd.DataFrame(data=rank_df)
    rank_df = rank_df.drop_duplicates(subset=['customer_id'])

    total_num = rank_df.shape[0]
            
    for i in range(total_num):
        #x=rank_df.iloc[i]
        for j in range(dialog_turns):
            if rank_df.iloc[i]["turn_"+str(j+1)]<top_k:
                for k in range(dialog_turns -j-1):
                    # there are three columns before the "turn_x" columns
                    rank_df.iloc[i,3+j+1+k]=0
            continue
    rank_df.to_csv(val_result_folder + '/rank_df.csv')
    
    # NDCG@topk
    ndcg={}
    for i in range(dialog_turns):
        label = "turn_"+str(i+1)
        tmp = 1.0/np.log2(rank_df[rank_df[label]<top_k].iloc[:,3:53]+2)
        ndcg[label]=format(tmp[label].sum()/total_num, '.4f')
    y=[]
    for i in range(dialog_turns):
        y.append(float(ndcg["turn_"+str(i+1)]))
    ndcg_topk = y
    ndcg_topk_mean = sum(y)/len(y)

    # NDCG@10
    ndcg={}
    for i in range(dialog_turns):
        label = "turn_"+str(i+1)
        tmp = 1.0/np.log2(rank_df[rank_df[label]<10].iloc[:,3:53]+2)
        ndcg[label]=format(tmp[label].sum()/total_num, '.4f')
    y=[]
    for i in range(dialog_turns):
        y.append(float(ndcg["turn_"+str(i+1)]))
    ndcg_10 = y
    ndcg_10_mean = sum(y)/len(y)

    # success rate
    list_success=[]
    for i in range(dialog_turns):
        label = "turn_"+str(i+1)
        list_success.append(rank_df[rank_df[label]<top_k].shape[0])
    y=[]
    for i in range(dialog_turns):
        y.append(float(format(list_success[i]/total_num, '.4f')))
    sr = y
    sr_mean = sum(y)/len(y)
    
    return ndcg_topk_mean, ndcg_10_mean, sr_mean, ndcg_topk, ndcg_10, sr


