import torch
from models.metrics import metrics
import math
import numpy as np
import pandas as pd
from torch.autograd import Variable
import os, sys


def evaluation(split, model_ege, model_gru, user, ranker, batch_size, top_k, turns, model_dir, result_folder):
    model_ege.eval()
    model_gru.eval()
    all_input = user.feature_input
    
    if split =='val':
        all_seq = user.val_seq
    elif split == 'test':
        all_seq = user.test_seq
    
    dialog_turns = turns

    user_img_idx = torch.LongTensor(batch_size)
    top_k_act_img_idx = torch.LongTensor(batch_size, top_k)
    history_idx = torch.LongTensor(batch_size,len(all_seq[0]["item_log"]))
    total_batch = math.ceil(len(all_seq) / batch_size)
    
    for batch_idx in range(1, total_batch + 1):
#         # sample data index
#         # user.sample_idx(user_img_idx,  split)
#         user.sample_target_idx(user_img_idx, split, batch_idx, batch_size, total_batch)
#         user.sample_k_idx(top_k_act_img_idx, top_k)
        
#         user_img_name = user.get_image_name(user_img_idx)
        
#         # print("user_img_idx",user_img_idx)
#         # print("user_img_name",user_img_name)
        
#         output_dict = {}
#         output_dict["target_img_idx"]= user_img_idx.cpu().numpy()
#         output_dict["target_img_name"]= np.array(user_img_name)

        model_ege.init_hid(batch_size)
        model_gru.init_hid(batch_size)
        if torch.cuda.is_available():
            model_ege.hx = model_ege.hx.cuda()
            model_gru.hx = model_gru.hx.cuda()

        # if torch.cuda.is_available():
        #     top_k_act_img_idx = top_k_act_img_idx.cuda()
        # act_emb = ranker.feat[top_k_act_img_idx]
        
        # update item embeddings
        feat_gru = model_gru.update_rep(all_input)
        ranker.update_rep(feat_gru)
        
        # sample data index
        history_input,user_name_list = user.sample_target_idx_history(user_img_idx, split, batch_idx, batch_size, total_batch, history_idx)
        user_img_name = user.get_image_name(user_img_idx)
        
        # print("user_img_idx",user_img_idx)
        # print("user_img_name",user_img_name)
        
        output_dict = {}
        # ------------------------
        # revise: Add customer_id
        # ------------------------
        output_dict["customer_id"]= np.array(user_name_list)
        output_dict["target_img_idx"]= user_img_idx.cpu().numpy()
        output_dict["target_img_name"]= np.array(user_img_name)

        # if torch.cuda.is_available():
        #     top_k_act_img_idx = top_k_act_img_idx.cuda()
        # act_emb = ranker.feat[top_k_act_img_idx]
        
        # history_input = all_input[history_idx]
        if torch.cuda.is_available():
            history_input = history_input.cuda()
        state = model_gru.init_forward(history_input)
        # sampling for the next turn
        top_k_act_img_idx = ranker.k_nearest_neighbors(state.data,K=top_k)
        
        # update item embeddings
        feat_ege = model_ege.update_rep(all_input)
        ranker.update_rep(feat_ege)

        # memory of act_img_idx
        memory_act_img_idx = torch.LongTensor(batch_size,1)
        memory_act_img_idx=torch.reshape(top_k_act_img_idx, (batch_size, top_k))
        if torch.cuda.is_available():
            memory_act_img_idx = memory_act_img_idx.cuda()
        # print("memory_act_img_idx",memory_act_img_idx)

        for k in range(dialog_turns):
            # print("k",k)
            # print("act_img_idx",act_img_idx)
            top_k_act_img_name = user.get_top_k_image_name(top_k_act_img_idx)
            
            p_act_img_idx, p_position, n_act_img_idx, n_position = ranker.nearest_neighbor_selector(user_img_idx, top_k_act_img_idx)
            act_input = all_input[p_act_img_idx]
            for i in range(top_k):
                output_dict["act_img_idx_"+str(i+1)]= top_k_act_img_idx.cpu().numpy()[:,i]
                output_dict["act_img_name_"+str(i+1)]= np.array(top_k_act_img_name)[:,i]
            
            txt_input, sents = user.get_feedback(p_act_img_idx, user_img_idx)
            output_dict["text_feedback"]= np.array(sents)
            output_dict["position"] = np.array(p_position)
            
            if torch.cuda.is_available():
                act_input = act_input.cuda()
                txt_input = txt_input.cuda()

            with torch.no_grad():
                state = model_ege.merge_forward(act_input, txt_input)

            ranking_candidate = ranker.compute_rank_correct(state.data, user_img_idx, memory_act_img_idx)
            output_dict["rank"]=ranking_candidate.numpy()

            # avoid repeated recommendations
            # print("******************************")
            # print("avoid repeated recommendations!")
            
            top_km_act_img_idx = ranker.k_nearest_neighbors(state.data,K=top_k*turns)
            for i in range(batch_size):
                # print(i)
                k_item=0
                for j in range(top_k*turns):
                    # print(j)
                    if top_km_act_img_idx[i,j].cpu().numpy() in memory_act_img_idx[i,:].cpu().numpy():
                        pass
                    else:
                        top_k_act_img_idx[i,k_item]=top_km_act_img_idx[i,j]
                        if k_item==top_k-1:
                            break #skip the rest in top-k
                        k_item=k_item+1
            memory_act_img_idx = torch.cat((memory_act_img_idx, torch.reshape(top_k_act_img_idx, (batch_size, top_k)).cuda()), 1)
            
            # p_act_img_idx, p_position, n_act_img_idx, n_position = ranker.nearest_neighbor_selector(user_img_idx, top_k_act_img_idx)
            top_k_act_img_name = user.get_top_k_image_name(top_k_act_img_idx)
            
            for i in range(top_k):
                output_dict["new_act_img_idx_"+str(i+1)]= top_k_act_img_idx.cpu().numpy()[:,i]
                output_dict["new_act_img_name_"+str(i+1)]= np.array(top_k_act_img_name)[:,i]
            df = pd.DataFrame(data=output_dict)
            
            result_path = os.path.join(model_dir, result_folder)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            
            save_filename = result_path+'/turn_'+ str(k+1) +'.csv'
            if not os.path.exists(save_filename):
                df.to_csv(save_filename, index = False)
            else:
                df.to_csv(save_filename, mode='a', header=False, index = False)

            # print("updated ranking_candidate",ranking_candidate)
            # print("updated act_img_idx",act_img_idx)

    # ------------------------
    # revise: more outputs of the metrics
    # ------------------------
    ndcg_topk_mean, ndcg_10_mean, sr_mean, ndcg_topk, ndcg_10, sr = metrics(model_dir, result_folder, top_k, turns)
    return ndcg_topk_mean, ndcg_10_mean, sr_mean, ndcg_topk, ndcg_10, sr


