import torch
import math
import time

def train_sl(epoch, optimizer, triplet_loss, model, user, ranker, batch_size, top_k, turns):
    print('train epoch #{}'.format(epoch))
    model.train()
    triplet_loss.train()
    split= 'train'
    # train / test
    all_input = user.feature_input
    all_seq = user.train_seq
    dialog_turns = turns

    user_img_idx = torch.LongTensor(batch_size)
    top_k_act_img_idx = torch.LongTensor(batch_size,top_k)
    history_idx = torch.LongTensor(batch_size,len(all_seq[0]["item_log"]))
    neg_img_idx = torch.LongTensor(batch_size)
    total_batch = math.ceil(len(all_seq) / batch_size)

    for batch_idx in range(1, total_batch + 1):
        start = time.time()

        # update item embeddings
        feat = model.update_rep(all_input)
        ranker.update_rep(feat)
        
        model.init_hid(batch_size)
        if torch.cuda.is_available():
            model.hx = model.hx.cuda()
            
        outs = []
            
        # sample target images
        # user.sample_idx(user_img_idx, split)
        history_input,_ = user.sample_target_idx_history(user_img_idx, split, batch_idx, batch_size, total_batch, history_idx)
        
        # sample initial top-k recommendation
        # user.sample_k_idx(top_k_act_img_idx, top_k)
        
        # history_input = all_input[history_idx]
                
        if torch.cuda.is_available():
            history_input = history_input.cuda()
        init_state = model.init_forward(history_input)
        # sampling for the next turn
        top_k_act_img_idx = ranker.k_nearest_neighbors(init_state.data,K=top_k)
        
        user_emb = ranker.feat[user_img_idx]
        user.sample_idx(neg_img_idx)
        neg_emb = ranker.feat[neg_img_idx]
        hist_loss = triplet_loss.forward(init_state, user_emb, neg_emb)
        outs.append(hist_loss)

        # finish dialog and update model parameters
        optimizer.zero_grad()
        outs = torch.stack(outs, dim=0).mean()
        outs.backward()
        optimizer.step()

        end = time.time()
        print('batch_idx:', batch_idx, '/', total_batch, ', time elapsed:{:.2f}'.format(end - start))
    return