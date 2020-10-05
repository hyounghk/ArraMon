from utils import LinearAct
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)

class InstructionEncoder(nn.Module):
    def __init__(self, ntoken, hidden_size):
        super(InstructionEncoder, self).__init__()

        self.hidden_size = hidden_size

        self.w_emb = nn.Embedding(ntoken, 300)
        self.langlstm = nn.LSTM(300, hidden_size, batch_first=True)


    def forward(self, insts, insts_len):

        insts = self.w_emb(insts)

        bsz = insts.size(0)

        with torch.no_grad():
            h0 = torch.zeros(1, bsz, self.hidden_size).cuda()
            c0 = torch.zeros(1, bsz, self.hidden_size).cuda()

        self.langlstm.flatten_parameters()
        insts_enc, (h1, c1) = self.langlstm(insts, (h0, c0))

        with torch.no_grad():
            insts_mask = torch.zeros(bsz, insts_enc.size(1)).cuda()

        for idx in range(bsz):
            insts_mask[idx,:insts_len[idx]] = 1

        return insts_enc, insts_mask

class ActionDecoder(nn.Module):
    def __init__(self, sim, hidden_size, ntoken, maxLen=150):
        super(ActionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.nActions = 5


        self.a_emb = nn.Embedding(5, 64)
        self.drop = nn.Dropout(0.3)

        self.act_lstm = nn.LSTMCell(64, self.hidden_size) 
        self.lang_int = LinearAct(hidden_size * 3, hidden_size)
        self.vis_int = LinearAct(hidden_size * 3, hidden_size)
        self.lang_vis_com = LinearAct(hidden_size, 1)
        self.projection = LinearAct(hidden_size * 3, self.nActions)
        self.projection_vfeat = LinearAct(2048, hidden_size)

        self.langEnc = InstructionEncoder(ntoken, self.hidden_size)

        self.sim = sim
        self.maxLen = maxLen

    def forward(self, epoch, mapids, turn, insts, insts_len, gt_acts, gt_path, gt_rot, phase='navi'):

        insts_enc, insts_mask = self.langEnc(insts, insts_len)

        if self.training:
            teacherForcing = True
            # greedy = True
        else:
            teacherForcing = False
            greedy = True
            # teacherForcing = True


        temperature = 1.0

        batch_size, seq_len = gt_acts.size()

        state = self.init_hidden(batch_size)
        
        act_seq = gt_acts.new_zeros((batch_size, seq_len-1), dtype=torch.long)
        pos_seq = gt_acts.new_zeros(batch_size, seq_len, 3).float()
        pos_seq_gt = gt_acts.new_zeros(batch_size, seq_len, 3).float()
        pos_len = gt_acts.new_ones((batch_size), dtype=torch.long) * seq_len
        seqLogprobs = gt_acts.new_zeros(batch_size, seq_len-1).float()
        logits = gt_acts.new_zeros(batch_size, seq_len-1, self.nActions).float()

        placed_obj_pos = gt_acts.new_zeros(batch_size, 3).float() - 1

        gt_track = gt_acts.new_zeros((batch_size), dtype=torch.long)
        pos_track = gt_acts.new_zeros((batch_size), dtype=torch.long) - 1



        if self.training and phase == 'navi':
            self.sim.sendActionW(batch_size, 'w')
        elif not self.training and phase == 'navi':
            self.sim.sendActionW(batch_size, 'w')
        elif phase == 'assembly':
            self.sim.sendActionW(batch_size, 'w')



        for t in range(self.maxLen):
            if t == 0: # input <pad>
                it = gt_acts[:,t]
                unfinished = (it != 4) 

            elif teacherForcing:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = gt_acts[:,t]
                    
            elif greedy:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()

            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)
                else:
                    pass

                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long()

            if t >= 1:

                act_seq[:,t-1] = it

                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)
                logits[:,t-1, :] = logit

                
                if phase == 'navi':
                    unfinished = unfinished * (pick_ids == -1)  
                else:
                    unfinished = unfinished * (place_pos[:,0] == -1)

                it = it * unfinished.type_as(it)
                if unfinished.sum() == 0:

                    break

            ##### sim #####

            record = False
            imgtag = ''

            pick_ids, place_pos, agent_pos, agent_rot, vis_feat = self.sim.sendActions(it, batch_size, record, imgtag)
            pick_ids = pick_ids.cuda() 
            place_pos = place_pos.cuda() 

            if phase == 'navi':
                P_act = (pick_ids != -1) * (pos_len == seq_len)
            else:
                P_act = (place_pos[:,0] != -1) * (pos_len == seq_len)
                placed_obj_pos[P_act] = place_pos[P_act]


            pos_len[P_act] = t+1

            ##### sim #####

            forward_gt = (gt_acts[:,t] == 1)
            
            forward_pos = ((pos_seq[range(batch_size),pos_track] - agent_pos.cuda()).sum(-1) != 0)
            
            gt_track[forward_gt] += 1
            pos_track[forward_pos] += 1
            
            pos_seq_gt[range(batch_size),gt_track] = gt_path[range(batch_size),gt_track].float()
            pos_seq[range(batch_size),pos_track] = agent_pos.cuda() 

            logprobs, logit, state = self.func(it, state, insts_enc, insts_mask, vis_feat) # (N, num_words)

        pos_track += 1
        pick_ids_k, _, agent_pos_k, agent_rot_k, vis_feat_k = self.sim.sendActionK(batch_size, record, imgtag)

        return seqLogprobs, logits, act_seq, pos_seq, pos_track, pos_seq_gt, pick_ids, placed_obj_pos, agent_pos

    def func(self, it, state, insts_enc, insts_mask, vis_feat):

        emb_it = self.a_emb(it) 
        emb_it = self.drop(emb_it)

        bsz, _, _, d = vis_feat.size()

        h_act, c_act = self.act_lstm(emb_it, (state[0], state[1]))

        vis_feat_proj = self.projection_vfeat(vis_feat.view(bsz, -1, d))

        insts_enc, vis_feat_proj = self.attention_cross(insts_enc, vis_feat_proj, insts_mask)

        h_att_inst = self.attention(h_act, insts_enc, insts_mask)
        h_att_vfeat = self.attention(h_act, vis_feat_proj, None)
    
        logit = self.projection(torch.cat([h_att_inst, h_att_vfeat, h_act], dim=-1))
        logit = self.drop(logit)

        logprobs = F.log_softmax(logit, dim=-1)

        state = (h_act, c_act)

        return logprobs, logit, state

    def attention_cross(self, insts_enc, visual, insts_mask):
        N,L,d = insts_enc.size()
        V = visual.size(1)

        insts_enc_ext = insts_enc.view(N, L, 1, d)
        visual_ext = visual.view(N, 1, V, d)

        sim = self.lang_vis_com(insts_enc_ext * visual_ext).view(N, L, V)

        sim_v = torch.softmax(sim, dim=-1) #(N, L, V)

        sim = mask_logits(sim, insts_mask.unsqueeze(-1))   
        sim_l = torch.softmax(sim, dim=1) #(N, L, V)

        ltv = torch.matmul(sim_v, visual) #(N, L, D)
        vtl = torch.matmul(sim_l.transpose(1,2), insts_enc) #(N, V, D)

        inst_new = self.lang_int(torch.cat([insts_enc, ltv, insts_enc*ltv], dim=-1))
        vis_new = self.vis_int(torch.cat([visual, vtl, visual*vtl], dim=-1))


        return inst_new, vis_new

    def attention(self, h_att, insts_enc, insts_mask):

        sim = torch.matmul(h_att.unsqueeze(1), insts_enc.transpose(1,2)) #(N, 1, L)

        if insts_mask is not None:
            sim = mask_logits(sim, insts_mask.unsqueeze(1))

        h_sofm = torch.softmax(sim, dim=-1) #(N, 1, L)
        att = torch.matmul(h_sofm, insts_enc).squeeze(1) #(N, D)

        return att
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.hidden_size),
                weight.new_zeros(bsz, self.hidden_size))