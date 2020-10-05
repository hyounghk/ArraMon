import random
import numpy as np

import torch
from torch import nn, optim
from data import NADataset, TorchDataset
import argparse
from tqdm import tqdm
from decoders_sim import ActionDecoder

from metric_dtw import DTW
from sim_mul import simulator
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='NA Task')
parser.add_argument('--workers', default=4, type=int, help='num of workers')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--batch_size_val', default=8, type=int, help='batch size')
parser.add_argument('--num_epochs', default=300, type=int, help='num of epochs')
parser.add_argument('--max_input', default=150, type=int, help='max input size')
parser.add_argument('--seed', default=1234, type=int, help='seeds')
parser.add_argument('--hsz', default=128, type=int, help='hidden size')
parser.add_argument('--lr', default=0.001, type=float, help='hidden size')
parser.add_argument('--port', default=1111, type=int, help='port number')
parser.add_argument('--sim_num', default=4, type=int, help='the num of sims')


s_turn = 0
e_turn = 2

def get_tuple(args, split, batch_size, shuffle=True, drop_last=True, max_length=100):
    dataset = NADataset(split)
    torch_ds = TorchDataset(dataset, max_length=max_length)

    print("The size of data split %s is %d" % (split, len(torch_ds)))
    loader = torch.utils.data.DataLoader(torch_ds,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True,
        drop_last=drop_last)

    return dataset, torch_ds, loader

def calLossAvg(loss):
    seq_len = (loss != 0.0).float().sum(-1)
    loss_avg = loss.sum(-1) / (seq_len + 0.000001)
    loss_avg = loss_avg.mean()
    return loss_avg

def train(args, sim, model_navi, model_assem, optimizer, train_tuple, valid_seen_tuple):
    
    sim.resnet_extractor.eval()
    train_ds, train_tds, train_loader = train_tuple

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    best_dtw_score = 0.0

    for epoch in range(args.num_epochs):

        iterator = tqdm(enumerate(train_loader), total=len(train_tds)//args.batch_size, unit="batch")

        pos_error = 0.0
        total_cnt = 0.0
        tot_init_dist = 0.0

        for k, (mapids, insts, actions, paths, rots, objPos) in iterator:

            res = sim.mapIDset(mapids, train=False, epoch=epoch)

            inst_navi, inst_assem, leng_navi, leng_assem = insts

            inst_navi, inst_assem, leng_navi, leng_assem = \
            inst_navi.cuda(), inst_assem.cuda(), leng_navi.cuda(), leng_assem.cuda() 

            acts_navi, acts_assem, leng_acts_navi, leng_acts_assem = actions
            acts_navi, acts_assem, leng_acts_navi, leng_acts_assem = \
            acts_navi.cuda(), acts_assem.cuda(), leng_acts_navi.cuda(), leng_acts_assem.cuda()

            path_navi, path_assem, path_leng_navi, path_leng_assem = paths
            rot_navi, rot_assem, rot_leng_navi, rot_leng_assem = rots

            pos_navi, pos_assem = objPos
            pos_navi, pos_assem = pos_navi.cuda(), pos_assem.cuda()

            bsz = pos_navi.size(0)
            losses_navi = []
            losses_assem = []
            object_pos = torch.zeros(bsz).cuda()
            init_dist = torch.zeros(bsz).cuda()

            global s_turn, e_turn
            for i in range(s_turn, e_turn):

                optimizer.zero_grad()

                _, logit_navi, _, _, _, pos_seq_gt, obj_ids, _, agent_pos  = model_navi(epoch, mapids, i, inst_navi[:,i,:], leng_navi[:,i], acts_navi[:,i,:], path_navi[:,i,:], rot_navi[:,i,:], phase='navi')

                res = sim.shift(epoch=epoch)

                diff_pow = (pos_navi[:,i,:] - agent_pos.cuda())**2
                innersum = torch.sum(diff_pow, dim=-1)
                object_pos += torch.sqrt(innersum)

                diff_pow = (pos_navi[:,i,:] - path_navi[:,i,0,:].cuda())**2
                innersum = torch.sum(diff_pow, dim=-1)
                init_dist += torch.sqrt(innersum)


                loss_navi = loss_func(logit_navi.contiguous().view(-1, 5), acts_navi[:,i,1:].contiguous().view(-1))
                  
                if True:          
                    bsz = logit_navi.size(0)
                    loss_navi = loss_navi.view(bsz, -1)
                    loss_navi = calLossAvg(loss_navi)

                loss_navi.backward()
                nn.utils.clip_grad_norm_(model_navi.parameters(), 5.)
                optimizer.step()


                optimizer.zero_grad()
                _, logit_assem, _, _, _, pos_seq_gt, _, obj_place, agent_pos = model_assem(epoch, mapids, i, inst_assem[:,i,:], leng_assem[:,i], acts_assem[:,i,:], path_assem[:,i,:], rot_assem[:,i,:], phase='assembly')
                
                res = sim.shift(epoch=epoch)

                loss_assem = loss_func(logit_assem.contiguous().view(-1, 5), acts_assem[:,i,1:].contiguous().view(-1))
                
                if True:  
                    loss_assem = loss_assem.view(bsz, -1)
                    loss_assem = calLossAvg(loss_assem)
                loss_assem.backward()
                nn.utils.clip_grad_norm_(model_assem.parameters(), 5.)
                optimizer.step()


                losses_navi.append(loss_navi)
                losses_assem.append(loss_assem)

            total_cnt += bsz
            pos_error += object_pos / 3
            loss_total = sum(losses_navi) + sum(losses_assem)

            iterator.set_postfix(loss=sum(losses_navi).item())

        dtw_score = evaluation(args, epoch, sim, model_navi, model_assem, valid_seen_tuple)

        if dtw_score > best_dtw_score:
            dtw_score = best_dtw_score
            save(model_navi, model_assem, "best_model", epoch)

def save(model_navi, model_assem, name, epoch):
        model_navi_path = os.path.join("best_models", '%s_model_navi_%s.pth' % (name, str(epoch)))
        model_assem_path = os.path.join("best_models", '%s_model_assem_%s.pth' % (name, str(epoch)))
        torch.save(model_navi.state_dict(), model_navi_path)
        torch.save(model_assem.state_dict(), model_assem_path)

def load(model_navi, model_assem, name, epoch):
        model_navi_path = os.path.join("best_models", '%s_model_navi_%s.pth' % (name, str(epoch)))
        model_assem_path = os.path.join("best_models", '%s_model_assem_%s.pth' % (name, str(epoch)))
        model_navi_state_dict = torch.load(model_navi_path)
        model_assem_state_dict = torch.load(model_assem_path)
        model_navi.load_state_dict(model_navi_state_dict)
        model_assem.load_state_dict(model_assem_state_dict)

def evaluation(args, epoch, sim, model_navi, model_assem, valid_tuple, log_name="scores.txt"):
    with torch.no_grad():
        valid_ds, valid_tds, valid_loader = valid_tuple

        model_navi.eval()
        model_assem.eval()
        sim.resnet_extractor.eval()
        dtw = DTW()

        total_outter_score = 0.0
        total_cnt = 0.0
        pos_error = 0.0
        pos_error_each = 0.0
        coc_3_total = 0.0
        coc_5_total = 0.0
        coc_7_total = 0.0
        tot_init_dist = 0.0
        placement_dist = 0.0
        placement_error = 0.0
        placement_error_0 = 0.0
        placement_error_3 = 0.0
        placement_error_5 = 0.0
        placement_error_7 = 0.0
        placement_success = 0.0
        placement_success_0 = 0.0
        placement_success_3 = 0.0
        placement_success_5 = 0.0
        placement_success_7 = 0.0
        pick_score_turn1 = 0.0
        pick_score_turn2 = 0.0

        dtw_score_each = torch.zeros(3)
        dtw_score_each_tot = 0.0

        map_path = {}

        iterator = tqdm(enumerate(valid_loader), total=len(valid_tds)//args.batch_size, unit="batch")

        for k, (mapids, insts, actions, paths, rots, objPos) in iterator:

            res = sim.mapIDset(mapids, epoch=epoch)

            inst_navi, inst_assem, leng_navi, leng_assem = insts
            inst_navi, inst_assem, leng_navi, leng_assem = \
            inst_navi.cuda(), inst_assem.cuda(), leng_navi.cuda(), leng_assem.cuda()

            acts_navi, acts_assem, leng_acts_navi, leng_acts_assem = actions
            acts_navi, acts_assem, leng_acts_navi, leng_acts_assem = \
            acts_navi.cuda(), acts_assem.cuda(), leng_acts_navi, leng_acts_assem 

            path_navi, path_assem, path_leng_navi, path_leng_assem = paths
            rot_navi, rot_assem, rot_leng_navi, rot_leng_assem = rots


            pos_navi, pos_assem = objPos
            pos_navi, pos_assem = pos_navi.cuda(), pos_assem.cuda()



            pos_seq_navi_list = []
            pos_len_navi_list = []
            pos_seq_navi_list_gt = []
            pos_len_navi_list_gt = []

            pos_seq_assem_list = []
            pos_len_assem_list = []
            pos_seq_assem_list_gt = []
            pos_len_assem_list_gt = []

            collected_object = [[], []]

            bsz = pos_navi.size(0)
            init_dist = torch.zeros(bsz).cuda()
            object_pos = torch.zeros(bsz).cuda()
            object_pos_each = torch.zeros(bsz, 3).cuda()
            coc_0 = torch.zeros(bsz, 2).cuda()
            coc_3 = torch.zeros(bsz, 2).cuda()
            coc_5 = torch.zeros(bsz, 2).cuda()
            coc_7 = torch.zeros(bsz, 2).cuda()
            placement = torch.zeros(bsz, 2, 3).cuda()
            object_dist = torch.zeros(bsz, 2).cuda()
            object_place = torch.zeros(bsz, 2).cuda()
            object_place_0 = torch.zeros(bsz, 2).cuda()
            object_place_3 = torch.zeros(bsz, 2).cuda()
            object_place_5 = torch.zeros(bsz, 2).cuda()
            object_place_7 = torch.zeros(bsz, 2).cuda()
            object_success = torch.zeros(bsz, 2).cuda()
            object_success_0 = torch.zeros(bsz, 2).cuda()
            object_success_3 = torch.zeros(bsz, 2).cuda()
            object_success_5 = torch.zeros(bsz, 2).cuda()
            object_success_7 = torch.zeros(bsz, 2).cuda()

            global s_turn, e_turn
            for i in range(s_turn, e_turn):
                _, logit_navi, _, pos_seq_navi, pos_len_navi, pos_seq_gt, obj_ids, obj_place, agent_pos  = model_navi(-1, mapids, i, inst_navi[:,i,:], leng_navi[:,i], acts_navi[:,i,:], path_navi[:,i,:], rot_navi[:,i,:], phase='navi')

                pos_seq_navi_list.append(pos_seq_navi.cpu())
                pos_len_navi_list.append(pos_len_navi.cpu())
                pos_seq_navi_list_gt.append(pos_seq_gt.cpu())
                collected_object[i].append(sum((obj_ids==i).float()))


                diff_pow = (pos_navi[:,i,:] - agent_pos.cuda())**2
                innersum = torch.sum(diff_pow, dim=-1)
                object_pos += torch.sqrt(innersum)
                object_pos_each[:, i] = torch.sqrt(innersum)

                coc_0[:,i] = (obj_ids==i).float()
                coc_3[:,i] = (torch.sqrt(innersum) < 3.0).float()
                coc_5[:,i] = (torch.sqrt(innersum) < 5.0).float()
                coc_7[:,i] = (torch.sqrt(innersum) < 7.0).float()


                diff_pow = (pos_navi[:,i,:] - path_navi[:,i,0,:].cuda())**2
                innersum = torch.sum(diff_pow, dim=-1)
                init_dist += torch.sqrt(innersum)

                res = sim.shift(epoch=epoch)

                _, logit_assem, _, pos_seq_assem, pos_len_assem, pos_seq_gt, _, obj_place, agent_pos = model_assem(-1, mapids, i, inst_assem[:,i,:], leng_assem[:,i], acts_assem[:,i,:], path_assem[:,i,:], rot_assem[:,i,:], phase='assembly')

                pos_seq_assem_list.append(pos_seq_assem.cpu())
                pos_len_assem_list.append(pos_len_assem.cpu())
                pos_seq_assem_list_gt.append(pos_seq_gt.cpu())

                obj_place[obj_place==-1] = 100

                placement[:,i,:] = obj_place.cuda()
                manhattanD = torch.abs(pos_assem[:,i,:] - obj_place.cuda())
                innersum = torch.sum(manhattanD, dim=-1)

                object_dist[:,i] += innersum
                object_place[:,i] += 1/(1 + innersum ** 2)
                object_place_0[:,i] += 1/(1 + innersum ** 2)
                object_place_3[:,i] += 1/(1 + innersum ** 2)
                object_place_5[:,i] += 1/(1 + innersum ** 2)
                object_place_7[:,i] += 1/(1 + innersum ** 2)
                object_success[:,i] += (innersum == 0).float()
                object_success_0[:,i] += (innersum == 0).float()
                object_success_3[:,i] += (innersum == 0).float()
                object_success_5[:,i] += (innersum == 0).float()
                object_success_7[:,i] += (innersum == 0).float()


                if True:
                    object_place_0[:,i][coc_0[:,i]!=1] = 0
                    object_success_0[:,i] *= coc_0[:,i] 

                    object_place_3[:,i][coc_3[:,i]!=1] = 0
                    object_success_3[:,i] *= coc_3[:,i] 

                    object_place_5[:,i][coc_5[:,i]!=1] = 0
                    object_success_5[:,i] *= coc_5[:,i] 

                    object_place_7[:,i][coc_7[:,i]!=1] = 0
                    object_success_7[:,i] *= coc_7[:,i] 


                res = sim.shift(epoch=epoch)

            tot_init_dist += init_dist / 2
            pos_error += object_pos / 2
            pos_error_each += object_pos_each
            coc_3_total += coc_3
            coc_5_total += coc_5
            coc_7_total += coc_7
            placement_dist += object_dist 
            placement_error += object_place 
            placement_error_0 += object_place_0 
            placement_error_3 += object_place_3 
            placement_error_5 += object_place_5 
            placement_error_7 += object_place_7 
            placement_success += object_success
            placement_success_0 += object_success_0
            placement_success_3 += object_success_3
            placement_success_5 += object_success_5
            placement_success_7 += object_success_7
            pick_score_turn1 += sum(collected_object[0])
            pick_score_turn2 += sum(collected_object[1]) 
            bsz = path_navi.size(0)
            total_cnt += bsz

            for idx in range(bsz):

                total_inner_score = 0.0

                mapid = mapids[idx].item()

                map_path[mapid] = {"path_gen":[], "path_gt":[], "dtw":[], "path_assem_gen":[], "path_assem_gt":[], "ptc":[], "ctc0":[], "ctc3":[], "ctc5":[], "ctc7":[], "placement":[]}

                for j in range(s_turn, e_turn):

                    dtw_score = dtw(pos_seq_navi_list[j][idx], pos_seq_navi_list_gt[j][idx],
                                 pos_len_navi_list[j][idx], path_leng_navi[idx][j], metric='ndtw')

                    path_gen = pos_seq_navi_list[j][idx][:pos_len_navi_list[j][idx]].tolist()
                    path_gt = pos_seq_navi_list_gt[j][idx][:path_leng_navi[idx][j]].tolist()


                    path_assem_gen = pos_seq_assem_list[j][idx][:pos_len_assem_list[j][idx]].tolist()
                    path_assem_gt = pos_seq_assem_list_gt[j][idx][:path_leng_assem[idx][j]].tolist()

                    map_path[mapid]["path_gen"].append(path_gen)
                    map_path[mapid]["path_gt"].append(path_gt)
                    map_path[mapid]["path_assem_gen"].append(path_assem_gen)
                    map_path[mapid]["path_assem_gt"].append(path_assem_gt)
                    map_path[mapid]["dtw"].append(dtw_score.item())
                    map_path[mapid]["ptc"].append(object_success[idx,j].item())
                    map_path[mapid]["ctc0"].append(coc_0[idx,j].item())
                    map_path[mapid]["ctc3"].append(coc_3[idx,j].item())
                    map_path[mapid]["ctc5"].append(coc_5[idx,j].item())
                    map_path[mapid]["ctc7"].append(coc_7[idx,j].item())
                    map_path[mapid]["placement"].append(placement[idx,j].tolist())

                    total_inner_score += dtw_score
                    dtw_score_each[j] += dtw_score

                total_inner_score /= 2
                total_outter_score += total_inner_score


        dtw_avg = total_outter_score / total_cnt
        pick_score_avg = (pick_score_turn1 +  pick_score_turn2)/2/ total_cnt
        with open("evalScores/" + log_name, 'a') as f:
            print("epoch", epoch, file=f)
            print("eval target-agent init dist", torch.sum(tot_init_dist)/total_cnt, file=f)
            print("eval target-agent dist", torch.sum(pos_error)/total_cnt, file=f)
            print(file=f)
            print("eval CTC-3 1 turn", torch.sum(coc_3_total[:,0])/total_cnt, file=f)
            print("eval CTC-3 2 turn", torch.sum(coc_3_total[:,1])/total_cnt, file=f)
            print("eval CTC-3 total", torch.sum(coc_3_total)/2/total_cnt, file=f)
            print(file=f)
            print("eval CTC-5 1 turn", torch.sum(coc_5_total[:,0])/total_cnt, file=f)
            print("eval CTC-5 2 turn", torch.sum(coc_5_total[:,1])/total_cnt, file=f)
            print("eval CTC-5 total", torch.sum(coc_5_total)/2/total_cnt, file=f)
            print(file=f)
            print("eval CTC-7 1 turn", torch.sum(coc_7_total[:,0])/total_cnt, file=f)
            print("eval CTC-7 2 turn", torch.sum(coc_7_total[:,1])/total_cnt, file=f) 
            print("eval CTC-7 total", torch.sum(coc_7_total)/2/total_cnt, file=f) 
            print(file=f)
            print("eval target-agent dist 1 turn", torch.sum(pos_error_each[:,0])/total_cnt, file=f)
            print("eval target-agent dist 2 turn", torch.sum(pos_error_each[:,1])/total_cnt, file=f)
            print(file=f)

            print("eval placement dist 1 turn", torch.sum(placement_dist[:,0])/total_cnt, file=f)
            print("eval placement dist 2 turn", torch.sum(placement_dist[:,1])/total_cnt, file=f)
            print("eval placement dist total", torch.sum(placement_dist)/2/total_cnt, file=f)
            print(file=f)
            print("eval rPOD 1 turn", torch.sum(placement_error[:,0])/total_cnt, file=f)
            print("eval rPOD 2 turn", torch.sum(placement_error[:,1])/total_cnt, file=f)
            print("eval rPOD total", torch.sum(placement_error)/2/total_cnt, file=f)
            print(file=f)
            print("eval rPOD_0 1 turn", torch.sum(placement_error_0[:,0])/total_cnt, file=f)
            print("eval rPOD_0 2 turn", torch.sum(placement_error_0[:,1])/total_cnt, file=f)
            print("eval rPOD_0 total", torch.sum(placement_error_0)/2/total_cnt, file=f)
            print(file=f)
            print("eval rPOD_3 1 turn", torch.sum(placement_error_3[:,0])/total_cnt, file=f)
            print("eval rPOD_3 2 turn", torch.sum(placement_error_3[:,1])/total_cnt, file=f)
            print("eval rPOD_3 total", torch.sum(placement_error_3)/2/total_cnt, file=f)
            print(file=f)
            print("eval rPOD_5 1 turn", torch.sum(placement_error_5[:,0])/total_cnt, file=f)
            print("eval rPOD_5 2 turn", torch.sum(placement_error_5[:,1])/total_cnt, file=f)
            print("eval rPOD_5 total", torch.sum(placement_error_5)/2/total_cnt, file=f)
            print(file=f)
            print("eval rPOD_7 1 turn", torch.sum(placement_error_7[:,0])/total_cnt, file=f)
            print("eval rPOD_7 2 turn", torch.sum(placement_error_7[:,1])/total_cnt, file=f)
            print("eval rPOD_7 total", torch.sum(placement_error_7)/2/total_cnt, file=f)
            print(file=f)
            

            print("eval PTC 1 turn", torch.sum(placement_success[:,0])/total_cnt, file=f)
            print("eval PTC 2 turn", torch.sum(placement_success[:,1])/total_cnt, file=f)
            print("eval PTC total", torch.sum(placement_success)/2/total_cnt, file=f)
            print(file=f)
            print("eval PTC_0 1 turn", torch.sum(placement_success_0[:,0])/total_cnt, file=f)
            print("eval PTC_0 2 turn", torch.sum(placement_success_0[:,1])/total_cnt, file=f)
            print("eval PTC_0 total", torch.sum(placement_success_0)/2/total_cnt, file=f)
            print(file=f)
            print("eval PTC_3 1 turn", torch.sum(placement_success_3[:,0])/total_cnt, file=f)
            print("eval PTC_3 2 turn", torch.sum(placement_success_3[:,1])/total_cnt, file=f)
            print("eval PTC_3 total", torch.sum(placement_success_3)/2/total_cnt, file=f)
            print(file=f)
            print("eval PTC_5 1 turn", torch.sum(placement_success_5[:,0])/total_cnt, file=f)
            print("eval PTC_5 2 turn", torch.sum(placement_success_5[:,1])/total_cnt, file=f)
            print("eval PTC_5 total", torch.sum(placement_success_5)/2/total_cnt, file=f)
            print(file=f)
            print("eval PTC_7 1 turn", torch.sum(placement_success_7[:,0])/total_cnt, file=f)
            print("eval PTC_7 2 turn", torch.sum(placement_success_7[:,1])/total_cnt, file=f)
            print("eval PTC_7 total", torch.sum(placement_success_7)/2/total_cnt, file=f)
            print(file=f)
            print("eval DTW 1 turn", dtw_score_each[0]/total_cnt, file=f)
            print("eval DTW 2 turn", dtw_score_each[1]/total_cnt, file=f)
            print("eval DTW total", (dtw_score_each[0] + dtw_score_each[1])/2/total_cnt, file=f)
            # print("dtw_avg", dtw_avg, file=f)
            print("pick_score 1 turn", pick_score_turn1/total_cnt, file=f)
            print("pick_score 2 turn", pick_score_turn2/total_cnt, file=f)
            print("pick_score_avg", pick_score_avg, file=f)


        model_navi.train()
        model_assem.train()
        sim.resnet_extractor.eval()
    return dtw_avg


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_tuple = get_tuple(args, 'train', args.batch_size, shuffle=True, drop_last=True, max_length=args.max_input)
    val_seen_tuple = get_tuple(args, 'valid_seen', args.batch_size, shuffle=False, drop_last=True, max_length=args.max_input)
    args.ntoken = train_tuple[0].tok.vocab_size
    sim = simulator([args.port], args.sim_num)
    res = sim.reset()
    assert  'd' in res

    model_navi = ActionDecoder(sim, args.hsz, args.ntoken).cuda()
    model_assem = ActionDecoder(sim, args.hsz, args.ntoken).cuda()

    optimizer = optim.Adam(list(model_navi.parameters()) + list(model_assem.parameters()),lr=args.lr)

    model_navi = nn.DataParallel(model_navi)
    model_assem = nn.DataParallel(model_assem)
    
    train(args, sim, model_navi, model_assem, optimizer, train_tuple, val_seen_tuple)




