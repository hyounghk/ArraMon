from torch.utils.data import Dataset
from tok import Tokenizer
import copy
import json
import os
import numpy as np
import torch

DATA_ROOT = "data/"


class NADataset:
    def __init__(self, split='train'):
        self.split = split

        if split == 'train':
            self.data = json.load(open(os.path.join(DATA_ROOT, "train.json")))
        elif split == 'valid_seen':
            self.data = json.load(open(os.path.join(DATA_ROOT, "val_seen.json")))
        elif split == 'valid_unseen':
            self.data = json.load(open(os.path.join(DATA_ROOT, "val_unseen.json")))
        # elif split == 'test_unseen':
        #     self.data = json.load(open(os.path.join(DATA_ROOT, "test_unseen.json")))

        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, "na_vocab.txt"))

class TorchDataset(Dataset):
    def __init__(self, dataset, max_length=80):
        self.dataset = dataset
        self.tok = dataset.tok
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, item):
        datum = self.dataset.data[item]



        ## instructions ##

        maids = datum["mapId"]

        insts = datum['instructions']
        inst_navi = insts['navi_inst_gt']
        inst_assem = insts['assem_inst_gt']


        a_navi = np.ones((3, self.max_length), np.int64) * self.tok.pad_id
        inst_len_navi = []
        for idx, sent in enumerate(inst_navi):
            inst = self.tok.encode(sent)
            length = len(inst) 
            inst_len_navi.append(length)
            a_navi[idx, :length] = inst

        a_assem = np.ones((3, self.max_length), np.int64) * self.tok.pad_id
        inst_len_assem = []
        for idx, sent in enumerate(inst_assem):
            inst = self.tok.encode(sent)
            length = len(inst) 
            inst_len_assem.append(length)
            a_assem[idx, :length] = inst


        # Lang: numpy --> torch
        inst_navi = torch.from_numpy(a_navi)
        inst_assem = torch.from_numpy(a_assem)
        leng_navi = torch.tensor(inst_len_navi)
        leng_assem = torch.tensor(inst_len_assem)

        ## Actions ##
        actions = datum['gt_actionIDs']
        acts_navi = actions['navi_actionIDs_gt']
        acts_assem = actions['assem_actionIDs_gt']


        act_navi = np.zeros((3, self.max_length)) - 1
        acts_leng_navi = []
        for idx, acts in enumerate(acts_navi):
            length = len(acts) 
            acts_leng_navi.append(length)
            act_navi[idx, 1:length+1] = acts

        act_assem = np.zeros((3, self.max_length)) - 1
        acts_leng_assem = []
        for idx, acts in enumerate(acts_assem):
            length = len(acts) 
            acts_leng_assem.append(length)
            act_assem[idx, 1:length+1] = acts 


        acts_navi = torch.tensor(act_navi).long() + 1
        acts_assem = torch.tensor(act_assem).long() + 1
        acts_leng_navi = torch.tensor(acts_leng_navi)
        acts_leng_assem = torch.tensor(acts_leng_assem)


        paths = datum['gt_paths']
        path_navi = paths['navi_path_gt']
        path_assem = paths['assem_path_gt']

        p_navi = np.zeros((3, self.max_length, 3))
        path_leng_navi = []
        for idx, path in enumerate(path_navi):
            length = len(path) 
            path_leng_navi.append(length)
            p_navi[idx, :length, :] = path

        p_assem = np.zeros((3, self.max_length, 3))
        path_leng_assem = []
        for idx, path in enumerate(path_assem):
            length = len(path) 
            path_leng_assem.append(length)
            p_assem[idx, :length, :] = path

        path_navi = torch.tensor(p_navi)
        path_assem = torch.tensor(p_assem)
        path_leng_navi = torch.tensor(path_leng_navi)
        path_leng_assem = torch.tensor(path_leng_assem)

        r_navi = np.zeros((3, self.max_length, 3))
        rot_leng_navi = [0]

        r_assem = np.zeros((3, self.max_length, 3))
        rot_leng_assem = [0]

        rot_navi = torch.tensor(r_navi)
        rot_assem = torch.tensor(r_assem)
        rot_leng_navi = torch.tensor(rot_leng_navi)
        rot_leng_assem = torch.tensor(rot_leng_assem)

        obj_pos = datum['gt_objects_pos']
        obj_pos_navi = obj_pos['navi_obj_pos_gt']
        obj_pos_assem = obj_pos['assem_obj_pos_gt']

        obj_p_navi = np.zeros((3, 3))
        for idx, pos in enumerate(obj_pos_navi):
            obj_p_navi[idx, :] = pos


        obj_p_assem = np.zeros((3, 3))
        for idx, pos in enumerate(obj_pos_assem):
            obj_p_assem[idx, :] = pos

        obj_p_navi = torch.tensor(obj_p_navi)
        obj_p_assem = torch.tensor(obj_p_assem)


        return maids, (inst_navi, inst_assem, leng_navi, leng_assem), \
                (acts_navi, acts_assem, acts_leng_navi, acts_leng_assem), \
                (path_navi, path_assem, path_leng_navi, path_leng_assem), \
                (rot_navi, rot_assem, rot_leng_navi, rot_leng_assem), \
                (obj_p_navi, obj_p_assem)
