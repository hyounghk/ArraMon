import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import math
from connection_mul_th import connection

import os
import io
from PIL import Image
from array import array
import base64
import json
import numpy as np


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

img_transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
])

cnt = 0

class simulator(nn.Module):
    def __init__(self, ports, sim_num):
        super(simulator, self).__init__()
        
        self.connection = connection(ports, sim_num)
        self.num_sim = sim_num
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-2]
        self.resnet_extractor = nn.Sequential(*modules).cuda()
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

        self.mapids = []

        self.infos_pre = []
        self.vfeat_pre = []

    def reset(self):

        b_list = ["b".encode() for _ in range(self.num_sim)]
        res = self.connection.start(b_list)
        return res

    def setMapIDs(self, mapids):

        self.mapids = []
        self.infos = []
        bsz = mapids.size(0)
        for i in range(bsz):
            self.mapids.append(str(mapids[i].item()))
            self.infos_pre.append([])
            self.vfeat_pre.append([])

        for j in range(len(self.mapids)):
            with open("./pos_sim_passed_sec/" + "pos_" + self.mapids[j] + ".json", 'r') as jf:
                self.infos.append(json.load(jf))

    def shift(self, train=False, epoch=0):

        if train:
            return 'd'

        s_list = ["s".encode() for _ in range(self.num_sim)]
        res = self.connection.start(s_list, epoch=epoch)
        return res

    def mapIDset(self, mapids, train=False, epoch=0):

        if train:
            self.setMapIDs(mapids)
        bsz = mapids.size(0)
        actions_list = []
        for i in range(self.num_sim):
            start = int(bsz/self.num_sim) * i
            end = int(bsz/self.num_sim) * (1+i)
            mapids_str = 'm,'
            for mid in mapids[start:end]:
                mapids_str += str(mid.item()) + ','

            mapids_str = mapids_str[:-1]
            actions_list.append(mapids_str.encode())
        res = self.connection.start(actions_list, epoch=epoch)
        return res

    def getFeats(self, bsz, imgtag):

        obj_ids = []
        obj_pos = []
        agent_pos = []
        agent_rot = []

        vfeat = torch.zeros(bsz, 7, 7, 2048)
        for i in range(bsz):
            k = self.mapids[i] + "_" + imgtag

            if self.infos[i].get(k) is not None:
                infos = self.infos[i][k]
                self.infos_pre[i] = infos

                vfeat_name = "vfeat_" + self.mapids[i] + "_" + imgtag + ".npy"
                np_feat = np.load("./vfeat_sim_passed_sec/" + vfeat_name)
                self.vfeat_pre[i] = np_feat
                vfeat[i,:,:,:] = torch.from_numpy(np_feat)
            else:
                infos = self.infos_pre[i]

            obj_ids.append(infos[0])
            obj_pos.append(infos[1])
            agent_pos.append(infos[2])
            agent_rot.append(infos[3])

        obj_ids = torch.tensor(obj_ids)
        obj_pos = torch.tensor(obj_pos)
        agent_pos = torch.tensor(agent_pos)
        agent_rot = torch.tensor(agent_rot)

        return obj_ids, obj_pos, agent_pos, agent_pos, vfeat.cuda()

    def sendActions(self, actions, bsz, record=False, imgtag=None):

        actions_list = []
        for i in range(self.num_sim):
            start = int(bsz/self.num_sim) * i
            end = int(bsz/self.num_sim) * (1+i)
            action_str = 'a,'
            for act in actions[start:end]:
                if act == 0:
                    act_char = 'N,'
                elif act == 1:
                    act_char = 'f,'
                elif act == 2:
                    act_char = 'l,'
                elif act == 3:
                    act_char = 'r,'
                elif act == 4:
                    act_char = 'P,'

                action_str += act_char


            action_str = action_str[:-1]
            actions_list.append(action_str.encode())
        obj_ids, obj_pos, agent_pos, agent_rot, img_feats = self.connection.send(actions_list, bsz)

        vis_feat = self.extractVisFeature(action_str, img_feats, record, imgtag)
        return obj_ids, obj_pos, agent_pos, agent_rot, vis_feat

    def sendActionK(self, bsz, record=False, imgtag=None):

        actions_list = []
        for i in range(self.num_sim):
            mini_bsz = int(bsz/self.num_sim)

            action_str = 'a,'
            for i in range(mini_bsz):
                action_str += 'k,'


            action_str = action_str[:-1]
            actions_list.append(action_str.encode())
        obj_ids, obj_pos, agent_pos, agent_rot, img_feats = self.connection.send(actions_list, bsz)

        vis_feat = self.extractVisFeature(action_str, img_feats, record, imgtag)
        return obj_ids, obj_pos, agent_pos, agent_rot, vis_feat


    def sendActionW(self, bsz, w):

        actions_list = []
        for i in range(self.num_sim):
            mini_bsz = int(bsz/self.num_sim)

            action_str = 'a,'
            for i in range(mini_bsz):
                action_str += w + ','

            action_str = action_str[:-1]
            actions_list.append(action_str.encode())
        obj_ids, obj_pos, agent_pos, agent_rot, img_feats = self.connection.send(actions_list, bsz)

        return True

    def extractVisFeature(self, action_str, imgs_str, record=False, imgtag=None):

        global cnt
        img_list = []
        for i, img_str in enumerate(imgs_str):
            imgdata = base64.b64decode(img_str)

            image = Image.open(io.BytesIO(imgdata))
            img_list.append(img_transform(image))

            if False:
                image.save("./images/img" + str(cnt) + "_" + str(imgtag) + "_" + str(i) + ".png")

            image.close()
            
        cnt += 1

        with torch.no_grad():
            img_tensor = torch.stack(img_list, dim=0).cuda()
            vis_feat = self.resnet_extractor(img_tensor)
            vis_feat = vis_feat.permute(0,2,3,1)

        return vis_feat






        


