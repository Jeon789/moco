# Integrating Multi-Label Contrastive Learning With Dual Adversarial Graph Neural Networks
# for Cross-Model Retrieval

import random
import torch
import torch.nn as nn


def calculate_class_wise_CL(feature, label, sim='dot', temp=0.1):
    """
    feature : B x ???
    label : multi-label. B x num_class. GT
    return :  CL Loss
    """

    nce_loss = nn.CrossEntropyLoss()
    batch_size = label.size()[0]
    num_class = label.size()[1]
    feature = feature.view(feature.size()[0], -1) # change into B x D dimension
    total_cl_loss = 0

    for batch_idx in range(batch_size):

        # operate CL for every class
        for class_idx in range(num_class):

            # get positive pair idx
            pos_idx = None
            # pos_idx_candidate = [(i+batch_idx+1)%batch_size for i in range(batch_size)] # 자기 다음 idx부터 돌면서 pos_idx를 찾는다
            pos_idx_candidate = list(range(batch_size))
            pos_idx_candidate.remove(batch_idx)
            random.shuffle(pos_idx_candidate)

            for i in pos_idx_candidate:
                if label[batch_idx, class_idx] == label[i, class_idx] :
                    pos_idx = i
                    break
            
            # if there is a positive pair in mini-batch
            if pos_idx !=  None:
                neg_idxs = list(range(batch_size))
                neg_idxs.remove(batch_idx)
                neg_idxs.remove(pos_idx)
                neg_pairs = feature[neg_idxs]
                pos_neg = torch.concat([feature[pos_idx].unsqueeze(0),neg_pairs],dim=0) # (B-1) x D

                if sim == 'dot':
                    sim = torch.mm(pos_neg, feature[batch_idx].unsqueeze(1)).squeeze(1) / temp # (B-1)
                if sim == 'cos':
                    sim = nn.functional.cosine_similarity(feature[batch_idx], pos_neg, dim=2) / temp


                cl_target = torch.tensor(0).long().to(sim.device)
                total_cl_loss += nce_loss(sim, cl_target)

    return total_cl_loss

## TODO neg_feature : 작명센스 발휘좀
# out_sider...?
def grouped_CL(feature, label, neg_feature=None ,sim='dot', temp=0.1):
    """
    feature : B x ???
    label : can be multi-label. B x num_class. GT
    return :  similarity, cl_label
    """
    def _tensor2str(tensor):
        "To make as a key in dict. dict('[1,0,0,1]': ~ )"
        tensor = tensor.long().cpu()
        if tensor.dim() == 1 :
            tensor = tensor.tolist()
            return ''.join(str(a) for a in tensor)
        elif tensor.dim() == 2:
            tensor = torch.unique(tensor, dim=0).tolist()
            return [ ''.join(str(a) for a in ts ) for ts in tensor]
    
    def _distribute_into_group(feature, label, batch_size):
        # 그룹화시키기
        G = {}
        groups = _tensor2str(label)
        for gp in groups: G[gp] = []
        for i in range(batch_size):       # distribute instance to corresponding label/sub-label group
            gp = _tensor2str(label[i])
            G[gp].append(feature[i])
        return G
        
    def get_centroid(tensors_list, sim='dot'):
        if sim == 'dot':
            # This is for 'dot' similarity
            # And for 'cos' but as d^1
            return torch.mean(torch.stack(tensors_list), dim=0)
        
        elif sim == 'cos' or sim == 'angle':
           # TODO Frechet Mean on Sphere 읽고 구현해야함
            for n in range(len(tensors_list)) :
                if n == 0 :
                    temp_cen = tensors_list[n]/torch.linalg.norm(tensors_list[n],2)
                else :
                    ############## arc of a circle ##############
                    # add_vec = tensors_list[n] / torch.linalg.norm(tensors_list[n], 2)
                    # inner = torch.dot(temp_cen,add_vec)
                    # theta = torch.arccos(inner)
                    
                    # y= torch.cos((n * theta) / (n + 1)) - torch.cos(theta) * torch.cos(theta / (n + 1))
                    # y= y / (1 - (torch.cos(theta)) ** 2)
                    
                    # x = torch.cos(theta / (n + 1)) - y
                    
                    # temp_cen = x * temp_cen + y * add_vec
                    # temp_cen = temp_cen / torch.linalg.norm(temp_cen, 2)
                    ############## arc of a circle ##############
                    
                    ############## geodesic ##############
                    next_vec = tensors_list[n] / torch.linalg.norm(tensors_list[n], 2)
                    theta = torch.arccos(torch.dot(temp_cen, next_vec))

                    tan_vec  = next_vec - torch.dot(next_vec, temp_cen) * temp_cen
                    tan_vec  = tan_vec / torch.linalg.norm(tan_vec, 2)
                    temp_cen = torch.cos(theta / (1 + n)) * temp_cen + torch.sin(theta / (1 + n)) * tan_vec
                    temp_cen = temp_cen / torch.linalg.norm(temp_cen, 2)
                    ############## geodesic ##############

            # This is for 'cos' and d^2 as Frechet Mean
            return temp_cen

    batch_size = label.size()[0]
    feature = feature.view(feature.size()[0], -1) # change into B x D dimension

    G = _distribute_into_group(feature, label, batch_size)
    groups = list(G.keys())


    Similarity = []
    # 각 gp 마다 한번씩 CL 진행
    for gp in G.keys():
        if len(G[gp]) <= 1 : continue
        mid = len(G[gp]) // 2
        centroid = get_centroid(G[gp][:mid], sim=sim)
        pos = get_centroid(G[gp][mid:], sim=sim)

        if neg_feature is not None :
            neg = torch.stack([get_centroid(G[neg_gp], sim=sim) for neg_gp in groups if neg_gp != gp])
            neg_ = get_centroid(neg_feature.tolist(), sim=sim)
            pos_neg = torch.concat([pos.unsqueeze(0),neg, neg_.unsqueeze(0)],dim=0) # (2^(num_deg)-1) x D
        else : 
            neg = torch.stack([get_centroid(G[neg_gp], sim=sim) for neg_gp in groups if neg_gp != gp])
            pos_neg = torch.concat([pos.unsqueeze(0),neg],dim=0)  # (2^(num_deg)-1) x D

        # calculate similarity
        if sim == 'dot':
            similarity = torch.mm(pos_neg, centroid.unsqueeze(1)).squeeze(1) / temp # (B-1)
            Similarity.append(similarity)
        if sim == 'cos':
            similarity = nn.functional.cosine_similarity(centroid, pos_neg) / temp
            Similarity.append(similarity)
        if sim == 'angle':
            angular_dist = torch.acos(nn.functional.cosine_similarity(centroid, pos_neg))
            similarity = (torch.pi - angular_dist) / temp
            Similarity.append(similarity)

    Similarity = torch.stack(Similarity) # (num_gp, num_gp)
    cl_target = torch.zeros(len(G)).to(Similarity.device).long()
    return Similarity, cl_target


def grouped_CL_origin(feature, label, sim='dot', temp=0.1):
    """
    feature : B x ???
    label : can be multi-label. B x num_class. GT
    return :  CL Loss
    """
    def _tensor2str(tensor):
        tensor = tensor.long().cpu()
        if tensor.dim() == 1 :
            tensor = tensor.tolist()
            return ''.join(str(a) for a in tensor)
        elif tensor.dim() == 2:
            tensor = torch.unique(tensor, dim=0).tolist()
            return [ ''.join(str(a) for a in ts ) for ts in tensor]
    
    def _distribute_into_group(feature, label, batch_size):
        # 그룹화시키기
        G = {}
        groups = _tensor2str(label)
        for gp in groups: G[gp] = []
        for i in range(batch_size):       # distribute instance to corresponding label/sub-label group
            gp = _tensor2str(label[i])
            G[gp].append(feature[i])
        return G
    
    def _angular_distance(C, X, p=1):
        """
        X : N X D / C : D
        """
        N  = X.size()[0]
        pi = torch.pi
        result = 0
        for i in range(N):
            C_ = nn.functional.normalize(C, dim=0)
            X_i = nn.functional.normalize(X[i], dim=0)
            result += ( torch.arccos(torch.dot(C_,X_i)) / pi ) ** p
        return result / N
        
    def get_centroid(tensors_list, sim='dot'):
        if sim == 'dot':
            # This is for 'dot' similarity
            # And for 'cos' but as d^1
            return torch.mean(torch.stack(tensors_list), dim=0)
        
        elif sim == 'cos' or sim == 'angle':
           # TODO Frechet Mean on Sphere 읽고 구현해야함
            for n in range(len(tensors_list)) :
                if n == 0 :
                    temp_cen = tensors_list[n]/torch.linalg.norm(tensors_list[n],2)
                else :
                    ############## arc of a circle ##############
                    # add_vec = tensors_list[n] / torch.linalg.norm(tensors_list[n], 2)
                    # inner = torch.dot(temp_cen,add_vec)
                    # theta = torch.arccos(inner)
                    
                    # y= torch.cos((n * theta) / (n + 1)) - torch.cos(theta) * torch.cos(theta / (n + 1))
                    # y= y / (1 - (torch.cos(theta)) ** 2)
                    
                    # x = torch.cos(theta / (n + 1)) - y
                    
                    # temp_cen = x * temp_cen + y * add_vec
                    # temp_cen = temp_cen / torch.linalg.norm(temp_cen, 2)
                    ############## arc of a circle ##############
                    
                    ############## geodesic ##############
                    next_vec = tensors_list[n] / torch.linalg.norm(tensors_list[n], 2)
                    theta = torch.arccos(torch.dot(temp_cen, next_vec))

                    tan_vec  = next_vec - torch.dot(next_vec, temp_cen) * temp_cen
                    tan_vec  = tan_vec / torch.linalg.norm(tan_vec, 2)
                    temp_cen = torch.cos(theta / (1 + n)) * temp_cen + torch.sin(theta / (1 + n)) * tan_vec
                    temp_cen = temp_cen / torch.linalg.norm(temp_cen, 2)
                    ############## geodesic ##############

            # This is for 'cos' and d^2 as Frechet Mean
            return temp_cen

    nce_loss = nn.CrossEntropyLoss()
    batch_size = label.size()[0]
    feature = feature.view(feature.size()[0], -1) # change into B x D dimension
    total_cl_loss = 0


    G = _distribute_into_group(feature, label, batch_size)
    groups, num_gp = list(G.keys()), len(G.keys())

    # 각 gp 마다 한번씩 CL 진행
    for gp in G.keys():
        if len(G[gp]) <= 1 :
            continue
        mid = len(G[gp]) // 2
        centroid = get_centroid(G[gp][:mid], sim=sim) # TODO Frechet Mean 처럼 구현하고 sim 풀어놔야함
        pos = get_centroid(G[gp][mid:], sim=sim)

        neg_pairs = torch.stack([get_centroid(G[neg_gp], sim=sim) for neg_gp in groups if neg_gp != gp]) 

        pos_neg = torch.concat([pos.unsqueeze(0),neg_pairs],dim=0) # (B-1) x D

        if sim == 'dot':
            similarity = torch.mm(pos_neg, centroid.unsqueeze(1)).squeeze(1) / temp # (B-1)
        if sim == 'cos':
            similarity = nn.functional.cosine_similarity(centroid, pos_neg) / temp
        ## TODO
        # 이걸로도 실험 돌려봐야함
        if sim == 'angle':
            angular_dist = torch.acos(nn.functional.cosine_similarity(centroid, pos_neg))
            similarity = (torch.pi - angular_dist) / temp

        cl_target = torch.tensor(0).long().to(similarity.device)
        total_cl_loss += nce_loss(similarity, cl_target)

    return total_cl_loss / num_gp


# TODO
# 미완성
# num_sub_label 변하게 이어줘야함
# nested = epoch 이 진행됨에 따라 num_sub_label을 1~4(=# of dgd) 까지 올려 나감
def nested_grouped_calculate_class_wise_CL(feature, label, sim='dot', temp=0.1, num_sub_label=3):
    """
    feature : B x ???
    label : multi-label. B x num_class. GT
    return :  CL Loss
    """
    def _tensor2str(tensor):
        tensor = tensor.long().cpu()
        if tensor.dim() == 1 :
            tensor = tensor.tolist()
            return ''.join(str(a) for a in tensor)
        elif tensor.dim() == 2:
            tensor = torch.unique(tensor, dim=0).tolist()
            return [ ''.join(str(a) for a in ts ) for ts in tensor]

    nce_loss = nn.CrossEntropyLoss()
    batch_size = label.size()[0]
    num_class = label.size()[1]
    feature = feature.view(feature.size()[0], -1) # change into B x D dimension
    total_cl_loss = 0


    # get random sub-label
    choosen_labels = random.sample(range(num_class), k=num_sub_label)
    sub_label = label[:, choosen_labels]

    # 그룹화시키기
    G = {}
    groups = _tensor2str(sub_label)
    for gp in groups: G[gp] = []
    for i in range(batch_size):       # distribute instance to corresponding sub-label group
        gp = _tensor2str(sub_label[i])
        G[gp].append(feature[i])

    # 각 gp 마다 한번씩 CL 진행
    for gp in groups:
        if len(G[gp]) <= 1 :
            continue
        mid = len(G[gp]) // 2
        centroid = torch.mean(torch.stack(G[gp][:mid]), dim=0)
        pos = torch.mean(torch.stack(G[gp][mid:]), dim=0)

        neg_pairs = torch.stack([torch.mean(torch.stack(G[neg_gp]), dim=0) for neg_gp in groups if neg_gp != gp])
        pos_neg = torch.concat([pos.unsqueeze(0),neg_pairs],dim=0) # (B-1) x D

        if sim == 'dot':
            sim = torch.mm(pos_neg, centroid.unsqueeze(1)).squeeze(1) / temp # (B-1)
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(centroid, pos_neg) / temp



        cl_target = torch.tensor(0).long().to(sim.device)
        total_cl_loss += nce_loss(sim, cl_target)

    return total_cl_loss



def Charbonnier_loss(X, Y, eps=1e-9):
    """
    Charbonnier_loss = ||z
    """
    diff = X-Y
    return torch.mean(torch.sqrt((diff*diff) + eps))


# Get from SwinIR code
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss
