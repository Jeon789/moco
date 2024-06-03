import torch

def _get_centroid4(tensors, sim='cos', lr=1e-5, num_iter=5):
    """ tensors =  tensor of size (num_gp,3,32,32)"""
    """ angle n:1 """
    if sim == 'dot':
        return torch.mean(tensors, dim=0)
    elif sim == 'cos' or sim == 'angle':
        temp_cen = torch.mean(tensors, dim=0)
        N  = tensors.size()[0]
        pi = torch.pi
        for _ in range(num_iter):
            angle_dist = angular_distance(temp_cen, tensors, p=2)
            angle_dist.backward()
            with torch.no_grad():
                temp_cen -= temp_cen.grad * lr
                temp_cen = torch.nn.functional.normalize(temp_cen)
        return temp_cen
    

def angular_distance(C, X, p=2):
    N  = X.size()[0]
    pi = torch.pi
    result = 0
    for i in range(N):
        C_ = nn.functional.normalize(C, dim=0)
        X_i = nn.functional.normalize(X[i], dim=0)
        result += ( torch.arccos(torch.dot(C_,X_i)) / pi ) ** p
        # result += ( torch.arccos(torch.dot(C,X[i])) / pi ) ** p
        # result += ( torch.arccos(torch.cos(C-X[i])) / pi ) ** p
    return result / N