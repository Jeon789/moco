def _get_centroid2(tensors, sim='cos'):
    """ tensors =  tensor of size (num_gp,3,32,32)"""
    """ Recursive one """
    if tensors.size()[0] == 1:
        return tensors[0]
    elif tensors.size()[0] == 2:
        return torch.nn.functional.normalize(tensors[0] + tensors[1], dim=0)
    elif tensors.size()[0] > 2 :
        mid = int(tensors.size()[0] / 2)
        temp1 = _get_centroid2(tensors[:mid], sim=sim)
        temp2 = _get_centroid2(tensors[mid:], sim=sim)
        return torch.nn.functional.normalize(temp1 + temp2, dim=0)