import torch

def fidelity_loss(pred, target):
    pred = pred / pred.norm()
    target = target / target.norm()
    overlap = torch.abs(torch.trace(pred @ target))
    return 1 - overlap

def physical_constraint_loss(pred):
    eigenvalues = torch.linalg.eigvalsh(pred)
    return torch.relu(-eigenvalues).mean()
