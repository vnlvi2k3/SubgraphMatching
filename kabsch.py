import torch
def kabsch_rmsd(P, Q):
    P_mean = P.mean(dim=0)
    Q_mean = Q.mean(dim=0)
    P_cen = P - P_mean
    Q_cen = Q - Q_mean
    h = P_cen.T@Q_cen
    u, s, vt = torch.linalg.svd(h)
    v = vt.T
    d = torch.sign(torch.det(v@u.T))
    e = torch.tensor([[1,0,0],[0,1,0],[0,0,d]])
    r = v@e@u.T
    tt = Q_mean - r@P_mean
    P = (r@P.T).T + tt
    rmsd = ((P - Q)**2).sum()
    return rmsd