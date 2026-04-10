import torch


_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
log_theta = torch.nn.LogSigmoid()


def get_logp(C, z, logdet_J, mu=None):
    # HGADの数式 (13), (14) に基づき、中心点をシフトさせる
    if mu is not None:
        # z: [N, C], mu: [1, C] のため、ブロードキャストで引き算
        diff = z - mu
    else:
        diff = z
        
    # 尤度計算: z**2 の部分を (z - mu)**2 に置き換え
    logp = C * _GCONST_ - 0.5 * torch.sum(diff**2, 1) + logdet_J
    return logp

def neg_relu(logps):
    zeros = logps.new_zeros(logps.shape)
    logps = torch.min(logps, zeros)
    
    return logps
