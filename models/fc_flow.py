from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def flow_model(args, in_channels, **kwargs):
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    coupling_layers = kwargs.get('coupling_layers', None)
    coupling_layers = coupling_layers if coupling_layers is not None else args.coupling_layers
    for k in range(coupling_layers):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def conditional_flow_model(args, in_channels, **kwargs):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    coupling_layers = kwargs.get('coupling_layers', None)
    coupling_layers = coupling_layers if coupling_layers is not None else args.coupling_layers
    for k in range(coupling_layers):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(args.pos_embed_dim,), subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_flow_model(args, in_channels, **kwargs):
    if args.flow_arch == 'flow_model':
        model = flow_model(args, in_channels, **kwargs)
    elif args.flow_arch == 'conditional_flow_model':
        model = conditional_flow_model(args, in_channels, **kwargs)
    else:
        raise NotImplementedError('{} is not supported Normalizing Flow!'.format(args.flow_arch))
    
    return model
class FCFlow(nn.Module):
    # ... 既存の __init__ ...

    def forward(self, x, mu=None):
        # x: 入力特徴 [Batch, C, H, W]
        # mu: 動的な中心点 [1, C, 1, 1] または [Batch, C, H, W]
        
        # 既存の処理で潜在変数 z と log_det_jac を取得
        z, jac = self.flow(x) 
        
        # z の形状に合わせて mu を拡張（必要に応じて）
        if mu is not None:
            # パッチごとの平均を使う場合は形状を合わせる
            if mu.dim() == 2: # [1, C] の場合
                mu = mu.unsqueeze(-1).unsqueeze(-1)
        else:
            mu = 0.0

        # 対数尤度の計算 (HGADの数式13, 14を参照 [cite: 593])
        # 標準ガウス分布 N(0, I) ではなく N(mu, I) を仮定
        log_prob_z = -0.5 * z.shape[1] * math.log(2 * math.pi) - 0.5 * torch.sum((z - mu)**2, dim=1)
        
        log_likelihood = log_prob_z + jac
        return z, log_likelihood
