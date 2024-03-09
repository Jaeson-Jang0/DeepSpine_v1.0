import torch
from torch import nn
import torch.nn.functional as F
from .utils import LayerNorm, Affine
from .utils import get_activation, get_normalization
from typing import MutableSequence

class MLP_module(nn.Module):
    def __init__(self, 
            in_size, 
            out_size, 
            dropout=0,
            normalization=False,
            activation='relu', 
            last_dropout=None,
            last_normalization=None,
            last_activation=None):
        super(MLP_module, self).__init__()
        
        if isinstance(out_size, int):
            out_size = [out_size]

        if last_dropout is None:
            last_dropout = dropout
        if last_normalization is None:
            last_normalization = normalization
        if last_activation is None:
            last_activation = activation
        
        assert isinstance(out_size, MutableSequence)     
        
        layers = []
        num_layers = len(out_size)
                
        for l, hid_size in enumerate(out_size):
            # add linear layer
            layers.append(nn.Linear(in_size, hid_size))

            # add normalization
            norm = normalization if l < num_layers - 1 else last_normalization
            if norm:
                layers.append(get_normalization(norm, hid_size))

            # add activation
            act = activation if l < num_layers - 1 else last_activation
            layers.append(get_activation(act))

            # add dropout
            p = dropout if l < num_layers - 1 else last_dropout
            if p > 0:
                layers.append(nn.Dropout(p))

            in_size = hid_size
    
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    
# class MLP_module(nn.Module):
#     def __init__(self, in_size, out_size, activation='relu', interm=None):
#         super(MLP_module, self).__init__()

#         if activation == 'relu':
#             activation = nn.ReLU(inplace=True)
#         elif activation == 'tanh':
#             activation = nn.Tanh()
#         else:
#             activation = None
        
#         mlp = []
# #         if interm is not None:
# #             if interm['activation'] == 'relu':
# #                 interm_activation = nn.ReLU(inplace=True)
# #             elif interm['activation'] == 'tanh':
# #                 interm_activation = nn.Tanh()
# #             else:
# #                 interm_activation = None
                
# #             for interm_features in interm['features']:
# #                 mlp.append(nn.Linear(in_features, interm_features))
# #                 if interm_activation is not None:
# #                     mlp.append(interm_activation)

# #                 in_features = interm_features
        
#         mlp.append(nn.Linear(in_size, out_size))
#         if activation is not None:
#             mlp.append(activation)

#         self.mlp = nn.Sequential(*mlp)

#     def forward(self, x):
#         y = self.mlp(x)

#         return y
    
    
class MLP(nn.Module):
    def __init__(self, in_MLP, hid1_MLP, hid2_MLP, out_MLP):
        super(MLP, self).__init__()
        self.MLP1 = MLP_module(
                    in_size = in_MLP,
                    out_size = hid1_MLP,
                    activation='relu'
                    )
        self.MLP2 = MLP_module(
                    in_size = hid1_MLP,
                    out_size = hid2_MLP,
                    activation = 'relu'
                    )
        self.MLP3 = MLP_module(
                    in_size = hid2_MLP,
                    out_size = out_MLP,
                    activation = 'relu'
                    )

    def forward(self, x):
        y_all = []
        for i in range(x.shape[2]):
            x_temp = torch.squeeze(x[:,:,i])

            h = self.MLP1(x_temp)
            h = self.MLP2(h)
            y_temp = self.MLP3(h)
            
            y_all.append(y_temp)
            
        y_all = torch.stack(y_all, dim=2)

        return y_all   


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True, pad=0):
        super(CausalConv1d, self).__init__()
        self.pad = pad
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, bias=bias, padding=pad)
        
    def forward(self, x):
        y = self.conv(x)
        if self.pad > 0:
            y = y[:,:,:-self.pad]

        return y


class ConvNet1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, use_norm=None, activation='relu'):
        super(ConvNet1d, self).__init__()

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = None
        
        self.activation = activation
        self.use_norm = use_norm
        self.groups = groups

        self.cnn = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, pad=kernel_size-1)

        if use_norm:
            self.norm = LayerNorm(dim=2, affine_shape=[groups, int(out_channels/groups), 1])

    def forward(self, x):
        y = self.cnn(x) # BxCxT
        if self.use_norm:
            batch_size = y.size(0)
            T = y.size(-1)

            y = y.view(batch_size, self.groups, -1, T)
            y = self.norm(y)
            y = y.view(batch_size, -1, T)
                
        if self.activation is not None:
            y = self.activation(y)
            
        return y

class SensoryEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, use_norm=None, activation='relu'):
        super(SensoryEncoder, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        
        self.use_norm = use_norm
        self.groups = groups

        self.kinematic2afferent = ConvNet1d(in_channels, out_channels, kernel_size, groups, use_norm, activation)
        self.spike2rate = CausalConv1d(1, 1, kernel_size=30, groups=1, pad=29)
        self.antidromic = CausalConv1d(1, 1, kernel_size=20, groups=1, pad=19)   
        self.affine = Affine(out_channels, init_gamma=0.1, init_beta=-26) 

    def forward(self, x, ees):
        ees_amp, _ = ees.max(2)
        ees_spike = ees.clamp_(max=1) ##### Spike train
        
        afferents = self.kinematic2afferent(x) # BxCxT
        ees_amp = ees_amp.repeat(1, afferents.size(1))          # BxC
        ees_recruitment = torch.sigmoid(self.affine(ees_amp)).unsqueeze(2)  # BxCx1
        ees_rate = F.relu(self.spike2rate(ees_spike))
        ees_ortho = ees_recruitment * ees_rate                             # BxCxT
           
        ees_anti = F.relu(self.antidromic(ees_ortho.view(-1,1,ees_ortho.shape[2])))
        
        ees_anti = ees_anti.view(*ees_ortho.shape)
        
        print((ees_ortho + F.relu(afferents - ees_anti)).shape)
        
        return ees_ortho + F.relu(afferents - ees_anti)