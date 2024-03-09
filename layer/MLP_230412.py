# revised from 'feedforward': To test MLP compared to spine_230314_withKineFB

import torch
from torch import nn
import torch.nn.functional as F

import math

##### 230202: From feedforward.py
import torch
from torch import nn
import torch.nn.functional as F
from .utils import LayerNorm, Affine, LinearLN
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


class MLP_230412(nn.Module):
    # simple stacked gru (no top-down pathway)
    def __init__(self,
                 emb_in_channels,
                 emb_out_channels,
                 emb_kernel_size,
                 emb2_in_channels,
                 emb2_out_channels,
                 input_size,
                 hidden_size,
                 Tmax,
                 rdo_in_channels,
                 rdo_out_channels,
                 rdo_kernel_size,
                 rdo_groups=1,
                 rdo_activation='none',
                 offset=0,
                 activation='relu',
                 dropout=None,
                 rdo_use_norm=None,
                 emb_groups=1,
                 emb_activation='relu',      
                 emb_use_norm=None,                 
                 output_norm=None
                ):
        super(MLP_230412, self).__init__()
        
        ##### Sensory encoder
        if emb_activation == 'relu':
            self.emb_activation = nn.ReLU(inplace=True)
        elif emb_activation == 'tanh':
            self.emb_activation = nn.Tanh()
        elif emb_activation == 'sigmoid':
            self.emb_activation = nn.Sigmoid()
        else:
            self.emb_activation = None
        
        self.emb_use_norm = emb_use_norm
        self.emb_groups = emb_groups

        self.emb_kinematics2ch = ConvNet1d(emb_in_channels, emb_out_channels, emb_kernel_size, emb_groups, emb_use_norm, emb_activation)
        self.emb_ch2afferent = ConvNet1d(emb2_in_channels, emb2_out_channels, emb_kernel_size, emb_groups, emb_use_norm, emb_activation)
        self.emb_spike2rate = CausalConv1d(1, 1, kernel_size=30, groups=1, pad=29)
        self.emb_antidromic = CausalConv1d(1, 1, kernel_size=20, groups=1, pad=19)   
        self.emb_affine = Affine(emb_out_channels, init_gamma=0.1, init_beta=-26) 
        
        ##### Core: MLP
        conv1d = nn.Conv1d
        self.dropout = dropout
        self.offset = offset

        self.MLP_input_size = input_size
        self.MLP_hid_size = hidden_size
        
        self.MLP1 = MLP_module(         
            in_size = emb2_out_channels,
            out_size = self.MLP_input_size,
            activation = 'relu'
        )
        self.MLP2 = MLP_module(
            in_size = self.MLP_input_size,
            out_size = self.MLP_hid_size,
            activation = 'relu'
        )
        self.MLP3 = MLP_module(
            in_size = self.MLP_hid_size,
            out_size = rdo_in_channels,
            activation = 'identity'
        )        
        
#         self.offset = offset
#         conv1d = nn.Conv1d
        
#         if isinstance(hidden_size, int):
#             hidden_size = [hidden_size]

#         self.hidden_size = hidden_size

#         grus = []
        
# #         for i in range(50):
# #             hidden_size = self.hidden_size[0]
#         for hidden_size in self.hidden_size:
#             gru = nn.GRUCell(input_size, hidden_size)
            
#             self._chrono_init(gru, hidden_size, Tmax)
#             grus.append(gru)

#             input_size = hidden_size

#         self.grus = nn.ModuleList(grus)
#         self.n_layers = len(grus)
        
#         if dropout is not None:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = None

#         if output_norm is not None:
#             self.output_norm = nn.InstanceNorm1d(input_size, affine=True)
#             nn.init.constant_(self.output_norm.weight, 0.1)
#             nn.init.constant_(self.output_norm.bias, 0)
#         else:
#             self.output_norm = None
            
        ##### Readout
        if rdo_activation == 'relu':
            rdo_activation = nn.ReLU(inplace=True)
        elif rdo_activation == 'tanh':
            rdo_activation = nn.Tanh()
        elif rdo_activation == 'sigmoid':
            rdo_activation = nn.Sigmoid()
        else:
            rdo_activation = None
        
        self.rdo_activation = rdo_activation
        self.rdo_use_norm = rdo_use_norm
        self.rdo_groups = rdo_groups

        self.rdo_cnn = CausalConv1d(rdo_in_channels, rdo_out_channels, kernel_size=rdo_kernel_size, groups=rdo_groups, pad=rdo_kernel_size-1)

        if rdo_use_norm:
            self.rdo_norm = LayerNorm(dim=2, affine_shape=[rdo_groups, int(rdo_out_channels/rdo_groups), 1])

        ##### 230209: Somatosensory feedback        
        self.emg2aff_connection = conv1d(rdo_out_channels, emb2_out_channels, kernel_size=1, groups=1)   
            
#     def _chrono_init(self, gru, hidden_size, Tmax):
#         gru.bias_ih.data.fill_(0)
#         gru.bias_hh.data.fill_(0)
        
#         torch.nn.init.uniform_(gru.bias_hh[hidden_size:2*hidden_size].data, 1, Tmax - 1)
#         gru.bias_hh[hidden_size:2*hidden_size].data.log_()#.mul_(-1)
            
            
#     def _init_zero_state(self, batch_size, device):
#         hiddens = []

#         for hidden_size in self.hidden_size:
#             hiddens.append(torch.zeros(batch_size, hidden_size).to(device))

#         return hiddens

    def forward(self, kinematics, ees, with_ees, verbose=False):
        ##### Common parameters
        t_bin = 50
        t_int = 25
        t_total = kinematics.shape[2]
        n_bin = int((t_total-(t_bin-t_int))/t_int)
        
        ##### Core
        if self.dropout is not None:
            afferents = self.dropout(afferents)
        
        # xs: [batch x dim x time]
        batch_size = kinematics.size(0)
        T = kinematics.size(2)
        device = kinematics.device

#         hiddens = self._init_zero_state(batch_size, device)
#         ys = []
#         last_hiddens = []

        if self.dropout is not None:
            xs = self.dropout(xs)

        ##### in time
        for t in range(n_bin):
            t_start = int(t*t_int)
            kinematics_bin = kinematics[:,:,t_start:t_start+t_bin]
            
            afferents_ch = self.emb_kinematics2ch(kinematics_bin)
            
            if with_ees:
                afferents_ch_ees = afferents_ch + ees[:,:,t_start:t_start+t_bin]
            else:
                afferents_ch_ees = afferents_ch
            
            afferents = self.emb_ch2afferent(afferents_ch_ees)               
             
            batch_size, T = afferents[0], afferents[-1]  

            ##### 230309
            if t > 17:                
                kinematics_bin_emg = torch.cat((torch.tile(emg_1, (1,1,25)), torch.tile(emg, (1,1,25))), dim=2) 
                afferents_ch_emg = self.emb_kinematics2ch(kinematics_bin_emg)
                              
                if with_ees:
                    afferents_ch_ees_emg = afferents_ch_emg + ees[:,:,t_start:t_start+t_bin]
                else:
                    afferents_ch_ees_emg = afferents_ch_emg

                afferents_emg = self.emb_ch2afferent(afferents_ch_ees_emg)
                       
        ##### Time series
#         for t in range(T):
            ##### 230227: Providing afferents before EES ##### 230209: Somatosensory feedback 
            if t == 0:
                afferents_t = afferents 
            elif t < 19: # 230309
                afferents_t = afferents + feedback                
            elif t == 20: # 230309       
                afferents_t = torch.cat((afferents[:, :, :25], afferents_emg[:,:,25:]), dim=2) + feedback
            else:
                afferents_t = afferents_emg + feedback
         
            y_all = []
            for i in range(50):
                x_temp = torch.squeeze(afferents_t[:,:,i])

                h = self.MLP1(x_temp)
                h = self.MLP2(h)
                y_temp = self.MLP3(h)

                y_all.append(y_temp)

            mns = torch.mean(torch.stack(y_all, dim=2), dim=2, keepdim=True) 
#             print(afferents_t.shape, y_all.shape, mns.shape)
            
#             for i in range(50):
# #             hidden_size = self.hidden_size[0]
#                 for gru in self.grus:
                
#                     hx = hiddens[-self.n_layers]
#                     h = gru(torch.squeeze(afferents_t[:,:,i]), hx)

#                     hiddens.append(h)
# #                 x = h #####
#             mns = torch.unsqueeze(h,2)
# #             mns = h

#             last_hiddens.append(h)
#             last_hiddens = torch.stack(last_hiddens, dim=2)
#             return last_hiddens
        
            ##### 230209: Readout
            if t > 2: emg_1 = emg ##### 230309
            
            emg = self.rdo_cnn(mns) # BxCxT
            
            if self.rdo_use_norm:
                batch_size = emg.size(0)
                T = y.size(-1)

                emg = emg.view(batc_size, self.groups, -1, T)
                emg = self.norm(emg)
                emg = emg.view(batch_size, -1, T)

            if self.rdo_activation is not None:
                emg = self.rdo_activation(emg)
                                    
            ##### 230209: Somatosensory feedback
            feedback = self.emg2aff_connection(emg)
                        
            if t == 0:
                emg_stack = emg
                                
            else:
                emg_stack = torch.cat((emg_stack, emg), 2)               
        
        if verbose:
            if self.offset > 0:
                return mns[:,:,self.offset:], Iais[:,:,self.offset:], exs[:,:,self.offset:]
            else:
                return mns, Iais, exs
        else:        
            if self.offset > 0:
                return emg_stack[:,:,self.offset:], mns_stack[:,:,self.offset:], exs_stack[:,:,self.offset:], Iais_stack[:,:,self.offset:]
            else:
                return emg_stack, [afferents_ch, afferents_ch_ees, afferents]
