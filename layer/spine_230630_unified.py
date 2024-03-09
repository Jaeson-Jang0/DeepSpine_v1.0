# Revised from "spine_230627_unified": To fix FB circuits (from motor neurons not from rdo)
# Revised from "spine_230605_simpleAct: To fix 1-neuron error + merged with synthetic
# Revised from "spine_230531_withKineFB: Saving all the variables
# Revised from "spine_230314_withKineFB: To add weight initialization and clipping on FB
# Revised from "spine_230314_withKineFB": To train with non-EES data
# Revised from "spine_230310": To train before EES until end of trial
# Revised from "spine_230309": To train before EES
# Revised from "spine_230307": To use feedback from kinematics prediction
# Revised from "spine_230227": 
# Revised from "spine_230223": To only use input before EES
# Revised from "spine_230216": To apply to treadmill data
# Revised from "spine_230209": To use afferents as input, kinematics as output
# Revised from 'spine_230202': To merge with sensory encoder for somatosensory feedback
# Revised from 'spine_230119': 
# Revised from "spine.py": Restricting weights within non-negative values + Adding somatosensory feedback

import torch
from torch import nn
import torch.nn.functional as F

from .utils import LayerNorm, Affine, LinearLN
import copy

##### 230202: From feedforward.py
import torch
from torch import nn
import torch.nn.functional as F
from .utils import LayerNorm, Affine
from .utils import get_activation, get_normalization
from typing import MutableSequence

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


class Integrator(nn.Module):
    def __init__(self, out_neurons, Tmax, activation):
        super(Integrator, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = None

        self.Tmax = Tmax
        self.x2h = nn.Sequential(  
            nn.Linear(out_neurons, out_neurons, bias=False),
            nn.LayerNorm(out_neurons)
        )

        self.h2h = nn.Sequential(  
            nn.Linear(out_neurons, 2*out_neurons, bias=False),
            nn.LayerNorm(2*out_neurons)
        )

        self._reset_parameters()        
     
    def _reset_parameters(self):
        # chrono initialization
        nn.init.constant_(self.x2h[-1].weight, 0.1)
        self.x2h[-1].bias.data = -torch.log(torch.nn.init.uniform_(self.x2h[-1].bias.data, 1, self.Tmax - 1))
        
        nn.init.constant_(self.h2h[-1].weight, 0.1)
        nn.init.constant_(self.h2h[-1].bias, 0)

    def forward(self, x, hx):
        batch_size = x.shape[0]

        h_i, h_g = self.h2h(hx).chunk(2,dim=1)        
      
        x = self.activation(x + h_i)

        # compute a dynamic stepsize (Eq. (4))
        _g = self.x2h(x) + h_g
        g = torch.sigmoid(_g)

        h = (1 - g) * hx + g * x    # (Eq. (7))
        
        # x, hx, h_i, h_g, _g, g, h
#         print(torch.squeeze(x)[0].cpu().numpy(), torch.squeeze(hx)[0].cpu().numpy(), torch.squeeze(h_i)[0].cpu().numpy(), torch.squeeze(h_g)[0].cpu().numpy(), torch.squeeze(_g)[0].cpu().numpy(), torch.squeeze(g)[0].cpu().numpy(), torch.squeeze(h)[0].cpu().numpy())
        
        return h


class Layer(nn.Module):
    def __init__(self, 
            in_pathways, 
            out_neurons, 
            Tmax,
            activation,
            reciprocal_inhibition=False):
        super(Layer, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = None
        
        self.norm = LayerNorm(dim=2, affine_shape=[in_pathways,out_neurons,1]) # v230119

        self.in_pathways = in_pathways
        self.out_neurons = out_neurons
        
        self.affine_layer = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)
        self.int_layer = Integrator(out_neurons, Tmax, activation)


    def forward(self, xs, init_h, EI=None):
        # B: batch size
        # P: # of input pathways either for flexor and extensor
        # N: # of output neurons
        # T: # of time steps
           
        xs = torch.stack(xs, dim=1)         # [Bx(2xP)xNxT]
#         xs = self.norm(xs) # apply layer normalization separately per pathway #### Critical
         
        batch_size, T = xs.shape[0], xs.shape[-1]  
        
        ##### v230119
        hx = init_h        # [Bx(2xN)]        
        xx = xs[...,-1]

        # compute a dynamic gain control (Eq. (6))
        g_pre = self.affine_layer(hx.unsqueeze(1).repeat(1,self.in_pathways,1))

        g = torch.sigmoid(g_pre)
        in_layer_pre = g*xx
        
        if EI is not None:                
            # EI: [P]
            # EI set to True for excitation and set to False for inhibition pathways
            in_layer = self.activation(in_layer_pre[:,EI].sum(1)) - self.activation(in_layer_pre[:,~EI].sum(1)) # [BxN]
#             in_layer = self.activation(in_layer[:,EI].sum(1) - in_layer[:,~EI].sum(1)) # [BxN]
        else:
            in_layer = self.activation(in_layer_pre.sum(1))
       
        h = self.int_layer(in_layer, hx)
        h = h.unsqueeze(2)
    
        return h, in_layer, hx, xx, g, g_pre, in_layer_pre


class SpinalCordCircuit_230630_unified(nn.Module):
    def __init__(self,
            emb_in_channels,
            emb_out_channels,
            emb_kernel_size,
            emb2_in_channels,
            emb2_out_channels,
            Ia_neurons, 
            II_neurons, 
            ex_neurons, 
            Iai_neurons, 
            mn_neurons,
            Tmax,
            rdo_in_channels, rdo_emg_channels, rdo_out_channels, rdo_kernel_size, rdo_groups=1, rdo_activation='none',
            offset=0, activation='relu', dropout=None, rdo_use_norm=None,
            emb_groups=1,
            emb_activation='relu',      
            emb_use_norm=None
                ):
        super(SpinalCordCircuit_230630_unified, self).__init__()
        
        ##### 230209: Sensory encoder
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

        ##### Core
        self.Ia_neurons = Ia_neurons
        self.II_neurons = II_neurons
        self.ex_neurons = ex_neurons
        self.Iai_neurons = Iai_neurons
        self.mn_neurons = mn_neurons

        self.offset = offset
        conv1d = nn.Conv1d

        # v230119
        self.ex_flxs = Layer(
            in_pathways=1, 
            out_neurons=ex_neurons, 
            Tmax=Tmax,
            activation=activation
        )
        self.Iai_flxs_pre = Layer(
            in_pathways=2, 
            out_neurons=Iai_neurons, 
            Tmax=Tmax,
            activation=activation,
            reciprocal_inhibition=True
        )
        self.Iai_flxs = Layer(
            in_pathways=3, 
            out_neurons=Iai_neurons, 
            Tmax=Tmax,
            activation=activation,
            reciprocal_inhibition=True
        )
        self.mn_flxs = Layer(
            in_pathways=3, 
            out_neurons=mn_neurons, 
            Tmax=Tmax,
            activation=activation
        )
        
        self.ex_exts = Layer(
            in_pathways=1, 
            out_neurons=ex_neurons, 
            Tmax=Tmax,
            activation=activation
        )
        self.Iai_exts_pre = Layer(
            in_pathways=2, 
            out_neurons=Iai_neurons, 
            Tmax=Tmax,
            activation=activation,
            reciprocal_inhibition=True
        )
        self.Iai_exts = Layer(
            in_pathways=3, 
            out_neurons=Iai_neurons, 
            Tmax=Tmax,
            activation=activation,
            reciprocal_inhibition=True
        )
        self.mn_exts = Layer(
            in_pathways=3, 
            out_neurons=mn_neurons, 
            Tmax=Tmax,
            activation=activation
        )
                
        self.IaF2mnFIaiF_connection = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)                
        self.IIF2exFIaiF_connection = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=1)        
        self.exF2mnF_connection = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=1)
        self.IaiF2IaiEmnE_connection = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)
        
        self.IaE2mnEIaiE_connection = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)                
        self.IIE2exEIaiE_connection = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=1)        
        self.exE2mnE_connection = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=1)
        self.IaiE2IaiFmnF_connection = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)
        
        torch.nn.init.trunc_normal_(self.IaF2mnFIaiF_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIF2exFIaiF_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exF2mnF_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiF2IaiEmnE_connection.weight, a=-float('inf'), b=0)
        
        torch.nn.init.trunc_normal_(self.IaE2mnEIaiE_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIE2exEIaiE_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exE2mnE_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiE2IaiFmnF_connection.weight, a=-float('inf'), b=0)
        
        ##### Saving initial weight matrices
        self.IaF2mnFIaiF_connection_ini = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)        
        self.IIF2exFIaiF_connection_ini = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=1)        
        self.exF2mnF_connection_ini = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=1)
        self.IaiF2IaiEmnE_connection_ini = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)
        
        self.IaE2mnEIaiE_connection_ini = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)          
        self.IIE2exEIaiE_connection_ini = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=1)        
        self.exE2mnE_connection_ini = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=1)
        self.IaiE2IaiFmnF_connection_ini = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)

        self.IaF2mnFIaiF_connection_ini.weight = torch.nn.parameter.Parameter(self.IaF2mnFIaiF_connection.weight)
        self.IIF2exFIaiF_connection_ini.weight = torch.nn.parameter.Parameter(self.IIF2exFIaiF_connection.weight)
        self.exF2mnF_connection_ini.weight = torch.nn.parameter.Parameter(self.exF2mnF_connection.weight)
        self.IaiF2IaiEmnE_connection_ini.weight = torch.nn.parameter.Parameter(self.IaiF2IaiEmnE_connection.weight)

        self.IaE2mnEIaiE_connection_ini.weight = torch.nn.parameter.Parameter(self.IaE2mnEIaiE_connection.weight)
        self.IIE2exEIaiE_connection_ini.weight = torch.nn.parameter.Parameter(self.IIE2exEIaiE_connection.weight)
        self.exE2mnE_connection_ini.weight = torch.nn.parameter.Parameter(self.exE2mnE_connection.weight)
        self.IaiE2IaiFmnF_connection_ini.weight = torch.nn.parameter.Parameter(self.IaiE2IaiFmnF_connection.weight)

        # excitatory pathways (True): Ia2mn, ex2mn || inhibitory pathways (False): Iai2mn
        self.register_buffer('Iai_connectivity', torch.BoolTensor([True, True, False])) 
        self.register_buffer('mn_connectivity', torch.BoolTensor([True, True, True])) ##### 230531: Already negative
#         self.register_buffer('mn_connectivity', torch.BoolTensor([True, True, False])) ##### 230531: Already negative

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
            
        ##### 230209: Readout
        if rdo_activation == 'relu':
            rdo_activation = nn.ReLU(inplace=True)
        elif rdo_activation == 'tanh':
            rdo_activation = nn.Tanh()
        elif rdo_activation == 'sigmoid':
            rdo_activation = nn.Sigmoid()
        else:
            rdo_activation = None
        
        self.rdo_activation = rdo_activation
        self.rdo_groups = rdo_groups
       
        self.rdo_mns2emg = CausalConv1d(rdo_in_channels, rdo_emg_channels, kernel_size=rdo_kernel_size, groups=rdo_groups, pad=rdo_kernel_size-1)
        self.rdo_emg2kine = CausalConv1d(rdo_emg_channels, rdo_out_channels, kernel_size=rdo_kernel_size, groups=rdo_groups, pad=rdo_kernel_size-1)
        
#         self.rdo_use_norm = rdo_use_norm
#         if rdo_use_norm:
#             self.rdo_norm = LayerNorm(dim=2, affine_shape=[rdo_groups, int(rdo_out_channels/rdo_groups), 1])
        
        ##### 230209: Somatosensory feedback        
        self.mns2aff_connection = conv1d(2*mn_neurons, 2*(Ia_neurons+II_neurons), kernel_size=1, groups=1) ##### 230630
        torch.nn.init.trunc_normal_(self.mns2aff_connection.weight, a=0.0, b=float('inf')) ##### 230531

        ##### 230531: Bias clipping
        torch.nn.init.trunc_normal_(self.IaF2mnFIaiF_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIF2exFIaiF_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exF2mnF_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiF2IaiEmnE_connection.bias, a=-float('inf'), b=0)
        
        torch.nn.init.trunc_normal_(self.IaE2mnEIaiE_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIE2exEIaiE_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exE2mnE_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiE2IaiFmnF_connection.bias, a=-float('inf'), b=0)
        
        torch.nn.init.trunc_normal_(self.mns2aff_connection.bias, a=0.0, b=float('inf')) ##### 230531

        
    def _init_hidden(self, x): ##### 230307
        batch_size = x.size(0)

        init_ex = x.new(batch_size, self.ex_neurons).zero_() # v230119
        init_Iai = x.new(batch_size, self.Iai_neurons).zero_()
        init_mn = x.new(batch_size, self.mn_neurons).zero_()
        
        return init_ex, init_Iai, init_mn

    def forward(self, kinematics, ees, with_ees, verbose=False): ##### 230307
        t_total = kinematics.shape[2]
        t0 = 500
                
        ##### Core
        init_ex, init_Iai, init_mn = self._init_hidden(kinematics)
        init_exF, init_exE = init_ex, init_ex
        init_IaiF, init_IaiE = init_Iai, init_Iai
        init_mnF, init_mnE = init_mn, init_mn
        
        if self.dropout is not None:
            afferents = self.dropout(afferents)
        
        for t in range(t_total):
#             print('T = ', t)
            kinematics_bin = kinematics[:,:,t:t+1]
            
            afferents_ch = self.emb_kinematics2ch(kinematics_bin)
            
            if with_ees:
                afferents_ch_ees = afferents_ch + ees[:,:,t:t+1]
            else:
                afferents_ch_ees = copy.deepcopy(afferents_ch)
            
            afferents = self.emb_ch2afferent(afferents_ch_ees)               
              
            # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)

            Ia, II = torch.split(afferents, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)              
            Ia_flxs, Ia_exts = torch.split(Ia, [self.Ia_neurons, self.Ia_neurons], dim=1)
            II_flxs, II_exts = torch.split(II, [self.II_neurons, self.II_neurons], dim=1) 

            batch_size, T = Ia_flxs.shape[0], Ia_flxs.shape[-1]  
            
            ##### 230309
            if t > 0:                
                afferents_ch_kine = self.emb_kinematics2ch(kine)
                              
                if with_ees:
                    afferents_ch_ees_kine = afferents_ch_kine + ees[:,:,t:t+1]
                    afferents_kine = self.emb_ch2afferent(afferents_ch_ees_kine)               
                else:
                    afferents_kine = self.emb_ch2afferent(afferents_ch_kine)               

                # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)
                Ia_kine, II_kine = torch.split(afferents_kine, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)              
                Ia_flxs_kine, Ia_exts_kine = torch.split(Ia_kine, [self.Ia_neurons, self.Ia_neurons], dim=1)
                II_flxs_kine, II_exts_kine = torch.split(II_kine, [self.II_neurons, self.II_neurons], dim=1) 
        
            ##### Time series
            # fb_Ia_flxs: FB from motor neurons, Ia_flxs_emg: Neural afferents from predicted kinematics
            if t == 0:            
                Ia_flxs_t = (Ia_flxs) # copy.deepcopy
                II_flxs_t = (II_flxs)
                Ia_exts_t = (Ia_exts)
                II_exts_t = (II_exts)
            elif t < int(t0/2):
                coef = t/int(t0/2)
                Ia_flxs_t = (1-coef)*Ia_flxs + coef*Ia_flxs_kine + fb_Ia_flxs # withKineFB
                II_flxs_t = (1-coef)*II_flxs + coef*II_flxs_kine + fb_II_flxs
                Ia_exts_t = (1-coef)*Ia_exts + coef*Ia_exts_kine + fb_Ia_exts
                II_exts_t = (1-coef)*II_exts + coef*II_exts_kine + fb_II_exts
            else:
                Ia_flxs_t = Ia_flxs_kine + fb_Ia_flxs
                II_flxs_t = II_flxs_kine + fb_II_flxs
                Ia_exts_t = Ia_exts_kine + fb_Ia_exts
                II_exts_t = II_exts_kine + fb_II_exts           
        
            ##### Ia & II neural afferents to neurons
            IaF2mnFIaiF = self.IaF2mnFIaiF_connection(Ia_flxs_t)
            IIF2exFIaiF = self.IIF2exFIaiF_connection(II_flxs_t)
            IaE2mnEIaiE = self.IaE2mnEIaiE_connection(Ia_exts_t)
            IIE2exEIaiE = self.IIE2exEIaiE_connection(II_exts_t)            
                       
            IaF2mnF, IaF2IaiF = IaF2mnFIaiF.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIF2exF, IIF2IaiF = IIF2exFIaiF.split([self.ex_neurons, self.Iai_neurons], dim=1)
            IaE2mnE, IaE2IaiE = IaE2mnEIaiE.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIE2exE, IIE2IaiE = IIE2exEIaiE.split([self.ex_neurons, self.Iai_neurons], dim=1)
            
#             IaF2mnF = torch.mean(IaF2mnF, dim=2, keepdim=True) ##### 230307
#             IaF2IaiF = torch.mean(IaF2IaiF, dim=2, keepdim=True)
#             IIF2exF = torch.mean(IIF2exF, dim=2, keepdim=True)
#             IIF2IaiF = torch.mean(IIF2IaiF, dim=2, keepdim=True)
#             IaE2mnE = torch.mean(IaE2mnE, dim=2, keepdim=True)
#             IaE2IaiE = torch.mean(IaE2IaiE, dim=2, keepdim=True)
#             IIE2exE = torch.mean(IIE2exE, dim=2, keepdim=True)
#             IIE2IaiE = torch.mean(IIE2IaiE, dim=2, keepdim=True)
            
            ##### Excitatory neuron activity
            exsF, exsF_inlayer, exsF_hx, exsF_x, exsF_g, exsF_gpre, exsF_inlayerpre = self.ex_flxs([IIF2exF], init_exF) ##### 230531
            exF2mnF = self.exF2mnF_connection(exsF)

            exsE, exsE_inlayer, exsE_hx, exsE_x, exsE_g, exsE_gpre, exsE_inlayerpre = self.ex_exts([IIE2exE], init_exE)
            exE2mnE = self.exE2mnE_connection(exsE)       

            ##### Inhibitory neuron activity
            IaisF, IaisF_inlayer, IaisF_hx, IaisF_x, IaisF_g, IaisF_gpre, IaisF_inlayerpre = self.Iai_flxs([IaF2IaiF, IIF2IaiF, torch.unsqueeze(init_IaiE,2)], init_IaiF, self.Iai_connectivity) ##### 230531
            IaisE, IaisE_inlayer, IaisE_hx, IaisE_x, IaisE_g, IaisE_gpre, IaisE_inlayerpre = self.Iai_exts([IaE2IaiE, IIE2IaiE, torch.unsqueeze(init_IaiF,2)], init_IaiE, self.Iai_connectivity)   

            IaiF2IaiEmnE = self.IaiF2IaiEmnE_connection(IaisF)
            IaiF2IaiE, IaiF2mnE = IaiF2IaiEmnE.split([self.Iai_neurons, self.mn_neurons], dim=1)

            IaiE2IaiFmnF = self.IaiE2IaiFmnF_connection(IaisE)
            IaiE2IaiF, IaiE2mnF = IaiE2IaiFmnF.split([self.Iai_neurons, self.mn_neurons], dim=1)
            
            ##### Motor neuron activity
            mnsF, mnsF_inlayer, mnsF_hx, mnsF_x, mnsF_g, mnsF_gpre, mnsF_inlayerpre = self.mn_flxs([IaF2mnF, exF2mnF, IaiE2mnF], init_mnF, self.mn_connectivity)
            mnsE, mnsE_inlayer, mnsE_hx, mnsE_x, mnsE_g, mnsE_gpre, mnsE_inlayerpre = self.mn_exts([IaE2mnE, exE2mnE, IaiF2mnE], init_mnE, self.mn_connectivity)
                        
            ##### 230605
            mnsF_hx = mnsF_hx.unsqueeze(2)
            mnsF_inlayer = mnsF_inlayer.unsqueeze(2)
            mnsE_hx = mnsE_hx.unsqueeze(2)
            mnsE_inlayer = mnsE_inlayer.unsqueeze(2)
            
            exsF_hx = exsF_hx.unsqueeze(2)
            exsF_inlayer = exsF_inlayer.unsqueeze(2)
            exsE_hx = exsE_hx.unsqueeze(2)
            exsE_inlayer = exsE_inlayer.unsqueeze(2)
            
            IaisF_hx = IaisF_hx.unsqueeze(2)
            IaisF_inlayer = IaisF_inlayer.unsqueeze(2)
            IaisE_hx = IaisE_hx.unsqueeze(2)
            IaisE_inlayer = IaisE_inlayer.unsqueeze(2)
            
            mnsF_x = mnsF_x.unsqueeze(3)
            mnsF_g = mnsF_g.unsqueeze(3)
            mnsF_gpre = mnsF_gpre.unsqueeze(3)
            mnsF_inlayerpre = mnsF_inlayerpre.unsqueeze(3)
            mnsE_x = mnsE_x.unsqueeze(3)
            mnsE_g = mnsE_g.unsqueeze(3)
            mnsE_gpre = mnsE_gpre.unsqueeze(3)
            mnsE_inlayerpre = mnsE_inlayerpre.unsqueeze(3)
            
            exsF_x = exsF_x.unsqueeze(3)
            exsF_g = exsF_g.unsqueeze(3)
            exsF_gpre = exsF_gpre.unsqueeze(3)
            exsF_inlayerpre = exsF_inlayerpre.unsqueeze(3)
            exsE_x = exsE_x.unsqueeze(3)
            exsE_g = exsE_g.unsqueeze(3)
            exsE_gpre = exsE_gpre.unsqueeze(3)
            exsE_inlayerpre = exsE_inlayerpre.unsqueeze(3)
            
            IaisF_x = IaisF_x.unsqueeze(3)
            IaisF_g = IaisF_g.unsqueeze(3)
            IaisF_gpre = IaisF_gpre.unsqueeze(3)
            IaisF_inlayerpre = IaisF_inlayerpre.unsqueeze(3)
            IaisE_x = IaisE_x.unsqueeze(3)
            IaisE_g = IaisE_g.unsqueeze(3)
            IaisE_gpre = IaisE_gpre.unsqueeze(3)
            IaisE_inlayerpre = IaisE_inlayerpre.unsqueeze(3)
            
            mns = torch.cat([mnsF, mnsE], dim=1)    # [Bx(2xN)]                

            init_exF = exsF[...,-1]
            init_exE = exsE[...,-1]
            init_IaiF = IaisF[...,-1]
            init_IaiE = IaisE[...,-1]
            init_mnF = mnsF[...,-1]
            init_mnE = mnsE[...,-1]
            
            ##### 230209: Readout          
            emg = self.rdo_mns2emg(mns) # BxCxT
            kine = self.rdo_emg2kine(emg)
            
#             print('Shape: ', mnsF.shape, mnsE.shape, mns.shape, emg.shape, kine.shape)
            
#             if self.rdo_use_norm:
#                 batch_size = emg.size(0)
#                 T = y.size(-1)

#                 emg = emg.view(batch_size, self.groups, -1, T)
#                 emg = self.norm(emg)
#                 emg = emg.view(batch_size, -1, T)

#             if self.rdo_activation is not None:
#                 emg = self.rdo_activation(emg)
                                    
            ##### 230209: Somatosensory feedback
            feedback = self.mns2aff_connection(mns)
            
            fb_Ia_flxs, fb_II_flxs, fb_Ia_exts, fb_II_exts = torch.split(feedback, \
                        [self.Ia_neurons, self.II_neurons, self.Ia_neurons, self.II_neurons], dim=1)                         

            exs = torch.cat([exsF, exsE], dim=1) 
            Iais = torch.cat([IaisF, IaisE], dim=1)                         
            
            if t == 0:
                device = kine.device
                emg_stack = torch.zeros(torch.cat((torch.tensor(emg.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                kine_stack = torch.zeros(torch.cat((torch.tensor(kine.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                exsF_stack = torch.zeros(torch.cat((torch.tensor(exsF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsE_stack = torch.zeros(torch.cat((torch.tensor(exsE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisF_stack = torch.zeros(torch.cat((torch.tensor(IaisF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisE_stack = torch.zeros(torch.cat((torch.tensor(IaisE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsF_stack = torch.zeros(torch.cat((torch.tensor(mnsF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsE_stack = torch.zeros(torch.cat((torch.tensor(mnsE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)         
                
                Ia_flxs_t_stack = torch.zeros(torch.cat((torch.tensor(Ia_flxs_t.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                II_flxs_t_stack = torch.zeros(torch.cat((torch.tensor(II_flxs_t.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                Ia_exts_t_stack = torch.zeros(torch.cat((torch.tensor(Ia_exts_t.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                II_exts_t_stack = torch.zeros(torch.cat((torch.tensor(II_exts_t.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                IaF2mnF_stack = torch.zeros(torch.cat((torch.tensor(IaF2mnF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaF2IaiF_stack = torch.zeros(torch.cat((torch.tensor(IaF2IaiF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IIF2exF_stack = torch.zeros(torch.cat((torch.tensor(IIF2exF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IIF2IaiF_stack = torch.zeros(torch.cat((torch.tensor(IIF2IaiF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaE2mnE_stack = torch.zeros(torch.cat((torch.tensor(IaE2mnE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaE2IaiE_stack = torch.zeros(torch.cat((torch.tensor(IaE2IaiE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IIE2exE_stack = torch.zeros(torch.cat((torch.tensor(IIE2exE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IIE2IaiE_stack = torch.zeros(torch.cat((torch.tensor(IIE2IaiE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                exF2mnF_stack = torch.zeros(torch.cat((torch.tensor(exF2mnF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaiE2mnF_stack = torch.zeros(torch.cat((torch.tensor(IaiE2mnF.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exE2mnE_stack = torch.zeros(torch.cat((torch.tensor(exE2mnE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaiF2mnE_stack = torch.zeros(torch.cat((torch.tensor(IaiF2mnE.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                Ia_flxs_stack = torch.zeros(torch.cat((torch.tensor(Ia_flxs.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                II_flxs_stack = torch.zeros(torch.cat((torch.tensor(II_flxs.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                Ia_exts_stack = torch.zeros(torch.cat((torch.tensor(Ia_exts.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                II_exts_stack = torch.zeros(torch.cat((torch.tensor(II_exts.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                fb_Ia_flxs_stack = torch.zeros(torch.cat((torch.tensor(fb_Ia_flxs.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                fb_II_flxs_stack = torch.zeros(torch.cat((torch.tensor(fb_II_flxs.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                fb_Ia_exts_stack = torch.zeros(torch.cat((torch.tensor(fb_Ia_exts.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                fb_II_exts_stack = torch.zeros(torch.cat((torch.tensor(fb_II_exts.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                ##### Internal activity
                mnsF_inlayer_stack = torch.zeros(torch.cat((torch.tensor(mnsF_inlayer.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)                
                mnsF_hx_stack = torch.zeros(torch.cat((torch.tensor(mnsF_hx.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsE_inlayer_stack = torch.zeros(torch.cat((torch.tensor(mnsE_inlayer.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsE_hx_stack = torch.zeros(torch.cat((torch.tensor(mnsE_hx.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                mnsF_x_stack = torch.zeros(torch.cat((torch.tensor(mnsF_x.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsF_g_stack = torch.zeros(torch.cat((torch.tensor(mnsF_g.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsF_gpre_stack = torch.zeros(torch.cat((torch.tensor(mnsF_gpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsF_inlayerpre_stack = torch.zeros(torch.cat((torch.tensor(mnsF_inlayerpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsE_x_stack = torch.zeros(torch.cat((torch.tensor(mnsE_x.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsE_g_stack = torch.zeros(torch.cat((torch.tensor(mnsE_g.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsE_gpre_stack = torch.zeros(torch.cat((torch.tensor(mnsE_gpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                mnsE_inlayerpre_stack = torch.zeros(torch.cat((torch.tensor(mnsE_inlayerpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                exsF_inlayer_stack = torch.zeros(torch.cat((torch.tensor(exsF_inlayer.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsF_hx_stack = torch.zeros(torch.cat((torch.tensor(exsF_hx.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsE_inlayer_stack = torch.zeros(torch.cat((torch.tensor(exsE_inlayer.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsE_hx_stack = torch.zeros(torch.cat((torch.tensor(exsE_hx.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                exsF_x_stack = torch.zeros(torch.cat((torch.tensor(exsF_x.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsF_g_stack = torch.zeros(torch.cat((torch.tensor(exsF_g.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsF_gpre_stack = torch.zeros(torch.cat((torch.tensor(exsF_gpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsF_inlayerpre_stack = torch.zeros(torch.cat((torch.tensor(exsF_inlayerpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsE_x_stack = torch.zeros(torch.cat((torch.tensor(exsE_x.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsE_g_stack = torch.zeros(torch.cat((torch.tensor(exsE_g.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsE_gpre_stack = torch.zeros(torch.cat((torch.tensor(exsE_gpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                exsE_inlayerpre_stack = torch.zeros(torch.cat((torch.tensor(exsE_inlayerpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                IaisF_inlayer_stack = torch.zeros(torch.cat((torch.tensor(IaisF_inlayer.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisF_hx_stack = torch.zeros(torch.cat((torch.tensor(IaisF_hx.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisE_inlayer_stack = torch.zeros(torch.cat((torch.tensor(IaisE_inlayer.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisE_hx_stack = torch.zeros(torch.cat((torch.tensor(IaisE_hx.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                
                IaisF_x_stack = torch.zeros(torch.cat((torch.tensor(IaisF_x.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisF_g_stack = torch.zeros(torch.cat((torch.tensor(IaisF_g.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisF_gpre_stack = torch.zeros(torch.cat((torch.tensor(IaisF_gpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisF_inlayerpre_stack = torch.zeros(torch.cat((torch.tensor(IaisF_inlayerpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisE_x_stack = torch.zeros(torch.cat((torch.tensor(IaisE_x.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisE_g_stack = torch.zeros(torch.cat((torch.tensor(IaisE_g.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisE_gpre_stack = torch.zeros(torch.cat((torch.tensor(IaisE_gpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)
                IaisE_inlayerpre_stack = torch.zeros(torch.cat((torch.tensor(IaisE_inlayerpre.shape[:-1]), torch.tensor([t_total]))).tolist()).to(device)    
        
            #### Saving for each t
            emg_stack[..., t:t+1] = emg
            kine_stack[..., t:t+1] = kine

            exsF_stack[..., t:t+1] = exsF
            IaisF_stack[..., t:t+1] = IaisF
            mnsF_stack[..., t:t+1] = mnsF

            exsE_stack[..., t:t+1] = exsE
            IaisE_stack[..., t:t+1] = IaisF
            mnsE_stack[..., t:t+1] = mnsE

            Ia_flxs_t_stack[..., t:t+1] = Ia_flxs_t
            II_flxs_t_stack[..., t:t+1] = II_flxs_t
            Ia_exts_t_stack[..., t:t+1] = Ia_exts_t
            II_exts_t_stack[..., t:t+1] = II_exts_t

            IaF2mnF_stack[..., t:t+1] = IaF2mnF
            IaF2IaiF_stack[..., t:t+1] = IaF2IaiF
            IIF2exF_stack[..., t:t+1] = IIF2exF
            IIF2IaiF_stack[..., t:t+1] = IIF2IaiF
            IaE2mnE_stack[..., t:t+1] = IaE2mnE
            IaE2IaiE_stack[..., t:t+1] = IaE2IaiE
            IIE2exE_stack[..., t:t+1] = IIE2exE
            IIE2IaiE_stack[..., t:t+1] = IIE2IaiE

            exF2mnF_stack[..., t:t+1] = exF2mnF
            IaiE2mnF_stack[..., t:t+1] = IaiE2mnF
            exE2mnE_stack[..., t:t+1] = exE2mnE
            IaiF2mnE_stack[..., t:t+1] = IaiF2mnE

            Ia_flxs_stack[..., t:t+1] = Ia_flxs
            II_flxs_stack[..., t:t+1] = II_flxs
            Ia_exts_stack[..., t:t+1] = Ia_exts
            II_exts_stack[..., t:t+1] = II_exts

            fb_Ia_flxs_stack[..., t:t+1] = fb_Ia_flxs
            fb_II_flxs_stack[..., t:t+1] = fb_II_flxs
            fb_Ia_exts_stack[..., t:t+1] = fb_Ia_exts
            fb_II_exts_stack[..., t:t+1] = fb_II_exts

            mnsF_inlayer_stack[..., t:t+1] = mnsF_inlayer
            mnsF_hx_stack[..., t:t+1] = mnsF_hx
            mnsE_inlayer_stack[..., t:t+1] = mnsE_inlayer
            mnsE_hx_stack[..., t:t+1] = mnsE_hx

            mnsF_x_stack[..., t:t+1] = mnsF_x
            mnsF_g_stack[..., t:t+1] = mnsF_g
            mnsF_gpre_stack[..., t:t+1] = mnsF_gpre
            mnsF_inlayerpre_stack[..., t:t+1] = mnsF_inlayerpre
            mnsE_x_stack[..., t:t+1] = mnsE_x
            mnsE_g_stack[..., t:t+1] = mnsE_g
            mnsE_gpre_stack[..., t:t+1] = mnsE_gpre
            mnsE_inlayerpre_stack[..., t:t+1] = mnsE_inlayerpre

            exsF_inlayer_stack[..., t:t+1] = exsF_inlayer
            exsF_hx_stack[..., t:t+1] = exsF_hx
            exsE_inlayer_stack[..., t:t+1] = exsE_inlayer
            exsE_hx_stack[..., t:t+1] = exsE_hx

            exsF_x_stack[..., t:t+1] = exsF_x
            exsF_g_stack[..., t:t+1] = exsF_g
            exsF_gpre_stack[..., t:t+1] = exsF_gpre
            exsF_inlayerpre_stack[..., t:t+1] = exsF_inlayerpre
            exsE_x_stack[..., t:t+1] = exsE_x
            exsE_g_stack[..., t:t+1] = exsE_g
            exsE_gpre_stack[..., t:t+1] = exsE_gpre
            exsE_inlayerpre_stack[..., t:t+1] = exsE_inlayerpre

            IaisF_inlayer_stack[..., t:t+1] = IaisF_inlayer
            IaisF_hx_stack[..., t:t+1] = IaisF_hx
            IaisE_inlayer_stack[..., t:t+1] = IaisE_inlayer
            IaisE_hx_stack[..., t:t+1] = IaisE_hx

            IaisF_x_stack[..., t:t+1] = IaisF_x
            IaisF_g_stack[..., t:t+1] = IaisF_g
            IaisF_gpre_stack[..., t:t+1] = IaisF_gpre
            IaisF_inlayerpre_stack[..., t:t+1] = IaisF_inlayerpre
            IaisE_x_stack[..., t:t+1] = IaisE_x
            IaisE_g_stack[..., t:t+1] = IaisE_g
            IaisE_gpre_stack[..., t:t+1] = IaisE_gpre
            IaisE_inlayerpre_stack[..., t:t+1] = IaisE_inlayerpre
                
        if verbose:
            if self.offset > 0:
                return mns[:,:,self.offset:], Iais[:,:,self.offset:], exs[:,:,self.offset:]
            else:
                return mns, Iais, exs
        else:        
            if self.offset > 0:
                return emg_stack[:,:,self.offset:], mns_stack[:,:,self.offset:], exs_stack[:,:,self.offset:], Iais_stack[:,:,self.offset:]
            else:
                return emg_stack, kine_stack, [afferents_ch, afferents_ch_ees, afferents, exsF_stack, IaisF_stack, mnsF_stack, exsE_stack, IaisE_stack, mnsE_stack, Ia_flxs_t_stack, II_flxs_t_stack, Ia_exts_t_stack, II_exts_t_stack,
                                   IaF2mnF_stack, IaF2IaiF_stack, IIF2exF_stack, IIF2IaiF_stack, IaE2mnE_stack, IaE2IaiE_stack, IIE2exE_stack, IIE2IaiE_stack,
                                   exF2mnF_stack, IaiE2mnF_stack, exE2mnE_stack, IaiF2mnE_stack,
                                   Ia_flxs_stack, II_flxs_stack, Ia_exts_stack, II_exts_stack, fb_Ia_flxs_stack, fb_II_flxs_stack, fb_Ia_exts_stack, fb_II_exts_stack,
                                   mnsF_inlayer_stack, mnsF_hx_stack, mnsE_inlayer_stack, mnsE_hx_stack,
                                   exsF_inlayer_stack, exsF_hx_stack, exsE_inlayer_stack, exsE_hx_stack,
                                   IaisF_inlayer_stack, IaisF_hx_stack, IaisE_inlayer_stack, IaisE_hx_stack,                  
                                   mnsF_x_stack, mnsF_g_stack, mnsF_gpre_stack, mnsF_inlayerpre_stack, mnsE_x_stack, mnsE_g_stack, mnsE_gpre_stack, mnsE_inlayerpre_stack,
                                   exsF_x_stack, exsF_g_stack, exsF_gpre_stack, exsF_inlayerpre_stack, exsE_x_stack, exsE_g_stack, exsE_gpre_stack, exsE_inlayerpre_stack,
                                   IaisF_x_stack, IaisF_g_stack, IaisF_gpre_stack, IaisF_inlayerpre_stack, IaisE_x_stack, IaisE_g_stack, IaisE_gpre_stack, IaisE_inlayerpre_stack]