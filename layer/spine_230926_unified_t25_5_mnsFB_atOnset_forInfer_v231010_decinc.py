# Revised from "spine_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010": To provide avg. gait cycles as preonset inputs
# Revised from "spine_230926_unified_t25_5_mnsFB_atOnset_forInfer": To provide virtual inputs
# Revised from "spine_230817_unified_t25_5_mnsFB_atOnset": To reduce computation when EMG is unnecessary
# Revised from "spine_230802_unified_t25_5_mnsFB": To use multiple DeepSpine (rather than simple combining in 230807)
# Revised frrm "spine_230727_unified_t25_5_mnsFB": Not mns->EMG->Kine, but mns->EMG and mns->Kine
# Revised from "spine_230718_unified_t25_5": to initialize bias as 0
# Revised from "spine_230703_unified_t25_5": Change kine2aff to mns2aff
# Revised from "spine_230630_unified": To apply to multiple tasks + multiple gpus
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

from .utils import get_activation, get_normalization
from typing import MutableSequence
import numpy as np

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


    def forward(self, xs, init_h, EI=None, with_layernorm=False):
        # B: batch size
        # P: # of input pathways either for flexor and extensor
        # N: # of output neurons
        # T: # of time steps
           
        xs = torch.stack(xs, dim=1)         # [Bx(2xP)xNxT]
        
#         s1 = xs.shape
#         print("Pre: ", xs)
        if with_layernorm:
            xs = self.norm(xs) # apply layer normalization separately per pathway #### Critical
#         print("Post: ", xs)
#         s2 = xs.shape
#         print(s1, s2)
#         print(xs)    
    
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


class SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc(nn.Module):
    def __init__(self,
            muscle_list, kine_list, ds_kine_list, ang_kine_list, with_emg, ##### 230817
                 
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
            rdo_in_channels, rdo_emg_channels, rdo_out_channels, rdo_kernel_size, ini_kine, ini_kine_dir, ##### inc, forInfer
            rdo_groups=1, rdo_activation='none',
            offset=0, activation='relu', dropout=None, rdo_use_norm=None,
            emb_groups=1,
            emb_activation='relu',      
            emb_use_norm=None
                ):
        super(SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc, self).__init__()
        
        self.ini_kine = ini_kine ##### v231010, forInfer
        self.ini_kine_dir = ini_kine_dir ##### inc

        self.muscle_list = muscle_list
        self.kine_list = kine_list
        self.ds_kine_list = ds_kine_list
        self.ang_kine_list = ang_kine_list
        self.ds_kine_keys = [int(a) for a in self.ds_kine_list.keys()]
        
        self.with_emg = with_emg
        
        self.n_ds = len(muscle_list)
        
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
        
        ##### 230817: List for multiple DeepSpines
        self.emb_kinematics2ch_list, self.emb_ch2afferent_list = [], []        
        self.ex_flxs_list, self.Iai_flxs_list, self.mn_flxs_list, self.ex_exts_list, self.Iai_exts_list, self.mn_exts_list = [], [], [], [], [], []        
        self.IaF2mnFIaiF_connection_list, self.IIF2exFIaiF_connection_list, self.exF2mnF_connection_list, self.IaiF2IaiEmnE_connection_list = [], [], [], []        
        self.IaE2mnEIaiE_connection_list, self.IIE2exEIaiE_connection_list, self.exE2mnE_connection_list, self.IaiE2IaiFmnF_connection_list = [], [], [], []               
        self.rdo_mns2emg_list, self.rdo_mns2kine_list, self.mns2aff_connection_list = [], [], []
     
        ##### Core
        self.Ia_neurons = Ia_neurons
        self.II_neurons = II_neurons
        self.ex_neurons = ex_neurons
        self.Iai_neurons = Iai_neurons
        self.mn_neurons = mn_neurons

        self.offset = offset
        conv1d = nn.Conv1d
            
        # excitatory pathways (True): Ia2mn, ex2mn || inhibitory pathways (False): Iai2mn
        self.register_buffer('ex_connectivity', torch.BoolTensor([True])) 
        self.register_buffer('Iai_connectivity', torch.BoolTensor([True, True, False])) 
        self.register_buffer('mn_connectivity', torch.BoolTensor([True, True, True])) ##### 230531: Already negative
#         self.register_buffer('mn_connectivity', torch.BoolTensor([True, True, False])) ##### 230531: Already negative

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

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
           
        for ds_idx in range(self.n_ds):
            emb_kinematics2ch = ConvNet1d(emb_in_channels, emb_out_channels, emb_kernel_size, emb_groups, emb_use_norm, emb_activation)
            emb_ch2afferent = ConvNet1d(emb2_in_channels, emb2_out_channels, emb_kernel_size, emb_groups, emb_use_norm, emb_activation)

            # v230119
            ex_flxs = Layer(in_pathways=1, out_neurons=ex_neurons, Tmax=Tmax, activation=activation)
            Iai_flxs = Layer(in_pathways=3, out_neurons=Iai_neurons, Tmax=Tmax, activation=activation, reciprocal_inhibition=True)
            mn_flxs = Layer(in_pathways=3, out_neurons=mn_neurons, Tmax=Tmax, activation=activation)

            ex_exts = Layer(in_pathways=1, out_neurons=ex_neurons, Tmax=Tmax, activation=activation)
            Iai_exts = Layer(in_pathways=3, out_neurons=Iai_neurons, Tmax=Tmax, activation=activation, reciprocal_inhibition=True)
            mn_exts = Layer(in_pathways=3, out_neurons=mn_neurons, Tmax=Tmax, activation=activation)
            
            IaF2mnFIaiF_connection = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)                 
            IIF2exFIaiF_connection = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=1)        
            exF2mnF_connection = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=1)
            IaiF2IaiEmnE_connection = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)

            IaE2mnEIaiE_connection = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)                 
            IIE2exEIaiE_connection = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=1)        
            exE2mnE_connection = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=1)
            IaiE2IaiFmnF_connection = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=1)
            
            ##### 230817
            rdo_emg_channels_temp = len(self.muscle_list[ds_idx]) ##### 230817                                     
            rdo_mns2emg = CausalConv1d(rdo_in_channels, rdo_emg_channels_temp, kernel_size=rdo_kernel_size, groups=rdo_groups, pad=rdo_kernel_size-1)
            
            if ds_idx in self.ds_kine_keys:
                rdo_in_channels_temp = rdo_in_channels * len(self.ds_kine_list[str(ds_idx)])
                rdo_out_channels_temp = len(self.ang_kine_list[list(self.ds_kine_keys).index(ds_idx)])
                
                rdo_mns2kine = CausalConv1d(rdo_in_channels_temp, rdo_out_channels_temp, kernel_size=rdo_kernel_size, groups=rdo_groups, pad=rdo_kernel_size-1)    
                self.rdo_mns2kine_list.append(rdo_mns2kine)
                                        
            mns2aff_connection = conv1d(2*mn_neurons, 2*(Ia_neurons+II_neurons), kernel_size=1, groups=1) ##### v1; 230630
            
            torch.nn.init.trunc_normal_(IaF2mnFIaiF_connection.weight, a=0.0, b=float('inf'))
            torch.nn.init.trunc_normal_(IIF2exFIaiF_connection.weight, a=0.0, b=float('inf'))
            torch.nn.init.trunc_normal_(exF2mnF_connection.weight, a=0.0, b=float('inf'))
            torch.nn.init.trunc_normal_(IaiF2IaiEmnE_connection.weight, a=-float('inf'), b=0)

            torch.nn.init.trunc_normal_(IaE2mnEIaiE_connection.weight, a=0.0, b=float('inf'))
            torch.nn.init.trunc_normal_(IIE2exEIaiE_connection.weight, a=0.0, b=float('inf'))
            torch.nn.init.trunc_normal_(exE2mnE_connection.weight, a=0.0, b=float('inf'))
            torch.nn.init.trunc_normal_(IaiE2IaiFmnF_connection.weight, a=-float('inf'), b=0)           
            
            torch.nn.init.trunc_normal_(IaF2mnFIaiF_connection.bias, a=0.0, b=0.0)
            torch.nn.init.trunc_normal_(IIF2exFIaiF_connection.bias, a=0.0, b=0.0)
            torch.nn.init.trunc_normal_(exF2mnF_connection.bias, a=0.0, b=0.0) ##### 230817
            torch.nn.init.trunc_normal_(IaiF2IaiEmnE_connection.bias, a=0.0, b=0) ##### 230817

            torch.nn.init.trunc_normal_(IaE2mnEIaiE_connection.bias, a=0.0, b=0.0)
            torch.nn.init.trunc_normal_(IIE2exEIaiE_connection.bias, a=0.0, b=0.0)
            torch.nn.init.trunc_normal_(exE2mnE_connection.bias, a=0.0, b=0.0) ##### 230817
            torch.nn.init.trunc_normal_(IaiE2IaiFmnF_connection.bias, a=0.0, b=0) ##### 230817
            
            ##### 230817: Appending each DeepSpine
            self.emb_kinematics2ch_list.append(emb_kinematics2ch)
            self.emb_ch2afferent_list.append(emb_ch2afferent)
            self.ex_flxs_list.append(ex_flxs)
            self.Iai_flxs_list.append(Iai_flxs)
            self.mn_flxs_list.append(mn_flxs)
            self.ex_exts_list.append(ex_exts)
            self.Iai_exts_list.append(Iai_exts)
            self.mn_exts_list.append(mn_exts)
            self.IaF2mnFIaiF_connection_list.append(IaF2mnFIaiF_connection)
            self.IIF2exFIaiF_connection_list.append(IIF2exFIaiF_connection)
            self.exF2mnF_connection_list.append(exF2mnF_connection)
            self.IaiF2IaiEmnE_connection_list.append(IaiF2IaiEmnE_connection)
            self.IaE2mnEIaiE_connection_list.append(IaE2mnEIaiE_connection)
            self.IIE2exEIaiE_connection_list.append(IIE2exEIaiE_connection)
            self.exE2mnE_connection_list.append(exE2mnE_connection)
            self.IaiE2IaiFmnF_connection_list.append(IaiE2IaiFmnF_connection)
            self.rdo_mns2emg_list.append(rdo_mns2emg)
            self.mns2aff_connection_list.append(mns2aff_connection)    
        
        ##### 230817: List to torch
        self.emb_kinematics2ch_list = nn.ModuleList(self.emb_kinematics2ch_list)
        self.emb_ch2afferent_list = nn.ModuleList(self.emb_ch2afferent_list)
        self.ex_flxs_list = nn.ModuleList(self.ex_flxs_list)
        self.Iai_flxs_list = nn.ModuleList(self.Iai_flxs_list)
        self.mn_flxs_list = nn.ModuleList(self.mn_flxs_list)
        self.ex_exts_list = nn.ModuleList(self.ex_exts_list)
        self.Iai_exts_list = nn.ModuleList(self.Iai_exts_list)
        self.mn_exts_list = nn.ModuleList(self.mn_exts_list)
        self.IaF2mnFIaiF_connection_list = nn.ModuleList(self.IaF2mnFIaiF_connection_list)
        self.IIF2exFIaiF_connection_list = nn.ModuleList(self.IIF2exFIaiF_connection_list)
        self.exF2mnF_connection_list = nn.ModuleList(self.exF2mnF_connection_list)
        self.IaiF2IaiEmnE_connection_list = nn.ModuleList(self.IaiF2IaiEmnE_connection_list)
        self.IaE2mnEIaiE_connection_list = nn.ModuleList(self.IaE2mnEIaiE_connection_list)
        self.IIE2exEIaiE_connection_list = nn.ModuleList(self.IIE2exEIaiE_connection_list)
        self.exE2mnE_connection_list = nn.ModuleList(self.exE2mnE_connection_list)
        self.IaiE2IaiFmnF_connection_list = nn.ModuleList(self.IaiE2IaiFmnF_connection_list)
        self.rdo_mns2emg_list = nn.ModuleList(self.rdo_mns2emg_list)
        self.rdo_mns2kine_list = nn.ModuleList(self.rdo_mns2kine_list)
        self.mns2aff_connection_list = nn.ModuleList(self.mns2aff_connection_list)  
           
    def _init_hidden(self, x): ##### 230307
        batch_size = x.size(0)

        init_ex = x.new(batch_size, self.ex_neurons).zero_() # v230119
        init_Iai = x.new(batch_size, self.Iai_neurons).zero_()
        init_mn = x.new(batch_size, self.mn_neurons).zero_()
        
        return init_ex, init_Iai, init_mn

    def forward(self, theta): ##### forInfer ##### 230703 ##### 230307
        t_total, t0 = 800, 500
#         t_total, t0 = 1500, 500
#         t_total, t0 = 1000, 500

        t_len = t_total       
        self.t0 = 500
        self.eesdur = 300
        self.amp_max = 1500
    
        ##### v231010, forInfer
        import json
        file_path = self.ini_kine_dir ##### inc
        with open(file_path, 'r') as json_file:
            virtual_kine = json.load(json_file)
        
        kinematics = torch.tensor(np.expand_dims(np.array(virtual_kine[self.ini_kine]), axis=0)).to(torch.float32) ##### v231010
        
        with_ees, with_layernorm = True, True
        
        ees = torch.zeros((1, 13, t_len))
        if torch.sum(theta) != 0:
            #                             elec, amp, freq = meta[idx,0], meta[idx,1], meta[idx,2]
            if len(theta.shape)==2:
                freq, amp = theta[0,0], theta[0,1]
                elec_idx = torch.argmax(theta[0,2:])                
#                 print(theta.shape, theta, freq, amp, elec_idx)
            else:
                theta = torch.unsqueeze(theta, 0)
                freq, amp = theta[0,0], theta[0,1]
                elec_idx = torch.argmax(theta[0,2:])

#                 if len(theta.shape)==1: theta = torch.unsqueeze(theta, 0)
#                 freq, amp = theta[:,0], theta[:,1]
#                 elec_idx = torch.argmax(theta[:,2:], dim=1, keepdim=True)
#                 print(theta.shape, freq, amp, elec_idx)

            ##### Inverse discretization
            freq = freq/0.09*10
            amp = amp/0.9*1500

            elec_idx = elec_idx.detach().cpu()
            amp = amp.detach().cpu()
            freq = freq.detach().cpu()

            ees_template = torch.zeros(t_len)
            ees_loc = torch.arange(int(freq*self.eesdur/1000))*(1000/freq) + self.t0
            ees_loc = torch.round(ees_loc)

            ees_template[ees_loc.numpy()] = amp/self.amp_max  

            ees[:, elec_idx, :] = ees_template

#         ees = ees.to(self.device)

        ############################################################################
    
        ##### Temporal merging
        t_bin = 50
        t_int = 25
        n_bin = int((t_total-(t_bin-t_int))/t_int)
        kine_crop = torch.zeros(1, int(np.array(virtual_kine[self.ini_kine]).shape[0]), n_bin) ##### v231010

        ##### Core
        init_ex, init_Iai, init_mn = self._init_hidden(kinematics)
        init_exF, init_exE = init_ex, init_ex
        init_IaiF, init_IaiE = init_Iai, init_Iai
        init_mnF, init_mnE = init_mn, init_mn
        
        if self.dropout is not None:
            afferents = self.dropout(afferents)
                
        for t in range(n_bin): ##### t25
            t_start = int(t*t_int)
            
            kine_crop[:,:,t] = torch.mean(kinematics[:,:,t_start:t_start+t_bin], dim=2) ##### v231010, t25
            
            for ds_idx in range(self.n_ds):
                kine_idx = np.array(self.kine_list[ds_idx])
                kinematics_bin = kinematics[:,kine_idx,t_start:t_start+t_bin] ##### t25
                
                afferents_ch = self.emb_kinematics2ch_list[ds_idx](kinematics_bin)

                if with_ees:
                    afferents_ch_ees = afferents_ch + ees[:,:,t_start:t_start+t_bin]
                else:
                    afferents_ch_ees = afferents_ch ##### 230713

                afferents = self.emb_ch2afferent_list[ds_idx](afferents_ch_ees)               

                # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)
                Ia, II = torch.split(afferents, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)              
                Ia_flxs, Ia_exts = torch.split(Ia, [self.Ia_neurons, self.Ia_neurons], dim=1)
                II_flxs, II_exts = torch.split(II, [self.II_neurons, self.II_neurons], dim=1) 

                batch_size, T = Ia_flxs.shape[0], Ia_flxs.shape[-1]  

                ##### 230309
                if t > int(t_bin/t_int) - 1: ##### 230817, 230718, t25
#                     kinematics_bin_fb = torch.cat((torch.tile(kine_stack[:,kine_idx,t-2:t-1], (1,1,t_int)), torch.tile(kine_stack[:,kine_idx,t-1:t], (1,1,t_int))), dim=2) ##### 230817, 230718
                    kinematics_bin_fb = torch.cat((kine_stack[:,kine_idx,t-2:t-1].repeat(1,1,t_int), kine_stack[:,kine_idx,t-1:t].repeat(1,1,t_int)), dim=2) ##### 230817: for Inference
                    
                    afferents_ch_kine = self.emb_kinematics2ch_list[ds_idx](kinematics_bin_fb) ##### t25

                    if with_ees:
                        afferents_ch_ees_kine = afferents_ch_kine + ees[:,:,t_start:t_start+t_bin]
                        afferents_kine = self.emb_ch2afferent_list[ds_idx](afferents_ch_ees_kine) ##### t25
                    else:
                        afferents_kine = self.emb_ch2afferent_list[ds_idx](afferents_ch_kine)               

                    # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)
                    Ia_kine, II_kine = torch.split(afferents_kine, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)              
                    Ia_flxs_kine, Ia_exts_kine = torch.split(Ia_kine, [self.Ia_neurons, self.Ia_neurons], dim=1)
                    II_flxs_kine, II_exts_kine = torch.split(II_kine, [self.II_neurons, self.II_neurons], dim=1) 

                ##### Time series
                # fb_Ia_flxs: FB from motor neurons, Ia_flxs_kine: Neural afferents from predicted kinematics
                if t == 0: ##### mnsFB, 230718            
                    Ia_flxs_t = (Ia_flxs) # copy.deepcopy
                    II_flxs_t = (II_flxs)
                    Ia_exts_t = (Ia_exts)
                    II_exts_t = (II_exts)
                elif t < int(t0/t_int): ##### 230817; 230718
                    Ia_flxs_t = Ia_flxs + fb_Ia_flxs ##### v5 # withKineFB
                    II_flxs_t = II_flxs + fb_II_flxs
                    Ia_exts_t = Ia_exts + fb_Ia_exts
                    II_exts_t = II_exts + fb_II_exts
                else:
                    Ia_flxs_t = Ia_flxs_kine + fb_Ia_flxs
                    II_flxs_t = II_flxs_kine + fb_II_flxs
                    Ia_exts_t = Ia_exts_kine + fb_Ia_exts
                    II_exts_t = II_exts_kine + fb_II_exts           

                ##### Ia & II neural afferents to neurons
                IaF2mnFIaiF = self.IaF2mnFIaiF_connection_list[ds_idx](Ia_flxs_t)
                IIF2exFIaiF = self.IIF2exFIaiF_connection_list[ds_idx](II_flxs_t)
                IaE2mnEIaiE = self.IaE2mnEIaiE_connection_list[ds_idx](Ia_exts_t)
                IIE2exEIaiE = self.IIE2exEIaiE_connection_list[ds_idx](II_exts_t)            

                IaF2mnF, IaF2IaiF = IaF2mnFIaiF.split([self.mn_neurons, self.Iai_neurons], dim=1)
                IIF2exF, IIF2IaiF = IIF2exFIaiF.split([self.ex_neurons, self.Iai_neurons], dim=1)
                IaE2mnE, IaE2IaiE = IaE2mnEIaiE.split([self.mn_neurons, self.Iai_neurons], dim=1)
                IIE2exE, IIE2IaiE = IIE2exEIaiE.split([self.ex_neurons, self.Iai_neurons], dim=1)

                IaF2mnF = torch.mean(IaF2mnF, dim=2, keepdim=True) ##### t25; 230307
                IaF2IaiF = torch.mean(IaF2IaiF, dim=2, keepdim=True)
                IIF2exF = torch.mean(IIF2exF, dim=2, keepdim=True)
                IIF2IaiF = torch.mean(IIF2IaiF, dim=2, keepdim=True)
                IaE2mnE = torch.mean(IaE2mnE, dim=2, keepdim=True)
                IaE2IaiE = torch.mean(IaE2IaiE, dim=2, keepdim=True)
                IIE2exE = torch.mean(IIE2exE, dim=2, keepdim=True)
                IIE2IaiE = torch.mean(IIE2IaiE, dim=2, keepdim=True)

                ##### Excitatory neuron activity
                exsF, exsF_inlayer, exsF_hx, exsF_x, exsF_g, exsF_gpre, exsF_inlayerpre = self.ex_flxs_list[ds_idx]([IIF2exF], init_exF, self.ex_connectivity, with_layernorm) ##### 230531
                exF2mnF = self.exF2mnF_connection_list[ds_idx](exsF)

                exsE, exsE_inlayer, exsE_hx, exsE_x, exsE_g, exsE_gpre, exsE_inlayerpre = self.ex_exts_list[ds_idx]([IIE2exE], init_exE, self.ex_connectivity, with_layernorm)
                exE2mnE = self.exE2mnE_connection_list[ds_idx](exsE)       

                ##### Inhibitory neuron activity
                IaisF, IaisF_inlayer, IaisF_hx, IaisF_x, IaisF_g, IaisF_gpre, IaisF_inlayerpre = self.Iai_flxs_list[ds_idx]([IaF2IaiF, IIF2IaiF, torch.unsqueeze(init_IaiE,2)], init_IaiF, self.Iai_connectivity, with_layernorm) ##### 230531
                IaisE, IaisE_inlayer, IaisE_hx, IaisE_x, IaisE_g, IaisE_gpre, IaisE_inlayerpre = self.Iai_exts_list[ds_idx]([IaE2IaiE, IIE2IaiE, torch.unsqueeze(init_IaiF,2)], init_IaiE, self.Iai_connectivity, with_layernorm)   

                IaiF2IaiEmnE = self.IaiF2IaiEmnE_connection_list[ds_idx](IaisF)
                IaiF2IaiE, IaiF2mnE = IaiF2IaiEmnE.split([self.Iai_neurons, self.mn_neurons], dim=1)

                IaiE2IaiFmnF = self.IaiE2IaiFmnF_connection_list[ds_idx](IaisE)
                IaiE2IaiF, IaiE2mnF = IaiE2IaiFmnF.split([self.Iai_neurons, self.mn_neurons], dim=1)

                ##### Motor neuron activity
                mnsF, mnsF_inlayer, mnsF_hx, mnsF_x, mnsF_g, mnsF_gpre, mnsF_inlayerpre = self.mn_flxs_list[ds_idx]([IaF2mnF, exF2mnF, IaiE2mnF], init_mnF, self.mn_connectivity, with_layernorm)
                mnsE, mnsE_inlayer, mnsE_hx, mnsE_x, mnsE_g, mnsE_gpre, mnsE_inlayerpre = self.mn_exts_list[ds_idx]([IaE2mnE, exE2mnE, IaiF2mnE], init_mnE, self.mn_connectivity, with_layernorm)

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
                if t>2: kine_1 = kine ##### t25

                if self.with_emg: emg = self.rdo_mns2emg_list[ds_idx](mns) # BxCxT ##### 230926
#                 kine = self.rdo_mns2kine_list[ds_idx](mns)   

                ##### 230209: Somatosensory feedback
                feedback = self.mns2aff_connection_list[ds_idx](mns) ##### v2

                fb_Ia_flxs, fb_II_flxs, fb_Ia_exts, fb_II_exts = torch.split(feedback, \
                            [self.Ia_neurons, self.II_neurons, self.Ia_neurons, self.II_neurons], dim=1)                         
                exs = torch.cat([exsF, exsE], dim=1) 
                Iais = torch.cat([IaisF, IaisE], dim=1)                         

                ##### 230817: Combining results across DeepSpines               
                if ds_idx == 0:
                    mns_t = mns
                    if self.with_emg: emg_t = emg
                else: 
                    mns_t = torch.cat([mns_t, mns], dim=1)
                    if self.with_emg: emg_t = torch.cat([emg_t, emg], dim=1)

                if ds_idx in np.array(self.ds_kine_keys):                    
                    mns_temp = mns_t[:, -mns.shape[1]*len(self.ds_kine_list[str(ds_idx)]):, :]                    
                    
                    kine = self.rdo_mns2kine_list[list(self.ds_kine_keys).index(ds_idx)](mns_temp)
                    
                    if ds_idx == self.ds_kine_keys[0]:
                        kine_t = kine
                    else:
                        kine_t = torch.cat([kine_t, kine], dim=1)           
                
                if ds_idx == self.n_ds-1:
                    if t == 0:
                        device = kine_t.device
                        if self.with_emg: emg_stack = torch.zeros(torch.cat((torch.tensor(emg_t.shape[:-1]), torch.tensor([n_bin]))).tolist()).to(device)
                        kine_stack = torch.zeros(torch.cat((torch.tensor(kine_t.shape[:-1]), torch.tensor([n_bin]))).tolist()).to(device)

                    #### Saving for each t
                    if self.with_emg: emg_stack[..., t:t+1] = emg_t
                    kine_stack[..., t:t+1] = kine_t
        
        if not self.with_emg: emg_stack = torch.zeros(1).to(device)
             
        #####
#         pred = torch.squeeze(kine_stack)
#         y_pred = kine_stack[:,0,22] - kine_stack[:,0,18]
        y_pred = kine_stack[:,:,22] - kine_stack[:,:,18] ##### v231010   
        y_base = kine_crop[:,:,24] - kine_crop[:,:,20]
        
        y_cal = y_pred - y_base       

#         print(pred.shape, kine_stack.shape)
#         y_pred = pred[:,22] - pred[:,18]
#         y_pred = torch.unsqueeze(y_pred[0],0)
        #####
        return y_cal