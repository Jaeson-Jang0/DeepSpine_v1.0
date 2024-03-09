# Revised from 'spine_230202': To merge with sensory encoder for somatosensory feedback
# Revised from 'spine_230119': 
# Revised from "spine.py": Restricting weights within non-negative values + Adding somatosensory feedback

import torch
from torch import nn
import torch.nn.functional as F

from .utils import LayerNorm, Affine, LinearLN

##### 230202: From feedforward.py
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
        
        return ees_ortho + F.relu(afferents - ees_anti)
#####


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

#         self.reciprocal_inhibition = reciprocal_inhibition
#         if self.reciprocal_inhibition:
#             self.div_inhibition = LinearLN(2*out_neurons, out_neurons)

        self.in_pathways = in_pathways
        self.out_neurons = out_neurons

        
        self.affine_layer = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)
#         self.affine_ext = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)
               
        self.int_layer = Integrator(out_neurons, Tmax, activation)
#         self.flexor = Integrator(out_neurons, Tmax, activation)
#         self.extensor = Integrator(out_neurons, Tmax, activation)

    def forward(self, xs, init_h, EI=None):
        # B: batch size
        # P: # of input pathways either for flexor and extensor
        # N: # of output neurons
        # T: # of time steps
        
        hiddens = [init_h]
        
        xs = torch.stack(xs, dim=1)         # [Bx(2xP)xNxT]
        xs = self.norm(xs) # apply layer normalization separately per pathway
    
        batch_size, T = xs.shape[0], xs.shape[-1]  

        ##### v230119
        hx = hiddens[-1]        # [Bx(2xN)]        
        xx = xs[...,-1]

        # compute a dynamic gain control (Eq. (6))
        g = self.affine_layer(hx.unsqueeze(1).repeat(1,self.in_pathways,1))

        g = torch.sigmoid(g)
        in_layer = g*xx

        if EI is not None:                
            # EI: [P]
            # EI set to True for excitation and set to False for inhibition pathways
            in_layer = self.activation(in_layer[:,EI].sum(1)) - self.activation(in_layer[:,~EI].sum(1)) # [BxN]
        else:
            in_layer = self.activation(in_layer.sum(1))

        h = self.int_layer(in_layer, hx)
 
        hiddens.append(h)

        return torch.stack(hiddens[1:], dim=2)    # exclude initial hidden state


class SpinalCordCircuit_230209(nn.Module):
    def __init__(self,
            emb_in_channels,
            emb_out_channels,
            emb_kernel_size,            
            Ia_neurons, 
            II_neurons, 
            ex_neurons, 
            Iai_neurons, 
            mn_neurons,
            Tmax,
            rdo_in_channels, rdo_out_channels, rdo_kernel_size, rdo_groups=1, rdo_activation='none',
            offset=0, activation='relu', dropout=None, rdo_use_norm=None,
            emb_groups=1,
            emb_activation='relu',      
            emb_use_norm=None
                ):
        super(SpinalCordCircuit_230209, self).__init__()
        
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

        self.emb_kinematic2afferent = ConvNet1d(emb_in_channels, emb_out_channels, emb_kernel_size, emb_groups, emb_use_norm, emb_activation)
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
        self.register_buffer('mn_connectivity', torch.BoolTensor([True, True, False])) 

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
        self.rdo_use_norm = rdo_use_norm
        self.rdo_groups = rdo_groups

        self.rdo_cnn = CausalConv1d(rdo_in_channels, rdo_out_channels, kernel_size=rdo_kernel_size, groups=rdo_groups, pad=rdo_kernel_size-1)

        if rdo_use_norm:
            self.rdo_norm = LayerNorm(dim=2, affine_shape=[rdo_groups, int(rdo_out_channels/rdo_groups), 1])
        
        ##### 230209: Somatosensory feedback        
        self.emg2aff_connection = conv1d(rdo_out_channels, 2*(Ia_neurons+II_neurons), kernel_size=1, groups=1)    
        
    def _init_hidden(self, x):
        batch_size = x.size(0)

        init_ex = x.new(batch_size, self.ex_neurons).zero_() # v230119
        init_Iai = x.new(batch_size, self.Iai_neurons).zero_()
        init_mn = x.new(batch_size, self.mn_neurons).zero_()

        return init_ex, init_Iai, init_mn

#     def forward(self, afferents, verbose=False):
    def forward(self, x, ees, verbose=False):
        ##### 230209: Variable declaration - Sensory encoder
        ees_amp, _ = ees.max(2)
        ees_spike = ees.clamp_(max=1) ##### Spike train
        
        afferents = self.emb_kinematic2afferent(x) # BxCxT
        ees_amp = ees_amp.repeat(1, afferents.size(1))          # BxC
        ees_recruitment = torch.sigmoid(self.emb_affine(ees_amp)).unsqueeze(2)  # BxCx1
        ees_rate = F.relu(self.emb_spike2rate(ees_spike))
        ees_ortho = ees_recruitment * ees_rate                             # BxCxT
           
        ees_anti = F.relu(self.emb_antidromic(ees_ortho.view(-1,1,ees_ortho.shape[2])))
        
        ees_anti = ees_anti.view(*ees_ortho.shape)
        
        afferents = ees_ortho + F.relu(afferents - ees_anti)        
        
        ##### Core
        init_ex, init_Iai, init_mn = self._init_hidden(afferents)
        init_exF, init_exE = init_ex, init_ex
        init_IaiF, init_IaiE = init_Iai, init_Iai
        init_mnF, init_mnE = init_mn, init_mn

        if self.dropout is not None:
            afferents = self.dropout(afferents)
            
        # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)
        
        Ia, II = torch.split(afferents, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)              
        Ia_flxs, Ia_exts = torch.split(Ia, [self.Ia_neurons, self.Ia_neurons], dim=1)
        II_flxs, II_exts = torch.split(II, [self.II_neurons, self.II_neurons], dim=1) 

        batch_size, T = Ia_flxs.shape[0], Ia_flxs.shape[-1]  
    
        ##### Time series
        for t in range(T):
            ##### 230209: Somatosensory feedback
            if t == 0:            
                Ia_flxs_t = Ia_flxs[..., t:(t+1)]
                II_flxs_t = II_flxs[..., t:(t+1)]
                Ia_exts_t = Ia_exts[..., t:(t+1)]
                II_exts_t = II_exts[..., t:(t+1)]     
            else:
                Ia_flxs_t = Ia_flxs[..., t:(t+1)] + fb_Ia_flxs
                II_flxs_t = II_flxs[..., t:(t+1)] + fb_II_flxs
                Ia_exts_t = Ia_exts[..., t:(t+1)] + fb_Ia_exts
                II_exts_t = II_exts[..., t:(t+1)] + fb_II_exts                              
            
            # compute inputs of Ex, Iai, Mn from afferents (Ia & II)
            IaF2mnFIaiF = self.IaF2mnFIaiF_connection(Ia_flxs_t) ##### 230209
            IIF2exFIaiF = self.IIF2exFIaiF_connection(II_flxs_t)
            IaE2mnEIaiE = self.IaE2mnEIaiE_connection(Ia_exts_t)
            IIE2exEIaiE = self.IIE2exEIaiE_connection(II_exts_t)
                        
            IaF2mnF, IaF2IaiF = IaF2mnFIaiF.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIF2exF, IIF2IaiF = IIF2exFIaiF.split([self.ex_neurons, self.Iai_neurons], dim=1)
            IaE2mnE, IaE2IaiE = IaE2mnEIaiE.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIE2exE, IIE2IaiE = IIE2exEIaiE.split([self.ex_neurons, self.Iai_neurons], dim=1)

            # compute excitatory neurons
            exsF = self.ex_flxs([IIF2exF], init_exF)
            exF2mnF = self.exF2mnF_connection(exsF)

            exsE = self.ex_exts([IIE2exE], init_exE)
            exE2mnE = self.exE2mnE_connection(exsE)       

            # compute inhibitory neurons
            IaisF = self.Iai_flxs([IaF2IaiF, IIF2IaiF, torch.unsqueeze(init_IaiE,2)], init_IaiF, self.Iai_connectivity)  
            IaisE = self.Iai_exts([IaE2IaiE, IIE2IaiE, torch.unsqueeze(init_IaiF,2)], init_IaiE, self.Iai_connectivity)   

            IaiF2IaiEmnE = self.IaiF2IaiEmnE_connection(IaisF)
            IaiF2IaiE, IaiF2mnE = IaiF2IaiEmnE.split([self.Iai_neurons, self.mn_neurons], dim=1)

            IaiE2IaiFmnF = self.IaiE2IaiFmnF_connection(IaisE)
            IaiE2IaiF, IaiE2mnF = IaiE2IaiFmnF.split([self.Iai_neurons, self.mn_neurons], dim=1)

            # compute motor neurons
            mnsF = self.mn_flxs([IaF2mnF, exF2mnF, IaiE2mnF], init_mnF, self.mn_connectivity)
            mnsE = self.mn_exts([IaE2mnE, exE2mnE, IaiF2mnE], init_mnE, self.mn_connectivity)

            mns = torch.cat([mnsF, mnsE], dim=1)    # [Bx(2xN)]                

            init_exF = exsF[...,-1]
            init_exE = exsE[...,-1]
            init_IaiF = IaisF[...,-1]
            init_IaiE = IaisE[...,-1]
            init_mnF = mnsF[...,-1]
            init_mnE = mnsE[...,-1]
            
            ##### 230209: Readout
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
            
            fb_Ia_flxs, fb_II_flxs, fb_Ia_exts, fb_II_exts = torch.split(feedback, \
                        [self.Ia_neurons, self.II_neurons, self.Ia_neurons, self.II_neurons], dim=1)                         

            exs = torch.cat([exsF, exsE], dim=1) 
            Iais = torch.cat([IaisF, IaisE], dim=1) 
            
            if t == 0:
                emg_stack = emg
                
                exsF_stack = exsF
                exsE_stack = exsE
                IaisF_stack = IaisF
                IaisE_stack = IaisE
                mnsF_stack = mnsF
                mnsE_stack = mnsE
                                
            else:
                emg_stack = torch.cat((emg_stack, emg), 2)
                
                exsF_stack = torch.cat((exsF_stack, exsF), 2)
                IaisF_stack = torch.cat((IaisF_stack, IaisF), 2)
                mnsF_stack = torch.cat((mnsF_stack, mnsF), 2)
                
                exsE_stack = torch.cat((exsE_stack, exsE), 2)
                IaisE_stack = torch.cat((IaisE_stack, IaisF), 2)
                mnsE_stack = torch.cat((mnsE_stack, mnsE), 2)
                
        exsF_mean = torch.mean(exsF_stack, dim=1, keepdim=True)
        IaisF_mean = torch.mean(IaisF_stack, dim=1, keepdim=True)
        mnsF_mean = torch.mean(mnsF_stack, dim=1, keepdim=True)
        
        exsE_mean = torch.mean(exsE_stack, dim=1, keepdim=True)
        IaisE_mean = torch.mean(IaisE_stack, dim=1, keepdim=True)
        mnsE_mean = torch.mean(mnsE_stack, dim=1, keepdim=True)
                
        if verbose:
            if self.offset > 0:
                return mns[:,:,self.offset:], Iais[:,:,self.offset:], exs[:,:,self.offset:]
            else:
                return mns, Iais, exs
        else:        
            if self.offset > 0:
                return emg_stack[:,:,self.offset:], mns_stack[:,:,self.offset:], exs_stack[:,:,self.offset:], Iais_stack[:,:,self.offset:]
            else:
                return emg_stack, [exsF_mean, exsE_mean, IaisF_mean, IaisE_mean, mnsF_mean, mnsE_mean]