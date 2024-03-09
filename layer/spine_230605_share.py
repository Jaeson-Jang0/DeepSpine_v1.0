import torch
from torch import nn
import torch.nn.functional as F

from .utils import LayerNorm, Affine, LinearLN

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

        self.in_pathways = in_pathways
        self.out_neurons = out_neurons
        
        self.affine_layer = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)
        self.int_layer = Integrator(out_neurons, Tmax, activation)


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
 
        hiddens.append(h)
    
        return torch.stack(hiddens[1:], dim=2), in_layer.unsqueeze(2), hx.unsqueeze(2), xx, g, g_pre, in_layer_pre';
    

class SpinalCordCircuit_230605(nn.Module):
    def __init__(self,
            emb_in_channels,
            emb_out_channels,
            emb_kernel_size,
            emb2_in_channels, ##### 230223
            emb2_out_channels,
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
        super(SpinalCordCircuit_230605, self).__init__()
        
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
        
        torch.nn.init.trunc_normal_(self.emg2aff_connection.weight, a=0.0, b=float('inf')) ##### 230531

        ##### 230531: Bias clipping
        torch.nn.init.trunc_normal_(self.IaF2mnFIaiF_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIF2exFIaiF_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exF2mnF_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiF2IaiEmnE_connection.bias, a=-float('inf'), b=0)
        
        torch.nn.init.trunc_normal_(self.IaE2mnEIaiE_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIE2exEIaiE_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exE2mnE_connection.bias, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiE2IaiFmnF_connection.bias, a=-float('inf'), b=0)
        
        torch.nn.init.trunc_normal_(self.emg2aff_connection.bias, a=0.0, b=float('inf')) ##### 230531

        
    def _init_hidden(self, x): 
        batch_size = x.size(0)

        init_ex = x.new(batch_size, self.ex_neurons).zero_()
        init_Iai = x.new(batch_size, self.Iai_neurons).zero_()
        init_mn = x.new(batch_size, self.mn_neurons).zero_()

        return init_ex, init_Iai, init_mn


    def forward(self, kinematics, ees, with_ees, verbose=False): ##### 230307       
        t_bin = 50
        t_int = 25
        t_total = kinematics.shape[2]
        n_bin = int((t_total-(t_bin-t_int))/t_int)
        
        ##### Core
        init_ex, init_Iai, init_mn = self._init_hidden(kinematics)
        init_exF, init_exE = init_ex, init_ex
        init_IaiF, init_IaiE = init_Iai, init_Iai
        init_mnF, init_mnE = init_mn, init_mn
        
        if self.dropout is not None:
            afferents = self.dropout(afferents)
        
        for t in range(n_bin):
            t_start = int(t*t_int)
            kinematics_bin = kinematics[:,:,t_start:t_start+t_bin]
            
            afferents_ch = self.emb_kinematics2ch(kinematics_bin)
            
            if with_ees:
                afferents_ch_ees = afferents_ch + ees[:,:,t_start:t_start+t_bin]
            else:
                afferents_ch_ees = afferents_ch
            
            afferents = self.emb_ch2afferent(afferents_ch_ees)               
              
            # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)

            Ia, II = torch.split(afferents, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)              
            Ia_flxs, Ia_exts = torch.split(Ia, [self.Ia_neurons, self.Ia_neurons], dim=1)
            II_flxs, II_exts = torch.split(II, [self.II_neurons, self.II_neurons], dim=1) 

            batch_size, T = Ia_flxs.shape[0], Ia_flxs.shape[-1]  
            
            if t > 17:                
                kinematics_bin_emg = torch.cat((torch.tile(emg_1, (1,1,25)), torch.tile(emg, (1,1,25))), dim=2) 
                afferents_ch_emg = self.emb_kinematics2ch(kinematics_bin_emg)
                              
                if with_ees:
                    afferents_ch_ees_emg = afferents_ch_emg + ees[:,:,t_start:t_start+t_bin]
                else:
                    afferents_ch_ees_emg = afferents_ch_emg

                afferents_emg = self.emb_ch2afferent(afferents_ch_ees_emg)               

                # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)

                Ia_emg, II_emg = torch.split(afferents_emg, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)              
                Ia_flxs_emg, Ia_exts_emg = torch.split(Ia_emg, [self.Ia_neurons, self.Ia_neurons], dim=1)
                II_flxs_emg, II_exts_emg = torch.split(II_emg, [self.II_neurons, self.II_neurons], dim=1) 
        
        ##### Time series
            if t == 0:            
                Ia_flxs_t = Ia_flxs #[..., t:(t+1)]
                II_flxs_t = II_flxs #[..., t:(t+1)]
                Ia_exts_t = Ia_exts #[..., t:(t+1)]
                II_exts_t = II_exts #[..., t:(t+1)]
            elif t < 19: 
                Ia_flxs_t = Ia_flxs + fb_Ia_flxs # withKineFB
                II_flxs_t = II_flxs + fb_II_flxs
                Ia_exts_t = Ia_exts + fb_Ia_exts
                II_exts_t = II_exts + fb_II_exts
            elif t == 20:      
                Ia_flxs_t = torch.cat((Ia_flxs[:, :, :25], Ia_flxs_emg[:,:,25:]), dim=2) + fb_Ia_flxs
                II_flxs_t = torch.cat((II_flxs[:, :, :25], II_flxs_emg[:,:,25:]), dim=2) + fb_II_flxs
                Ia_exts_t = torch.cat((Ia_exts[:, :, :25], Ia_exts_emg[:,:,25:]), dim=2) + fb_Ia_exts
                II_exts_t = torch.cat((II_exts[:, :, :25], II_exts_emg[:,:,25:]), dim=2) + fb_II_exts 
            else:
                Ia_flxs_t = Ia_flxs_emg + fb_Ia_flxs
                II_flxs_t = II_flxs_emg + fb_II_flxs
                Ia_exts_t = Ia_exts_emg + fb_Ia_exts
                II_exts_t = II_exts_emg + fb_II_exts           
        
            # compute inputs of Ex, Iai, Mn from afferents (Ia & II)
            IaF2mnFIaiF = self.IaF2mnFIaiF_connection(Ia_flxs_t) ##### 230209
            IIF2exFIaiF = self.IIF2exFIaiF_connection(II_flxs_t)
            IaE2mnEIaiE = self.IaE2mnEIaiE_connection(Ia_exts_t)
            IIE2exEIaiE = self.IIE2exEIaiE_connection(II_exts_t)
                        
            IaF2mnF, IaF2IaiF = IaF2mnFIaiF.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIF2exF, IIF2IaiF = IIF2exFIaiF.split([self.ex_neurons, self.Iai_neurons], dim=1)
            IaE2mnE, IaE2IaiE = IaE2mnEIaiE.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIE2exE, IIE2IaiE = IIE2exEIaiE.split([self.ex_neurons, self.Iai_neurons], dim=1)            
           
            IaF2mnF = torch.mean(IaF2mnF, dim=2, keepdim=True)
            IaF2IaiF = torch.mean(IaF2IaiF, dim=2, keepdim=True)
            IIF2exF = torch.mean(IIF2exF, dim=2, keepdim=True)
            IIF2IaiF = torch.mean(IIF2IaiF, dim=2, keepdim=True)
            IaE2mnE = torch.mean(IaE2mnE, dim=2, keepdim=True)
            IaE2IaiE = torch.mean(IaE2IaiE, dim=2, keepdim=True)
            IIE2exE = torch.mean(IIE2exE, dim=2, keepdim=True)
            IIE2IaiE = torch.mean(IIE2IaiE, dim=2, keepdim=True)
            
            # compute excitatory neurons
            exsF, exsF_inlayer, exsF_hx, exsF_x, exsF_g, exsF_gpre, exsF_inlayerpre = self.ex_flxs([IIF2exF], init_exF) ##### 230531
            exF2mnF = self.exF2mnF_connection(exsF)

            exsE, exsE_inlayer, exsE_hx, exsE_x, exsE_g, exsE_gpre, exsE_inlayerpre = self.ex_exts([IIE2exE], init_exE)
            exE2mnE = self.exE2mnE_connection(exsE)       

            # compute inhibitory neurons
            IaisF, IaisF_inlayer, IaisF_hx, IaisF_x, IaisF_g, IaisF_gpre, IaisF_inlayerpre = self.Iai_flxs([IaF2IaiF, IIF2IaiF, torch.unsqueeze(init_IaiE,2)], init_IaiF, self.Iai_connectivity) ##### 230531
            IaisE, IaisE_inlayer, IaisE_hx, IaisE_x, IaisE_g, IaisE_gpre, IaisE_inlayerpre = self.Iai_exts([IaE2IaiE, IIE2IaiE, torch.unsqueeze(init_IaiF,2)], init_IaiE, self.Iai_connectivity)   

            IaiF2IaiEmnE = self.IaiF2IaiEmnE_connection(IaisF)
            IaiF2IaiE, IaiF2mnE = IaiF2IaiEmnE.split([self.Iai_neurons, self.mn_neurons], dim=1)

            IaiE2IaiFmnF = self.IaiE2IaiFmnF_connection(IaisE)
            IaiE2IaiF, IaiE2mnF = IaiE2IaiFmnF.split([self.Iai_neurons, self.mn_neurons], dim=1)
            
            ##### 230531
            mnsF, mnsF_inlayer, mnsF_hx, mnsF_x, mnsF_g, mnsF_gpre, mnsF_inlayerpre = self.mn_flxs([IaF2mnF, exF2mnF, IaiE2mnF], init_mnF, self.mn_connectivity)
            mnsE, mnsE_inlayer, mnsE_hx, mnsE_x, mnsE_g, mnsE_gpre, mnsE_inlayerpre = self.mn_exts([IaE2mnE, exE2mnE, IaiF2mnE], init_mnE, self.mn_connectivity)
                        
            ##### 230605
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
            if t > 2: emg_1 = emg ##### 230309
            
            emg = self.rdo_cnn(mns) # BxCxT
                                    
            ##### 230209: Somatosensory feedback
            feedback = self.emg2aff_connection(emg)
            
            fb_Ia_flxs, fb_II_flxs, fb_Ia_exts, fb_II_exts = torch.split(feedback, \
                        [self.Ia_neurons, self.II_neurons, self.Ia_neurons, self.II_neurons], dim=1)                         

            exs = torch.cat([exsF, exsE], dim=1) 
            Iais = torch.cat([IaisF, IaisE], dim=1)                         
            
            ##### 230523
            if t == 0:
                emg_stack = emg
                
                exsF_stack = exsF
                exsE_stack = exsE
                IaisF_stack = IaisF
                IaisE_stack = IaisE
                mnsF_stack = mnsF
                mnsE_stack = mnsE
                
                ##### 230523
                Ia_flxs_t_stack = Ia_flxs_t
                II_flxs_t_stack = II_flxs_t
                Ia_exts_t_stack = Ia_exts_t
                II_exts_t_stack = II_exts_t
                
                IaF2mnF_stack = IaF2mnF
                IaF2IaiF_stack = IaF2IaiF
                IIF2exF_stack = IIF2exF
                IIF2IaiF_stack = IIF2IaiF
                IaE2mnE_stack = IaE2mnE
                IaE2IaiE_stack = IaE2IaiE
                IIE2exE_stack = IIE2exE
                IIE2IaiE_stack = IIE2IaiE
                
                exF2mnF_stack = exF2mnF
                IaiE2mnF_stack = IaiE2mnF
                exE2mnE_stack = exE2mnE
                IaiF2mnE_stack = IaiF2mnE
                
                ##### 230531
                Ia_flxs_stack = Ia_flxs
                II_flxs_stack = II_flxs
                Ia_exts_stack = Ia_exts
                II_exts_stack = II_exts
                
                fb_Ia_flxs_stack = fb_Ia_flxs
                fb_II_flxs_stack = fb_II_flxs
                fb_Ia_exts_stack = fb_Ia_exts
                fb_II_exts_stack = fb_II_exts
                
                mnsF_inlayer_stack = mnsF_inlayer
                mnsF_hx_stack = mnsF_hx
                mnsE_inlayer_stack = mnsE_inlayer
                mnsE_hx_stack = mnsE_hx
                
                mnsF_x_stack = mnsF_x
                mnsF_g_stack = mnsF_g
                mnsF_gpre_stack = mnsF_gpre
                mnsF_inlayerpre_stack = mnsF_inlayerpre
                mnsE_x_stack = mnsE_x
                mnsE_g_stack = mnsE_g
                mnsE_gpre_stack = mnsE_gpre
                mnsE_inlayerpre_stack = mnsE_inlayerpre
                
                exsF_inlayer_stack = exsF_inlayer
                exsF_hx_stack = exsF_hx
                exsE_inlayer_stack = exsE_inlayer
                exsE_hx_stack = exsE_hx
                
                exsF_x_stack = exsF_x
                exsF_g_stack = exsF_g
                exsF_gpre_stack = exsF_gpre
                exsF_inlayerpre_stack = exsF_inlayerpre
                exsE_x_stack = exsE_x
                exsE_g_stack = exsE_g
                exsE_gpre_stack = exsE_gpre
                exsE_inlayerpre_stack = exsE_inlayerpre
                
                IaisF_inlayer_stack = mnsF_inlayer
                IaisF_hx_stack = mnsF_hx
                IaisE_inlayer_stack = mnsE_inlayer
                IaisE_hx_stack = mnsE_hx
                
                IaisF_x_stack = IaisF_x
                IaisF_g_stack = IaisF_g
                IaisF_gpre_stack = IaisF_gpre
                IaisF_inlayerpre_stack = IaisF_inlayerpre
                IaisE_x_stack = IaisE_x
                IaisE_g_stack = IaisE_g
                IaisE_gpre_stack = IaisE_gpre
                IaisE_inlayerpre_stack = IaisE_inlayerpre
                
            else:
                emg_stack = torch.cat((emg_stack, emg), 2)
                
                exsF_stack = torch.cat((exsF_stack, exsF), 2)
                IaisF_stack = torch.cat((IaisF_stack, IaisF), 2)
                mnsF_stack = torch.cat((mnsF_stack, mnsF), 2)
                
                exsE_stack = torch.cat((exsE_stack, exsE), 2)
                IaisE_stack = torch.cat((IaisE_stack, IaisF), 2)
                mnsE_stack = torch.cat((mnsE_stack, mnsE), 2)
                
                #### 230523
                Ia_flxs_t_stack = torch.cat((Ia_flxs_t_stack, Ia_flxs_t), 2)
                II_flxs_t_stack = torch.cat((II_flxs_t_stack, II_flxs_t), 2)
                Ia_exts_t_stack = torch.cat((Ia_exts_t_stack, Ia_exts_t), 2)
                II_exts_t_stack = torch.cat((II_exts_t_stack, II_exts_t), 2)
                
                IaF2mnF_stack = torch.cat((IaF2mnF_stack, IaF2mnF), 2)
                IaF2IaiF_stack = torch.cat((IaF2IaiF_stack, IaF2IaiF), 2)
                IIF2exF_stack = torch.cat((IIF2exF_stack, IIF2exF), 2)
                IIF2IaiF_stack = torch.cat((IIF2IaiF_stack, IIF2IaiF), 2)
                IaE2mnE_stack = torch.cat((IaE2mnE_stack, IaE2mnE), 2)
                IaE2IaiE_stack = torch.cat((IaE2IaiE_stack, IaE2IaiE), 2)
                IIE2exE_stack = torch.cat((IIE2exE_stack, IIE2exE), 2)
                IIE2IaiE_stack = torch.cat((IIE2IaiE_stack, IIE2IaiE), 2)
                
                exF2mnF_stack = torch.cat((exF2mnF_stack, exF2mnF), 2)
                IaiE2mnF_stack = torch.cat((IaiE2mnF_stack, IaiE2mnF), 2)
                exE2mnE_stack = torch.cat((exE2mnE_stack, exE2mnE), 2)
                IaiF2mnE_stack = torch.cat((IaiF2mnE_stack, IaiF2mnE), 2)
                
                ##### 230531
                Ia_flxs_stack = torch.cat((Ia_flxs_stack, Ia_flxs), 2)
                II_flxs_stack = torch.cat((II_flxs_stack, II_flxs), 2)
                Ia_exts_stack = torch.cat((Ia_exts_stack, Ia_exts), 2)
                II_exts_stack = torch.cat((II_exts_stack, II_exts), 2)
                
                fb_Ia_flxs_stack = torch.cat((fb_Ia_flxs_stack, fb_Ia_flxs), 2)
                fb_II_flxs_stack = torch.cat((fb_II_flxs_stack, fb_II_flxs), 2)
                fb_Ia_exts_stack = torch.cat((fb_Ia_exts_stack, fb_Ia_exts), 2)
                fb_II_exts_stack = torch.cat((fb_II_exts_stack, fb_II_exts), 2)
                
                mnsF_inlayer_stack = torch.cat((mnsF_inlayer_stack, mnsF_inlayer), 2)
                mnsF_hx_stack = torch.cat((mnsF_hx_stack, mnsF_hx), 2)
                mnsE_inlayer_stack = torch.cat((mnsE_inlayer_stack, mnsE_inlayer), 2)
                mnsE_hx_stack = torch.cat((mnsE_hx_stack, mnsE_hx), 2)
                
                ##### 230605
                mnsF_x_stack = torch.cat((mnsF_x_stack, mnsF_x), 3)
                mnsF_g_stack = torch.cat((mnsF_g_stack, mnsF_g), 3)
                mnsF_gpre_stack = torch.cat((mnsF_gpre_stack, mnsF_gpre), 3)
                mnsF_inlayerpre_stack = torch.cat((mnsF_inlayerpre_stack, mnsF_inlayerpre), 3)
                mnsE_x_stack = torch.cat((mnsE_x_stack, mnsE_x), 3)
                mnsE_g_stack = torch.cat((mnsE_g_stack, mnsE_g), 3)
                mnsE_gpre_stack = torch.cat((mnsE_gpre_stack, mnsE_gpre), 3)
                mnsE_inlayerpre_stack = torch.cat((mnsE_inlayerpre_stack, mnsE_inlayerpre), 3)
                
                exsF_inlayer_stack = torch.cat((exsF_inlayer_stack, exsF_inlayer), 2)
                exsF_hx_stack = torch.cat((exsF_hx_stack, exsF_hx), 2)
                exsE_inlayer_stack = torch.cat((exsE_inlayer_stack, exsE_inlayer), 2)
                exsE_hx_stack = torch.cat((exsE_hx_stack, exsE_hx), 2)
                
                exsF_x_stack = torch.cat((exsF_x_stack, exsF_x), 3)
                exsF_g_stack = torch.cat((exsF_g_stack, exsF_g), 3)
                exsF_gpre_stack = torch.cat((exsF_gpre_stack, exsF_gpre), 3)
                exsF_inlayerpre_stack = torch.cat((exsF_inlayerpre_stack, exsF_inlayerpre), 3)
                exsE_x_stack = torch.cat((exsE_x_stack, exsE_x), 3)
                exsE_g_stack = torch.cat((exsE_g_stack, exsE_g), 3)
                exsE_gpre_stack = torch.cat((exsE_gpre_stack, exsE_gpre), 3)
                exsE_inlayerpre_stack = torch.cat((exsE_inlayerpre_stack, exsE_inlayerpre), 3)
                
                IaisF_inlayer_stack = torch.cat((IaisF_inlayer_stack, IaisF_inlayer), 2)
                IaisF_hx_stack = torch.cat((IaisF_hx_stack, IaisF_hx), 2)
                IaisE_inlayer_stack = torch.cat((IaisE_inlayer_stack, IaisE_inlayer), 2)
                IaisE_hx_stack = torch.cat((IaisE_hx_stack, IaisE_hx), 2)
            
                IaisF_x_stack = torch.cat((IaisF_x_stack, IaisF_x), 3)
                IaisF_g_stack = torch.cat((IaisF_g_stack, IaisF_g), 3)
                IaisF_gpre_stack = torch.cat((IaisF_gpre_stack, IaisF_gpre), 3)
                IaisF_inlayerpre_stack = torch.cat((IaisF_inlayerpre_stack, IaisF_inlayerpre), 3)
                IaisE_x_stack = torch.cat((IaisE_x_stack, IaisE_x), 3)
                IaisE_g_stack = torch.cat((IaisE_g_stack, IaisE_g), 3)
                IaisE_gpre_stack = torch.cat((IaisE_gpre_stack, IaisE_gpre), 3)
                IaisE_inlayerpre_stack = torch.cat((IaisE_inlayerpre_stack, IaisE_inlayerpre), 3)
            
            Ia_flxs_t = torch.mean(Ia_flxs_t, dim=2, keepdim=True) ##### 230531: Just moved
            II_flxs_t = torch.mean(II_flxs_t, dim=2, keepdim=True)

            Ia_exts_t = torch.mean(Ia_exts_t, dim=2, keepdim=True)
            II_exts_t = torch.mean(II_exts_t, dim=2, keepdim=True)
              
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
                return emg_stack, [afferents_ch, afferents_ch_ees, afferents, exsF_stack, IaisF_stack, mnsF_stack, exsE_stack, IaisE_stack, mnsE_stack, Ia_flxs_t_stack, II_flxs_t_stack, Ia_exts_t_stack, II_exts_t_stack, IaF2mnF_stack, IaF2IaiF_stack, IIF2exF_stack, IIF2IaiF_stack, IaE2mnE_stack, IaE2IaiE_stack, IIE2exE_stack, IIE2IaiE_stack, exF2mnF_stack, IaiE2mnF_stack, exE2mnE_stack, IaiF2mnE_stack, Ia_flxs_stack, II_flxs_stack, Ia_exts_stack, II_exts_stack, fb_Ia_flxs_stack, fb_II_flxs_stack, fb_Ia_exts_stack, fb_II_exts_stack, mnsF_inlayer_stack, mnsF_hx_stack, mnsE_inlayer_stack, mnsE_hx_stack, mnsF_x_stack, mnsF_g_stack, mnsF_gpre_stack, mnsF_inlayerpre_stack, mnsE_x_stack, mnsE_g_stack, mnsE_gpre_stack, mnsE_inlayerpre_stack,
                                   exsF_inlayer_stack, exsF_hx_stack, exsE_inlayer_stack, exsE_hx_stack, exsF_x_stack, exsF_g_stack, exsF_gpre_stack, exsF_inlayerpre_stack, exsE_x_stack, exsE_g_stack, exsE_gpre_stack, exsE_inlayerpre_stack,
                                   IaisF_inlayer_stack, IaisF_hx_stack, IaisE_inlayer_stack, IaisE_hx_stack, IaisF_x_stack, IaisF_g_stack, IaisF_gpre_stack, IaisF_inlayerpre_stack, IaisE_x_stack, IaisE_g_stack, IaisE_gpre_stack, IaisE_inlayerpre_stack]