# Revised from 'spine_230119': 
# Revised from "spine.py": Restricting weights within non-negative values + Adding somatosensory feedback

import torch
from torch import nn
import torch.nn.functional as F

from .utils import LayerNorm, Affine, LinearLN

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


class SpinalCordCircuit_230202(nn.Module):
    def __init__(self, 
            Ia_neurons, 
            II_neurons, 
            ex_neurons, 
            Iai_neurons, 
            mn_neurons,
            Tmax,
            offset=0,
            activation='relu',
            dropout=None):
        super(SpinalCordCircuit_230202, self).__init__()
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
        
#          print(self.IaF2mnFIaiF_connection.weight)

        torch.nn.init.trunc_normal_(self.IaF2mnFIaiF_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIF2exFIaiF_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exF2mnF_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiF2IaiEmnE_connection.weight, a=-float('inf'), b=0)
        
        torch.nn.init.trunc_normal_(self.IaE2mnEIaiE_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IIE2exEIaiE_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.exE2mnE_connection.weight, a=0.0, b=float('inf'))
        torch.nn.init.trunc_normal_(self.IaiE2IaiFmnF_connection.weight, a=-float('inf'), b=0)

#         print(self.IaF2mnFIaiF_connection.weight)

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

    def _init_hidden(self, x):
        batch_size = x.size(0)

        init_ex = x.new(batch_size, self.ex_neurons).zero_() # v230119
        init_Iai = x.new(batch_size, self.Iai_neurons).zero_()
        init_mn = x.new(batch_size, self.mn_neurons).zero_()

        return init_ex, init_Iai, init_mn

    def forward(self, afferents, verbose=False):
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

        print('Core: ', Ia_flxs.shape)
        
        for t in range(T):
            # compute inputs of Ex, Iai, Mn from afferents (Ia & II)
            IaF2mnFIaiF = self.IaF2mnFIaiF_connection(Ia_flxs[..., t:(t+1)])
            IIF2exFIaiF = self.IIF2exFIaiF_connection(II_flxs[..., t:(t+1)])
            IaE2mnEIaiE = self.IaE2mnEIaiE_connection(Ia_exts[..., t:(t+1)])
            IIE2exEIaiE = self.IIE2exEIaiE_connection(II_exts[..., t:(t+1)])
            
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
            
            if t == 0:
                mns_stack = mns
            else:
                mns_stack = torch.cat((mns_stack, mns), 2)
            
            
        print(mns.shape, mns_stack.shape)
            
        if verbose:
            if self.offset > 0:
                return mns[:,:,self.offset:], Iais[:,:,self.offset:], exs[:,:,self.offset:]
            else:
                return mns, Iais, exs
        else:        
            if self.offset > 0:
                return mns_stack[:,:,self.offset:]
            else:
                return mns_stack #, hiddens