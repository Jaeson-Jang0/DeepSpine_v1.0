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

#         print(in_pathways, out_neurons)
        
        self.affine_layer = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)
#         self.affine_ext = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)
               
        self.int_layer = Integrator(out_neurons, Tmax, activation)
#         self.flexor = Integrator(out_neurons, Tmax, activation)
#         self.extensor = Integrator(out_neurons, Tmax, activation)
        
#         print(self.affine_flx)
#         print(self.affine_ext)
#         print(self.flexor)
#         print(self.extensor)

    def forward(self, xs, init_h, EI=None):
        # B: batch size
        # P: # of input pathways either for flexor and extensor
        # N: # of output neurons
        # T: # of time steps
        
#         print('Asdf')
#         print(init_h, EI)
        
        hiddens = [init_h]
        
#         print("Size check: ", len(xs))
        xs = torch.stack(xs, dim=1)         # [Bx(2xP)xNxT]
#         print(xs.shape)
        xs = self.norm(xs) # apply layer normalization separately per pathway
#         print(xs.shape)
    
        batch_size, T = xs.shape[0], xs.shape[-1]  

        ##### v230119
#         x_flxs, x_exts = xs.chunk(2, dim=1) # x_flxs: [BxPxNxT], x_exts: [BxPxNxT]
        
#         for t in range(T):
        hx = hiddens[-1]        # [Bx(2xN)]
        
#         print("Here :", xs.shape)
        xx = xs[...,-1]
#         print(xx.shape)
#             hx_flx, hx_ext = hx.chunk(2, dim=1)
#             x_flx = x_flxs[...,t]   # [BxPxN]
#             x_ext = x_exts[...,t]   # [BxPxN]

        # compute a dynamic gain control (Eq. (6))
        g = self.affine_layer(hx.unsqueeze(1).repeat(1,self.in_pathways,1))
#             g_flx = self.affine_flx(hx_flx.unsqueeze(1).repeat(1,self.in_pathways,1))
#             g_ext = self.affine_ext(hx_ext.unsqueeze(1).repeat(1,self.in_pathways,1))

        g = torch.sigmoid(g)
#             g_flx = torch.sigmoid(g_flx)
#             g_ext = torch.sigmoid(g_ext)
#         print('GG :', g.shape)
        in_layer = g*xx
#             in_flx = (g_flx * x_flx)
#             in_ext = (g_ext * x_ext)

#             print(self.in_pathways)
#             print(hx_flx.shape, g_flx.shape, in_flx.shape)

#         print('Pre :', in_layer.shape)
        if EI is not None:                
            # EI: [P]
            # EI set to True for excitation and set to False for inhibition pathways
#                 print(EI)
            in_layer = self.activation(in_layer[:,EI].sum(1)) - self.activation(in_layer[:,~EI].sum(1)) # [BxN]
#                 in_flx = self.activation(in_flx[:,EI].sum(1)) - self.activation(in_flx[:,~EI].sum(1)) # [BxN]
#                 in_ext = self.activation(in_ext[:,EI].sum(1)) - self.activation(in_ext[:,~EI].sum(1)) # [BxN]

#                 print('EI: ', in_flx.shape)
        else:
            in_layer = self.activation(in_layer.sum(1))
#                 in_flx = self.activation(in_flx.sum(1)) # [BxN]
#                 in_ext = self.activation(in_ext.sum(1)) # [BxN]

#                 print('None: ', in_flx.shape)

#             if self.reciprocal_inhibition:
#                 # reciprocal inhibition betwen Iai flexors and extensors
#                 gate = torch.sigmoid(self.div_inhibition(hx))   # [BxN]
#                 in_flx = in_flx * gate
#                 in_ext = in_ext * (1 - gate)
        
#         print('In_layer :', in_layer.shape, hx.shape)
        h = self.int_layer(in_layer, hx)
#             h_flx = self.flexor(in_flx, hx_flx)   # [BxN]
#             h_ext = self.extensor(in_ext, hx_ext)   # [BxN]

#             h = torch.cat([h_flx, h_ext], dim=1)    # [Bx(2xN)]        
        hiddens.append(h)

        return torch.stack(hiddens[1:], dim=2)    # exclude initial hidden state


class SpinalCordCircuit_230119(nn.Module):
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
        super(SpinalCordCircuit_230119, self).__init__()
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
                
        self.IaF2mnFIaiF_connection = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=2)                
        self.IIF2exFIaiF_connection = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=2)        
        self.exF2mnF_connection = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=2)
        self.IaiF2IaiEmnE_connection = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=2)
        
        self.IaE2mnEIaiE_connection = conv1d(Ia_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=2)                
        self.IIE2exEIaiE_connection = conv1d(II_neurons, (Iai_neurons + ex_neurons), kernel_size=1, groups=2)        
        self.exE2mnE_connection = conv1d(ex_neurons, mn_neurons, kernel_size=1, groups=2)
        self.IaiE2IaiFmnF_connection = conv1d(Iai_neurons, (Iai_neurons + mn_neurons), kernel_size=1, groups=2)

#         self.ex = Layer(
#             in_pathways=1, 
#             out_neurons=ex_neurons, 
#             Tmax=Tmax,
#             activation=activation
#         )
#         self.Iai = Layer(
#             in_pathways=2, 
#             out_neurons=Iai_neurons, 
#             Tmax=Tmax,
#             activation=activation,
#             reciprocal_inhibition=True
#         )
#         self.mn = Layer(
#             in_pathways=3, 
#             out_neurons=mn_neurons, 
#             Tmax=Tmax,
#             activation=activation
#         )
           
#         self.Ia2mnIai_connection = conv1d(2*Ia_neurons, 2*(Iai_neurons + mn_neurons), kernel_size=1, groups=2)               
#         self.II2exIai_connection = conv1d(2*II_neurons, 2*(Iai_neurons + ex_neurons), kernel_size=1, groups=2)        
#         self.ex2mn_connection = conv1d(2*ex_neurons, 2*mn_neurons, kernel_size=1, groups=2)        
#         self.Iai2mn_connection = conv1d(2*Iai_neurons, 2*mn_neurons, kernel_size=1, groups=2)
        
       
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

#         print(init_ex.shape, init_Iai.shape, init_mn.shape)
        if self.dropout is not None:
            afferents = self.dropout(afferents)
        
        # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)
        
#         print('Input size: ', afferents.shape)
        Ia, II = torch.split(afferents, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)  
        
#         print(Ia.shape, II.shape)
              
        Ia_flxs, Ia_exts = torch.split(Ia, [self.Ia_neurons, self.Ia_neurons], dim=1)
        II_flxs, II_exts = torch.split(II, [self.II_neurons, self.II_neurons], dim=1) 
        
#         print(Ia_flxs.shape, Ia_exts.shape)
#         print(Ia_flxs[..., 0:1].shape)
        batch_size, T = Ia_flxs.shape[0], Ia_flxs.shape[-1]  

        for t in range(T):
#             print('TTTTTT = ', t)
            # compute inputs of Ex, Iai, Mn from afferents (Ia & II)
            IaF2mnFIaiF = self.IaF2mnFIaiF_connection(Ia_flxs[..., t:(t+1)])
            IIF2exFIaiF = self.IIF2exFIaiF_connection(II_flxs[..., t:(t+1)])
            IaE2mnEIaiE = self.IaE2mnEIaiE_connection(Ia_exts[..., t:(t+1)])
            IIE2exEIaiE = self.IIE2exEIaiE_connection(II_exts[..., t:(t+1)])
            
#             IaF2mnFIaiF = self.IaF2mnFIaiF_connection(Ia_flxs)
#             IIF2exFIaiF = self.IIF2exFIaiF_connection(II_flxs)
#             IaE2mnEIaiE = self.IaE2mnEIaiE_connection(Ia_exts)
#             IIE2exEIaiE = self.IIE2exEIaiE_connection(II_exts)

            IaF2mnF, IaF2IaiF = IaF2mnFIaiF.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIF2exF, IIF2IaiF = IIF2exFIaiF.split([self.ex_neurons, self.Iai_neurons], dim=1)
            IaE2mnE, IaE2IaiE = IaE2mnEIaiE.split([self.mn_neurons, self.Iai_neurons], dim=1)
            IIE2exE, IIE2IaiE = IIE2exEIaiE.split([self.ex_neurons, self.Iai_neurons], dim=1)

#             print('init_exF :', init_exF.shape, init_IaiF.shape, init_mnF.shape)
            # compute excitatory neurons
            exsF = self.ex_flxs([IIF2exF], init_exF)
            exF2mnF = self.exF2mnF_connection(exsF)

            exsE = self.ex_exts([IIE2exE], init_exE)
            exE2mnE = self.exE2mnE_connection(exsE)       

            # compute inhibitory neurons
#             IaisF = self.Iai_flxs_pre([IaF2IaiF, IIF2IaiF], init_Iai)       
#             IaiF2IaiEmnE = self.IaiF2IaiEmnE_connection(IaisF)
#             IaiF2IaiE_pre, IaiF2mnE = IaiF2IaiEmnE.split([self.Iai_neurons, self.mn_neurons], dim=1)

#             IaisE = self.Iai_exts_pre([IaE2IaiE, IIE2IaiE], init_Iai)       
#             IaiE2IaiFmnF = self.IaiE2IaiFmnF_connection(IaisE)
#             IaiE2IaiF_pre, IaiE2mnF = IaiE2IaiFmnF.split([self.Iai_neurons, self.mn_neurons], dim=1)

    #         IaiE2IaiF_ = ?
    #         IaiF2IaiE = ?
            
#             print('Shape :', IaF2IaiF.shape, IIF2IaiF.shape, init_IaiE.shape, init_IaiF.shape)
#             if t == 0:
            IaisF = self.Iai_flxs([IaF2IaiF, IIF2IaiF, torch.unsqueeze(init_IaiE,2)], init_IaiF, self.Iai_connectivity)  
            IaisE = self.Iai_exts([IaE2IaiE, IIE2IaiE, torch.unsqueeze(init_IaiF,2)], init_IaiE, self.Iai_connectivity)   
#             else:
#                 IaisF = self.Iai_flxs([IaF2IaiF, IIF2IaiF, init_IaiE], init_IaiF, self.Iai_connectivity)  
#                 IaisE = self.Iai_exts([IaE2IaiE, IIE2IaiE, init_IaiF], init_IaiE, self.Iai_connectivity)                

            IaiF2IaiEmnE = self.IaiF2IaiEmnE_connection(IaisF)
            IaiF2IaiE, IaiF2mnE = IaiF2IaiEmnE.split([self.Iai_neurons, self.mn_neurons], dim=1)

            IaiE2IaiFmnF = self.IaiE2IaiFmnF_connection(IaisE)
            IaiE2IaiF, IaiE2mnF = IaiE2IaiFmnF.split([self.Iai_neurons, self.mn_neurons], dim=1)

            # compute motor neurons
            mnsF = self.mn_flxs([IaF2mnF, exF2mnF, IaiE2mnF], init_mnF, self.mn_connectivity)
            mnsE = self.mn_exts([IaE2mnE, exE2mnE, IaiF2mnE], init_mnE, self.mn_connectivity)

            mns = torch.cat([mnsF, mnsE], dim=1)    # [Bx(2xN)]                
#             print("No problem: ", mnsF.shape, mns.shape)

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
                            
    #         # compute inputs of Ex, Iai, Mn from afferents (Ia & II)
    #         Ia2mnIai = self.Ia2mnIai_connection(Ia)
    #         II2exIai = self.II2exIai_connection(II)        

    #         ## split into flexor and extensor
    #         Ia2mnIai_flxs, Ia2mnIai_exts = Ia2mnIai.chunk(2, dim=1)
    #         II2exIai_flxs, II2exIai_exts = II2exIai.chunk(2, dim=1)

    #         print(Ia2mnIai.shape, Ia2mnIai_flxs.shape)

    #         Ia2mn_flxs, Ia2Iai_flxs = Ia2mnIai_flxs.split([self.mn_neurons, self.Iai_neurons], dim=1)
    #         II2ex_flxs, II2Iai_flxs = II2exIai_flxs.split([self.ex_neurons, self.Iai_neurons], dim=1)
    #         Ia2mn_exts, Ia2Iai_exts = Ia2mnIai_exts.split([self.mn_neurons, self.Iai_neurons], dim=1)
    #         II2ex_exts, II2Iai_exts = II2exIai_exts.split([self.ex_neurons, self.Iai_neurons], dim=1)

    #         # compute excitatory neurons
    #         exs = self.ex(
    #             [II2ex_flxs, II2ex_exts],
    #             init_ex
    #         )
    #         ex2mn = self.ex2mn_connection(exs)
    #         ex2mn_flxs, ex2mn_exts = ex2mn.chunk(2, dim=1)        

    #         # compute inhibitory neurons
    #         Iais = self.Iai(
    #             [Ia2Iai_flxs, II2Iai_flxs, Ia2Iai_exts, II2Iai_exts],
    #             init_Iai
    #         )
    #         Iai2mn = self.Iai2mn_connection(Iais)
    #         Iai2mn_exts, Iai2mn_flxs = Iai2mn.chunk(2, dim=1) # swap the order of flexor and extensor for lateral inhibition

    #         # compute motor neurons
    #         mns = self.mn(
    #             [Ia2mn_flxs, ex2mn_flxs, Iai2mn_flxs, Ia2mn_exts, ex2mn_exts, Iai2mn_exts], 
    #             init_mn,
    #             self.mn_connectivity
    #         )
    
#         print("Output: ", mns_stack.shape)        
        
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