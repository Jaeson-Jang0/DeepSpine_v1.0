import torch

def get_optimizer(core, core_cfg):
    optimizer = getattr(torch.optim, core_cfg['type'])
    optim_args = dict(core_cfg['args'])
    
    optim_core = optimizer(core.parameters(), **optim_args)
    
    return optim_core


    # optim_embeddings = []
    # optimizer = getattr(torch.optim, config['optimizer']['embedding']['type'])
    # optim_args = dict(config['optimizer']['embedding']['args'])
    # for embedding in embeddings:
    #     optim_embeddings.append(optimizer(embedding.parameters(), **optim_args))

    # optimizer = getattr(torch.optim, config['optimizer']['core']['type'])
    # optim_args = dict(config['optimizer']['core']['args'])
    # optim_core = optimizer(core.parameters(), **optim_args)
    
    # optim_readouts = []
    # optimizer = getattr(torch.optim, config['optimizer']['readout']['type'])
    # optim_args = dict(config['optimizer']['readout']['args'])
    # for readout in readouts:
    #     optim_readouts.append(optimizer(readout.parameters(), **optim_args)) 
