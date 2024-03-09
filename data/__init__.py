from data import datasets_unified
from torch.utils.data import DataLoader
def get_loader(config):
    dataset_unified = getattr(datasets_unified, config['dataset']['type'])(**dict(config['dataset']['args']))
    loader = DataLoader(dataset_unified, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=config['shuffle'])

    # compability
    loaders = [loader]
    return loaders