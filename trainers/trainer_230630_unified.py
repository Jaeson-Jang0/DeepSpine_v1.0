# Revised from "trainer_230627_unified": To apply to revised architecture
# Revised from "trainer_230531_sling": To apply to unified version
# Revised from "trainer_230406_sling": To add parameters from config file
# Revised from "trainer_230316_sling": Add testing before training + 
# Revised from "trainer_230316": To train for multiple joints + Training without EES/Validating with EES
# Revised from "trainer_230313": To train with non-EES data
# Revised from "trainer_230310": To train before EES until end of trial
# Revised from "trainer_230309": To train before EES
# Revised from "trainer_230307": To use feedback from kinematics prediction
# Revised from "trainer_230307":
# Revised from "trainer_230223": To apply to treadmill data + batch for training data
# Revised from "trainer_230209": To use afferents as input, kinematics as output
# Revised from "trainer_230202": To merge with sensory encoder for somatosensory feedback
# Revised from "synthetic_beta_trainer.py"

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as scistats
from statistics import mean 
import matplotlib.pyplot as plt
import os

class Trainer: 
    def __init__(self, 
            model,
            criterion,
            optimizer,
            data_loader,
            config
        ):
        self.embeddings = model['embeddings']
        self.core = model['core']
        self.readouts = model['readouts']
        
        print("# of core parameters:", sum(p.numel() for p in self.core.parameters() if p.requires_grad))

        # import pdb
        # pdb.set_trace()
        self.optim_embeddings = optimizer['embeddings']
        self.optim_core = optimizer['core']
        self.optim_readouts = optimizer['readouts']

        self.criterion = criterion

        self.train_loaders = data_loader['train']
        self.valid_loaders = data_loader['valid']

        # init batch iterators
        batch_iterators = []
        for train_loader in self.train_loaders:
            batch_iterator = iter(train_loader)
            batch_iterators.append(batch_iterator)
       
        self.batch_iterators = batch_iterators

        self.config = config

        self.device = config['device']['type']
        self.n_sessions = len(self.train_loaders)

        trainer_cfg = config['trainer']['args']
        self.max_iter = trainer_cfg['max_iter']
        self.max_norm = trainer_cfg['max_norm']

        self.saveckpt_period = trainer_cfg['saveckpt_period']
        self.saveact_period = trainer_cfg['saveact_period']
        self.save_dir = trainer_cfg['save_dir']
        self.eval_period = trainer_cfg['eval_period']
        
        if 'is_sliding_window' in trainer_cfg.keys():
            self.is_sliding_window = trainer_cfg['is_sliding_window']
        else:
            self.is_sliding_window = False
        if self.is_sliding_window:
            self.window_size = trainer_cfg['sliding_window_size']
            self.stride = trainer_cfg['stride']

        if 'resume' in trainer_cfg.keys():
            print('resume from %d' % trainer_cfg['resume'])
            state = torch.load(os.path.join(self.save_dir, 'checkpoint_iter{}.pth'.format(trainer_cfg['resume'])))
            self._load_model(state['state_dict'])
            self._load_optim(state['optimizer'])

            self.start_iter = trainer_cfg['resume'] + 1
        elif 'load_model_dir' in trainer_cfg.keys(): # 230316
            print('load model from ' + trainer_cfg['load_model_dir'])
            state = torch.load(trainer_cfg['load_model_dir'])
            self._load_model(state['state_dict'])
            self._load_optim(state['optimizer'])
            
            self.start_iter = 0
        else:
            self.start_iter = 0
            
        self.with_ees_train = trainer_cfg['with_ees_train']
        self.with_ees_valid = trainer_cfg['with_ees_valid']
        
        self.with_weight_clipping = config['with_weight_clipping'] ##### 230531
        self.with_bias_clipping = config['with_bias_clipping']
    
    def _load_model(self, state):
        for embedding, state_dict in zip(self.embeddings, state['embeddings']):
            embedding.load_state_dict(state_dict)

        for readout, state_dict in zip(self.readouts, state['readouts']):
            readout.load_state_dict(state_dict)

        self.core.load_state_dict(state['core'])

    def _load_optim(self, state):
        for optim_embedding, state_dict in zip(self.optim_embeddings, state['embeddings']):
            optim_embedding.load_state_dict(state_dict)

        for optim_readout, state_dict in zip(self.optim_readouts, state['readouts']):
            optim_readout.load_state_dict(state_dict)

        self.optim_core.load_state_dict(state['core'])
            
    def _train_session(self, session, kinematics, meta): # 230227
        embedding = self.embeddings[session]
        readout = self.readouts[session]
        optim_embedding = self.optim_embeddings[session]
        optim_readout = self.optim_readouts[session]
    
        # [batch x dim x time]
        kinematics = kinematics.to(self.device)
        meta = meta.to(self.device)
        
        ##### 230223:                    
        nch = self.config["nch"]
        amp_max = torch.tensor(self.config["amp_max"])
        
        tdur = torch.tensor(self.config["tdur"])
        t0 = torch.tensor(self.config["t0"])
        tbin = torch.tensor(self.config["tbin"])
        eesdur = torch.tensor(self.config["eesdur"])
        
        ch_list = np.array(self.config["ch_list"])
        
        ees = torch.zeros((kinematics.shape[0], len(ch_list), kinematics.shape[2]))
        for idx in range(meta.shape[0]):
            if torch.sum(meta[idx,:]) > 0: ##### 230314
                elec, amp, freq = meta[idx,0], meta[idx,1], meta[idx,2]
                elec = elec.detach().cpu()
                amp = amp.detach().cpu()
                freq = freq.detach().cpu()

                elec_idx = np.argwhere(ch_list==elec.numpy())[0][0]

                ees_template = torch.zeros(kinematics.shape[2])
                ees_loc = torch.arange(int(freq*eesdur/1000))*(1000/freq) + t0
                ees_loc = torch.round(ees_loc)

                ees_template[ees_loc.numpy()] = amp/amp_max  

                ees[idx, elec_idx, :] = ees_template

        ees = ees.to(self.device)            
                     
        predicted_emg, predicted_kinematics, predicted_neural = self.core(kinematics, ees, self.with_ees_train)
        
        loss = self.criterion(predicted_kinematics[:,:,int(t0/2):], kinematics[:,:,int(t0/2):])
        loss.backward()

#         emb_norm = nn.utils.clip_grad_norm_(embedding.parameters(), self.max_norm)
#         readout_norm = nn.utils.clip_grad_norm_(readout.parameters(), self.max_norm)

#         optim_embedding.step()
#         optim_readout.step()

        return loss.item()

    def _train_iteration(self, iteration):
        self.core.train()
        
        list_session = np.arange(self.n_sessions)
        np.random.shuffle(list_session)
        list_loss = []
        
        train_loader = self.train_loaders[0]
        
        batch_id = 0

        for session in list_session:
            for kinematics, meta in train_loader:
                print('Training BATCH: {}'.format(batch_id))
                
                from datetime import datetime
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)
                
                batch_id += 1
                
                self.optim_core.zero_grad()

                loss = self._train_session(session, kinematics, meta)
                list_loss.append(loss)

                core_norm = nn.utils.clip_grad_norm_(self.core.parameters(), self.max_norm)
                self.optim_core.step()

            #### Weight clipping # Recovered at 230531       
            with torch.no_grad():
                if self.with_weight_clipping:
                    self.core.IaF2mnFIaiF_connection.weight = torch.nn.parameter.Parameter(abs(self.core.IaF2mnFIaiF_connection.weight))
                    self.core.IIF2exFIaiF_connection.weight = torch.nn.parameter.Parameter(abs(self.core.IIF2exFIaiF_connection.weight))
                    self.core.exF2mnF_connection.weight = torch.nn.parameter.Parameter(abs(self.core.exF2mnF_connection.weight))
                    self.core.IaiF2IaiEmnE_connection.weight = torch.nn.parameter.Parameter(-abs(self.core.IaiF2IaiEmnE_connection.weight))

                    self.core.IaE2mnEIaiE_connection.weight = torch.nn.parameter.Parameter(abs(self.core.IaE2mnEIaiE_connection.weight))
                    self.core.IIE2exEIaiE_connection.weight = torch.nn.parameter.Parameter(abs(self.core.IIE2exEIaiE_connection.weight))
                    self.core.exE2mnE_connection.weight = torch.nn.parameter.Parameter(abs(self.core.exE2mnE_connection.weight))
                    self.core.IaiE2IaiFmnF_connection.weight = torch.nn.parameter.Parameter(-abs(self.core.IaiE2IaiFmnF_connection.weight))
                    
                    self.core.mns2aff_connection.weight = torch.nn.parameter.Parameter(abs(self.core.mns2aff_connection.weight)) ##### 230630                   
                 
                if self.with_bias_clipping: ##### 230531
                    self.core.IaF2mnFIaiF_connection.bias = torch.nn.parameter.Parameter(abs(self.core.IaF2mnFIaiF_connection.bias))
                    self.core.IIF2exFIaiF_connection.bias = torch.nn.parameter.Parameter(abs(self.core.IIF2exFIaiF_connection.bias))
                    self.core.exF2mnF_connection.bias = torch.nn.parameter.Parameter(abs(self.core.exF2mnF_connection.bias))
                    self.core.IaiF2IaiEmnE_connection.bias = torch.nn.parameter.Parameter(-abs(self.core.IaiF2IaiEmnE_connection.bias))

                    self.core.IaE2mnEIaiE_connection.bias = torch.nn.parameter.Parameter(abs(self.core.IaE2mnEIaiE_connection.bias))
                    self.core.IIE2exEIaiE_connection.bias = torch.nn.parameter.Parameter(abs(self.core.IIE2exEIaiE_connection.bias))
                    self.core.exE2mnE_connection.bias = torch.nn.parameter.Parameter(abs(self.core.exE2mnE_connection.bias))
                    self.core.IaiE2IaiFmnF_connection.bias = torch.nn.parameter.Parameter(-abs(self.core.IaiE2IaiFmnF_connection.bias))
                    
                    self.core.mns2aff_connection.bias = torch.nn.parameter.Parameter(abs(self.core.mns2aff_connection.bias))
            
        return torch.mean(torch.FloatTensor(list_loss))

    def _eval_pearsonr(self, output, target, mask=None):
        output_dim = output.size(1)
        T = output.size(2)

        output = output.detach().cpu()                      # [batch x dim x time]
        target = target.detach().cpu()                      # [batch x dim x time]
        
        # poisson_output = torch.distributions.poisson.Poisson(output.exp())
        # sampled_output = poisson_output.sample()
        
        output = output.view(-1, T)                         # [(batch x dim) x time]
        # output = sampled_output.view(-1, T)                         # [(batch x dim) x time]
        target = target.view(-1, T)                         # [(batch x dim) x time]

        if mask is not None:
            mask = mask.detach().cpu()                          # [batch x 1 x time]
            mask = mask.repeat(1, output_dim, 1).view(-1, T)    # [(batch x dim) x time]

            length = mask.sum(1, keepdims=True)                 # [(batch x dim) x 1]
        else:
            mask = 1
            length = T

        masked_output = mask * output
        masked_target = mask * target
        output_mean = masked_output.sum(1, keepdims=True) / length # [(batch x dim) x 1]
        target_mean = masked_target.sum(1, keepdims=True) / length # [(batch x dim) x 1]

        centered_output = mask * (output - output_mean)
        centered_target = mask * (target - target_mean)
        output_std = torch.sqrt(torch.sum(centered_output ** 2, dim=1))
        target_std = torch.sqrt(torch.sum(centered_target ** 2, dim=1))

        cov = torch.sum(centered_output * centered_target, dim=1)
        r = cov / (output_std * target_std)                         # [(batch x dim)]
        r = r.mean().item()

        return r

    def _valid_session(self, iteration, session, is_save=False, with_pred=False):  
        embedding = self.embeddings[session]
        readout = self.readouts[session]
        valid_loader = self.valid_loaders[session]

        list_loss = []
        if is_save:
            list_emg, list_predicted_emg, list_meta, list_loss = [], [], [], []
            list_kine, list_predicted_kine = [], []
            list_neural = []

        batch_id = 0
        for kinematics, meta in valid_loader: ##### 230216
            print('BATCH: {}'.format(batch_id))
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)            
            
            batch_id += 1

            kinematics = kinematics.to(self.device)
            meta = meta.to(self.device)
           
            nch = self.config["nch"]
            amp_max = torch.tensor(self.config["amp_max"])

            tdur = torch.tensor(self.config["tdur"])
            t0 = torch.tensor(self.config["t0"])
            tbin = torch.tensor(self.config["tbin"])
            eesdur = torch.tensor(self.config["eesdur"])

            ch_list = np.array(self.config["ch_list"])

            ees = torch.zeros((kinematics.shape[0], len(ch_list), kinematics.shape[2])) ##### sling
            for idx in range(meta.shape[0]):
                if torch.sum(meta[idx,:]) > 0: ##### 230314
                    elec, amp, freq = meta[idx,0], meta[idx,1], meta[idx,2]
                    elec = elec.detach().cpu()
                    amp = amp.detach().cpu()
                    freq = freq.detach().cpu()

                    elec_idx = np.argwhere(ch_list==elec.numpy())[0][0]

                    ees_template = torch.zeros(kinematics.shape[2])
                    ees_loc = torch.arange(int(freq*eesdur/1000))*(1000/freq) + t0
                    ees_loc = torch.round(ees_loc)

                    ees_template[ees_loc.numpy()] = amp/amp_max  

                    ees[idx, elec_idx, :] = ees_template

            # [batch x dim x time]
            ees = ees.to(self.device)              
    
            predicted_emg, predicted_kinematics, predicted_neural = self.core(kinematics, ees, self.with_ees_valid) # 230307       
            
            loss = self.criterion(predicted_kinematics[:,:,int(t0/2):], kinematics[:,:,int(t0/2):], reduc='none') ##### 230313
           
            if is_save:
                list_loss.append(loss)
                list_meta.append(meta)
                
#                 list_emg.append(emg)
                list_predicted_emg.append(predicted_emg)
                
                list_kine.append(kinematics)
                list_predicted_kine.append(predicted_kinematics)
                list_neural.append(predicted_neural)

        loss = torch.cat(list_loss, dim=0)

        if is_save and with_pred: # 230209: Saving prediction only for last iteration
            metas = torch.cat(list_meta, dim=0)      

            predicted_emg = torch.cat(list_predicted_emg, dim=0)
            
            kinematics = torch.cat(list_kine, dim=0)
            predicted_kinematics = torch.cat(list_predicted_kine, dim=0)
            
            # eval pearson r score
            # pearsonr = self._eval_pearsonr(predicted_emg, emg)
            
            state = {
                'predicted_emg': predicted_emg,
#                 'emg': emg,
                'predicted_kine': predicted_kinematics,
                'kine': kinematics,
                'loss': loss,
                'meta': metas,
                'neural': list_neural
            }

            save_dir = os.path.join(self.save_dir, str(iteration))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)    
        
            filename = os.path.join(save_dir, 'session_%d.pth' % (session))
            torch.save(state, filename)

#         import ipdb; ipdb.set_trace()
        return loss#, pearsonr

    def _valid_iteration(self, iteration, is_save=False, with_pred=False):
        self.core.eval()
        list_loss, list_pearsonr = [], []

        # fig, ax = plt.subplots(3, self.n_sessions, constrained_layout=False)
        # fig.tight_layout(pad=0) 

        for session in range(self.n_sessions):
            loss = self._valid_session(iteration, session, is_save, with_pred)
            list_loss.append(loss)
            # list_pearsonr.append(pearsonr)

        # plt.subplots_adjust(wspace=0, hspace=0)
        # fig.savefig('raw_emg_%d.png' % (iteration), dpi=600.)
        # plt.close()

        return list_loss#, list_pearsonr
            
    def _save_checkpoint(self, iteration, save_best=False):
        state = {
            'iteration': iteration,
            'state_dict': {
                "embeddings": [embedding.state_dict() for embedding in self.embeddings],
                "core": self.core.state_dict(),
                "readouts": [readout.state_dict() for readout in self.readouts]
            },
            'optimizer': {
                "embeddings": [optim_embedding.state_dict() for optim_embedding in self.optim_embeddings],
                "core": self.optim_core.state_dict(),
                "readouts": [optim_readout.state_dict() for optim_readout in self.optim_readouts]
            },
            # 'monitor_best': self.mnt_best,
            'config': self.config
        }

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)    
        
        filename = os.path.join(self.save_dir, 'checkpoint_iter{}.pth'.format(iteration))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        
        #####
#         w_IaF2mnF = np.squeeze(state['IaF2mnFIaiF_connection.weight'][:nmn, :, 0].detach().cpu().numpy())

        
#         print(

        # if save_best:
        #     best_path = str(self.checkpoint_dir / 'model_best.pth')
        #     torch.save(state, best_path)
        #     self.logger.info("Saving current best: model_best.pth ...")


    def train(self):
        ##### 230406: Testing before training               
        with torch.no_grad():
            ##### 230209: Saving prediction only for last iteration
            iteration = -1
            valid_loss = self._valid_iteration(iteration, is_save=True, with_pred=True)
            ##### 

            log= ''
            for session, v_loss in enumerate(valid_loss):
                log += '[sess: %d][iter: %d]:   | %02.6f \n' % (session, iteration, torch.mean(v_loss)) ##### 230316 ##### 230727                
            log_file = open(os.path.join(self.save_dir, "log.txt"), "a")
            log_file.write(log) 
            log_file.close() 
        #####
        
        train_loss = [[] for _ in range(self.n_sessions)]
        for iteration in range(self.start_iter, self.max_iter):
            
            # import time
            # start_time = time.time()
            iter_loss = self._train_iteration(iteration)
            # print(time.time()-start_time)
            print('iter {}/{}, loss={}'.format(iteration, self.max_iter, iter_loss))

            for session in range(self.n_sessions):
                train_loss[session].append(iter_loss)#[session])

            if (iteration+1) % self.saveckpt_period == 0:
                self._save_checkpoint(iteration)

            if (iteration+1) % self.eval_period == 0:
                
                with torch.no_grad():
                    ##### 230209: Saving prediction only for last iteration
                    if (iteration+1) % self.saveact_period == 0: #== self.max_iter: % 100
                        valid_loss = self._valid_iteration(iteration, is_save=True, with_pred=True)
                    else:
                        valid_loss = self._valid_iteration(iteration, is_save=True, with_pred=False)
                    ##### 
                    
                    log= ''
                    for session, (t_loss, v_loss) in enumerate(zip(train_loss, valid_loss)):
                        log += '[sess: %d][iter: %d]: %02.6f | %02.6f \n' % (session, iteration, iter_loss, torch.mean(v_loss)) ##### 230316 ##### 230727                

                    log_file = open(os.path.join(self.save_dir, "log.txt"), "a")
                    log_file.write(log) 
                    log_file.close() 

                train_loss = [[] for _ in range(self.n_sessions)]


    def test(self, iteration):
        state = torch.load(os.path.join(self.save_dir, 'checkpoint_iter{}.pth'.format(iteration)))
        self._load_model(state['state_dict'])

        with torch.no_grad():
            valid_loss = self._valid_iteration(iteration, is_save=True, with_pred=True) # 230209: Saving prediction only for last iteration

            log= ''
            
            for session, v_loss in enumerate(valid_loss):
                
                log += '[sess: %d][iter: %d]: %02.3f \n' % (session, iteration, v_loss)
        
            log_file = open(os.path.join(self.save_dir, 'test_log_%d.txt' % (iteration)), "a")
            log_file.write(log) 
            log_file.close() 
