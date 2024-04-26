import torch
from copy import deepcopy
class FedAVG():
    def __init__(self, device) -> None:
        self.device = device
        self.f_net = None
        self.density_net = None
        self.radiance_net = None
    def avg(self, share_data_one, share_data_two):
        w_avg = {'f_net':[], 'density_net':[], 'radiance_net':[]}
        w_avg['f_net'] = deepcopy(share_data_one.f_net.state_dict())
        w_avg['density_net'] = deepcopy(share_data_one.density_net.state_dict())
        w_avg['radiance_net'] = deepcopy(share_data_one.radiance_net.state_dict())
        
        for k in w_avg['f_net'].keys():  #average f_net
            w_avg['f_net'][k] += share_data_two.f_net.state_dict()[k]
            w_avg['f_net'][k] = torch.true_divide(w_avg['f_net'][k], 2)
            
        for k in w_avg['density_net'].keys():  #average density_net
            w_avg['density_net'][k] += share_data_two.density_net.state_dict()[k]
            w_avg['density_net'][k] = torch.true_divide(w_avg['density_net'][k], 2)

        for k in w_avg['radiance_net'].keys():  #average radiance_net
            w_avg['radiance_net'][k] += share_data_two.radiance_net.state_dict()[k]
            w_avg['radiance_net'][k] = torch.true_divide(w_avg['radiance_net'][k], 2)
        
        return w_avg
    def federate(self,share_data_one, share_data_two, event_one, event_two):
        while(1):
            if not (event_one.is_set() or event_two.is_set()):
                w_avg = self.avg(share_data_one, share_data_two)
                
                self.f_net = deepcopy(share_data_one.f_net)
                self.density_net = deepcopy(share_data_one.density_net)
                self.radiance_net = deepcopy(share_data_one.radiance_net)
                self.f_net.load_state_dict(w_avg['f_net'])
                self.density_net.load_state_dict(w_avg['density_net'])
                self.radiance_net.load_state_dict(w_avg['radiance_net'])
                                
                share_data_one.f_net = deepcopy(self.f_net) 
                share_data_one.density_net = deepcopy(self.density_net)
                share_data_one.radiance_net = deepcopy(self.radiance_net)
                
                share_data_two.f_net = deepcopy(self.f_net) 
                share_data_two.density_net = deepcopy(self.density_net)
                share_data_two.radiance_net = deepcopy(self.radiance_net)
                
                event_one.set()
                event_two.set()



