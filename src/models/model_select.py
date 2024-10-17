import torch
import torch.nn as nn
import torch.nn.functional as F


class AllModels(nn.Module):
    def __init__(self, gpus, model_name):
        super(AllModels, self).__init__()

        if model_name == 'CNN':
            from .cnn import Network as net
            from .model_config import CnnNetworkConfiguration 
            net_conf = CnnNetworkConfiguration()
        else:
            from .fusion_model import Network as net
        
        self.model_name = model_name
        self.cur_net = net(net_conf)

        if len(gpus)>1:
            self.cur_net = torch.nn.DataParallel(self.cur_net, device_ids=gpus).cuda()
        else:
            self.cur_net=self.cur_net.cuda()
        self.loss_fn = nn.CrossEntropyLoss().cuda()

    @property
    def loss(self):
        return self.loss_mse
 
    def build_loss(self, pred, labels):
        loss_mse = self.loss_fn(pred, labels)  
        return loss_mse
    
    def forward(self, samples, labels):                               
        pred, softmaxed_pred = self.cur_net(samples)                          
        self.loss_mse= self.build_loss(pred, labels)               
        return pred, softmaxed_pred