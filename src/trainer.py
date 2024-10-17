import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import loss_function

import pdb
import os
import matplotlib.pyplot as pyplot
from configs import default_configs as conf
from models.model_select import AllModels
from utils import Timer, logger

best_accuracy = 0.0

class Trainer():
    def __init__(self, dataloader, pwd):
        self.save_path = "/output/"
        if not os.path.exists(self.save_path):
            os.system('mkdir '+self.save_path)
        else:
            os.system('rm -rf ' + self.save_path)
            os.system('mkdir ' + self.save_path)

        self.conf = conf
        self.pwd = pwd

        self.net_name = conf.net
        self.net = AllModels(conf.gpu_id, self.net_name).cuda()
        self.optimizer = optim.Adam(self.net.cur_net.parameters(), lr = conf.learning_rate, weight_decay = 1e-4)
        self.scheduler = StepLR(self.optimizer, step_size = conf.num_epoch_lr_decay, gamma = conf.learning_rate_decay)
        self.train_record = {'best_accuracy': 0.01, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 
        
        self.exp_path = conf.exp_path
        self.exp_name = conf.exp_name

        self.epoch = 0
        self.i_tb = 0

        self.confidence_constraint = loss_function.ConfidenceConstraint()
        # self.smoothness_constraint = loss_function.SmoothnessConstraint(self.net.cur_net, 0.01)

        if conf.is_load:
            self.net.load_state_dict(torch.load(conf.load_model_path))

        self.train_loader, self.val_loader, self.test_loader = dataloader(conf.data_dir, conf.class_num)

        self.writer = logger(self.exp_path, self.exp_name, self.pwd)

    def forward(self):
        for epoch in range(self.epoch, self.conf.max_epoch):
            self.epoch = epoch

            # training
            self.timer['train time'].tic()
            self.train(epoch)
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if epoch % self.conf.val_freq == 0 or epoch > self.conf.val_every_start:
                self.timer['val time'].tic()
                self.validate(epoch)
                self.timer['val time'].toc(average=False)           
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )

            if epoch > self.conf.start_lr_decay:
                self.scheduler.step()
    
    def train(self, epoch):
        self.net.train()
        total_correct = 0
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            emg_gram = data[0]
            emg_signal = data[1]
            emg_user_label = data[2]
            emg_state_label = data[3]

            emg_gram = emg_gram.cuda()
            emg_signal = emg_signal.cuda()
            emg_user_label = emg_user_label.cuda()
            emg_state_label = emg_state_label.cuda()

            self.optimizer.zero_grad()
            preds, softmaxed_preds = self.net(emg_gram, emg_state_label)
            loss = self.net.loss
            loss += self.confidence_constraint(preds, emg_state_label)
            loss.backward()
            self.optimizer.step()
            total_correct += self.get_num_correct(preds, emg_state_label)

            if (i + 1) % self.conf.print_freq == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print( '[epoch %d][iteration %d][loss %.4f][lr %.4f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff) )
        print("epoch:", epoch+1, "train_total_correct:", total_correct, "train_accuracy:", total_correct/len(self.train_loader.dataset))

    def validate(self, epoch):
        global best_accuracy
        self.net.eval()
        total_correct = 0
        val_loss = 0
        if not os.path.exists(self.save_path):
            os.system('mkdir '+self.save_path)
        else:
            os.system('rm -rf ' + self.save_path)
            os.system('mkdir ' + self.save_path)

        for vi, data in enumerate(self.val_loader, 0):
            emg_gram = data[0]
            emg_signal = data[1]
            emg_user_label = data[2]
            emg_state_label = data[3]

            with torch.no_grad():
                emg_gram = emg_gram.cuda()
                emg_signal = emg_signal.cuda()
                emg_user_label = emg_user_label.cuda()
                emg_state_label = emg_state_label.cuda()   

                preds, softmaxed_pred = self.net(emg_gram, emg_state_label)
                # preds = preds.cpu().numpy()
                val_loss += self.net.loss
                total_correct += self.get_num_correct(preds, emg_state_label)
                
        val_loss /= len(self.val_loader.dataset)
        accuracy = total_correct/len(self.val_loader.dataset)

        print("epoch:", epoch+1, "val_total_correct:", total_correct, "val_accuracy:", accuracy)

        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('val_accuracy', accuracy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if not os.path.exists(self.conf.output_dir):
                os.system('mkdir '+self.conf.output_dir)
            else:
                os.system('rm -rf ' + self.conf.output_dir)
                os.system('mkdir ' + self.conf.output_dir)
            model_path = os.path.join(self.conf.output_dir, 'best_model.pth')
            torch.save(self.net.state_dict(), model_path)  # 保存模型参数
            print(f'Saving model with accuracy {accuracy:.2f} at epoch {epoch}\n')


    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()




