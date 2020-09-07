import os
import logging
import torch
import warnings
import numpy as np
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from ..graphs.models.U_Net import U_Net
from ..utils import U_Net_utils as utils
from ..datasets.MISData import MISDataset_Loader

class U_Net_Agent:
    def __init__(self, config, pretrained=False):
        '''
        This is the brain of the operation. It specifies model, loss, optimizer ....,
        takes care of data and model loading,
        lets you monitor the training process with a logger and summary writers (->tensorboard),
        and implements the training/validation cycle

        Arguments:  
            config: as specified in utils
            pretrained: boolean
                - True:  load checkpoint   
                - False: basic weight init   
        '''

        self.logger = logging.getLogger("Agent")

        # save config
        self.config = config

        # create model
        self.model = U_Net(config=self.config)

        # dataloader
        self.data_loader = MISDataset_Loader(self.config)

        # pixel-wise L2 Loss 
        self.loss = torch.nn.MSELoss(reduction='none').cuda()
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.config.optimizer.learning_rate, 
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2), 
            eps=self.config.optimizer.eps, weight_decay=self.config.optimizer.weight_decay, 
            amsgrad=self.config.optimizer.amsgrad)

        # learning rate decay scheduler
        if self.config.optimizer.lr_scheduler.want:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.config.optimizer.lr_scheduler.every_n_epochs, 
                gamma=self.config.optimizer.lr_scheduler.gamma)

        # initialize counters; updated in load_checkpoint
        self.current_epoch = 0
        self.current_train_iteration = 0
        self.current_val_iteration = 0
        self.best_val_acc = 0

        # if cuda is available export model to gpu
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.agent.seed)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.agent.seed)
            self.logger.info("Operation will be on *****CPU***** ")
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        # try loading pretrained model
        if pretrained:
            self.load_checkpoint()

        # Tensorboard Writers
        Path(self.config.dir.current_run).mkdir(exist_ok=True, parents=True)
        self.train_summary_writer = SummaryWriter(log_dir=self.config.dir.current_run, comment='U_Net')
        self.val_summary_writer = SummaryWriter(log_dir=self.config.dir.current_run, comment='U_Net')

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=False):
        """
        Saving the latest checkpoint of the training
        
        Argument:
            filename: filename which will contain the state
            is_best: flag is it is the best model
        """

        #aggregate important data
        state = {
            self.config.agent.checkpoint.epoch: self.current_epoch,
            self.config.agent.checkpoint.train_iteration: self.current_train_iteration,
            self.config.agent.checkpoint.val_iteration: self.current_val_iteration,
            self.config.agent.checkpoint.best_val_acc: self.best_val_acc,
            self.config.agent.checkpoint.state_dict: self.model.state_dict(),
            self.config.agent.checkpoint.optimizer: self.optimizer.state_dict()
        }
        
        if is_best:
            filename = self.config.agent.best_checkpoint_name

        # create dir if not exists
        Path(self.config.dir.current_run).mkdir(exist_ok=True, parents=True)

        # Save the state
        torch.save(state, os.path.join(self.config.dir.current_run, filename))
    
    def load_checkpoint(self, filename=None):
        '''
        load checkpoint from file
        should contain following keys: 
            'epoch', 'iteration', 'best_val_iou', 'state_dict', 'optimizer'
            where state_dict is model statedict
            and optimizer is optimizer statesict
        
        Arguments:
            filename: just the name, dir specified in config
        '''

        # use best if not specified
        if filename is None:
            filename = self.config.agent.best_checkpoint_name

        filepath = os.path.join(self.config.dir.current_run, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filepath)

            self.current_epoch = checkpoint[self.config.agent.checkpoint.epoch]
            self.current_train_iteration = checkpoint[
                self.config.agent.checkpoint.train_iteration]
            self.current_val_iteration = checkpoint[
                self.config.agent.checkpoint.val_iteration]
            self.best_val_acc = checkpoint[
                self.config.agent.checkpoint.best_val_acc]
            self.model.load_state_dict(checkpoint[
                self.config.agent.checkpoint.state_dict])
            self.optimizer.load_state_dict(checkpoint[
                self.config.agent.checkpoint.optimizer])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.dir.current_run, checkpoint['epoch'], checkpoint['train_iteration']))
        except OSError:
            warnings.warn("No checkpoint exists from '{}'. Skipping...".format(filepath))
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filepath))
            self.logger.info("**First time to train**")

    def run(self):
        '''
        starts training are testing: specify under config.loader.mode
        can handle keyboard interuptt
        '''

        print('starting ' + self.config.loader.mode + ' at ' + str(datetime.now()))
        try:
            if self.config.loader.mode == 'test':
                with torch.no_grad():
                    self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        '''
        training one epoch at a time
        validating after each epoch
        saving checkpoint after each epoch
        check if val acc is best and store separatelye
        '''

        # add selected loss and optimizer to config  | not added in init as may be changed before training
        self.config.loss = str(self.loss)
        self.config.optimizer.func = str(self.optimizer)

        # make sure to remember the hyper params
        self.add_hparams_summary_writer()
        self.save_hparams_json()

        # Iterate epochs | train one epoch | validate | save checkpoint
        for epoch in range(self.current_epoch, self.config.agent.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            with torch.no_grad():
                avg_val_acc = self.validate()
            is_best = avg_val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = avg_val_acc
            self.save_checkpoint(is_best=is_best)

        self.train_summary_writer.close()
        self.val_summary_writer.close()

    def train_one_epoch(self):
        '''
        One epoch training function
        '''

        # Initialize progress visualization and get batch
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        
        # Set the model to be in training mode
        self.model.train()

        current_batch = 0
        number_of_batches = int(self.data_loader.train_loader.dataset.__len__()/self.config.loader.batch_size)
        epoch_loss = torch.zeros(number_of_batches).to(self.device)
        epoch_acc = torch.zeros(number_of_batches).to(self.device)
        for rgb_image, depth_map, normals_map in tqdm_batch:
            # push to gpu if possible
            if self.cuda:
                rgb_image = rgb_image.cuda(non_blocking=self.config.loader.async_loading)
                depth_map = depth_map.cuda(non_blocking=self.config.loader.async_loading)
                normals_map = normals_map.cuda(non_blocking=self.config.loader.async_loading)

            # forward pass
            prediction = self.model(rgb_image)
            
            # pixel-wise loss
            current_loss = self.loss(prediction, depth_map)
            epoch_loss[current_batch] = torch.sum(current_loss.detach())
            
            # compute class-wise accuracy of current batch
            epoch_acc[current_batch] = utils.compute_accuracy(depth_map.detach(), prediction.detach())

            # backprop
            self.optimizer.zero_grad()
            current_loss.backward(torch.ones_like(current_loss.detach(), device=self.device))                            # , retain_graph=True?
            self.optimizer.step()

            # logging for visualization during training: separate plots for loss, acc, iou | each-classwise + overall
            self.train_summary_writer.add_scalar("Training/Loss", epoch_loss[current_batch], self.current_train_iteration)
            self.train_summary_writer.add_scalar("Training/Accuracy", epoch_acc[current_batch], self.current_train_iteration)

            # counters
            self.current_train_iteration += 1
            current_batch += 1

        tqdm_batch.close()

        # learning rate decay update; after validate; after each epoch
        if self.config.optimizer.lr_scheduler.want:
            self.lr_scheduler.step()

        avg_epoch_loss = torch.mean(epoch_loss, axis=0).tolist()
        avg_epoch_acc = torch.mean(epoch_acc, axis=0).tolist()
        
        self.logger.info("Training at Epoch-" + str(self.current_epoch) + " | " + "Average Loss: " + str(
             avg_epoch_loss) + " | " + 'Average Accuracy: ' + str(avg_epoch_acc))

    def validate(self):
        '''
        One epoch validation

        return: 
            average acc per class
        '''

        # Initialize progress visualization and get batch
        # !self.data_loader.valid_loader works for both valid and test 
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        current_batch = 0
        number_of_batches = int(self.data_loader.valid_loader.dataset.__len__()/self.config.loader.batch_size)
        epoch_loss = torch.zeros(number_of_batches).to(self.device)
        epoch_acc = torch.zeros(number_of_batches).to(self.device)
        for rgb_image, depth_map, normals_map in tqdm_batch:
            # push to gpu if possible
            if self.cuda:
                rgb_image = rgb_image.cuda(non_blocking=self.config.loader.async_loading)
                depth_map = depth_map.cuda(non_blocking=self.config.loader.async_loading)
                normals_map = normals_map.cuda(non_blocking=self.config.loader.async_loading)

            # forward pass
            prediction = self.model(rgb_image)
            
            # pixel-wise loss
            current_loss = self.loss(prediction, depth_map)
            epoch_loss[current_batch] = torch.sum(current_loss.detach())
            
            # compute class-wise accuracy of current batch
            epoch_acc[current_batch] = utils.compute_accuracy(depth_map.detach(), prediction.detach())

            # logging for visualization during training: separate plots for loss, acc, iou | each-classwise + overall
            self.val_summary_writer.add_scalar("Validation/Loss", epoch_loss[current_batch], self.current_val_iteration)
            self.val_summary_writer.add_scalar("Validation/Accuracy", epoch_acc[current_batch], self.current_val_iteration)

            # counters
            self.current_val_iteration += 1
            current_batch += 1

        avg_epoch_loss = torch.mean(epoch_loss, axis=0).tolist()
        avg_epoch_acc = torch.mean(epoch_acc, axis=0).tolist()
        
        self.logger.info("Validation at Epoch-" + str(self.current_epoch) + " | " + "Average Loss: " + str(
             avg_epoch_loss) + " | " + 'Average Accuracy: ' + str(avg_epoch_acc))

        tqdm_batch.close()
        
        return avg_epoch_acc
    
    def add_hparams_summary_writer(self):
        '''
        Add Hyperparamters to tensorboard summary writers using .add_hparams
        Can be accessed under the Hyperparameter tab in Tensorboard
        '''

        hyper_params = {
            'loss': self.config.loss,
            'optimizer': self.config.optimizer.func,
            'learning_rate': self.config.optimizer.learning_rate,
            'beta1': self.config.optimizer.beta1,
            'beta2': self.config.optimizer.beta2,
            'eps': self.config.optimizer.eps,
            'amsgrad': self.config.optimizer.amsgrad,
            'weight_decay': self.config.optimizer.weight_decay,
            'lr_scheduler': self.config.optimizer.lr_scheduler.want,
            'lr_scheduler_every_n_epochs': self.config.optimizer.lr_scheduler.every_n_epochs,
            'lr_scheduler_gamma': self.config.optimizer.lr_scheduler.gamma,
        }
       
        self.train_summary_writer.add_hparams(hyper_params, {})
        self.val_summary_writer.add_hparams(hyper_params, {})

    def save_hparams_json(self):
        '''
        Uses config information to generate a hyperparameter dict and saves it as a json file
        into the current_run directory
        '''

        hparams = {
            'loss': self.config.loss,
            'optimizer': self.config.optimizer.__dict__
        }

        utils.save_json_file(os.path.join(self.config.dir.current_run, 'hyperparams.json'), 
                                hparams , indent=4)

    def finalize(self):
        '''
        Close writers and print time
        '''

        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.train_summary_writer.close()
        self.val_summary_writer.close()
        print('ending ' + self.config.loader.mode + ' at ' + str(datetime.now()))
