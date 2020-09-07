import torch
import torch.nn as nn
from collections import deque

from ...utils.U_Net_utils import crop_around_center

class _U_Net_Layer(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, kernel_size, batch_norm, stride, bias):
        '''
        Sequence of batch norm (if wanted), relu, conv

        Arguments:
            exactly what they are called
        '''
        super().__init__()

        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, 
                                        kernel_size=kernel_size, stride=stride, bias=bias))

    def forward(self, input):
        return super().forward(input)

class _U_Net_Block(nn.Sequential):

    def __init__(self, block_config, want_transpconv):
        '''
        One block equals the processing at one size level of U-Net. the output of each encoder block is 
        fed to the corresponding block of the decoder 

        Arguments:
            block_config: dict with following keys: 
                'layers': list of layer depths
                'kernel_size': int/ list of ints specifying kernel size (w.r.t to layer)
                'stride': int/ list of ints specifying stride (w.r.t to layer)
                'batch_norm': boolean/ list of booleans specifying if want batchnorm (w.r.t to layer)
        '''
        super().__init__()

        # for better readability, assign vars from dict
        layers = block_config['layers']
        kernel_size = block_config['kernel_size']
        stride = block_config ['stride']
        batch_norm = block_config['batch_norm']

        # make lists if only single values were given
        if type(kernel_size) is int:
            kernel_size = [kernel_size]*(len(layers)-1)
        if type(stride) is int:
            stride = [stride]*(len(layers)-1)
        if not batch_norm:
            batch_norm = [batch_norm]*(len(layers)-1)

        # create layers according to specifications | skip last if that specifies up conv
        for i in range(len(layers[:-1])):
            layer = _U_Net_Layer(
                num_input_features=layers[i],
                num_output_features=layers[i+1],
                kernel_size=kernel_size[i],
                batch_norm=batch_norm[i],
                stride=stride[i],
                bias=not batch_norm[i]
            )
            self.add_module('layer%d' % (i + 1), layer)
        
        # add transp conv at the end if desired
        if want_transpconv:
            transpconv_layer = nn.ConvTranspose2d(in_channels=layers[-1], out_channels=layers[-1]//2, 
                                            kernel_size=2, stride=2, padding=0, bias=False)
            self.add_module('TransposedConv', transpconv_layer)

    def forward(self, input):
        return super().forward(input)

class U_Net(nn.Module):
    def __init__(self, config):
        '''
        U-Net implementation according to paper

        max pooling not part of graph (extra call in forward) because this is the cleanest version I could think of
        
        Arguments:
            config according to utils.U_Net_utils
        '''

        super().__init__()

        # Encoder
        self.enc = nn.Sequential()
        for i, block_config in enumerate(config.model.encoder):
            self.enc.add_module('block%d' % (i+1), _U_Net_Block(block_config, want_transpconv=False))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle
        self.lowest_level_block = _U_Net_Block(config.model.bottom, want_transpconv=True)

        # Decoder
        self.dec = nn.Sequential()
        for i, block_config in enumerate(config.model.decoder):
            not_is_last = not i == len(config.model.decoder)-1
            self.dec.add_module('block%d' % (i+1), _U_Net_Block(block_config, want_transpconv=not_is_last))

        self.init_weights()

        # get number of parameters of model
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, rgb_images):
        '''
        Feed image through network

        Arguments:
            rgb_image: Expected input size: batch, 3, y, x
        
        return:
            sample-wise 2D Map 
        '''

        features = rgb_images
        encoder_outputs = deque()

        # Encoder
        for block in self.enc:
            features = block(features)
            encoder_outputs.append(features)
            features = self.maxpool(features)

        # Middle
        features = self.lowest_level_block(features)

        # Decoder
        for block in self.dec:
            enc_out_crop = crop_around_center(encoder_outputs[-1].shape, features.shape)
            features = torch.cat((features, encoder_outputs.pop()[enc_out_crop]), 1)
            features = block(features)
        
        return features

    def init_weights(self):
        '''
        Official weight init from torch repo
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)           # TODO check if correct if no batchnorm
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)