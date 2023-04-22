from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json, math
import numpy as np

import os, sys
from six.moves import cPickle

from sys import path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'captioning/')
# print('relative captioning is called')

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *

import argparse
import captioning.utils.misc as utils
import torch

import skimage.io
from torch.autograd import Variable
from torchvision import transforms as trn

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from captioning.utils.resnet_utils import myResnet
from captioning.utils.resnet_utils import ResNetBatch
import captioning.utils.resnet as resnet

import wget
import tempfile

class object:
    def __init__(self):
        self.input_fc_dir = ''
        self.input_json = ''
        self.batch_size = ''
        self.id = ''
        self.sample_max = 1
        self.cnn_model = 'resnet101'
        self.model = ''
        self.language_eval = 0
        self.beam_size = 1
        self.temperature = 1.0
        return


class Captioner():

    def __init__(self, is_relative=True, model_path=None, image_feat_params=None, data_type=None, load_resnet=True, diff_feat=None):
        opt = object()

        if image_feat_params==None:
            image_feat_params = {}
            image_feat_params['model'] = 'resnet101'
            image_feat_params['model_root'] = ''
            image_feat_params['att_size'] = 7

        # inputs specific to shoe dataset
        infos_path = os.path.join(model_path, 'infos_best.pkl')
        model_path = os.path.join(model_path, 'model_best.pth')

        opt.infos_path = infos_path
        opt.model_path = model_path
        opt.beam_size = 1
        opt.load_resnet = load_resnet

        # load pre-trained model, adjusting if URL
        if opt.infos_path.startswith("http:") or opt.infos_path.startswith("https:"):
            # create a folder to store the checkpoints for downloading
            if not os.path.exists('./checkpoints_usersim'):
                os.mkdir('./checkpoints_usersim')

            checkpoint_path = os.path.join('./checkpoints_usersim', data_type)
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

            # set the location for infos
            infos_loc = os.path.join(checkpoint_path, 'infos_best.pkl')

            if not os.path.exists(infos_loc):
                try:
                    wget.download(opt.infos_path, infos_loc)
                except Exception as err:
                    print(f"[{err}]")
        else:
            infos_loc = infos_path

        if opt.model_path.startswith("http:") or opt.model_path.startswith("https:"):
            # create a folder to store the checkpoints for downloading
            if not os.path.exists('./checkpoints_usersim'):
                os.mkdir('./checkpoints_usersim')

            checkpoint_path = os.path.join('./checkpoints_usersim', data_type)
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

            # set the location for models
            model_loc = os.path.join(checkpoint_path, 'model_best.pth')

            if not os.path.exists(model_loc):
                try:
                    wget.download(opt.model_path, model_loc)
                except Exception as err:
                    print(f"[{err}]")
                opt.model = model_loc
        else:
            opt.model = model_path

        if os.path.exists(infos_loc):
            # load existing infos
            with open(infos_loc, 'rb') as f:
                infos = cPickle.load(f)

        self.caption_model = infos["opt"].caption_model

        # override and collect parameters
        if len(opt.input_fc_dir) == 0:
            opt.input_fc_dir = infos['opt'].input_fc_dir
            opt.input_att_dir = infos['opt'].input_att_dir
            opt.input_label_h5 = infos['opt'].input_label_h5
        if len(opt.input_json) == 0:
            opt.input_json = infos['opt'].input_json
        if opt.batch_size == 0:
            opt.batch_size = infos['opt'].batch_size
        if len(opt.id) == 0:
            opt.id = infos['opt'].id
        ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "model"]
        for k in vars(infos['opt']).keys():
            if k not in ignore:
                if k in vars(opt):
                    assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
                else:
                    vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

        vocab = infos['vocab']  # ix -> word mapping

        #         print('opt:', opt)

        # Setup the model
        opt.vocab = vocab
        model = models.setup(opt)
        del opt.vocab
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.model))
            model.cuda()
        else:
            model.load_state_dict(torch.load(opt.model, map_location={'cuda:0': 'cpu'}))

        model.eval()

        self.is_relative = is_relative
        self.model = model
        self.vocab = vocab
        self.opt = vars(opt)

        # Load ResNet for processing images
        if opt.load_resnet:
            if image_feat_params['model_root']=='':
                net = getattr(resnet, image_feat_params['model'])(pretrained=True)
            else:
                net = getattr(resnet, image_feat_params['model'])()
                net.load_state_dict(
                    torch.load(os.path.join(image_feat_params['model_root'], image_feat_params['model'] + '.pth')))
            my_resnet = myResnet(net)
            if torch.cuda.is_available():
                my_resnet.cuda()
            my_resnet.eval()

            my_resnet_batch = ResNetBatch(net)
            if torch.cuda.is_available():
                my_resnet_batch.cuda()

            self.my_resnet_batch = my_resnet_batch
            self.my_resnet = my_resnet
        self.att_size = image_feat_params['att_size']

        # Control the input features of the model
        if diff_feat == None:
            if self.caption_model == "show_attend_tell":
                self.diff_feat = True
            else:
                self.diff_feat = False
        else:
            self.diff_feat = diff_feat

    def gen_caption_from_feat(self, feat_target, feat_reference=None):
        if self.is_relative and feat_reference == None:
            return None, None

        if not self.is_relative and not feat_reference == None:
            return None, None

        if self.is_relative:
            if self.diff_feat:
                fc_feat = torch.cat((feat_target[0], feat_target[0] - feat_reference[0]), dim=-1)
                att_feat = torch.cat((feat_target[1], feat_target[1] - feat_reference[1]), dim=-1)
            else:
                fc_feat = torch.cat((feat_target[0], feat_reference[0]), dim=-1)
                att_feat = torch.cat((feat_target[1], feat_reference[1]), dim=-1)
        else:
            fc_feat = feat_target[0]
            att_feat = feat_target[1]

        # Reshape to B x K x C (128,14,14,4096) --> (128,196,4096)
        att_feat = att_feat.view(att_feat.shape[0], att_feat.shape[1] * att_feat.shape[2], att_feat.shape[-1])

        att_masks = np.zeros(att_feat.shape[:2], dtype='float32')
        for i in range(len(att_feat)):
            att_masks[i, :att_feat[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if att_masks.sum() == att_masks.size:
            att_masks = None

        if self.caption_model == 'show_attend_tell':
            seq, _ = self.model.sample(fc_feat, att_feat, self.opt)
        else:
            seq, _ = self.model(fc_feat, att_feat, att_masks=att_masks, opt=self.opt, mode='sample')
        sents = utils.decode_sequence(self.vocab, seq)

        return seq, sents

    def get_vocab_size(self):
        return len(self.vocab)

    def get_img_feat(self, img_name):
        # load the image
        I = skimage.io.imread(img_name)

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose([2, 0, 1]))
        if torch.cuda.is_available(): I = I.cuda()
        # I = Variable(preprocess(I), volatile=True)
        with torch.no_grad():
            I = preprocess(I)
            fc, att = self.my_resnet(I, self.att_size)

        return fc, att

    def get_img_feat_batch(self, img_names, batchsize=32):
        if not isinstance(img_names, list):
            img_names = [img_names]

        num_images = len(img_names)
        num_batches = math.ceil(np.float(num_images) / np.float(batchsize))

        feature_fc = []
        feature_att = []

        for id in range(num_batches):
            startInd = id * batchsize
            endInd = min((id + 1) * batchsize, num_images)

            img_names_current_batch = img_names[startInd:endInd]
            I_current_batch = []

            for img_name in img_names_current_batch:
                I = skimage.io.imread(img_name)

                if len(I.shape) == 2:
                    I = I[:, :, np.newaxis]
                    I = np.concatenate((I, I, I), axis=2)

                I = I.astype('float32') / 255.0
                I = torch.from_numpy(I.transpose([2, 0, 1]))
                I_current_batch.append(preprocess(I))

            I_current_batch = torch.stack(I_current_batch, dim=0)
            if torch.cuda.is_available(): I_current_batch = I_current_batch.cuda()
            # I_current_batch = Variable(I_current_batch, volatile=True)
            with torch.no_grad():
                fc, att = self.my_resnet_batch(I_current_batch, self.att_size)

            feature_fc.append(fc)
            feature_att.append(att)

        feature_fc = torch.cat(feature_fc, dim=0)
        feature_att = torch.cat(feature_att, dim=0)

        return feature_fc, feature_att



