from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F


class TestParams(object):
    gpus = [ ]  # default to use CPU mode
    ckpt = None     # path to the ckpt file
    testdata_dir = None

class Tester(object):
    TestParams = TestParams
    def __init__(self, model, test_params):
        assert isinstance(test_params, TestParams)
        self.params = test_params

        # load model
        self.model = model
        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(self.params.gpus) > 0:
            gpu_test = str(self.params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = self.model.cuda()

        self.model.eval()

    def _load_ckpt(self, ckpt ):
        print(  torch.load(ckpt).keys()  )
        state_dict = torch.load( ckpt )

        #去除keys前面的module
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        self.model.load_state_dict( new_state_dict )

    def test(self, img_mean, img_std, img_size):

        img_list = os.listdir(self.params.testdata_dir)

        for img_name in img_list:
            print('Processing image: ' + img_name)

            img = Image.open(os.path.join(self.params.testdata_dir, img_name))
            img = tv_F.to_tensor(tv_F.resize(img, ( img_size, img_size )))
            img = tv_F.normalize(img, img_mean, img_std )
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print( "predict:{}".format( prediction.item() ) )