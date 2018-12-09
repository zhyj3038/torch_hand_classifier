from torch import nn
from utils import Tester
from network import resnet34, resnet101, MobileNetV2, squeezenet1_1


img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
img_size = 224 #resnet是224
num_class = 6



# Set Test parameters
params = Tester.TestParams()
params.gpus = []  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/ckpt_epoch_200.pth'  #'./models/ckpt_epoch_400_res34.pth'
params.testdata_dir = './testimg/'

# models
# model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
# model.fc = nn.Linear(512, 6)
# model = resnet101(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
# model.fc = nn.Linear(512*4, 6)
# model = MobileNetV2( n_class = 6, input_size=224, width_mult=1. )

model = squeezenet1_1( pretrained = False)

# Test
tester = Tester(model, params)
tester.test( img_mean, img_std, img_size )
