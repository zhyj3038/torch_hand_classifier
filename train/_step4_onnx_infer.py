
#===============直接读取onnx的, 可以正常工作==========================
import onnx
import torch
import caffe2.python.onnx.backend as bc
from torch.autograd import Variable
import os 

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F





img_size = 224
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]



img = Image.open( "/zdata/zhangyajun/torch/hand_classifier/train/testimg/img_0100.png" )
img = tv_F.to_tensor(tv_F.resize(img, ( img_size, img_size )))
img = tv_F.normalize(img, img_mean, img_std )
img_input = Variable(torch.unsqueeze(img, 0))
print(img_input.size() )

model = onnx.load("squeezenet.onnx")
prepared_backend = bc.prepare(model)
W = {model.graph.input[0].name: img_input.data.numpy()}
c2_out = prepared_backend.run(W)[0]
print(c2_out)




#----------------------生成caffe的pb---------------------
# extract the workspace and the model proto from the internal representation
c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net

# Now import the caffe2 mobile exporter
from caffe2.python.predictor import mobile_exporter

# call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
init_net, predict_net = mobile_exporter.Export( c2_workspace, c2_model, c2_model.external_input)

# Let's also save the init_net and predict_net to a file that we will later use for running them on mobile
with open('init_net.pb', "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open('predict_net.pb', "wb") as fopen:
    fopen.write(predict_net.SerializeToString())

##------------------------------------------------
# from caffe2.python.onnx.backend import Caffe2Backend
# from caffe2.python.onnx.helper import c2_native_run_net, save_caffe2_net, load_caffe2_net, \
#     benchmark_caffe2_model, benchmark_pytorch_model


# model = onnx.load("squeezenet.onnx")
# onnx.checker.check_model(model)
# init_net, predict_net = bc.Caffe2Backend.onnx_graph_to_caffe2_net(model, device='CPU')
# save_caffe2_net(init_net, '/zdata/zhangyajun/torch/AICamera/app/src/main/assets/init_net.pb')
# save_caffe2_net(predict_net,  '/zdata/zhangyajun/torch/AICamera/app/src/main/assets/predict_net.pb', output_txt=True)

