import torch
import torch.onnx
from torch.autograd import Variable
import onnx

from network import resnet34, resnet101, MobileNetV2, squeezenet1_1




#step1

batch_size = 1    # just a random number
# Input to the model
# x = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

model = squeezenet1_1( pretrained = False)
state_dict = torch.load( './models/ckpt_epoch_200.pth' )


#去除keys前面的module
from collections import OrderedDict
new_state_dict = OrderedDict()
for k,v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict( new_state_dict )
model.train(False)

torch.onnx.export(model, x, "squeezenet.onnx", export_params=True)


# Load the ONNX model
model = onnx.load("squeezenet.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))




#step2 
#没必要运行
#convert-onnx-to-caffe2 squeezenet.onnx --output predict_net.pb --init-net-output init_net.pb