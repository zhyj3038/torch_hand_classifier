
from caffe2.python import workspace
import operator
import numpy as np
import cv2

from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils


img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

class Normalize(object):
    def __init__(self,mean, std):
        '''
        :param mean: RGB order
        :param std:  RGB order
        '''
        self.mean = np.array(mean).reshape(3,1,1)
        self.std = np.array(std).reshape(3,1,1)
    def __call__(self, image):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        # plt.figure(1)
        # plt.imshow(image)
        # plt.show()
        return (image.transpose((2, 0, 1)) / 255. - self.mean) / self.std


Normalize_img = Normalize( img_mean , img_std )

def get_test_img(img_path ="/zdata/zhangyajun/torch/pytorch_hand_classifier/testimg/img_0100.png"):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img =Normalize_img(img)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img
img =get_test_img()



with open("init_net.pb", 'rb' ) as f:
    init_net = f.read()
with open("predict_net.pb", 'rb' ) as f:
    predict_net = f.read()

#===========================读取caffe的======================================
p = workspace.Predictor(init_net, predict_net)


results = p.run([img])

results = np.asarray(results)
preds = np.squeeze(results)
print(preds)
curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
print("Prediction: ", curr_pred)
print("Confidence: ", curr_conf)


# #==========================
# workspace.RunNetOnce(init_net)
# workspace.RunNetOnce(predict_net)
# print(net_printer.to_string(predict_net))

# # Now, let's also pass in the resized cat image for processing by the model.
# workspace.FeedBlob("0", img )

# # run the predict_net to get the model output
# workspace.RunNetOnce(predict_net)

# # Now let's get the model output blob
# img_out = workspace.FetchBlob("127")
# print(img_out)

