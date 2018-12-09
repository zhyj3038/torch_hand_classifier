
#include <string>
#include <iostream>
#include <map>


#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/workspace.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/init.h"


// feel free to define USE_GPU if you want to use gpu



// headers for opencv 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace caffe2;
using namespace cv;


// define flags
// CAFFE2_DEFINE_string(init_net, "./init_net.pb",
//                      "The given path to the init protobuffer.");
// CAFFE2_DEFINE_string(predict_net, "./predict_net.pb",
//                      "The given path to the predict protobuffer.");
// CAFFE2_DEFINE_string(file, "./image_file.jpg", "The image file.");





namespace caffe2{

    string FLAGS_init_net = "/zdata/zhangyajun/torch/hand_classifier/test/init_net.pb";
string FLAGS_predict_net = "/zdata/zhangyajun/torch/hand_classifier/test/predict_net.pb";
string FLAGS_file = "/zdata/zhangyajun/torch/hand_classifier/test/horse.jpg";
void loadImage(std::string file_name, float* imgArray){

    auto image = cv::imread(file_name);    // CV_8UC3
    std::cout << "== image size: " << image.size()
              << " ==" << std::endl;

    // scale image to fit
    cv::Size scale(32,32);
    cv::resize(image, image, scale);
    std::cout << "== simply resize: " << image.size() 
              << " ==" << std::endl;
    
    // convert [unsigned int] to [float]
    image.convertTo(image, CV_32FC3);

    // convert NHWC to NCHW
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    std::vector<float> data;
    for (auto &c : channels) {
        data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
    }
    
    // do normalization & copy to imgArray
    int dim = 0;
    float image_mean[3] = {113.865, 122.95, 125.307};
    float image_std[3] = {66.7048, 62.0887, 62.9932};
        
    for(auto i = 0; i < data.size();++i){
        if(i > 0 && i % (32*32) == 0) dim++;
        imgArray[i] = (data[i] - image_mean[dim]) / image_std[dim];
        // std::cout << imgArray[i] << std::endl;
    }
}



void run(){

    // define a caffe2 Workspace
    Workspace workSpace;

    // define initNet and predictNet
    NetDef initNet, predictNet;

    // read protobuf
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &initNet));
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predictNet));

    // set device type
    predictNet.mutable_device_option()->set_device_type(PROTO_CPU);
    initNet.mutable_device_option()->set_device_type(PROTO_CPU);

    for(int i = 0; i < predictNet.op_size(); ++i){
        predictNet.mutable_op(i)->mutable_device_option()->set_device_type(PROTO_CPU);
    }
    for(int i = 0; i < initNet.op_size(); ++i){
        initNet.mutable_op(i)->mutable_device_option()->set_device_type(PROTO_CPU);
    }


    // load network
    CAFFE_ENFORCE(workSpace.RunNetOnce(initNet));
    CAFFE_ENFORCE(workSpace.CreateNet(predictNet));

    // load image from file, then convert it to float array.
    float imgArray[3 * 32 * 32];
    loadImage(FLAGS_file, imgArray);

    // define a Tensor which is used to stone input data
    vector<int64_t> dims({1,3,32,32});
	
    Tensor input(dims, CPU);
    // input.Resize(std::vector<int64_t>({1, 3, 32, 32}));
    input.ShareExternalPointer(imgArray);

    // get "data" blob

    auto data = workSpace.GetBlob("data")->GetMutable<TensorCPU>();


    // copy from input data
    data->CopyFrom(input);

    // forward
    workSpace.RunNet(predictNet.name());

    // get softmax blob and show the results
    std::vector<std::string> labelName = {"airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"};


    auto softmax = workSpace.GetBlob("softmax")->Get<TensorCPU>();


    std::vector<float> probs(softmax.data<float>(),
        softmax.data<float>() + softmax.size());
    
    auto max = std::max_element(probs.begin(), probs.end());
    auto index = std::distance(probs.begin(), max);
    std::cout << "== predicted label: " << labelName[index] 
              << " ==\n== with probability: " << (*max * 100)
              << "% ==" << std::endl;

              
}//run

}    // namespace caffe2

// main function
int main(int argc, char** argv) {
    caffe2::GlobalInit(&argc, &argv);
    caffe2::run();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
