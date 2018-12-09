cmake_minimum_required(VERSION 3.6)
add_definitions(-std=c++11)

project(hand_classify)

#caffe
include_directories(/zdata/zhangyajun/torch/hand_classifier/libpytorch/include )

#opencv
include_directories(/usr/local/include )




add_executable( hand_classify   main.cpp )

#caffe
target_link_libraries( hand_classify
                         /zdata/zhangyajun/torch/hand_classifier/libpytorch/lib/libcaffe2.so
                        /usr/local/lib/libopencv_core.so 
                        /usr/local/lib/libopencv_imgcodecs.so
                        /usr/local/lib/libopencv_highgui.so
                        /usr/local/lib/libopencv_imgproc.so

                        /zdata/zhangyajun/torch/hand_classifier/libpytorch/lib/libc10.so
                        /zdata/zhangyajun/torch/hand_classifier/libpytorch/lib/libqnnpack.a
                        /zdata/zhangyajun/torch/hand_classifier/libpytorch/lib/libprotobuf.a
                        )