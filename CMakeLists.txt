cmake_minimum_required(VERSION 3.10)
project(crowd_counting_cpp)

set(CMAKE_CXX_STANDARD 11)

link_directories(/home/czj/anaconda2/envs/tf/lib)
include_directories(
        /home/czj/tensorflow
        /home/czj/tensorflow/bazel-genfiles
        /home/czj/tensorflow/bazel-bin/tensorflow
        /home/czj/tools/tf-C/eigen3
)

add_executable(crowd_counting_cpp main.cpp crowd_counting.cpp crowd_counting.h)

find_package(OpenCV REQUIRED)
target_link_libraries(crowd_counting_cpp tensorflow_cc tensorflow_framework ${OpenCV_LIBS})