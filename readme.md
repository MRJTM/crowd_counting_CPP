
# Intruduction
CrowdCountingAPI is to input a image with a OpenCV Mat format, 
and return a heatmap which shows people distribution and the 
number of people.

# Installation
### 1.compile the C++ API of tensorflow1.8
which will generate 2 .so files:
 - libtensorflow_cc.so
 - libtensorflow_framework.so  
 to compile tensorflow you also have to have Eigen3  
 we assume that you put Eigens in /home/xxx/

### 2.compile OpenCV3.4 from source

### 3.compile this API
we assume that you git clone this project into /home/xxx/
```sh
$ cd /home/xxx
$ git clone https://git.light-field.tech/caozhijie/crowd_counting_CPP.git
$ cd crowd_counting_CPP
#compile the cpp file to object file
$ g++ -std=c++11 -fPIC -c CrowdCountingAPI.cpp -o CrowdCountingAPI.o \
-I /home/xxx/eigen3 -I /home/xxx/tensorflow \
-I /home/czj/tensorflow/bazel-genfiles -I /home/czj/tensorflow/bazel-bin/tensorflow
#compile the object file to .so file 
$ g++ -shared -o libCrowdCountingAPI.so CrowdCountingAPI.o
#copy the .so file into /usr/lib
$ sudo cp libCrowdCountingAPI.so /usr/lib
#if you want to test if everything is ok, then compile the test code
$  g++ -o main main.cpp  -L. -lCrowdCountingAPI -L/usr/local/lib -ltensorflow_cc \
-ltensorflow_framework -lopencv_core -lopencv_imgcodecs -lopencv_highgui
#run the executable file
$ ./main
```

# API Usage
this API contains a class with two functions:
### 1.CrowdCounter(std::string model_path)
this is the constructor which is to load the model,create a session and heat up the GPU
 - model_path: a String contains the absolute path of your .pb file,not a relative path  

### 2.void process(Mat inputImg,Mat& heatMap,double& number, int kernel_size,int sigma,int out_enlarge_rate)
this is the function to input a image and return the heatmap and number of people
 - inputImg: the input image with OpenCV Mat format
 - heatMap: the output heatmap with OpenCV Mat format
 - number: the output number of people
 - kernel_size: the guassian kernel size
 - sigma: the parameter sigma of guassian blur
 - out_enlarge_rate: the enlargement factor of the heatmap with which you can see 
a larger heatmap

### for more details of usage, you can check the main.cpp in my project
