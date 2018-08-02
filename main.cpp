#include <fstream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "CrowdCountingAPI.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    String image_path="/home/czj/Project/crowd_counting_cpp/test_images/frame_01120.jpg";
    String model_path="/home/czj/Project/crowd_counting_cpp/hongqiao_dt5_lr1e5_bs1_ep50.pb";
    Mat img=imread(image_path,0);


    Mat heatmap;
    double num=0;
    int kernel_size=11;
    int sigma=4;
    int enlarge_rate=2;


    CrowdCounter crowdCounter(model_path);
    crowdCounter.process(img,heatmap,num,kernel_size,sigma,enlarge_rate);

    //显示heatmap
    imshow("heatmap",heatmap);
    cout<<"num of people:"<<num<<endl;

    waitKey(0);
    return 0;
}


