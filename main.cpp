#include <fstream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "crowd_counting.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    String image_path="../test_images/frame_01120.jpg";
    String model_path="../hongqiao_dt5_lr1e5_bs1_ep50.pb";
    Mat img=imread(image_path,0);
//    imshow("test image",img);

    Mat heatmap;
    double num=0;
    int kernel_size=11;
    int sigma=4;
    int enlarge_rate=2;

    //初始化，导入模型，创建session，GPU热身
    Session* session=tensorflow_init(model_path);

    //用这个session去跑图片
    crowd_couting(img,heatmap,num,kernel_size,sigma,enlarge_rate,session);

    //显示heatmap
    imshow("heatmap",heatmap);
    cout<<"num of people:"<<num<<endl;

    waitKey(0);
    return 0;
}


