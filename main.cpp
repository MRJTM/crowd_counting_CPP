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
    String video_path="/home/czj/Videos/videos4/stream0.mp4";
    CrowdCounter crowdCounter(model_path);

    double num=0;
    int kernel_size=11;
    int sigma=4;
    int enlarge_rate=2;

      ///用图片测试
//    Mat img=imread(image_path,0);
//    Mat heatmap;

//    crowdCounter.process(img,heatmap,num,kernel_size,sigma,enlarge_rate);
//    imshow("heatmap",heatmap);
//    cout<<"num people:"<<num<<endl;
//    waitKey(0);

      ///用视频测试
    VideoCapture cap(video_path);
    if(!cap.isOpened())
    {
        cout<<"Can't open the video"<<endl;
        return -1;
    }
    for(;;)
    {
        Mat img;
        cap>>img;
        Mat heatmap;
        Mat gray;
        cvtColor(img,gray,CV_BGR2GRAY);
        crowdCounter.process(gray,heatmap,num,kernel_size,sigma,enlarge_rate);

        //显示heatmap
        imshow("heatmap",heatmap);
        cout<<"num of people:"<<num<<endl;

        //在图片上显示文字
        resize(img,img,Size(1280,720));
        putText(img,format("num peole:%d",int(num)),Point(50,80),CV_FONT_NORMAL,3,Scalar(0,0,255),2,8);
        imshow("1",img);
        int k=waitKey(1);
        if(k==27)
            break;
    }

    return 0;
}


