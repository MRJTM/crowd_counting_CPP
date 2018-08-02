#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
using namespace cv;


class CrowdCounter
{
private:
    void* temp_session;
public:
    // construction fu
    // nction, initialize session
    CrowdCounter(std::string model_path);
    
    // count one image
    void process(Mat inputImg,Mat& heatMap,double& number, int kernel_size,
                   int sigma,int out_enlarge_rate);
                   
    // release session, free memory... and so on              
    ~CrowdCounter();
};