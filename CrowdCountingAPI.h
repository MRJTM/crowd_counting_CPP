#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>



class CrowdCounter
{
private:
    void* session;
public:
    // construction function, initialize session
    CrowdCounter(std::string model_path);
    
    // count one image
    void process(Mat inputImg,Mat& heatMap,double& number, int kernel_size,
                   int sigma,int out_enlarge_rate);
                   
    // release session, free memory... and so on              
    ~CrowdCounter();
}