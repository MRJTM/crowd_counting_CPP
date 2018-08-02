#include <iostream>
#include <string>
#include <fstream>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "opencv2/opencv.hpp"

using namespace tensorflow::ops;
using namespace tensorflow;
using namespace std;
using namespace cv;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32 ;


class CrowdCounter
{
private:
    Session* session;
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