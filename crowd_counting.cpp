/*
 * 本函数作用是输入一张图片，输出一个density map的热力图，以及人数
 */

#include "crowd_counting.h"

// 定义一个函数讲OpenCV的Mat数据转化为tensor，python里面只要对cv2.read读进来的矩阵进行np.reshape之后，
// 数据类型就成了一个tensor，即tensor与矩阵一样，然后就可以输入到网络的入口了，但是C++版本，我们网络开放的入口
// 也需要将输入图片转化成一个tensor，所以如果用OpenCV读取图片的话，就是一个Mat，然后就要考虑怎么将Mat转化为
// Tensor了
void CVMat_to_Tensor(Mat img,Tensor* output_tensor,int input_rows,int input_cols)
{
    //图像进行resize处理
    cout<<"1--resizing the img"<<endl;
    resize(img,img,cv::Size(input_cols,input_rows));

    //归一化
    cout<<"2--Normalizing the img"<<endl;
    img.convertTo(img,CV_32FC1);
    img=img/255;

    //加上mask
    cout<<"3--add a mask"<<endl;
    int mask_height=35;
    for(int i=0;i<mask_height;i++)
        for(int j=0;j<input_cols;j++)
            img.at<float>(i,j)=0;
    imshow("masked img",img);

    //创建一个指向tensor的内容的指针
    float *p = output_tensor->flat<float>().data();

    //创建一个Mat，与tensor的指针绑定
    cout<<"4--transfer data from cvMat to tensor"<<endl;
    cv::Mat cameraImg(input_rows, input_cols, CV_32FC1, p);
    img.convertTo(cameraImg,CV_32FC1);


}

//载入模型和创建session
Session* tensorflow_init(String model_path)
{
    ///*--------------------------------创建session------------------------------*///
    Session* session;
    NewSession(SessionOptions(), &session);

    ///*--------------------------------从pb文件中读取模型--------------------------------*///
    GraphDef graphdef;
    //从pb文件中读取图模型;
    ReadBinaryProto(Env::Default(), model_path, &graphdef);
    //将模型导入会话Session中;
    session->Create(graphdef);
    cout << "<----Successfully created session and load graph.------->"<< endl;

    ///*--------------------------------GPU热身---------------------------------------*///
    cout << "<---------GPU heat up-------->"<< endl;
    //创建一个tensor
    Tensor test_tensor(DT_FLOAT, TensorShape({1,300,480,1}));
    //创建tensor的指针
    float *p = test_tensor.flat<float>().data();
    //产生一张480*300的图片
    Mat img(300,480,CV_32FC1,p);

    //跑5次网络热身
    string input_tensor_name="input_1";
    string output_tensor_name="conv2d_5/Relu";
    vector<tensorflow::Tensor> outputs;
    for(int i=0;i<2;i++)
    {
        session->Run({{input_tensor_name, test_tensor}}, {output_tensor_name}, {}, &outputs);
    }
    return session;
}

void crowd_couting(Mat inputImg,Mat& heatMap,double& number, int kernel_size,
                   int sigma,int out_enlarge_rate,Session* session)
{
    ///*---------------------------------配置路径参数-----------------------------*///
    int input_height =300;
    int input_width=480;
    int output_height=75;
    int output_width=120;
    string input_tensor_name="input_1";
    string output_tensor_name="conv2d_5/Relu";


    ///*---------------------------------把图片转化为tensor-------------------------------------*///
    //创建一个tensor作为输入网络的接口
    Tensor resized_tensor(DT_FLOAT, TensorShape({1,input_height,input_width,1}));

    //将Opencv的Mat格式的图片存入tensor
    cout<<">>----Converting the cvMat to a tensor----"<<endl;
    CVMat_to_Tensor(inputImg,&resized_tensor,input_height,input_width);
    cout<<">>----Successfully converted the cvMat to a tensor----"<<endl;

    ///*-----------------------------------用网络跑一次前向-----------------------------------------*///
    cout<<endl<<"<-------------Running the model with test_image--------------->"<<endl;
    clock_t startTime,endTime;
    startTime=clock();

    //前向运行，输出结果一定是一个tensor的vector
    vector<tensorflow::Tensor> outputs;
    session->Run({{input_tensor_name, resized_tensor}}, {output_tensor_name}, {}, &outputs);

    endTime=clock();
    cout<<"foward run takes time:"<<(float)(endTime-startTime)/CLOCKS_PER_SEC*1000<<"ms"<<endl;

    ///*-----------------------------------处理输出值-----------------------------------------*///
    //获取输出的tensor的内容
    Tensor t = outputs[0];
    cv::Mat out_den(output_height, output_width, CV_32FC1, t.flat<float>().data());

    //积分求和计算人数
    Scalar sum=cv::sum(out_den);
    number=sum(0);

    ///制作热力图
    //高斯模糊
    GaussianBlur(out_den,out_den,cv::Size(kernel_size,kernel_size),sigma,sigma);
    //处理density map
    double min,max;
    minMaxIdx(out_den,&min,&max);
    out_den=out_den/max*255;
    resize(out_den,out_den,cv::Size(120*out_enlarge_rate,75*out_enlarge_rate));
    out_den.convertTo(out_den,CV_8UC3);
    //显示非热力图效果的dentity map
    imshow("density map",out_den);
    applyColorMap(out_den,heatMap,COLORMAP_JET);

}