#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/flann/dist.h>
#include <opencv2/highgui/highgui.hpp>
#include <keypoints/keypoints.h>
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/highgui.h"
#include <salience/salience.h>

#define PI 3.1415

using namespace std;
using namespace cv;
using namespace vislab::keypoints;

double WrappingAngle(double x){
    x = fmod(x,360);
    if (x < 0)
        x += 360;
    return x;
}

//function sumstack
Mat sumstack( vector<Mat> stack, std::vector<double> weights)
{
    Mat result = Mat::zeros(stack[0].rows, stack[0].cols, stack[0].type()); //initialising the result to zeros of rows and cols

    if(weights.empty())
        for(unsigned i=0;i<stack.size();i++)
            weights.push_back(1./stack.size());

    if(weights.size() != stack.size())
    {
        std::cerr <<"Number of weights must match the number of images" << std::endl;
        return result;
    }

    for(unsigned i=0;i<stack.size(); i++)  //i is declared in order to access the stacks
    {
        Mat temp;
        threshold(stack[i], temp, 0, 0, THRESH_TOZERO);  //the value of stack[i] is stored in temp (thresholded)
        result = result + temp*weights[i]; //stack[i];  // the temp value is added to the result
//        result = result + temp; //stack[i];  // the temp value is added to the result
    }

    return result;
}

//function blobkernel
vector<Mat> createBlobKernels(vector<double> sigmas ) // Modified one
{
    vector<Mat> result; //calling the result of sumstack

    for(unsigned i=0; i<sigmas.size(); i++)     //initialising to access the sigmas
    {
        double sigma = sigmas[i];   //initialisig the value of sigma to sigma[i]
        int filtersize = sigma*4;
        if( filtersize % 2 == 0) filtersize++;        // Ensure that the kernel size is odd
        int offsetf = floor(filtersize/2);    //floor is used to round off to integer value

        Mat kernel( filtersize, filtersize, CV_32F );   //initialising the kernel and putting the filtersize into kernel
        Mat kernel_ex( filtersize, filtersize, CV_32F );   //initialising the kernel and putting the filtersize into kernel
        Mat kernel_inh( filtersize, filtersize, CV_32F );   //initialising the kernel and putting the filtersize into kernel
        float kernelsum = 0;

        // create the kernel we will be using (difference of gaussians
        for(int row=0; row<kernel.rows; row++)
        {
            for(int col=0; col<kernel.cols; col++)
            {   // return 0;
                double x = col - offsetf;
                double y = row - offsetf;

                double gauss1 = exp( -(x*x+y*y) / (sigma*sigma)   );
                double gauss2 = exp( -(x*x+y*y) / (sigma*sigma*4) );

//                kernel.at<float>(row,col) = gauss1 - gauss2;
//                kernelsum += gauss1 - gauss2;
                kernel_ex.at<float>(row,col) = gauss1;
                kernel_inh.at<float>(row,col) = gauss2;

            }
        }
        // Remove the DC component of the even kernel
//        kernel -= kernelsum/(filtersize*filtersize);
        kernel_ex *= 1/cv::sum(kernel_ex).val[0];
        kernel_inh *= 1/cv::sum(kernel_inh).val[0];
        kernel = kernel_ex-kernel_inh;

        result.push_back(kernel);
    }
    return result;
}

vector<Mat> filterStack( const vector<Mat>& inputstack, const vector<Mat> filterstack )
{
    vector<Mat> result;

    for(unsigned i=0; i<inputstack.size(); i++)
    {
        Mat response, tempresponse;
        response = Mat::zeros(inputstack[i].rows, inputstack[i].cols, CV_32F);

        for(unsigned j=0; j<filterstack.size(); j++)
        {
            Mat temp = inputstack[i];
             GaussianBlur(inputstack[i], temp, Size(21,21), 4.0);
//             cout<<"loop.....\n";
            filter2D( temp, tempresponse, inputstack[i].depth(), filterstack[j]);
//            filter2D( temp, tempresponse, response.type(), filterstack[j]);
//            cout<<"loopend....\n";
            response += tempresponse;
        }
        result.push_back(response);
    }
    return result;
}

 //-----------------------------Texture stack-----------------------------------------------
 vector<Mat> createTextureStack(Mat& img, vector<KPData>& datas)
 {
     vector<Mat> result, result1;
     Mat spec,spec1;
     Mat sumrows, sumcols;
     Point minLoc_r, maxLoc_r, minLoc_c, maxLoc_c;
     double max_val_o, max_val_f;

     if(datas.empty())
         return result;
     int freqs = datas.size();
     int oris  = datas[0].C_array.size();
     int rows = img.rows;
     int cols = img.cols;

      Mat Std_o(rows, cols, CV_32F);
      Mat Std_f(rows, cols, CV_32F);
      Mat Mu_o(rows, cols, CV_32F);
      Mat Mu_f(rows, cols, CV_32F);
      Mat eps(rows, cols, CV_32F);

     vector<int> pyrscales;
     for(unsigned i=0; i<datas.size(); i++)
         pyrscales.push_back(round(float(img.rows)/float(datas[i].C_array[0].rows)));

     for(unsigned i=0; i<4; i++)
         result.push_back(Mat::zeros(rows, cols, CV_32F));

     for(unsigned i=0; i<2; i++)
         result1.push_back(Mat::zeros(rows, cols, CV_32F));

     for(int y=0; y<rows; y++)
         for(int x=0; x<cols; x++)
             {
             Mat spectrum(freqs, oris, CV_32F);
             for(int sy=0; sy<spectrum.rows; sy++)
                 for(int sx=0; sx<spectrum.cols; sx++)
                     spectrum.at<float>(sy,sx) = datas[sy].C_array[sx](y/pyrscales[sy],x/pyrscales[sy]);

//             cout<<"ty = "<<img.type()<<"\n";
//             if(x==211)
//                 if(y==210)
//                 {
//                     imwrite("/home/saikrishna/Desktop/GCPR/insidespectrumwhite.png",spectrum);
//                     Scalar intensity = img.at<uchar>(y, x);
////                     Vec3b intensity1 = img.at<Vec3b>(y, x);
////                     uchar blue = intensity1.val[0];
////                     uchar green = intensity1.val[1];
////                     uchar red = intensity1.val[2];
//                     cout<<"blue = "<<intensity.val[0]<<"\n";
//                     cout<<"green = "<<intensity.val[1]<<"\n";
////                     cout<<"red = "<<intensity.val[2]<<"\n";
////                     cout<<int(blue)<<"\n";
////                     cout<<int(green)<<"\n";
////                     cout<<int(red)<<"\n";
//                     cout<<"=====================\n";
//                 }

             blur(spectrum, spec, Size(3, 3));
             spec1=spec/3;
             reduce( spec1, sumrows, 0, CV_REDUCE_AVG);
             reduce( spec1, sumcols, 1, CV_REDUCE_AVG );

             minMaxLoc(sumrows, NULL, &max_val_f, &minLoc_r, &maxLoc_r); // rows
             minMaxLoc(sumcols, NULL, &max_val_o, &minLoc_c, &maxLoc_c); // cols

             int r=sumrows.rows;
             int c=sumrows.cols;

             float v1, v2, Mu=0, Mu1, val=0, var, stddev;
             float v12, v22, Mu2=0, Mu12, val2=0, var2, stddev2;
             float f1, out, stddevr, total = 0;
             float fc, outc,stddevc, totalc=0;

             for (int i1=0; i1<r; i1++)
                 for (int j1=0; j1<c; j1++)
                 {
                     v1 = sumrows.at<float>(j1);
                     Mu = Mu + v1;
                 }
             Mu1 = Mu/c;

             for (int i2=0; i2<r; i2++)
                 for (int j2=0; j2<c; j2++)
                 {
                     v2 = pow((sumrows.at<float>(j2)-Mu1),2);
                     val = val + v2;
                 }
             var = val/c;
             stddev = sqrt(var);

//             float kk = 0, kk1 = 0, k1;

//             for(int a=0; a<r; a++)
//                 {
//                 for(int b=0;b<c;b++)
//                     {
//                     f1 = sumrows.at<float>(b);
//                     kk = kk + f1;
//                 }
//             }

//             for(int a=0; a<r; a++)
//                 {
//                 for(int b=0;b<c;b++)
//                     {
//                     k1 = sumrows.at<float>(b)/(kk);
//                     kk1 = kk1 + k1;
//                     out = (k1* (pow((b-maxLoc_r.x),2)));
//                     total = (total + out);
//                 }
//             }
//             total = fabs(total);
//             stddevr = sqrt(total);



             Std_o.at<float>(y,x) = stddev;  // Method 1 //Orientation
             //Mu_o.at<float>(y,x) = Mu1; //Original command

             double ang = WrappingAngle(Mu1); // Modified Command
             Mu_o.at<float>(y,x) = ang; //          for Wrapping angle

//             ort.at<float>(y,x) = stddevr;  // Method 2
//             Mu_o.at<float>(y,x) = max_val_o;

             int ro =sumcols.rows;
             int co=sumcols.cols;

             for (int i12=0; i12<ro; i12++)
                 for (int j12=0; j12<co; j12++)
                 {
                     v12 = sumcols.at<float>(i12);
                     Mu2 = Mu2 + v12;
                 }
             Mu12 = Mu2/ro;

             for (int i22=0; i22<ro; i22++)
                 for (int j22=0; j22<co; j22++)
                 {
//                     cout<<"hi "<<i22<<"\n";
                     v22 = pow((sumcols.at<float>(i22)-Mu12),2);
                     val2 = val2 + v22;
                 }
             var2 = val2/ro;
             stddev2 = sqrt(var2);

//             float sk = 0, kkk1 = 0, sk1;

//             for(int c=0; c<ro; c++)
//                 {
//                 for(int d=0;d<co;d++)
//                     {
//                     fc = sumrows.at<float>(c);
//                     sk = sk + fc;
//                 }
//             }

//             for(int m=0; m<ro; m++)
//                 {
//                 for(int n=0;n<co;n++)
//                     {
//                     sk1 = sumrows.at<float>(m)/(sk);
//                     kkk1 = kkk1 + sk1;
//                     outc = (sk1* (pow((m-maxLoc_c.y),2)));
//                     totalc = totalc + outc;
//                 }
//             }

//             totalc = fabs(totalc);
//             stddevc = sqrt(totalc);

             Std_f.at<float>(y,x) = stddev2; // Method 1 // Frequency
             Mu_f.at<float>(y,x) = Mu12;

//             fre.at<float>(y,x) = stddevc; // Method 2
//             Mu_f.at<float>(y,x) = max_val_f;

//             eps.at<float>(y,x) = (stddev2-stddev)/(stddev2+stddev);
//             if(eps.at<float>(y,x)>0.9)
//                 cout<<"hi = "<<eps.at<float>(y,x)<<"\n";

        }

     for(int ch=0; ch<result.size(); ch++)
         {
              for(int y=0; y<rows; y++)
                  for(int x=0; x<cols; x++)
                  {
                      switch(ch)
                      {
                      case 0: result[ch].at<float>(y,x) = std::abs(Std_o.at<float>(y,x));
                      case 1: result[ch].at<float>(y,x) = std::abs(Std_f.at<float>(y,x));
                      case 2: result[ch].at<float>(y,x) = std::abs(Mu_o.at<float>(y,x));
                      case 3: result[ch].at<float>(y,x) = std::abs(Mu_f.at<float>(y,x));
//                      case 0: result[ch].at<float>(y,x) = Std_o.at<float>(y,x)>0 && Std_o.at<float>(y,x)<15 ? std::abs(Std_o.at<float>(y,x)) : 0 ; break;
//                      case 1: result[ch].at<float>(y,x) = Std_f.at<float>(y,x)>0 && Std_f.at<float>(y,x)<15 ? std::abs(Std_f.at<float>(y,x)) : 0 ;break;
//                      case 2: result[ch].at<float>(y,x) = Mu_o.at<float>(y,x)>0 && Mu_o.at<float>(y,x)<15 ? std::abs(Mu_o.at<float>(y,x)) : 0 ;break;
//                      case 3: result[ch].at<float>(y,x) = Mu_f.at<float>(y,x)>0 && Mu_f.at<float>(y,x)<15 ? std::abs(Mu_f.at<float>(y,x)) : 0 ;break;
//                      case 0: result[ch].at<float>(y,x) = Std_o.at<float>(y,x)>0 && Std_o.at<float>(y,x)<100 ? std::abs(Std_o.at<float>(y,x)) : 0 ; break;
//                      case 1: result[ch].at<float>(y,x) = Std_f.at<float>(y,x)>0 && Std_f.at<float>(y,x)<100 ? std::abs(Std_f.at<float>(y,x)) : 0 ;break;
//                      case 2: result[ch].at<float>(y,x) = Mu_o.at<float>(y,x)>0 && Mu_o.at<float>(y,x)<100 ? std::abs(Mu_o.at<float>(y,x)) : 0 ;break;
//                      case 3: result[ch].at<float>(y,x) = Mu_f.at<float>(y,x)>0 && Mu_f.at<float>(y,x)<100 ? std::abs(Mu_f.at<float>(y,x)) : 0 ;break;
                      }
                  }
              //              int top, bottom, left, right;
              //              Mat src, dst;
              //              src = result[ch];
              //              dst = result[ch];
              //              top = (int) (0.10*src.rows); bottom = (int) (0.10*src.rows);
              //              left = (int) (0.10*src.cols); right = (int) (0.10*src.cols);

              //              copyMakeBorder( src, result[ch], top, bottom, left, right, BORDER_REPLICATE );
         }

//     result[0] = Std_o;
//     result[1] = Std_f;
//     result[2] = Mu_o;
//     result[3] = Mu_f;

//     Mat leo,leo1,leo2,leo3;
//     result[0].convertTo(leo, CV_8UC1);
//     result[1].convertTo(leo1, CV_8UC1);
//     result[2].convertTo(leo2, CV_8UC1);
//     result[3].convertTo(leo3, CV_8UC1);

//     imshow("result0", leo);
//     imshow("result1", leo1);
//     imshow("result2", leo2);
//     imshow("result3", leo3);

//     imshow("Orientation STD", result[0]/255.f);
//     imshow("Frequency STD", result[1]/255.f);
//     imshow("Orientation Mean", result[2]/255.f);
//     imshow("Frequency Mean", result[3]/255.f);


//     Mat leo,leo1,leo2,leo3;
//     result[0].convertTo(leo, CV_8UC1);
//     result[1].convertTo(leo1, CV_8UC1);
//     result[2].convertTo(leo2, CV_8UC1);
//     result[3].convertTo(leo3, CV_8UC1);

//     Mat src = result[0];
//     double minVal, maxVal;
//     Point min_loc, max_loc;
//     minMaxLoc(src, &minVal, &maxVal, &min_loc, &max_loc);
//     Mat res = src;
//     res = res/maxVal;
//     res = res*255;
//     Mat src1 = result[1];
//     double minVal1, maxVal1;
//     Point min_loc1, max_loc1;
//     minMaxLoc(src1, &minVal1, &maxVal1, &min_loc1, &max_loc1);
//     Mat res1 = src1;
//     res1 = res1/maxVal1;
//     res1 = res1*255;
//     Mat src2 = result[2];
//     double minVal2, maxVal2;
//     Point min_loc2, max_loc2;
//     minMaxLoc(src2, &minVal2, &maxVal2, &min_loc2, &max_loc2);
//     Mat res2 = src2;
//     res2 = res2/maxVal2;
//     res2 = res2*255;
//     Mat src3 = result[3];
//     double minVal3, maxVal3;
//     Point min_loc3, max_loc3;
//     minMaxLoc(src3, &minVal3, &maxVal3, &min_loc3, &max_loc3);
//     Mat res3 = src3;
//     res3 = res3/maxVal3;
//     res3 = res3*255;
//     double minVal4, maxVal4;
//     Point min_loc4, max_loc4;
//     minMaxLoc(res3, &minVal4, &maxVal4, &min_loc4, &max_loc4);
//     cout<<"low = "<<minVal4<<" high = "<<maxVal4<<"\n";

//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Orientation STD.png",res);
//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Frequency STD.png",res1);
//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Orientation Mean.png",res2);
//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Frequency Mean.png",res3);
//     imshow("result0", leo);
//     imshow("result1", leo1);
//     imshow("result2", leo2);
//     imshow("result3", leo3);

    return result;
 }
 //-----------------------------Enf of Texture stack-----------------------------------------------
