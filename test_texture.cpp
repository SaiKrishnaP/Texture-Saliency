#include "opencv/highgui.h"
#include <keypoints/keypoints.h>
#include <salience/salience.h>
#include <iomanip>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h"
#include <wchar.h>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <fstream>
#include <iostream>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

//using namespace cimg_library;

#define PI 3.1415

using namespace std;
using namespace cv;
using namespace vislab::keypoints;

Mat weightedfiltersum(Mat inputsaliency, Mat inputedge, Mat inputcont, int numberoftimes);

int main()
  {
  string tst_dir, tst_gt, filepath;
  tst_dir = "/home/saikrishna/Desktop/name_change/col"; //  AC, GB, IG, IT, MZ, SF
  int im = 0;
  Mat img;
  DIR *dp;
  struct dirent *dirp;
  struct stat filestat;

  Mat image1, lab, gray, gt_img;

  vector<double> blobSigmas;

  blobSigmas.push_back(45);
  blobSigmas.push_back(90);
  blobSigmas.push_back(180);

  vector<Mat> blobKernels = createBlobKernels(blobSigmas);    //calling the create blob kernel function in main


 dp = opendir( tst_dir.c_str() );
  while ((dirp = readdir( dp )))
    {

    filepath = tst_dir + "/" + dirp->d_name;

    // If the file is a directory (or is in some way invalid) we'll skip it
    if (stat( filepath.c_str(), &filestat )) continue;
    if (S_ISDIR( filestat.st_mode ))         continue;

    img = cv::imread(filepath.c_str());
    image1 = img;

    cvtColor(image1, lab, CV_BGR2Lab);                                               //converting the resized image to LAB colorspace
    cvtColor(image1, gray, CV_BGR2GRAY);                                             //converting the resized image to Gray colorspace

    vislab::keypoints::BimpFeatureDetector f(vislab::keypoints::makeLambdasLin(4,4,1));
    std::vector<cv::KeyPoint> point;
    std::vector<vislab::keypoints::KPData> data;
    f.detect(gray, point, data);


    //----------------------------Texture--------------------------------------
    vector<double> lambdas = makeLambdasLog(4,64,2);
    vector<KPData> datas;
    vector<KeyPoint> points = keypoints(gray,lambdas,datas,8,true);
    vector<Mat> textureStack = createTextureStack( gray, datas );
    vector<Mat> textureBlobStack = filterStack( textureStack, blobKernels );
    Mat TextureSaliency = sumstack(textureBlobStack);
    TextureSaliency = TextureSaliency/5;
    //----------------------------end Texture--------------------------------------

//    int rowss = data[0].C_array[0].rows;
//    int colss = data[0].C_array[0].cols;
//    Mat_<double> result( Mat::zeros(rowss, colss, CV_32F));

//    for(int i = 0; i<data[0].C_array.size(); i++)
//    {
//        result = result + data[0].C_array[i];
//    }

//    Mat edges = Mat::zeros(rowss,colss, CV_32FC1);
////    result= result/50;
//    result.convertTo(edges, CV_8UC1);
//    double minVal1, maxVal1;
//    Point min_loc1, max_loc1;
//    minMaxLoc(edges, &minVal1, &maxVal1, &min_loc1, &max_loc1);
//    edges = edges/maxVal1;
//    Mat edge1 = edges;
//    threshold( edge1, edge1, 0, 255, THRESH_BINARY | THRESH_OTSU );
////    double otus = threshold( gray, gray, 0, 255, THRESH_BINARY | THRESH_OTSU );
////    cout<<"thresh = "<<otus<<"\n";
////    waitKey(50);
////    imshow("edges",edge1);

//    int ro = edges.rows;
//    int co = edges.cols;

//    Mat cont = Mat::zeros(ro, co, CV_32F);
//    Mat cont1 = Mat::zeros(ro, co, CV_32F);
//    int largest_area=0;
//    int largest_contour_index=0;
//    Rect bounding_rect;
//    Mat contourOutput;
//    edge1.convertTo(contourOutput, CV_8UC1);
//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;

//    findContours( edge1, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image

//    for( int i = 0; i< contours.size(); i++ )
//    {
//        double a=contourArea( contours[i],false);  //  Find the area of contour
//        if(a>largest_area)
//        {
//            largest_area=a;
//            largest_contour_index=i;                //Store the index of largest contour
//            bounding_rect=boundingRect(contours[i]);
//        }
//    }

//    Scalar color( 255,255,255);
//    drawContours( cont, contours, largest_contour_index, color, 1, 8, hierarchy ); // Draw the largest contour using previously stored index.
////    waitKey(50);
////    imshow("contours",cont);

//    int dilation_size = 2;
//    Mat element = getStructuringElement(2, Size(2 * dilation_size + 1, 2 * dilation_size + 1),
//    Point(dilation_size, dilation_size));     // dilation_type = MORPH_ELLIPSE
//    dilate(cont, cont1, element, Point(-1,-1),1);

//    Mat res = Mat::zeros(ro, co, CV_32F);
//    Mat src = Mat::zeros(ro, co, CV_32F);

    Mat src = TextureSaliency;
//    res = weightedfiltersum(src, edges, cont, 500);

//    Mat src1 = res;
    double minVal, maxVal;
    Point min_loc, max_loc;
    minMaxLoc(src, &minVal, &maxVal, &min_loc, &max_loc);
    Mat res1 = src;
    res1 = res1/maxVal;
    res1 = res1*255;

//    string fn = dirp->d_name;
//    mystring.replace(dirp->d_name.find(".bmp"),4,".jpg");
//    cout<<"name = " <<getext(dirp->d_name)<<"\n";
//    cout<<"name = " <<strrchr(dirp->d_name)<<"\n";

//    fn.replace(fn.find(".jpg"),4,".png");

//    std::stringstream ss;
//    string type = ".jpg";
//    ss <<"/home/saikrishna/Desktop/Natural_dataset/syn_tex/"<<dirp->d_name ;

////     ss <<"/home/saikrishna/Desktop/synthatic dataset/tex/"<<im+1<<type ;
//    cv::imwrite(ss.str(),res1);

    size_t lastindex = filepath.find_last_of(".");
    string rawname = filepath.substr(0, lastindex);

//    cout<<"hi = "<<filepath<<"\n";
//    cout<<"ji = "<<rawname<<"\n";

    std::stringstream ss;
    string type = ".png";
    ss <<rawname<<type ;
    cv::imwrite(ss.str(),res1);

    im = im+1;
    cout<<"im = "<<im<<"\n";
  }
  closedir(dp);
  waitKey(0);

  return 0;
}

    Mat weightedfiltersum(Mat inputsaliency, Mat inputedge, Mat inputcont, int numberoftimes)
    {
     int r = inputsaliency.rows;
     int c = inputsaliency.cols;
     Mat output =  Mat::zeros(r,c, CV_32F);

     for(int k=0; k<numberoftimes;k++)
     {
     for(int i=1; i<r-1; i++)
         for(int j=1; j<c-1; j++)
         {
             float i1,i2,i3,i4,i5,i6,i7,i8;
             float w1,w2,w3,w4,w5,w6,w7,w8;
             float l1,l2,l3,l4,l5,l6,l7,l8;

             i1 = inputsaliency.at<float>(i-1,j-1);
             i2 = inputsaliency.at<float>(i-1,j);
             i3 = inputsaliency.at<float>(i-1,j+1);
             i4 = inputsaliency.at<float>(i,j+1);
             i5 = inputsaliency.at<float>(i+1,j+1);
             i6 = inputsaliency.at<float>(i+1,j);
             i7 = inputsaliency.at<float>(i+1,j-1);
             i8 = inputsaliency.at<float>(i,j-1);

             w1 = 1-inputcont.at<float>(i-1,j-1)/255;
             w2 = 1-inputcont.at<float>(i-1,j)/255;
             w3 = 1-inputcont.at<float>(i-1,j+1)/255;
             w4 = 1-inputcont.at<float>(i,j+1)/255;
             w5 = 1-inputcont.at<float>(i+1,j+1)/255;
             w6 = 1-inputcont.at<float>(i+1,j)/255;
             w7 = 1-inputcont.at<float>(i+1,j-1)/255;
             w8 = 1-inputcont.at<float>(i,j-1)/255;

             l1 = w1*i1;
             l2 = w2*i2;
             l3 = w3*i3;
             l4 = w4*i4;
             l5 = w5*i5;
             l6 = w6*i6;
             l7 = w7*i7;
             l8 = w8*i8;

             output.at<float>(i,j) = (l1+l2+l3+l4+l5+l6+l7+l8)/8;
         }
     inputsaliency = output;
     }
     return output;
    }




