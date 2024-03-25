#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iterator>

#include "ORBExtractor.h"
#include <iostream>

using namespace std;
using namespace cv;
using namespace Goudan;

int main(int argc, char** argv){
    if ( argc != 2 )
    {
        cout << "Please input ./orb_extractor image" << endl;
        return -1;
    }
    // 测试提取特征值

    // -----grid based orb extractor
    int nfeatures = 1000;
    int nlevels = 8;
    float fscaleFactor = 1.2;
    float fIniThFAST = 20;
    float fMinThFAST = 7;
    // default parameters printf
    cout << "Default parameters are : " << endl;
    cout << "nfeature : " << nfeatures << ", nlevels : " << nlevels << ", fscaleFactor : " << fscaleFactor << endl;
    cout << "fIniThFAST : " << fIniThFAST << ", fMinThFAST : " << fMinThFAST << endl;

    cout << "Read image.." <<endl;
    Mat image = imread( argv[1], CV_LOAD_IMAGE_UNCHANGED );
    Mat grayImg, mask;
    cvtColor( image, grayImg, CV_RGB2GRAY );
    imshow( "grayImg", grayImg );
    cout << "Read image finish"<<endl;

    // ORB 提取器初始化
    cout << "ORBextractor initialize..." << endl;
    ORBExtractor* pORBextractor;
    pORBextractor = new ORBExtractor( nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST );
    cout << "ORBextractor initialize finished!" << endl;

    cout << "Extract orb descriptors..." << endl;
    Mat desc;
    vector<KeyPoint> kps;
    (*pORBextractor)( grayImg, mask, kps, desc);
    cout << "Extract orb descriptors finished!" << endl;
    cout << "The number of keypoints are = " << kps.size() << endl;

    // draw keypoints in output image
    Mat outImg;
    drawKeypoints( grayImg, kps, outImg, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow( "GridOrbKpsImg", outImg );
    cout << "Finished! Congratulations!" << endl;


    // ----original orb extractor for comparation
    // orb initialization 
    cout << "Using original orb extractor to extract orb descriptors for comparation." << endl;
    Ptr<ORB> orb_ = ORB::create( 1000, 1.2f, 8, 19 );

    // orb extract
    vector<KeyPoint> orb_kps;
    Mat orb_desc;
    orb_->detectAndCompute( grayImg, mask, orb_kps, orb_desc );

    // draw keypoints in output image
    Mat orbOutImg;
    drawKeypoints( grayImg, orb_kps, orbOutImg, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow( "OrbKpsImg", orbOutImg );
    waitKey(0);



/*  验证分裂节点 函数
    Goudan::ExtractorNode* node1 = new Goudan::ExtractorNode();
    node1->UL = cv::Point2i(0, 0);
    node1->UR = cv::Point2i(image.cols, 0);
    node1->BL = cv::Point2i(0, image.rows);
    node1->BR = cv::Point2i(image.cols, image.rows);

    Goudan::ExtractorNode node_d1, node_d2, node_d3, node_d4;
    node1->DivideNode(node_d1, node_d2, node_d3, node_d4);

    // 在图上画出来矩形？
    cv::rectangle(image, cv::Rect2i(node_d1.UL, node_d1.BR), cv::Scalar(255, 0, 0, 255));
    cv::rectangle(image, cv::Rect2i(node_d2.UL, node_d2.BR), cv::Scalar(0, 255, 0, 255));
    cv::rectangle(image, cv::Rect2i(node_d3.UL, node_d3.BR), cv::Scalar(255, 255, 0, 255));
    cv::rectangle(image, cv::Rect2i(node_d4.UL, node_d4.BR), cv::Scalar(0, 0, 255, 255));

    imshow("divide", image);
    waitKey(0);
*/


    return 0;
}

