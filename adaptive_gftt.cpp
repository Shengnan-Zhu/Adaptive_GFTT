#include "gftt.h"
#include <iostream>

int main(){
    cv::Mat image = cv::imread("/home/shane/Adaptive_GFTT/1.png", 0);
    if(image.empty()){
        std::cerr << "load image failed ..." << std::endl;
        return -1;
    }
    // cv::imshow("src", image);
    cv::Mat img_show1, img_show2;
    image.copyTo(img_show1);
    image.copyTo(img_show2);
    cv::cvtColor(image, img_show1, CV_GRAY2BGR);
    cv::cvtColor(image, img_show2, CV_GRAY2BGR);
    
    std::vector<cv::Point2f> pts1, pts2;
    int max_cnt = 200;
    int min_dist = 30;
    double quality_level = 0.01;

    cv::goodFeaturesToTrack(image, pts1, max_cnt, quality_level, min_dist);
    printf("oencv gftt extract %d features \n", int(pts1.size()));
    for(auto p : pts1){
        cv::circle(img_show1, p, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("oencv gftt", img_show1);

    MY_GFTT gftt;
    gftt.goodFeaturesToTrack(image, pts2, max_cnt, quality_level, min_dist, cv::Mat(), 1);
    printf("my gftt extract %d features \n", int(pts2.size()));
    for(auto p : pts2){
        cv::circle(img_show2, p, 2, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("my gftt", img_show2);
    


    cv::waitKey(0);
    return 0;
}