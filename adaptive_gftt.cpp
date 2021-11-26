#include "gftt.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;

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
    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    auto t0 = steady_clock::now();
    cv::goodFeaturesToTrack(image, pts1, max_cnt, quality_level, min_dist, mask);
    auto t1 = steady_clock::now();
    auto time_cost = duration_cast<duration<double>>(t1 - t0);
    printf("oencv gftt extract %d features, with %f ms \n", int(pts1.size()), 1000 * time_cost.count());
    for(auto p : pts1){
        cv::circle(img_show1, p, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("oencv gftt", img_show1);

    MY_GFTT gftt;
    t0 = steady_clock::now();
    gftt.goodFeaturesToTrack(image, pts2, max_cnt, quality_level, min_dist, mask, 5);
    t1 = steady_clock::now();
    time_cost = duration_cast<duration<double>>(t1 - t0);
    printf("my gftt extract %d features, with %f ms \n", int(pts2.size()), 1000 * time_cost.count());
    for(auto p : pts2){
        cv::circle(img_show2, p, 2, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("my gftt", img_show2);
    


    cv::waitKey(0);
    return 0;
}