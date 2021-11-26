#pragma once

#include <opencv2/opencv.hpp>

using namespace std;


class MY_GFTT{
    public:

        void goodFeaturesToTrack(cv::InputArray _image, cv::OutputArray _corners, int maxCorners, 
                                    double qualityLevel, double minDistance, cv::InputArray _mask = cv::noArray(), int rejectBorder = 1,
                                    int blockSize = 3, int gradientSize = 3,
                                    bool useHarrisDetector = false, double harrisK = 0.04);

        void re_extractor(cv::InputArray _image, cv::InputArray &_mask, double minDistance, int maxCorners,
                            int &rejectBorder, vector<cv::Point2f> &_corners, std::vector<float> &_scores,
                            cv::InputArray _eigenMat, cv::InputArray _dilateMat);
};