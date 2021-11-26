#include "gftt.h"


struct greaterThanPtr
{
    bool operator () (const float * a, const float * b) const
    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

void MY_GFTT::goodFeaturesToTrack(cv::InputArray _image, cv::OutputArray _corners, int maxCorners, 
                                    double qualityLevel, double minDistance, cv::InputArray _mask, int rejectBorder,
                                    int blockSize, int gradientSize, bool useHarrisDetector, double harrisK){
    
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(_mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)) );
    
    cv::Mat image = _image.getMat(), eig, tmp;
    if(image.empty()){
        _corners.release();
        return;
    }

    if(useHarrisDetector)
        cv::cornerHarris(image, eig, blockSize, gradientSize, harrisK);
    else
        cv::cornerMinEigenVal(image, eig, blockSize, gradientSize);

    double maxVal = 0;
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, _mask);
    cv::threshold(eig, eig, maxVal* qualityLevel, 0, cv::THRESH_TOZERO);
    cv::dilate(eig, tmp, cv::Mat());

    cv::Size img_size = image.size(); 
    std::vector<const float*> tmpCorners;

    // collect list of points to features - put them into temporary image
    cv::Mat mask = _mask.getMat();

    for(int y = rejectBorder; y < img_size.height - rejectBorder; y++){
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for(int x = rejectBorder; x < img_size.width - rejectBorder; x++){
            float val = eig_data[x];
            if(val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]))
                tmpCorners.push_back(eig_data + x);
        }
    }
    std::vector<cv::Point2f> corners;
    std::vector<float> scores;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if(total == 0){
        _corners.release();
        return;
    }

    std::sort(tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());
    if (minDistance >= 1)
    {
         // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width * grid_height);

        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for(int yy = y1; yy <= y2; yy++)
            {
                for(int xx = x1; xx <= x2; xx++)
                {
                    std::vector <cv::Point2f> &m = grid[yy * grid_width + xx];

                    if(m.size())
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx * dx + dy * dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                corners.push_back(cv::Point2f((float)x, (float)y));
                scores.push_back(*tmpCorners[i]);
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for(i = 0; i < total; i++)
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            corners.push_back(cv::Point2f((float)x, (float)y));
            scores.push_back(*tmpCorners[i]);

            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }

    cv::Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);

}