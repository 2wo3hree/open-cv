//
// Created by Константин Гончаров on 20.03.2025.
//

#include "saturation.h"
#include <opencv2/opencv.hpp>
#include <cmath>

cv::Mat computeSaturationMap(const cv::Mat &img) {
    cv::Mat satMap(img.rows, img.cols, CV_32FC1);
    for (int y = 0; y < img.rows; y++){
        for (int x = 0; x < img.cols; x++){
            cv::Vec3f p = img.at<cv::Vec3f>(y, x);
            float m = (p[0] + p[1] + p[2]) / 3.0f;
            float s = std::sqrt((p[0] - m) * (p[0] - m) +
                                (p[1] - m) * (p[1] - m) +
                                (p[2] - m) * (p[2] - m));
            satMap.at<float>(y, x) = s;
        }
    }
    return satMap;
}

cv::Mat visualizeSaturation(const cv::Mat &satMap) {
    double minVal, maxVal;
    cv::minMaxLoc(satMap, &minVal, &maxVal);
    cv::Mat normSat;
    // Нормализуем карту в диапазон [0,1]
    satMap.convertTo(normSat, CV_32F, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));
    cv::Mat satGray;
    // Преобразуем в 8-битное изображение
    normSat.convertTo(satGray, CV_8U, 255.0);
    return satGray;
}

double computeMSE(const cv::Mat &saturation, const cv::Mat &mask) {
    double sumSq = 0.0;
    int count = 0;
    for (int y = 0; y < saturation.rows; y++){
        const float* satRow = saturation.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < saturation.cols; x++){
            if (maskRow[x] == 255) {
                double val = satRow[x];
                sumSq += val * val;
                count++;
            }
        }
    }
    if (count == 0) return 0.0;
    return sumSq / count;
}

