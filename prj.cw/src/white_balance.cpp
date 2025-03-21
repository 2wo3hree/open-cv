//
// Created by Константин Гончаров on 20.03.2025.
//

#include "white_balance.h"
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>


float sRGBToLin(float c) {
    return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

float linToSRGB(float c) {
    return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * std::pow(c, 1.0f/2.4f) - 0.055f);
}

cv::Mat convertToLinRGB(const cv::Mat &img) {
    cv::Mat linImg = img.clone(); // Копируем изображение
    for (int y = 0; y < linImg.rows; y++){
        for (int x = 0; x < linImg.cols; x++){
            cv::Vec3f &px = linImg.at<cv::Vec3f>(y, x);
            px[0] = sRGBToLin(px[0]);
            px[1] = sRGBToLin(px[1]);
            px[2] = sRGBToLin(px[2]);
        }
    }
    return linImg;
}

cv::Mat convertToSRGB(const cv::Mat &linImg) {
    cv::Mat sImg = linImg.clone();
    for (int y = 0; y < sImg.rows; y++){
        for (int x = 0; x < sImg.cols; x++){
            cv::Vec3f &px = sImg.at<cv::Vec3f>(y, x);
            px[0] = linToSRGB(px[0]);
            px[1] = linToSRGB(px[1]);
            px[2] = linToSRGB(px[2]);
        }
    }
    return sImg;
}

/* Вспомогательные функции, используемые внутри whiteBalanceCorrection */
static cv::Vec3f computeMean(const cv::Mat &img) {
    cv::Scalar m = cv::mean(img);
    return cv::Vec3f(m[0], m[1], m[2]);
}

static cv::Mat computeCovariance(const cv::Mat &img, const cv::Vec3f &meanColor) {
    cv::Mat cov = cv::Mat::zeros(3, 3, CV_32F);
    int total = img.rows * img.cols;
    for (int y = 0; y < img.rows; y++){
        for (int x = 0; x < img.cols; x++){
            cv::Vec3f px = img.at<cv::Vec3f>(y, x);
            cv::Vec3f d = px - meanColor;
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j++){
                    cov.at<float>(i, j) += d[i] * d[j];
                }
            }
        }
    }
    cov /= float(total);
    return cov;
}

cv::Mat whiteBalanceCorrection(const cv::Mat &linImg) {
    cv::Vec3f meanColor = computeMean(linImg);
    cv::Mat cov = computeCovariance(linImg, meanColor);

    cv::Mat eigenVals, eigenVects;
    cv::eigen(cov, eigenVals, eigenVects);
    cv::Vec3f axis(eigenVects.at<float>(0, 0),
                   eigenVects.at<float>(0, 1),
                   eigenVects.at<float>(0, 2));

    cv::Vec3f p0(0.5f, 0.5f, 0.5f);
    float sumP0 = p0[0] + p0[1] + p0[2];
    float sumMean = meanColor[0] + meanColor[1] + meanColor[2];
    float sumAxis = axis[0] + axis[1] + axis[2];
    float t = 0.f;
    if (std::fabs(sumAxis) > 1e-6f) {
        t = (sumP0 - sumMean) / sumAxis;
    }
    cv::Vec3f pGray = meanColor + axis * t;

    cv::Vec3f k;
    for (int i = 0; i < 3; i++){
        k[i] = (std::fabs(pGray[i]) < 1e-6f) ? 1.f : (p0[i] / pGray[i]);
    }

    cv::Mat corrected = linImg.clone();
    for (int y = 0; y < linImg.rows; y++){
        for (int x = 0; x < linImg.cols; x++){
            cv::Vec3f px = linImg.at<cv::Vec3f>(y, x);
            cv::Vec3f nx;
            for (int c = 0; c < 3; c++){
                nx[c] = k[c] * (px[c] - pGray[c]) + p0[c];
                nx[c] = std::min(std::max(nx[c], 0.f), 1.f);
            }
            corrected.at<cv::Vec3f>(y, x) = nx;
        }
    }
    return corrected;
}
