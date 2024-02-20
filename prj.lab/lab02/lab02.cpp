//
// Created by Константин Гончаров on 20.02.2024.
//
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

int main() {
    const int w = 209;
    uint8_t color_low = 20;
    uint8_t color_mid = 127;
    uint8_t color_hig = 255 - color_low;
    cv::Mat m = cv::Mat1b(256, 256);
    m = color_low;
    const cv::Rect2i rc = { (256 - w) / 2, (256 - w) / 2, w, w };
    cv::rectangle(m, rc, cv::Scalar(color_mid), cv::FILLED);
    cv::circle(m, {128, 128}, 83, cv::Scalar(color_hig), cv::FILLED);

    auto mask_low = m == color_low;
    auto mask_mid = m == color_mid;
    auto mask_hig = m == color_hig;

    std::cout
            << "low:" << cv::countNonZero(mask_low)
            << " mid:" << cv::countNonZero(mask_mid)
            << " hig:" << cv::countNonZero(mask_hig)
            << std::endl;

    cv::imshow("mask_low", mask_low);
    cv::imshow("mask_mid", mask_mid);
    cv::imshow("mask_hig", mask_hig);
    cv::imshow("example", m);

    cv::Mat noise(256, 256, CV_8SC1);
    cv::randn(noise, 0, 15);
    m += noise;

    cv::imshow("noised", m);
    cv::imwrite("lab02.png", m);

    cv::waitKey(0);
}