//
// Created by Константин Гончаров on 12.02.2024.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>



void gammaCorrection(cv::Mat& src, cv::Mat& dst, double gamma) {
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);

    cv::LUT(src, lookUpTable, dst);
}

void generate_gradient_image(int s, int h, double gamma, const std::string& filename = "") {
    int width = s * 256;
    int height = h;

    cv::Mat gradient(height, width, CV_8UC1);
    for (int i = 0; i < width; ++i) {
        int intensity = i * 255 / width;
        gradient.col(i).setTo(intensity);
    }

    cv::Mat gamma_corrected;
    gammaCorrection(gradient, gamma_corrected, gamma);

    cv::Mat combined(height * 2, width, CV_8UC1);
    gradient.copyTo(combined(cv::Rect(0, 0, width, height)));
    gamma_corrected.copyTo(combined(cv::Rect(0, height, width, height)));

    if (filename.empty()) {
        cv::imshow("Image", combined);
        cv::waitKey(0);
    } else {
        cv::imwrite(filename, combined);
        std::cout << filename << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int s = 3;
    int h = 30;
    double gamma = 2.4;
    std::string filename;

    if (argc >= 2) {
        s = atoi(argv[1]);
    }
    if (argc >= 3) {
        h = atoi(argv[2]);
    }
    if (argc >= 4) {
        gamma = atof(argv[3]);
    }

    if (argc >= 5) {
        filename = argv[4];
    }
    generate_gradient_image(s, h, gamma, filename);

    return 0;
}
