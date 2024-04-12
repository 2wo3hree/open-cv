//
// Created by Константин Гончаров on 20.02.2024.
//
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include <cmath>



cv::Mat generateTestImage (int intensive1, int intensive2, int intensive3) {
    cv::Mat image (256, 256, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image, cv::Rect(0, 0, 256, 256), cv::Scalar(intensive1), -1);
    cv::rectangle(image, cv::Rect((256 - 209) / 2, (256 - 209) / 2, 209, 209), cv::Scalar(intensive2), -1);
    cv::circle(image, cv::Point(256/2, 256/2), 83, cv::Scalar(intensive3), -1);
    return image;
}

cv::Mat drawHistogram (cv::Mat image) {
    cv::Mat background(256, 256, CV_8UC1, cv::Scalar(230));
    cv::Mat hist;
    const int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);
    for (int i = 0; i < histSize; i++) {
        cv::rectangle(background, cv::Rect(i, 256 - hist.at<float>(i, 0), 1, hist.at<float>(i, 0)), cv::Scalar(0), -1);
    }

    return background;
}

cv::Mat addNoise(cv::Mat& image, double stddev) {
    cv::Mat cloneImage = image.clone();

    int rows = cloneImage.rows;
    int cols = cloneImage.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double u = (double)rand() / RAND_MAX;
            double v = (double)rand() / RAND_MAX;
            double rand_numb_box_muller = sqrt(-2 * log(v)) * cos(2 * M_PI * u);
            cloneImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(cloneImage.at<uchar>(i, j) + (stddev * rand_numb_box_muller));
        }
    }

    return cloneImage;
}


int main() {
    int intensive[4][3] = {{0, 127, 255}, {20, 127, 235}, {55, 127, 200}, {90, 127, 165}};
    int noise[3] = {3, 7, 15};
    cv::Mat baseMat = cv::Mat::zeros(256 * 8 , 256 * 4, CV_8UC1);

    for (int i = 0; i < sizeof(intensive) / sizeof(intensive[0]); ++i) {
        int y = 0;
        cv::Mat testimg = generateTestImage (intensive[i][0], intensive[i][1], intensive[i][2]);
        testimg.copyTo(baseMat(cv::Rect(i * 256, y, 256, 256)));
        y += 256;

        cv::Mat hist = drawHistogram(testimg);
        hist.copyTo(baseMat(cv::Rect(i * 256, y, 256, 256)));
        y += 256;

        for (int j = 0; j < sizeof(noise) / sizeof(noise[0]); ++j) {
            cv::Mat noiseimage = addNoise(testimg, noise[j]);
            noiseimage.copyTo(baseMat(cv::Rect(i * 256, y, 256, 256)));
            y += 256;
            cv::Mat histNoise = drawHistogram(noiseimage);
            histNoise.copyTo(baseMat(cv::Rect(i * 256, y, 256, 256)));
            y += 256;
        }
    }
    cv::imshow("Image", baseMat);
    cv::waitKey(0);

    return 0;
}
