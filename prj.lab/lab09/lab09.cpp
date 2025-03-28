//
// Created by Константин Гончаров on 14.05.2024.
//
#include <opencv2/opencv.hpp>

//double calculateQuality(const cv::Mat& original, const cv::Mat& corrected) {
//
//}
//

void greyWorldCorrection(cv::Mat& image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Scalar meanValue = cv::mean(image);
    double meanGray = (meanValue[0] + meanValue[1] + meanValue[2]) / 3.0;

    for (int i = 0; i < 3; ++i) {
        double scale = meanGray / meanValue[i];
        channels[i] = channels[i] * scale;
    }
    cv::merge(channels, image);
}

int main() {
    cv::Mat originalImage = cv::imread("../prj.lab/lab01/images/lab09.png");
//    cv::Mat image2 = cv::imread("../prj.lab/lab01/images/lab09.png");

    if (originalImage.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
    cv::Mat correctedImage = originalImage.clone();
    cv::imshow("Input image", originalImage);

    greyWorldCorrection(correctedImage);

//    double quality = calculateQuality(originalImage, correctedImage);

    cv::imshow("Grey World", correctedImage);
    cv::imwrite("../prj.lab/lab01/images/lab09_output.png", correctedImage);
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}
