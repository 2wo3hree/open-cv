//
// Created by Константин Гончаров on 12.04.2024.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

cv::Mat autoContrastByQuantiles(const cv::Mat& src, double lowerQuantile, double upperQuantile) {
    CV_Assert(src.channels() == 1);
    CV_Assert(lowerQuantile < upperQuantile);

    cv::Mat flat = src.reshape(1, src.total());

    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);

    int lowerIndex = static_cast<int>(std::floor(lowerQuantile * sorted.total()));
    int upperIndex = static_cast<int>(std::ceil(upperQuantile * sorted.total())) - 1;

    lowerIndex = std::max(lowerIndex, 0);
    upperIndex = std::min(upperIndex, sorted.rows * sorted.cols - 1);

    double lowerBound = sorted.at<uchar>(lowerIndex);
    double upperBound = sorted.at<uchar>(upperIndex);

    cv::Mat contrasted = src.clone();

    for (int i = 0; i < contrasted.rows; ++i) {
        for (int j = 0; j < contrasted.cols; ++j) {
            double pixelValue = static_cast<double>(contrasted.at<uchar>(i, j));
            pixelValue = 255.0 * (pixelValue - lowerBound) / (upperBound - lowerBound);
            pixelValue = std::min(std::max(pixelValue, 0.0), 255.0);
            contrasted.at<uchar>(i, j) = static_cast<uchar>(pixelValue);
        }
    }

    return contrasted;
}

cv::Mat autoContrastColorPerChannel(const cv::Mat& src, double lowerQuantile, double upperQuantile) {
    CV_Assert(lowerQuantile < upperQuantile);

    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    for (int i = 0; i < src.channels(); ++i) {
        channels[i] = autoContrastByQuantiles(channels[i], lowerQuantile, upperQuantile);
    }

    cv::Mat contrasted;
    cv::merge(channels, contrasted);

    return contrasted;
}

cv::Mat autoContrastColorCombined(const cv::Mat& src, double lowerQuantile, double upperQuantile) {
    CV_Assert(lowerQuantile < upperQuantile);

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);

    hsvChannels[2] = autoContrastByQuantiles(hsvChannels[2], lowerQuantile, upperQuantile);

    cv::Mat contrastedHSV;
    cv::merge(hsvChannels, contrastedHSV);

    cv::Mat contrastedBGR;
    cv::cvtColor(contrastedHSV, contrastedBGR, cv::COLOR_HSV2BGR);

    return contrastedBGR;
}

bool isImageColor(const cv::Mat& image) {
    CV_Assert(!image.empty());

    if (image.channels() == 1) {
        return false;
    }

    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    cv::Mat diff1, diff2;
    cv::absdiff(channels[0], channels[1], diff1);
    cv::absdiff(channels[1], channels[2], diff2);

    double maxVal1, maxVal2;
    cv::minMaxLoc(diff1, nullptr, &maxVal1);
    cv::minMaxLoc(diff2, nullptr, &maxVal2);

    return maxVal1 > 10 || maxVal2 > 10;
}



int main(int argc, char** argv) {
    const char* keys =
            "{help h usage ? |      | print this message }"
            "{@image         |<none>| image to process }"
            "{lowerQuantile l| 0.02 | lower quantile for auto contrast adjustment }"
            "{autoContrastMethod m | combined | auto contrast method (perChannel or combined) }"
            "{outputImagePath o |../prj.lab/lab01/images/lab03.jpg | output image path}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Auto Contrast Adjustment");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string imagePath = parser.get<std::string>("@image");
    if (imagePath.empty()) {
        std::cerr << "Error: No image path provided." << std::endl;
        return -1;
    }

    double lowerQuantile = parser.get<double>("lowerQuantile");
    double upperQuantile = 1.0 - lowerQuantile;
    std::string method = parser.get<std::string>("autoContrastMethod");

    cv::Mat src = cv::imread(imagePath, cv::IMREAD_ANYCOLOR);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    cv::Mat contrasted;
    if (isImageColor(src) && method == "combined") {
        contrasted = autoContrastColorCombined(src, lowerQuantile, upperQuantile);
    } else if (isImageColor(src) && method == "perChannel") {
        contrasted = autoContrastColorPerChannel(src, lowerQuantile, upperQuantile);
    } else {
        if (src.channels() > 1) {
            cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        }
        contrasted = autoContrastByQuantiles(src, lowerQuantile, upperQuantile);
    }

    cv::imshow("Original", src);
    cv::imshow("Contrasted", contrasted);


    std::string outputImagePath = "../prj.lab/lab01/images/lab03.jpg";
    bool saved = cv::imwrite(outputImagePath, contrasted);
    if (!saved) {
        std::cerr << "Error: Could not save the contrasted image." << std::endl;
        return -1;
    }

    std::cout << "Contrasted image saved as: " << outputImagePath << std::endl;

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}
