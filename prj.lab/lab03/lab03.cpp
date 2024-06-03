#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

cv::Mat drawHistogram (cv::Mat image) {
    cv::Mat background(256, 260, CV_8UC1, cv::Scalar(230));

    cv::Mat hist;

    const int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    //указатель на изображение, количество изображений, индекс канала, маска, выходной массив, количество измерений, указател) на количество бинов, указатель на диапазон значений
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    //нормализация
    cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);

    //отрисовка гистограмы
    for (int i = 0; i < histSize; i++) {
        cv::rectangle(background, cv::Rect(i+2, 256-hist.at<float>(i, 0), 1, hist.at<float>(i, 0)), cv::Scalar(0), -1);
    }

    return background;
}

cv::Mat drawHistogram3Channels(cv::Mat image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat background(256, 3*260, CV_8UC1, cv::Scalar(230));

    for (int i = 0; i < 3; i++) {
        cv::Mat hist;
        cv::Mat background0(256, 260, CV_8UC1, cv::Scalar(230));

        const int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };

        //указатель на изображение, количество изображений, индекс канала, маска, выходной массив, количество измерений, указател) на количество бинов, указатель на диапазон значений
        cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

        //нормализация
        cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX);

        //отрисовка гистограмы
        for (int j = 0; j < histSize; j++) {
            cv::rectangle(background0, cv::Rect(j + 2, 256-hist.at<float>(j, 0), 1, hist.at<float>(j, 0)), cv::Scalar(0), -1);
        }

        if (i == 0) {
            cv::putText(background0, "Blue", cv::Point(124, 253), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        } else if (i == 1) {
            cv::putText(background0, "Green", cv::Point(124, 253), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        } else if (i == 2) {
            cv::putText(background0, "Red", cv::Point(124, 253), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        }

        background0.copyTo(background(cv::Rect(i*256, 0, 260, 256)));
    }

    return background;
}


cv::Mat autoContrastByQuantiles(const cv::Mat& src, double lowerQuantile, double upperQuantile) {
    CV_Assert(src.channels() == 1);
    CV_Assert(lowerQuantile < upperQuantile);

    cv::Mat flat = src.reshape(1, src.total());
    std::vector<uchar> sorted(flat.begin<uchar>(), flat.end<uchar>());
    std::sort(sorted.begin(), sorted.end());

    int lowerIndex = static_cast<int>(std::floor(lowerQuantile * sorted.size()));
    int upperIndex = static_cast<int>(std::ceil(upperQuantile * sorted.size())) - 1;

    lowerIndex = std::max(lowerIndex, 0);
    upperIndex = std::min(upperIndex, static_cast<int>(sorted.size() - 1));

    double lowerBound = sorted[lowerIndex];
    double upperBound = sorted[upperIndex];

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
            "{outputImagePath o |../prj.lab/lab01/images/lab03.jpg | output image path}"
            "{outputImagePath2 u |../prj.lab/lab01/images/lab03_histogram.jpg | output histogram image path}";

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
    cv::Mat hist1;
    cv::imshow("Original", src);
    cv::imshow("Contrasted", contrasted);
    if (src.channels() == 1) {
        cv::Mat imageClone = src.clone();
        hist1 = drawHistogram(imageClone);
        cv::Mat baseMat(256, src.channels() * 260, CV_8UC1, cv::Scalar(0));
        hist1.copyTo(baseMat(cv::Rect(0, 0, hist1.cols, hist1.rows)));
        cv::imshow("Histogram", baseMat);

    } else if (src.channels() == 3) {
        cv::Mat imageClone = src.clone();
        hist1 = drawHistogram3Channels(imageClone);
        cv::Mat baseMat(256, src.channels() * 260, CV_8UC1, cv::Scalar(0));
        hist1.copyTo(baseMat(cv::Rect(0, 0, hist1.cols, hist1.rows)));
        cv::imshow("Histogram", baseMat);

    }



    std::string outputImagePath = parser.get<std::string>("outputImagePath");
    bool saved = cv::imwrite(outputImagePath, contrasted);
    if (!saved) {
        std::cerr << "Error: Could not save the contrasted image." << std::endl;
        return -1;
    }

    std::cout << "Contrasted image saved as: " << outputImagePath << std::endl;

    std::string outputImagePath2 = parser.get<std::string>("outputImagePath2");
    bool saved2 = cv::imwrite(outputImagePath2, hist1);
    if (!saved2) {
        std::cerr << "Error: Could not save the histogram image." << std::endl;
        return -1;
    }

    std::cout << "Histogram image saved as: " << outputImagePath2 << std::endl;

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
