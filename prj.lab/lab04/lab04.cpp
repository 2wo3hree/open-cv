#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <map>
#include <cmath>

cv::Mat outputImage(std::string name, cv::Mat& image, bool show = true) {
    cv::Mat image_clone = image.clone();
    double minVal, maxVal;

    cv::minMaxLoc(image_clone, &minVal, &maxVal);
    double a = 255.0 / (maxVal - minVal);
    double b = (-minVal * 255.0 / (maxVal - minVal));
    image_clone.convertTo(image_clone, CV_8UC1, a, b);

    if (show) {
        cv::imshow(name, image_clone);
    }
    cv::imwrite("../prj.lab/lab01/images/lab04_gg.jpg", image_clone);
    return image_clone;
}

void addNoise(cv::Mat& image, float* noise) {
    std::random_device device;
    std::mt19937 generator(device());

    double min = 0.0;
    double max = 1.0;
    double stddev = static_cast<double>(noise[0]);
    double shift = static_cast<double>(noise[1]);

    std::uniform_real_distribution<double> distribution(min, max);

    int rows = image.rows;
    int cols = image.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double u = distribution(generator);
            double v = distribution(generator);
            double rand_numb_box_muller = sqrt(-2 * log(v)) * cos(2 * M_PI * u);

            image.at<uchar>(i, j) = cv::saturate_cast<uchar>(image.at<uchar>(i, j) + (stddev * rand_numb_box_muller) + shift);
        }
    }
}

cv::Mat testImage (int color_background, int nobj, float* radius, std::vector<float>& contrast, float* blur, std::string file_name) {
    cv::FileStorage js( file_name + ".json", cv::FileStorage::WRITE);

    int size_contrast = contrast.size();
    int indentation = 20;
    int size_radius = 2;
    float step = (radius[size_radius - 1] - radius[0]) / nobj;
    int w = indentation;
    for (int i = 0; i < nobj; i++) {
        w += static_cast<int>(2*(radius[0] + i*step) + indentation + 0.5);
    }

    int h = indentation;
    for (int i = 0; i < size_contrast; i++) {
        h += static_cast<int>(2*(radius[size_radius - 1]) + indentation + 0.5);
    }
    js << "h" << h;
    js << "w" << w;
    js << "array" << "[";
    cv::Mat image(h, w, CV_8UC1, cv::Scalar(color_background));
    float sum_distance_h = 0;
    for (int i = 0; i < size_contrast; i++) {
        float radius_max = radius[size_radius - 1];
        sum_distance_h += indentation + radius_max;
        float sum_distance_w = 0;
        for (int j = 0; j < nobj; j++) {
            js << "{";

            float radius_circlej = radius[0] + step*j;
            sum_distance_w += indentation + radius_circlej;
            cv::circle(image, cv::Point(sum_distance_w, sum_distance_h), radius_circlej, cv::Scalar(contrast[i] * 255 + color_background), -1, cv::LINE_8);
            js << "circle" << "[" << sum_distance_w  <<  sum_distance_h  << radius_circlej << "]";
            js << "contrast" << contrast[i];
            js << "quality" << std::string();
            js << "}";

            sum_distance_w += radius_circlej;
        }

        sum_distance_h += radius_max;
    }
    cv::blur(image, image, cv::Size(blur[0], blur[1]));
    js.release();
    cv::imwrite(file_name + ".jpg", image);
    return image;
}

void detectBlobsLoG(std::vector<cv::Point>& centers, std::vector<float>& radii, cv::Mat& image, float min_sigma, float max_sigma, float step_sigma, int pyr_depth, float min_radius, float max_radius, int threshold) {
    std::vector<cv::Mat> pyramid;
    pyramid.push_back(image.clone());

    for (int i = 1; i < pyr_depth; i++) {
        cv::Mat pyr_prev = pyramid[i - 1];
        cv::Mat pyr_cur;
        cv::pyrDown(pyr_prev, pyr_cur);
        pyramid.push_back(pyr_cur);
    }

    struct LoG_kernels {
        cv::Mat kernel;
        float sigmak;
    };

    std::vector<LoG_kernels> kernels;
    for (float sigma = min_sigma; sigma <= max_sigma; sigma += step_sigma) {
        int kernel_size = static_cast<int>(6*sigma) / 2 - 1;
        if (kernel_size % 2 == 0) {
            kernel_size += 1;
        }

        std::cout << kernel_size << std::endl;
        cv::Mat kernel1D = cv::getGaussianKernel(kernel_size, sigma, CV_32F);
        cv::Mat kernel2D = kernel1D * kernel1D.t();
        cv::Mat laplacian;
        cv::Laplacian(kernel2D, laplacian, CV_32F, kernel_size);

        LoG_kernels kernel_cur;
        kernel_cur.kernel = laplacian;
        kernel_cur.sigmak = sigma;
        kernels.push_back(kernel_cur);
    }

    struct Blob {
        cv::Point center;
        float radius;
        float response;
        int flag;
    };

    std::vector<Blob> blobList;

    for (int i = 0; i < pyr_depth; i++) {
        for (float k = 0; k < kernels.size(); k++) {
            cv::Mat laplacian = kernels[k].kernel;
            float sigma = kernels[k].sigmak;
            float radius = sigma * sqrt(2.0);

            cv::Mat image_clone;

            cv::filter2D(pyramid[i], image_clone, CV_32F, laplacian);

            outputImage("gg", image_clone);


            cv::Mat erode;
            cv::Mat image_compare;
            cv::erode(image_clone, erode, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*radius + 4, 2*radius + 4)));
            cv::compare(image_clone, erode, image_compare, cv::CMP_EQ);
            std::vector<cv::Point> idxpoints;


            cv::Mat image_temp;
            cv::Mat stats;
            cv::Mat centroids;
            int numLabels = cv::connectedComponentsWithStats(image_compare, image_temp, stats, centroids);

            for (int i = 1; i < numLabels; i++) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > threshold) {
                    std::cout << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;
                    cv::Mat componentMask = image_temp == i;
                    image_compare.setTo(cv::Scalar::all(1), componentMask);
                }
            }

            cv::findNonZero(image_compare, idxpoints);

            for (int j = 0; j < idxpoints.size(); j++) {
                float temp_radius = pow(2, i) * radius;
                float normalize_response = sigma*sigma*image_clone.at<float>(idxpoints[j]);
                Blob blob;
                blob.center = pow(2, i) * idxpoints[j];
                blob.radius = temp_radius;
                blob.response = normalize_response;
                blob.flag = 1;
                blobList.push_back(blob);
            }
        }
    }

    cv::Mat clear = image.clone();

    cv::cvtColor(image, clear, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < blobList.size(); i++) {
        if (blobList[i].flag == 1) {
            cv::circle(clear, blobList[i].center, blobList[i].radius, cv::Scalar(0, 0, 255), 3);
        }
    }

    cv::imshow("image_kddk", clear);
    cv::imwrite("../prj.lab/lab01/images/lab04_image_kddk.jpg", clear);

    for (int i = 0; i < blobList.size(); i++) {
        for (int j = i + 1; j < blobList.size(); j++) {
            int xi = blobList[i].center.x;
            int yi = blobList[i].center.y;
            int xj = blobList[j].center.x;
            int yj = blobList[j].center.y;
            float dist = sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));

            if (dist < (blobList[i].radius + blobList[j].radius)) {
                if (blobList[i].response < blobList[j].response) {
                    blobList[j].flag = 0;
                } else {
                    blobList[i].flag = 0;
                }
            }
        }
    }

    for (int i = 0; i < blobList.size(); i++) {
        if (blobList[i].flag == 1) {
            centers.push_back(blobList[i].center);
            radii.push_back(blobList[i].radius);
        }
    }
}

cv::Mat thresholdBinary(const cv::Mat& image, int thresh) {
    cv::Mat binary;
    cv::threshold(image, binary, thresh, 255, cv::THRESH_BINARY);
    return binary;
}

cv::Mat adaptiveThresholdBinary(const cv::Mat& image, int blockSize, int C) {
    cv::Mat binary;
    cv::adaptiveThreshold(image, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, C);
    return binary;
}


void displayBinaryImages(cv::Mat& image) {
    int binary_thresh = 128;
    int adaptive_blockSize = 11;
    int adaptive_C = 2;



    while (true) {
        cv::Mat binary = thresholdBinary(image, binary_thresh);
        cv::Mat adaptive = adaptiveThresholdBinary(image, adaptive_blockSize, adaptive_C);

        cv::imshow("Binary Image", binary);
        cv::imwrite("../prj.lab/lab01/images/lab04_binary.jpg", binary);
        cv::imshow("Adaptive Binary Image", adaptive);
        cv::imwrite("../prj.lab/lab01/images/lab04_adaptive_binary.jpg", adaptive);

        int key = cv::waitKey(1);
        if (key == 27) break; // ESC key to exit
    }
}

float IoU(cv::Rect& truth, cv::Rect& detection) {
    int xA = std::max(truth.x, detection.x);
    int yA = std::max(truth.y, detection.y);
    int xB = std::min(truth.x + truth.width, detection.x + detection.width);
    int yB = std::min(truth.y + truth.height, detection.y + detection.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int boxAArea = truth.width * truth.height;
    int boxBArea = detection.width * detection.height;

    float iou = float(interArea) / float(boxAArea + boxBArea - interArea);

    return iou;
}

void evaluateDetections(std::vector<cv::Rect>& truths, std::vector<cv::Rect>& detections, float iou_threshold) {
    int TP = 0, FP = 0, FN = 0;

    for (size_t i = 0; i < truths.size(); i++) {
        bool detected = false;
        for (size_t j = 0; j < detections.size(); j++) {
            if (IoU(truths[i], detections[j]) > iou_threshold) {
                TP++;
                detected = true;
                break;
            }
        }
        if (!detected) FN++;
    }

    FP = detections.size() - TP;

    std::cout << "TP: " << TP << ", FP: " << FP << ", FN: " << FN << std::endl;
}

int main () {
    int color_background = 30;
    int nobj = 10;
    float radius[2] = { 8.0, 150.0 };
    std::vector<float> contrast{ 0.4, 0.45, 0.57, 0.6, 0.7};
    float blur[2] = {5.0, 7.2};
    float noise[2] = {6, 2};

    cv::Mat image = testImage(color_background, nobj, radius, contrast, blur, "../prj.lab/lab01/images/lab04");
    addNoise(image, noise);

    cv::imshow("image", image);
    cv::imwrite("../prj.lab/lab01/images/lab04_image.jpg", image);

    std::vector<cv::Point> centers;
    std::vector<float> radii;
    detectBlobsLoG(centers, radii, image, 6.5, 8, 0.4, 1, 11.0, 70.0, 1);

    cv::Mat clear;

    cv::cvtColor(image, clear, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < centers.size(); i++) {
        cv::circle(clear, centers[i], radii[i], cv::Scalar(0, 0, 255), 3);
    }
    cv::imshow("image_clone", clear);
    cv::imwrite("../prj.lab/lab01/images/lab04_image_clone.jpg", clear);

    displayBinaryImages(image);
    std::cout<<"hello";

    std::vector<cv::Rect> groundTruths = { cv::Rect(50, 50, 100, 100) }; // пример значений ground truth
    std::vector<cv::Rect> detections = { cv::Rect(55, 55, 90, 90) }; // пример значений детекций
//    std::cout<<"hello";
    evaluateDetections(groundTruths, detections, 0.5);

    cv::waitKey(0);
    return 0;
}
