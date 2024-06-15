//
// Created by Константин Гончаров on 14.06.2024.
//
#include <opencv2/opencv.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace lab08 {
    struct Params {
        std::string input;
        std::string output;
        int size;
    };

    cv::Vec3f normalize(const cv::Vec3f& v) {
        return v / cv::norm(v, cv::NORM_L2);
    }

    std::vector<cv::Point2f> projectColors(const cv::Mat& src) {
        std::vector<cv::Point2f> result{};
        const cv::Vec3f ox{ normalize({ -1.0f, 1.0f, 0.0f }) };
        const cv::Vec3f oy{ normalize({ -1.0f, -1.0f, 2.0f }) };
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                const auto& color{ src.at<cv::Vec3f>(i, j) };
                const float alpha = 1.5f / (color.dot({ 1.0f, 1.0f, 1.0f }));
                if (std::isinf(alpha)) {
                    continue;
                }
                const auto proj{ alpha * color - cv::Vec3f{ 0.5f, 0.5f, 0.5f } };
                result.emplace_back(proj.dot(ox), proj.dot(oy));
            }
        }
        return result;
    }

    std::vector<cv::Point2f> projectAllColors() {
        std::vector<cv::Point2f> result{};
        const cv::Vec3f ox{ normalize({ -1.0f, 1.0f, 0.0f }) };
        const cv::Vec3f oy{ normalize({ -1.0f, -1.0f, 2.0f }) };
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < 256; ++j) {
                for (int k = 0; k < 256; ++k) {
                    const auto &color{cv::Vec3f(i / 255.0f, j / 255.0f, k / 255.0)};
                    const float alpha = 1.5f / (color.dot({1.0f, 1.0f, 1.0f}));
                    if (std::isinf(alpha)) {
                        continue;
                    }
                    const auto proj{alpha * color - cv::Vec3f{0.5f, 0.5f, 0.5f}};
                    result.emplace_back(proj.dot(ox), proj.dot(oy));
                }
            }
        }
        return result;
    }

    cv::Mat getProj(const std::vector<cv::Point2f>& points, const int size) {
        const int height = static_cast<int>(std::sqrt(3.0) / 2 * size);
        cv::Mat result{ cv::Mat::zeros(height, size, CV_16UC1) };
        for (const auto& p : points) {
            const int x = static_cast<int>((p.x + 0.75f * std::sqrt(2.0f)) / (1.5f * std::sqrt(2.0f)) * static_cast<float>(size - 1));
            const int y = static_cast<int>((p.y + 0.25f * std::sqrt(6.0f)) / (0.75f * std::sqrt(6.0f)) * static_cast<float>(height - 1));
            ++result.at<ushort>(height - y - 1, x);
        }
        double max = 0.0;
        cv::minMaxLoc(result, nullptr, &max);
        result.convertTo(result, CV_32FC1, 1.0 / max);
        return result;
    }
}

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser{
            argc, argv,
            "{ input | | input }"
            "{ output | | output }"
            "{ size | 256 | size }"
    };
    const lab08::Params params{
            .input = parser.get<std::string>("input"),
            .output = parser.get<std::string>("output"),
            .size = parser.get<int>("size"),
    };

    cv::Mat img{ cv::imread(params.input) };
    img.convertTo(img, CV_32FC3, 1.0f / std::numeric_limits<uchar>::max());
    const auto points{ lab08::projectColors(img) };
    cv::Mat result{ lab08::getProj(points, params.size) };
    result.convertTo(result, CV_8UC1, std::numeric_limits<uchar>::max());
    cv::imwrite(params.output, result);
    return 0;
}
