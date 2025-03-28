//
// Created by Константин Гончаров on 20.03.2025.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include "white_balance.h"
#include "saturation.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0] << " <исходное изображение> [<маска серой области> <маска голограммы (ориг.)> <маска голограммы (коррект.)>]\n";
        return -1;
    }

    // Обязательная загрузка исходного изображения
    cv::Mat orig = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (orig.empty()) {
        std::cerr << "Ошибка загрузки исходного изображения!\n";
        return -1;
    }

    // Опциональная загрузка масок
    bool masksProvided = (argc == 5);
    cv::Mat maskGray, maskHologramOrig, maskHologramCorr;

    if (masksProvided) {
        maskGray = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        maskHologramOrig = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        maskHologramCorr = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);

        if (maskGray.empty() || maskHologramOrig.empty() || maskHologramCorr.empty()) {
            std::cerr << "Ошибка загрузки одной или нескольких масок!\n";
            return -1;
        }
    } else {
        std::cout << "Маски не загружены. Метрики рассчитываться не будут.\n";
    }

    // Коррекция баланса белого
    orig.convertTo(orig, CV_32FC3, 1.0 / 255.0);
    cv::Mat linOrig = convertToLinRGB(orig);
    cv::Mat linCorrected = whiteBalanceCorrection(linOrig);
    cv::Mat corrected = convertToSRGB(linCorrected);

    // Карты насыщенности
    cv::Mat satOrig = computeSaturationMap(orig);
    cv::Mat satCorrected = computeSaturationMap(corrected);

    // Визуализация карт насыщенности
    cv::Mat satColorOrig = visualizeSaturation(satOrig);
    cv::Mat satColorCorrected = visualizeSaturation(satCorrected);

    // Приведение изображений к CV_8UC3
    cv::Mat showOrig, showCorrected;
    orig.convertTo(showOrig, CV_8UC3, 255.0);
    corrected.convertTo(showCorrected, CV_8UC3, 255.0);

    // Отображение изображений
    cv::imshow("Original", showOrig);
    cv::imshow("White balance", showCorrected);
    cv::imshow("Sat Original", satColorOrig);
    cv::imshow("Sat White balance", satColorCorrected);

    // Если маски предоставлены, рассчитываем метрики
    if (masksProvided) {
        double mseGray = computeMSE(satOrig, maskGray);
        double mseHoloOrig = computeMSE(satOrig, maskHologramOrig);
        double mseHoloCorr = computeMSE(satOrig, maskHologramCorr);

        std::cout << "MSE для серой области = " << mseGray << std::endl;
        std::cout << "MSE для голограммы (оригинал) = " << mseHoloOrig << std::endl;
        std::cout << "MSE для голограммы (коррект.) = " << mseHoloCorr << std::endl;

        double ratio = (mseHoloOrig > 1e-9) ? (mseGray / mseHoloOrig) : 0.0;
        std::cout << "Rg/c = " << ratio << std::endl;

        double ratioCorr = (mseHoloCorr > 1e-9) ? (mseGray / mseHoloCorr) : 0.0;
        std::cout << "Rg/c (после коррекции) = " << ratioCorr << std::endl;
    }

    cv::waitKey(0);
    return 0;
}


