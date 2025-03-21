//
// Created by Константин Гончаров on 20.03.2025.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include "white_balance.h"
#include "saturation.h"

int main(int argc, char** argv) {
    // Загрузка исходного изображения (например, "pivo_3.jpg")
    cv::Mat orig = cv::imread("../prj.cw/images/pivo_3.jpg", cv::IMREAD_COLOR);
    if (orig.empty()){
        std::cout << "Ошибка при загрузке изображения!" << std::endl;
        return -1;
    }

    // Преобразование исходного изображения в CV_32FC3 с нормировкой [0,1]
    orig.convertTo(orig, CV_32FC3, 1.0/255.0);

    // Выполнение коррекции баланса белого
    cv::Mat linOrig = convertToLinRGB(orig);
    cv::Mat linCorrected = whiteBalanceCorrection(linOrig);
    cv::Mat corrected = convertToSRGB(linCorrected);

    // Вычисление карты насыщенности для исходного и скорректированного изображений
    cv::Mat satOrig = computeSaturationMap(orig);
    cv::Mat satCorrected = computeSaturationMap(corrected);

    // Визуализация карты насыщенности
    cv::Mat satColorOrig = visualizeSaturation(satOrig);
    cv::Mat satColorCorrected = visualizeSaturation(satCorrected);

    // Преобразование изображений для отображения (возвращаем в CV_8UC3)
    cv::Mat showOrig, showCorrected;
    orig.convertTo(showOrig, CV_8UC3, 255.0);
    corrected.convertTo(showCorrected, CV_8UC3, 255.0);

    // Отображение результатов
    cv::imshow("Original", showOrig);
    cv::imshow("White balance", showCorrected);
    cv::imshow("Sat Original", satColorOrig);
    cv::imshow("Sat White balance", satColorCorrected);

    // Сохранение результатов
//    cv::imwrite("../prj.cw/images/pivo_3_white_balance4.jpg", showCorrected);
//    cv::imwrite("../prj.cw/images/pivo_3_sat_original4.jpg", satColorOrig);
//    cv::imwrite("../prj.cw/images/pivo_3_sat_white_balance4.jpg", satColorCorrected);

    // Загрузка масок (маска серой области, маска голограммы из исходного и скорректированного изображений)
    cv::Mat maskGray = cv::imread("../prj.cw/images/pivo_3_mask_gray.png", cv::IMREAD_GRAYSCALE);
    if (maskGray.empty()){
        std::cerr << "Ошибка при загрузке mask_gray.png!" << std::endl;
        return -1;
    }
    cv::Mat maskHologramOrig = cv::imread("../prj.cw/images/pivo_3_mask_orig.png", cv::IMREAD_GRAYSCALE);
    if (maskHologramOrig.empty()){
        std::cerr << "Ошибка при загрузке mask_hologram_orig.png!" << std::endl;
        return -1;
    }
    cv::Mat maskHologramCorr = cv::imread("../prj.cw/images/pivo_3_mask_corr.png", cv::IMREAD_GRAYSCALE);
    if (maskHologramCorr.empty()){
        std::cerr << "Ошибка при загрузке mask_hologram_corr.png!" << std::endl;
        return -1;
    }

    // Вычисление MSE на основе карты насыщенности исходного изображения
    cv::Mat satImg = computeSaturationMap(orig);
    double mseGray = computeMSE(satImg, maskGray);
    double mseHoloOrig = computeMSE(satImg, maskHologramOrig);
    double mseHoloCorr = computeMSE(satImg, maskHologramCorr);

    std::cout << "MSE для серой области = " << mseGray << std::endl;
    std::cout << "MSE для голограммы (оригинал) = " << mseHoloOrig << std::endl;
    std::cout << "MSE для голограммы (коррект.) = " << mseHoloCorr << std::endl;

    double ratio = (mseHoloOrig > 1e-9) ? (mseGray / mseHoloOrig) : 0.0;
    std::cout << "Rg/c = " << ratio << std::endl;

    double ratioCorr = (mseHoloCorr > 1e-9) ? (mseGray / mseHoloCorr) : 0.0;
    std::cout << "Rg/c (после коррекции) = " << ratioCorr << std::endl;

    cv::waitKey(0);
    return 0;
}

