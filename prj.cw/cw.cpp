//
// Created by Константин Гончаров on 18.04.2024.
//
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Структура для представления пикселя в изображении
struct Pixel {
    int red;
    int green;
    int blue;
};

// Функция для балансировки белого изображения
void whiteBalance(std::vector<std::vector<Pixel>>& image) {
    // Параметры изображения (ширина и высота)
    int width = image.size();
    int height = image[0].size();

    // Параметр дискретизации шкалы яркости
    int N = 256; // можно изменить в зависимости от разрешения

    // Рассчитываем М-оценку с ядром Уэлша для каждого цветового канала
    std::vector<double> M_estimate_red(N, 0.0);
    std::vector<double> M_estimate_green(N, 0.0);
    std::vector<double> M_estimate_blue(N, 0.0);

    // Рассчитываем статистики для М-оценки
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            M_estimate_red[image[i][j].red]++;
            M_estimate_green[image[i][j].green]++;
            M_estimate_blue[image[i][j].blue]++;
        }
    }

    // Преобразуем статистики в вероятности
    for (int i = 0; i < N; ++i) {
        M_estimate_red[i] /= (width * height);
        M_estimate_green[i] /= (width * height);
        M_estimate_blue[i] /= (width * height);
    }

    // Применяем М-оценку с ядром Уэлша
    double sum_red = 0.0, sum_green = 0.0, sum_blue = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_red += i * M_estimate_red[i];
        sum_green += i * M_estimate_green[i];
        sum_blue += i * M_estimate_blue[i];
    }

    // Вычисляем сдвиги для каждого канала
    double shift_red = N / 2.0 - sum_red;
    double shift_green = N / 2.0 - sum_green;
    double shift_blue = N / 2.0 - sum_blue;

    // Применяем баланс белого с учетом сдвигов
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            image[i][j].red = std::max(0, std::min(255, static_cast<int>(image[i][j].red + shift_red)));
            image[i][j].green = std::max(0, std::min(255, static_cast<int>(image[i][j].green + shift_green)));
            image[i][j].blue = std::max(0, std::min(255, static_cast<int>(image[i][j].blue + shift_blue)));
        }
    }
}

int main() {
    // Загрузка изображения из файла
    cv::Mat cv_image = cv::imread("image.png");

    // Проверяем, удалось ли загрузить изображение
    if (cv_image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    // Преобразуем изображение OpenCV в вектор векторов Pixel
    std::vector<std::vector<Pixel>> image(cv_image.rows, std::vector<Pixel>(cv_image.cols));
    for (int i = 0; i < cv_image.rows; ++i) {
        for (int j = 0; j < cv_image.cols; ++j) {
            cv::Vec3b pixel = cv_image.at<cv::Vec3b>(i, j);
            image[i][j].red = pixel[2]; // OpenCV хранит каналы в порядке BGR
            image[i][j].green = pixel[1];
            image[i][j].blue = pixel[0];
        }
    }

    // Балансировка белого
    whiteBalance(image);

    // Преобразуем вектор векторов Pixel обратно в cv::Mat
    for (int i = 0; i < cv_image.rows; ++i) {
        for (int j = 0; j < cv_image.cols; ++j) {
            cv_image.at<cv::Vec3b>(i, j) = cv::Vec3b(image[i][j].blue, image[i][j].green, image[i][j].red);
        }
    }

    // Сохранение изображения в файл или вывод на экран...
    cv::imshow("Balanced Image", cv_image);
    cv::waitKey(0); // Ожидание нажатия клавиши

    return 0;
}
