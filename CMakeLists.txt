cmake_minimum_required(VERSION 3.26)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
#        CACHE STRING "Vcpkg toolchain file")
project(misis2024s_21_02_goncharov_k_a)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin.dbg)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin.rel)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin.relwithdbg)

find_package(OpenCV CONFIG REQUIRED)

add_subdirectory(prj.lab)
add_subdirectory(prj.cw)
# Укажите путь к вашим фотографиям
set(IMAGES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/prj.lab/lab01/images")

# Копирование фотографий в директорию сборки
file(COPY ${IMAGES_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(IMAGES_DIR2 "${CMAKE_CURRENT_SOURCE_DIR}/prj.cw/images")

# Копирование фотографий в директорию сборки
file(COPY ${IMAGES_DIR2} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})