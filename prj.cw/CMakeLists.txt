# prj.cw/CMakeLists.txt

add_library(wb_lib STATIC
        src/white_balance.cpp
        src/saturation.cpp
)

target_include_directories(wb_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Линкуем библиотеку с OpenCV
target_link_libraries(wb_lib PRIVATE opencv_core opencv_imgproc opencv_highgui)

# Создаем исполняемый файл "cw" из main.cpp
add_executable(cw src/main.cpp)

# Линкуем исполняемый файл с нашей библиотекой и OpenCV
target_link_libraries(cw PRIVATE wb_lib opencv_core opencv_imgproc opencv_highgui)
