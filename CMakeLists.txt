cmake_minimum_required(VERSION 2.8)

project(CVlab)

set(OpenCV_DIR ..\\..\\libs\\opencv\\build)
find_package(OpenCV CONFIG REQUIRED)

set(SRC main.cpp)
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})