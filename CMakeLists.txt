cmake_minimum_required(VERSION 3.8)
project(hailort_yolo_common)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/yolo_hailortpp.cpp
)
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort ${OpenCV_LIBS} stdc++fs)
target_include_directories(${PROJECT_NAME} PUBLIC include ${OpenCV_INCLUDE_DIRS})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()