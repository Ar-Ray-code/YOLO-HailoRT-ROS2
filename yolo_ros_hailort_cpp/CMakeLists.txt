cmake_minimum_required(VERSION 3.8)
project(yolo_ros_hailort_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Threads)
find_package(HailoRT REQUIRED)

ament_auto_add_library(yolo_ros_hailort_cpp SHARED
  src/yolo_ros_hailort_cpp.cpp
)
target_link_libraries(yolo_ros_hailort_cpp
    Threads::Threads
    HailoRT::libhailort
    ${OpenCV_LIBRARIES}
)
rclcpp_components_register_node(
  yolo_ros_hailort_cpp
  PLUGIN "yolo_ros_hailort_cpp::YoloNode"
  EXECUTABLE yolo_ros_hailort_cpp_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)