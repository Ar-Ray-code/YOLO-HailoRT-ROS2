#ifndef HAILORT_YOLO_COMMON_OBJECT_HPP_
#define HAILORT_YOLO_COMMON_OBJECT_HPP_

#include <opencv2/core.hpp>

namespace yolo_cpp {

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

} // namespace yolo_cpp

#endif // HAILORT_YOLO_COMMON_OBJECT_HPP_