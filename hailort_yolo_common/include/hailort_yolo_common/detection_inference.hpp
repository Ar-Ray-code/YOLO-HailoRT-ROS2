#ifndef HAILO_YOLO_COMMON_DETECTION_INFERENCE_HPP_
#define HAILO_YOLO_COMMON_DETECTION_INFERENCE_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

#include "hailo/hailort.hpp"
#include "hailort_yolo_common/hailo_objects.hpp"
#include "hailort_yolo_common/yolo_hailortpp.hpp"
#include "hailort_yolo_common/double_buffer.hpp"
#include "hailort_yolo_common/object.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>
#include <thread>

constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
using namespace hailort;

namespace yolo_cpp {

class FeatureData {
public:
    FeatureData(uint32_t buffers_size, float32_t qp_zp, float32_t qp_scale, uint32_t width, hailo_vstream_info_t vstream_info) :
    m_buffers(buffers_size), m_qp_zp(qp_zp), m_qp_scale(qp_scale), m_width(width), m_vstream_info(vstream_info)
    {}
    static bool sort_tensors_by_size (std::shared_ptr<FeatureData> i, std::shared_ptr<FeatureData> j) { return i->m_width < j->m_width; };

    DoubleBuffer m_buffers;
    float32_t m_qp_zp;
    float32_t m_qp_scale;
    uint32_t m_width;
    hailo_vstream_info_t m_vstream_info;
};

class YoloHailoRT {
typedef std::pair<std::vector<InputVStream>, std::vector<OutputVStream>> VStreams;
public:
    explicit YoloHailoRT(const std::string &, const float, const float);
    ~YoloHailoRT() = default;

    hailo_status init_device(const std::string &, std::shared_ptr<VDevice> &, std::shared_ptr<ConfiguredNetworkGroup> &, std::shared_ptr<VStreams> &);
    
    std::vector<Object> post_processing(std::shared_ptr<FeatureData> &, cv::Mat &, const double, const double);

    
    hailo_status read_all(OutputVStream &, std::shared_ptr<FeatureData>);
    hailo_status create_feature(hailo_vstream_info_t, const size_t, std::shared_ptr<FeatureData> &);
    std::vector<Object> inference(const cv::Mat &);

    Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(VDevice &, const std::string &);

private:
    std::string m_model_path_;
    float m_nms_;
    float m_conf_;

    std::shared_ptr<VDevice> m_device_;
    std::shared_ptr<ConfiguredNetworkGroup> m_network_group_;
    std::shared_ptr<VStreams> m_vstreams_;
};

} // namespace yolo_cpp

#endif // HAILO_YOLO_COMMON_DETECTION_INFERENCE_HPP_