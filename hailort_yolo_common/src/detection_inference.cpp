#include "hailort_yolo_common/detection_inference.hpp"

// constexpr bool QUANTIZED = true;
// constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
using namespace hailort;

namespace yolo_cpp {

hailo_status YoloHailoRT::init_device(
    const std::string &model_path,
    std::shared_ptr<VDevice> &vdevice,
    std::shared_ptr<ConfiguredNetworkGroup> &network_group,
    // VStreams &vstreams)
    std::shared_ptr<VStreams> &vstreams)
{
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    vdevice = vdevice_exp.release();
    auto network_group_exp = configure_network_group(*vdevice, model_path);
    if (!network_group_exp) {
        std::cerr << "Failed to configure network group " << model_path << std::endl;
        return network_group_exp.status();
    }
    network_group = network_group_exp.release();

    auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);
    if (!vstreams_exp) {
        std::cerr << "Failed creating vstreams " << vstreams_exp.status() << std::endl;
        return vstreams_exp.status();
    }
    vstreams = std::make_shared<VStreams>(std::move(vstreams_exp.value()));
    return HAILO_SUCCESS;
}


std::vector<Object> YoloHailoRT::post_processing(std::shared_ptr<FeatureData> &feature,
                                cv::Mat& frame, 
                                const double org_height,
                                const double org_width) {
    std::vector<Object> objects;
    HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
    roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(feature->m_buffers.get_read_buffer().data()), feature->m_vstream_info));

    filter(roi);
    feature->m_buffers.release_read_buffer();

    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
    cv::resize(frame, frame, cv::Size((int)org_width, (int)org_height), 1);
    for (auto &detection : detections) {
        if (detection->get_confidence()==0) {
            continue;
        }

        HailoBBox bbox = detection->get_bbox();
        Object obj;
        obj.rect = cv::Rect_<float>(bbox.xmin() * float(org_width), bbox.ymin() * float(org_height), 
                                    (bbox.xmax() - bbox.xmin()) * float(org_width), (bbox.ymax() - bbox.ymin()) * float(org_height));
        obj.label = detection->get_class_id() - 1;
        obj.prob = detection->get_confidence();
        if (obj.prob > m_conf_) {
            objects.push_back(obj);
        }
    }
    return objects;
}

hailo_status YoloHailoRT::read_all(OutputVStream& output_vstream, std::shared_ptr<FeatureData> feature) { 
    std::vector<uint8_t>& buffer = feature->m_buffers.get_write_buffer();
    hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
    feature->m_buffers.release_write_buffer();
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed reading with status = " <<  status << std::endl;
        return status;
    }
    return HAILO_SUCCESS;
}

hailo_status YoloHailoRT::create_feature(hailo_vstream_info_t vstream_info, const size_t output_frame_size, std::shared_ptr<FeatureData> &feature) {
    feature = std::make_shared<FeatureData>(static_cast<uint32_t>(output_frame_size), vstream_info.quant_info.qp_zp,
        vstream_info.quant_info.qp_scale, vstream_info.shape.width, vstream_info);

    return HAILO_SUCCESS;
}

std::vector<Object> YoloHailoRT::inference(const cv::Mat &frame) {

    InputVStream& input_vstream = m_vstreams_->first[0];
    OutputVStream& output_vstream = m_vstreams_->second[0];

    std::shared_ptr<FeatureData> feature(nullptr);
    hailo_vstream_info_t vstream_info = output_vstream.get_info();
    feature = std::make_shared<FeatureData>(static_cast<uint32_t>(output_vstream.get_frame_size()), vstream_info.quant_info.qp_zp,
        vstream_info.quant_info.qp_scale, vstream_info.shape.width, vstream_info);
    cv::Mat _frame = frame.clone();
    cv::resize(frame, _frame, cv::Size(input_vstream.get_info().shape.width, input_vstream.get_info().shape.height), 1);
    input_vstream.write(MemoryView(_frame.data, input_vstream.get_frame_size()));

    std::vector<uint8_t>& buffer = feature->m_buffers.get_write_buffer();
    hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
    feature->m_buffers.release_write_buffer();

    return post_processing(feature, _frame, frame.rows, frame.cols);
}

Expected<std::shared_ptr<ConfiguredNetworkGroup>> YoloHailoRT::configure_network_group(VDevice &vdevice, const std::string &yolo_hef)
{
    auto hef_exp = Hef::create(yolo_hef);
    if (!hef_exp) return make_unexpected(hef_exp.status());
    auto hef = hef_exp.release();

    auto configure_params = hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!configure_params) return make_unexpected(configure_params.status());

    auto network_groups = vdevice.configure(hef, configure_params.value());
    if (!network_groups) return make_unexpected(network_groups.status());

    if (1 != network_groups->size()) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}

YoloHailoRT::YoloHailoRT(const std::string &model_path, const float nms, const float conf):
    m_model_path_(model_path),
    m_nms_(nms),
    m_conf_(conf)
{
    this->init_device(m_model_path_, m_device_, m_network_group_, m_vstreams_);
}

} // namespace yolo_cpp