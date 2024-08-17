#include "hailort_yolo_common/hailo_nms_decode.hpp"
#include "hailort_yolo_common/yolo_hailortpp.hpp"
#include "hailort_yolo_common/labels/coco_eighty.hpp"

#include <regex>

void filter(HailoROIPtr roi, void * params_void_ptr)
{
  if (!roi->has_tensors()) {
    return;
  }
  std::map<uint8_t, std::string> labels_map;
  labels_map = common::coco_eighty;

  for (auto tensor : roi->get_tensors()) {
    if (std::regex_search(tensor->name(), std::regex("nms"))) {
      auto post = HailoNMSDecode(tensor, labels_map);
      auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
      hailo_common::add_detections(roi, detections);
    }
  }
}
