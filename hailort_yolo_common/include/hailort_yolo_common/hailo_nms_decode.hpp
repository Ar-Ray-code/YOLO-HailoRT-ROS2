/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

#include <vector>
#include <string>
#include <iostream>

#include "hailo_objects.hpp"
#include "structures.hpp"

static const int DEFAULT_MAX_BOXES = 100;
static const float DEFAULT_THRESHOLD = 0.4f;

class HailoNMSDecode
{
private:
  HailoTensorPtr _nms_output_tensor;
  std::map<uint8_t, std::string> labels_dict;
  float _detection_thr;
  uint _max_boxes;
  bool _filter_by_score;
  const hailo_vstream_info_t _vstream_info;

  common::hailo_bbox_float32_t dequantize_hailo_bbox(const auto * bbox_struct)
  {
    // Dequantization of common::hailo_bbox_t (uint16_t) to common::hailo_bbox_float32_t (float32_t)
    common::hailo_bbox_float32_t dequant_bbox = {
      .y_min = _nms_output_tensor->fix_scale(bbox_struct->y_min),
      .x_min = _nms_output_tensor->fix_scale(bbox_struct->x_min),
      .y_max = _nms_output_tensor->fix_scale(bbox_struct->y_max),
      .x_max = _nms_output_tensor->fix_scale(bbox_struct->x_max),
      .score = _nms_output_tensor->fix_scale(bbox_struct->score)};

    return dequant_bbox;
  }

  void parse_bbox_to_detection_object(
    auto dequant_bbox, uint32_t class_index,
    std::vector<HailoDetection> & _objects)
  {
    float confidence = CLAMP(dequant_bbox.score, 0.0f, 1.0f);
    if (!_filter_by_score || dequant_bbox.score > _detection_thr) {
      float32_t w, h = 0.0f;
      std::tie(w, h) = get_shape(&dequant_bbox);
      _objects.push_back(
        HailoDetection(
          HailoBBox(dequant_bbox.x_min, dequant_bbox.y_min, w, h),
          class_index, labels_dict[(unsigned char)class_index], confidence));
    }
  }

  std::pair<float, float> get_shape(auto * bbox_struct)
  {
    float32_t w = (float32_t)(bbox_struct->x_max - bbox_struct->x_min);
    float32_t h = (float32_t)(bbox_struct->y_max - bbox_struct->y_min);
    return std::pair<float, float>(w, h);
  }

public:
  HailoNMSDecode(
    HailoTensorPtr tensor, std::map<uint8_t, std::string> & labels_dict,
    float detection_thr = DEFAULT_THRESHOLD, uint max_boxes = DEFAULT_MAX_BOXES,
    bool filter_by_score = false)
  : _nms_output_tensor(tensor), labels_dict(labels_dict), _detection_thr(detection_thr), _max_boxes(
      max_boxes), _filter_by_score(filter_by_score), _vstream_info(tensor->vstream_info())
  {
    if (HAILO_FORMAT_ORDER_HAILO_NMS != _vstream_info.format.order) {
      throw std::invalid_argument(
              "Output tensor " + _nms_output_tensor->name() + " is not an NMS type");
    }
  }

  template<typename T, typename BBoxType>
  std::vector<HailoDetection> decode()
  {
    std::vector<HailoDetection> _objects;
    if (!_nms_output_tensor) {
      return _objects;
    }

    _objects.reserve(_max_boxes);
    uint8_t * src_ptr = _nms_output_tensor->data();
    uint32_t actual_frame_size = 0;

    uint32_t num_of_classes = _vstream_info.nms_shape.number_of_classes;
    uint32_t max_bboxes_per_class = _vstream_info.nms_shape.max_bboxes_per_class;

    for (uint32_t class_index = 1; class_index <= num_of_classes; class_index++) {
      T bbox_count = *reinterpret_cast<const T *>(src_ptr + actual_frame_size);

      if ((int)bbox_count > max_bboxes_per_class) {
        throw std::runtime_error(
                (
                  "Runtime error - Got more than the maximum bboxes per class in the nms buffer"));
      }

      if (bbox_count > 0) {
        uint8_t * class_ptr = src_ptr + actual_frame_size + sizeof(bbox_count);

        // iterate over the boxes and parse each box to common::hailo_bbox_t
        for (uint8_t box_index = 0; box_index < bbox_count; box_index++) {
          BBoxType * bbox_struct = (BBoxType *)(class_ptr + (box_index * sizeof(BBoxType)));

          if (std::is_same<T, uint16_t>::value) {
            // output type (T) is uint16, so we need to do dequantization before parsing
            common::hailo_bbox_float32_t dequant_bbox = dequantize_hailo_bbox(bbox_struct);
            parse_bbox_to_detection_object(dequant_bbox, class_index, _objects);
          } else {
            parse_bbox_to_detection_object(*bbox_struct, class_index, _objects);
          }
        }
      }

      // calculate the frame size of the class - sums up the size of the output during iteration
      T class_frame_size = static_cast<uint8_t>(sizeof(bbox_count) + bbox_count * sizeof(BBoxType));
      actual_frame_size += static_cast<uint32_t>(class_frame_size);
    }

    return _objects;
  }
};
