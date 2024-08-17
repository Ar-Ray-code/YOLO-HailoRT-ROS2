#ifndef YOLO_ROS_HAILORT_CPP_YOLO_ROS_HAILORT_CPP_HPP_
#define YOLO_ROS_HAILORT_CPP_YOLO_ROS_HAILORT_CPP_HPP_

#include <cmath>
#include <chrono>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>


#include "hailort_yolo_common/detection_inference.hpp"
#include "hailort_yolo_common/utils.hpp"
#include "hailort_yolo_common/coco_names.hpp"
#include "yolo_param/yolo_param.hpp"

namespace yolo_ros_hailort_cpp{
    class YoloNode : public rclcpp::Node
    {
    public:
        YoloNode(const rclcpp::NodeOptions &);
    private:
        void onInit();
        void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);

        static vision_msgs::msg::Detection2DArray objects_to_detection2d(const std::vector<yolo_cpp::Object> &, const std_msgs::msg::Header &);

    protected:
        std::shared_ptr<yolo_parameters::ParamListener> param_listener_;
        yolo_parameters::Params params_;
    private:
        std::unique_ptr<yolo_cpp::YoloHailoRT> yolo_;
        std::vector<std::string> class_names_;

        rclcpp::TimerBase::SharedPtr init_timer_;
        image_transport::Subscriber sub_image_;

        rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_detection2d_;
        image_transport::Publisher pub_image_;
    };
}

#endif // YOLO_ROS_HAILORT_CPP_YOLO_ROS_HAILORT_CPP_HPP_