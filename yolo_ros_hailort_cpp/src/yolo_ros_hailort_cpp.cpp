#include "yolo_ros_hailort_cpp/yolo_ros_hailort_cpp.hpp"

namespace yolo_ros_hailort_cpp
{
    YoloNode::YoloNode(const rclcpp::NodeOptions &options)
        : Node("yolo_ros_hailort_cpp", options)
    {
        using namespace std::chrono_literals; // NOLINT
        this->init_timer_ = this->create_wall_timer(
            0s, std::bind(&YoloNode::onInit, this));
    }

    void YoloNode::onInit()
    {
        this->init_timer_->cancel();
        this->param_listener_ = std::make_shared<yolo_parameters::ParamListener>(
            this->get_node_parameters_interface());

        this->params_ = this->param_listener_->get_params();

        if (this->params_.imshow_isshow)
        {
            cv::namedWindow("yolo", cv::WINDOW_AUTOSIZE);
        }

        if (this->params_.class_labels_path != "")
        {
            RCLCPP_INFO(this->get_logger(), "read class labels from '%s'", this->params_.class_labels_path.c_str());
            this->class_names_ = yolo_cpp::utils::read_class_labels_file(this->params_.class_labels_path);
        }
        else
        {
            this->class_names_ = yolo_cpp::COCO_CLASSES;
        }

        this->yolo_ = std::make_unique<yolo_cpp::YoloHailoRT>(
                this->params_.model_path,
                this->params_.conf,
                this->params_.nms);

        RCLCPP_INFO(this->get_logger(), "model loaded");

        this->sub_image_ = image_transport::create_subscription(
            this, this->params_.src_image_topic_name,
            std::bind(&YoloNode::colorImageCallback, this, std::placeholders::_1),
            "raw");


        this->pub_detection2d_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            this->params_.publish_boundingbox_topic_name,
            10);

        if (this->params_.publish_resized_image) {
            this->pub_image_ = image_transport::create_publisher(this, this->params_.publish_image_topic_name);
        }
    }

    void YoloNode::colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &ptr)
    {
        auto img = cv_bridge::toCvCopy(ptr, "bgr8");
        cv::Mat frame = img->image;

        auto now = std::chrono::system_clock::now();
        auto objects = this->yolo_->inference(frame);

        auto end = std::chrono::system_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - now);
        RCLCPP_INFO(this->get_logger(), "Inference time: %5ld ms", elapsed.count());

        yolo_cpp::utils::draw_objects(frame, objects, this->class_names_);
        if (this->params_.imshow_isshow)
        {
            cv::imshow("yolo", frame);
            auto key = cv::waitKey(1);
            if (key == 27)
            {
                rclcpp::shutdown();
            }
        }

        vision_msgs::msg::Detection2DArray detections = objects_to_detection2d(objects, img->header);
        this->pub_detection2d_->publish(detections);

        if (this->params_.publish_resized_image) {
            sensor_msgs::msg::Image::SharedPtr pub_img =
                cv_bridge::CvImage(img->header, "bgr8", frame).toImageMsg();
            this->pub_image_.publish(pub_img);
        }
    }

    vision_msgs::msg::Detection2DArray YoloNode::objects_to_detection2d(const std::vector<yolo_cpp::Object> &objects, const std_msgs::msg::Header &header)
    {
        vision_msgs::msg::Detection2DArray detection2d;
        detection2d.header = header;
        for (const auto &obj : objects)
        {
            vision_msgs::msg::Detection2D det;
            det.bbox.center.position.x = obj.rect.x + obj.rect.width / 2;
            det.bbox.center.position.y = obj.rect.y + obj.rect.height / 2;
            det.bbox.size_x = obj.rect.width;
            det.bbox.size_y = obj.rect.height;

            det.results.resize(1);
            det.results[0].hypothesis.class_id = std::to_string(obj.label);
            det.results[0].hypothesis.score = obj.prob;
            detection2d.detections.emplace_back(det);
        }
        return detection2d;
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_ros_hailort_cpp::YoloNode)