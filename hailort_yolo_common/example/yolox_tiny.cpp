#include "hailort_yolo_common/detection_inference.hpp"

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model> <path_to_image>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    cv::Mat frame = cv::imread(image_path);
    // size
    std::cout << "Image size: " << frame.size() << std::endl;
    if (frame.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return 1;
    }

    float detection_threshold = 0.4f;
    float nms_threshold = 0.4f;
    std::cout << "Model path: " << model_path << std::endl;
    yolo_cpp::YoloHailoRT yolo(model_path, detection_threshold, nms_threshold);
    std::cout << "Inference started" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << "====================" << std::endl;
        std::vector<yolo_cpp::Object> bboxes = yolo.inference(frame);
        for (auto &bbox : bboxes) {
            std::cout << "Label: " << bbox.label << " Prob: " << bbox.prob << std::endl;
        }
    }
    std::cout << "Inference completed" << std::endl;

    return 0;
}