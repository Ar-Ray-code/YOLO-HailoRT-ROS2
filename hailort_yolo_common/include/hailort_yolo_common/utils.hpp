#ifndef _YOLO_CPP_UTILS_HPP
#define _YOLO_CPP_UTILS_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include "hailort_yolo_common/object.hpp"
#include "hailort_yolo_common/coco_names.hpp"

namespace yolo_cpp
{
    namespace utils
    {
        static std::vector<std::string> read_class_labels_file(const std::string &file_name)
        {
            std::vector<std::string> class_names;
            std::ifstream ifs(file_name);
            std::string buff;
            if (ifs.fail())
            {
                return class_names;
            }
            while (getline(ifs, buff))
            {
                if (buff == "")
                    continue;
                class_names.push_back(buff);
            }
            return class_names;
        }

        static void draw_objects(cv::Mat bgr, const std::vector<Object> &objects, const std::vector<std::string> &class_names = COCO_CLASSES)
        {

            for (const Object &obj : objects)
            {
                const int color_index = obj.label % 80;
                cv::Scalar color = cv::Scalar(color_list[color_index][0], color_list[color_index][1], color_list[color_index][2]);
                float c_mean = cv::mean(color)[0];
                cv::Scalar txt_color;
                if (c_mean > 0.5)
                {
                    txt_color = cv::Scalar(0, 0, 0);
                }
                else
                {
                    txt_color = cv::Scalar(255, 255, 255);
                }

                cv::rectangle(bgr, obj.rect, color * 255, 2);

                char text[256];
                sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

                cv::Scalar txt_bk_color = color * 0.7 * 255;

                int x = obj.rect.x;
                int y = obj.rect.y + 1;
                if (y > bgr.rows)
                    y = bgr.rows;

                cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                              txt_bk_color, -1);

                cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
            }
        }
    }
}
#endif