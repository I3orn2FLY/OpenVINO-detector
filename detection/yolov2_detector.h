//
// Created by kenny on 15/01/2021.
//

#ifndef OPENVINO_DETECTOR_YOLOV2_DETECTOR_H
#define OPENVINO_DETECTOR_YOLOV2_DETECTOR_H

#include "detector.h"

class YoloV2Detector : public Detector {
public:
    explicit YoloV2Detector(const std::string &model_name);

    void Predict(const cv::Mat &img, Detections &out_dets) override;
};

#endif //OPENVINO_DETECTOR_YOLOV2_DETECTOR_H
