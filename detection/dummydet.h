//
// Created by kenny on 28/01/2021.
//

#ifndef OPENVINO_DETECTOR_DUMMYDET_H
#define OPENVINO_DETECTOR_DUMMYDET_H

#include "detector.h"

class DummyDetector : public Detector {
public:
    explicit DummyDetector(const std::string &model_name);

    void Predict(const cv::Mat &img, Detections &out_dets) override;
};

#endif //OPENVINO_DETECTOR_DUMMYDET_H
