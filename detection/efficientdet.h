//
// Created by kenny on 28/01/2021.
//

#ifndef OPENVINO_DETECTOR_EFFICIENTDET_H
#define OPENVINO_DETECTOR_EFFICIENTDET_H

#include "detector.h"

class EfficientDet : public Detector {
public:
    explicit EfficientDet(const std::string &model_name);

    void Predict(const cv::Mat &img, Detections &out_dets) override;

private:
    std::vector<std::string> idx2cls;
};

#endif //OPENVINO_DETECTOR_EFFICIENTDET_H
