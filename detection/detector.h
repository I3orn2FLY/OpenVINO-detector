//
// Created by kenny on 15/01/2021.
//

#ifndef OPENVINO_DETECTOR_DETECTOR_H
#define OPENVINO_DETECTOR_DETECTOR_H

#include "ie_network.h"
#include <string>

struct Detection {
    float conf, x0, y0, x1, y1;
    size_t label;
};
typedef std::vector<Detection> Detections;

class Detector {
public:
    explicit Detector(const std::string &model_name);

    virtual void Predict(const cv::Mat &input, Detections &out_dets) = 0;


private:
    INetwork::Ptr net;
    NetworkOptions netOptions;
};

#endif //OPENVINO_DETECTOR_DETECTOR_H
