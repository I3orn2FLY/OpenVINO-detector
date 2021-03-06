//
// Created by kenny on 15/01/2021.
//

#ifndef OPENVINO_DETECTOR_DETECTOR_H
#define OPENVINO_DETECTOR_DETECTOR_H

#include "ie_network.h"
#include <string>
#include "Timestamp.h"


struct Detection {
    float conf = 0.f, x0 = 0.f, y0 = 0.f, x1 = 1.f, y1 = 1.f;
    size_t label = 0;

    Detection() = default;

    [[nodiscard]] float intersect_area(const Detection &box) const;

    [[nodiscard]] float area() const;
};

typedef std::vector<Detection> Detections;

class Detector {
public:
    explicit Detector(const std::string &model_name);

    virtual void Predict(const cv::Mat &input, Detections &out_dets);


protected:
    INetwork::Ptr net;
    NetworkOptions netOptions;
};

#endif //OPENVINO_DETECTOR_DETECTOR_H
