//
// Created by kenny on 15/01/2021.
//
#include "detector.h"

Detector::Detector(const std::string &model_name) {
    net = std::make_shared<IENetwork>("../data/models/" + model_name + ".xml", "../data/models/" + model_name + ".bin");
    net->Build(netOptions);
}

void Detector::Predict(const cv::Mat &input, Detections &out_dets) {
    net->SetInput(input);
    net->Predict();
}

float Detection::intersect_area(const Detection &box) const {
    float xmin = std::max(x0, box.x0);
    float xmax = std::min(x1, box.x1);
    if (xmin > xmax) return 0;
    float ymin = std::max(y0, box.y0);
    float ymax = std::min(y1, box.y1);
    if (ymin > ymax) return 0;
    return (ymax - ymin) * (xmax - xmin);
}

float Detection::area() const {
    return (x1 - x0) * (y1 - y0);
}
