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
