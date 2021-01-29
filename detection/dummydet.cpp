//
// Created by kenny on 28/01/2021.
//

#include "dummydet.h"

DummyDetector::DummyDetector(const std::string &model_name): Detector(model_name) {
    auto shapes = net->GetOutputShapes();
}

void DummyDetector::Predict(const cv::Mat &img, Detections &out_dets) {
    net->SetInput(img);
    net->Predict();
}