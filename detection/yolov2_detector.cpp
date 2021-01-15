//
// Created by kenny on 15/01/2021.
//

#include "yolov2_detector.h"

void YOLOV2Detector::Predict(const cv::Mat &img, Detections &out_dets) {
    cv::Mat inp;
    cv::cvtColor(img, inp, cv::COLOR_BGR2RGB);
    cv::resize(inp, inp, {416, 416});
    net->SetInput(inp);
    net->Predict();
}

YOLOV2Detector::YOLOV2Detector(const std::string &model_name) : Detector(model_name) {
    auto shapes = net->GetOutputShapes();
}
