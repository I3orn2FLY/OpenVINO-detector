//
// Created by kenny on 15/01/2021.
//
#include <iostream>
#include "efficientdet.h"
#include <chrono>

int main() {
    auto cap = cv::VideoCapture("../data/videos/demo.mp4");
//    auto detector = Detector("efficientdet-post-proc");
    auto detector = Detector("yolov5s");
//    auto detector = Detector("efficientdet-d0_v0.0.1");
//    auto detector = Detector("yolov4-csp");
//    auto detector = EfficientDet("efficientdet-d0-c90-384x640_v0.0.1");
    cv::Mat frame;

    float frame_n = 0;
    while (cap.read(frame)) {
        frame_n += 1;
        Detections dets;
        auto bf = ST_GET_TIMESTAMP();

//        frame = cv::imread("../data/images/zidane.jpg");
//        frame = cv::imread("../data/images/img.png");

        detector.Predict(frame, dets);
        auto spend = ST_GET_TIMESTAMP() - bf;
        std::cout << "\rSpeed in ms:" << spend;
        std::flush(std::cout);
    }
}