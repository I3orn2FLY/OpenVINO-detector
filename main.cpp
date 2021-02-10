//
// Created by kenny on 15/01/2021.
//
#include <iostream>
#include "efficientdet.h"
#include <chrono>

int main() {
    auto cap = cv::VideoCapture("../data/videos/demo.mp4");
    auto detector = Detector("efficientdet-d0_frozen");
//    auto detector = EfficientDet("efficientdet-256x512");
//    auto detector = EfficientDet("efficientdet-d0-512x512");
//    auto detector = EfficientDet("efficientdet-d0-384x384");
    cv::Mat frame;

    float frame_n = 0;
    while (cap.read(frame)) {
        frame_n += 1;
        Detections dets;
        auto bf = std::chrono::high_resolution_clock::now();

//        frame = cv::imread("../data/images/zidane.jpg");
//        frame = cv::imread("../data/images/img.png");

        detector.Predict(frame, dets);
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds spend = std::chrono::duration_cast<std::chrono::milliseconds>(now - bf);
        std::cout << "\rSpeed in ms:" << spend.count();
        std::flush(std::cout);
    }
}