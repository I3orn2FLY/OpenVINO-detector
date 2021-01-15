//
// Created by kenny on 15/01/2021.
//
#include <iostream>
#include "yolov2_detector.h"

int main() {
    auto cap = cv::VideoCapture("../data/videos/demo.mp4");

    auto detector = YOLOV2Detector("yolo-v2-ava-0001");
    cv::Mat frame;
    while (cap.read(frame)) {
        Detections dets;
        detector.Predict(frame, dets);
        cv::imshow("Window", frame);
        cv::waitKey(1);
    }
}