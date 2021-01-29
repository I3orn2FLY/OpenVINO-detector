//
// Created by kenny on 15/01/2021.
//
#include <iostream>
#include "yolov2_detector.h"
#include "dummydet.h"
#include <chrono>

int main() {
    auto cap = cv::VideoCapture("../data/videos/demo.mp4");

//    auto detector = YOLOV2Detector("yolo-v2-ava-0001");
//    auto detector = YoloV2Detector("yolo-v2-tiny-ava-sparse-30-0001");
    auto detector = DummyDetector("efficientdet-d0");
//    auto detector = DummyDetector("yolov5s");
    cv::Mat frame;

    float frame_n = 0;
    while (cap.read(frame)) {
        frame_n += 1;
        Detections dets;
        auto bf = std::chrono::high_resolution_clock::now();
        detector.Predict(frame, dets);
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds spend = std::chrono::duration_cast<std::chrono::milliseconds>(now - bf);
        std::cout << "\rSpeed in ms:" << spend.count();
        std::flush(std::cout);

//        cv::imshow("Window", frame);
//        cv::waitKey(1);
    }
}