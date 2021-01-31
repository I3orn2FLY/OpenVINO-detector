//
// Created by kenny on 15/01/2021.
//
#include <iostream>
#include "efficientdet.h"
#include <chrono>

int main() {
    auto cap = cv::VideoCapture("../data/videos/demo.mp4");

//    auto detector = YOLOV2Detector("yolo-v2-ava-0001");
//    auto detector = YoloV2Detector("yolo-v2-tiny-ava-sparse-30-0001");
    auto detector = EfficientDet("efficientdet-d0");
//    auto detector = DummyDetector("yolov5s");
    cv::Mat frame;

    float frame_n = 0;
    while (cap.read(frame)) {
        frame_n += 1;
        Detections dets;
        auto bf = std::chrono::high_resolution_clock::now();

        frame = cv::imread("../data/images/zidane.jpg");
//        frame = cv::imread("../data/images/img.png");

        detector.Predict(frame, dets);
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds spend = std::chrono::duration_cast<std::chrono::milliseconds>(now - bf);
//        std::cout << "\rSpeed in ms:" << spend.count();
//        std::flush(std::cout);

        for (const auto &det:dets) {
//            auto p0 = cv::Point2f{det.x0 * frame.cols, det.y0 * frame.rows};
//            auto p1 = cv::Point2f{det.x1 * frame.cols, det.y1 * frame.rows};

            auto p0 = cv::Point2f{det.x0, det.y0};
            auto p1 = cv::Point2f{det.x1, det.y1};
            cv::rectangle(frame, p0, p1, {0, 255, 255}, 2);
        }
        cv::imshow("Window", frame);
        cv::waitKey(0);
    }
}