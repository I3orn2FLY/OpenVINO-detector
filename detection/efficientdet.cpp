//
// Created by kenny on 28/01/2021.
//

#include "efficientdet.h"

EfficientDet::EfficientDet(const std::string &model_name) : Detector(model_name) {
    auto shapes = net->GetOutputShapes();

    idx2cls = {"person", "car", ""};
}

float computeIOU(const Detection &box1, const Detection &box2) {
    float area_of_overlap = box1.intersect_area(box2);
    if (area_of_overlap <= 0) return 0;
    float area_of_union = box1.area() + box2.area() - area_of_overlap;
    return area_of_overlap / area_of_union;
}

void NMS(Detections &detections, const float &iou_threshold) {
    std::sort(detections.begin(), detections.end(), [](const auto &a, const auto &b) { return a.conf > b.conf; });

    std::vector<int> mask(detections.size(), 1);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!mask[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!mask[j] or j == i) continue;
            auto box_i = detections[i];
            auto box_j = detections[j];
            auto iou = computeIOU(box_i, box_j);
            if (iou > iou_threshold) mask[j] = 0;
        }
    }
    for (int i = 0; i < mask.size(); ++i) {
        if (mask[i] == 1) {
            detections.push_back(detections[i]);
        }
    }
    detections.erase(detections.begin(), detections.begin() + mask.size());
}

void EfficientDet::Predict(const cv::Mat &img, Detections &out_dets) {
    out_dets.clear();
    auto bf0 = ST_GET_TIMESTAMP();
    cv::Mat inp;
    img.copyTo(inp);

    auto inp_shape = net->GetInputShape();

    int inp_h = inp_shape[2];
    int inp_w = inp_shape[3];

    auto img_w = static_cast<float>(img.cols);
    auto img_h = static_cast<float>(img.rows);


    int scaled_h, scaled_w;
    if (img_w / img_h > static_cast<float>(inp_w) / static_cast<float>(inp_h)) {
        scaled_w = inp_w;
        scaled_h = static_cast<int>(static_cast<float>(inp_w) * (img_h / img_w));
    } else {
        scaled_h = inp_h;
        scaled_w = static_cast<int>(static_cast<float>(inp_h) * (img_w / img_h));
    }
    cv::resize(inp, inp, {scaled_w, scaled_h});
    cv::cvtColor(inp, inp, cv::COLOR_RGB2BGR);

    cv::Mat padded(inp_h, inp_w, CV_8UC3);
    padded = 0;
    inp.copyTo(padded(cv::Rect(0, 0, scaled_w, scaled_h)));

//    cv::imshow("Padded", padded);
//    cv::waitKey(0);

    auto spend_pre = ST_GET_TIMESTAMP() - bf0;

    auto bf1 = ST_GET_TIMESTAMP();
    net->SetInput(padded);
    net->Predict();

    auto spend_nn = ST_GET_TIMESTAMP() - bf1;

    auto bf2 = ST_GET_TIMESTAMP();
    auto shapes = net->GetOutputShapes();
    size_t n_cls = 90;
    size_t box_reg_n = 4;
    size_t num_proposals = net->GetOutputShape(0)[1];
    auto *reg_res = net->GetOutput(1);
    auto *cls_res = net->GetOutput(0);
    for (size_t i = 0; i < num_proposals; ++i) {
        auto box = reg_res + box_reg_n * i;
        auto cls = cls_res + n_cls * i;
        auto max_score = std::max_element(cls, cls + n_cls);
        if (*max_score < 0.2) continue;
        Detection det = {*max_score, box[0] / inp_w, box[1] / inp_h, box[2] / inp_w, box[3] / inp_h};
        out_dets.push_back(det);
    }


    NMS(out_dets, 0.2);


    float x_max = static_cast<float>(scaled_w) / static_cast<float>(inp_w);
    float y_max = static_cast<float>(scaled_h) / static_cast<float>(inp_h);
    for (auto &det:out_dets) {
        det.x0 = std::max(det.x0 / x_max, 0.f);
        det.y0 = std::max(det.y0 / y_max, 0.f);
        det.x1 = std::min(det.x1 / x_max, 1.f);
        det.y1 = std::min(det.y1 / y_max, 1.f);
    }

    auto spend_post = ST_GET_TIMESTAMP() - bf2;
    auto spend_ovr = ST_GET_TIMESTAMP() - bf0;
//    std::cout << "\rSpend Preprocess:" << spend_pre << "      Spend NN:" << spend_nn << "     Spend POST:" << spend_post
//              << " Spend Overall:" << spend_ovr;
//    std::flush(std::cout);

    for (auto &det:out_dets) {
        auto p0 = cv::Point2f{det.x0 * img.cols, det.y0 * img.rows};
        auto p1 = cv::Point2f{det.x1 * img.cols, det.y1 * img.rows};
        cv::rectangle(img, p0, p1, {0, 255, 255}, 2);
    }
    cv::imshow("Detector Out", img);
    cv::waitKey(1);
//    std::cout << std::endl;
}