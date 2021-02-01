//
// Created by kenny on 28/01/2021.
//

#include "efficientdet.h"

EfficientDet::EfficientDet(const std::string &model_name) : Detector(model_name) {
    auto shapes = net->GetOutputShapes();

    idx2cls = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
               "boat", "traffic light", "fire hydrant", "", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
               "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "", "backpack", "umbrella", "", "",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "", "wine glass", "cup", "fork",
               "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
               "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "", "dining table", "", "",
               "toilet", "", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
               "toaster", "sink", "refrigerator", "", "book", "clock", "vase", "scissors", "teddy bear",
               "hair drier", "toothbrush"};
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
    cv::Mat inp;
    img.copyTo(inp);

    auto inp_shape = net->GetInputShape();

    int new_height, new_width;
    if (inp.rows > inp.cols) {
        new_height = inp_shape[2];
        new_width = int(inp.cols * (inp_shape[3] * 1.f / inp.rows));
    } else {
        new_width = inp_shape[3];
        new_height = int(inp.rows * (inp_shape[2] * 1.f / inp.cols));
    }
    cv::resize(inp, inp, {new_width, new_height});
    cv::cvtColor(inp, inp, cv::COLOR_RGB2BGR);

    cv::Mat padded(inp_shape[3], inp_shape[2], CV_8UC3);
    padded = 0;

    inp.copyTo(padded(cv::Rect(0, 0, new_width, new_height)));

    net->SetInput(padded);
    net->Predict();
    auto shapes = net->GetOutputShapes();

    std::vector<std::vector<float>> ratios = {{1.f, 1.f},
                                              {1.4, 0.7},
                                              {0.7, 1.4}};
    std::vector<float> scales = {
            1,
            1.2599210498948732,
            1.5874010519681994
    };

    size_t n_cls = 90;
    size_t box_reg_n = 4;
    auto reg_cell_size = (box_reg_n * scales.size() * ratios.size());
    auto cls_cell_size = (n_cls * scales.size() * ratios.size());


    for (size_t p = 0; p < 5; p++) {
        auto *cls_res = net->GetOutput(p);
        auto *reg_res = net->GetOutput(p + 5);
        auto shape = net->GetOutputShape(p + 5);
        auto rows = shape[1];
        auto cols = shape[2];
        float stride_y = 1.f / rows;
        float stride_x = 1.f / cols;


        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                auto *reg_cell = reg_res + (rows * i + j) * reg_cell_size;
                auto *cls_cell = cls_res + (rows * i + j) * cls_cell_size;

                for (size_t sc_idx = 0; sc_idx < scales.size(); sc_idx++) {
                    for (size_t r_idx = 0; r_idx < ratios.size(); r_idx++) {
                        auto reg_vals = reg_cell + (sc_idx * scales.size() + r_idx) * box_reg_n;
                        auto cls_vals = cls_cell + (sc_idx * scales.size() + r_idx) * n_cls;

                        auto p_max = std::max_element(cls_vals, cls_vals + n_cls);
                        float score = *p_max;
                        if (score < 0.3) {
                            continue;
                        }
                        size_t cls_idx = std::distance(p_max, cls_vals);

                        auto scale = scales[sc_idx];
                        auto ratio = ratios[r_idx];
                        float anchor_y = 4.f * scale * ratio[1] * stride_y;
                        float anchor_x = 4.f * scale * ratio[0] * stride_x;
                        float w = exp(reg_vals[3]) * anchor_x;
                        float h = exp(reg_vals[2]) * anchor_y;
                        float y_c = reg_vals[1] * anchor_y + (float(i) + 0.5f) * stride_y;
                        float x_c = reg_vals[0] * anchor_x + (float(j) + 0.5f) * stride_x;

                        float x0 = std::max(x_c - w / 2, 0.f);
                        float y0 = std::max(y_c - h / 2, 0.f);
                        float x1 = std::min(x_c + w / 2, 1.f);
                        float y1 = std::min(y_c + h / 2, 1.f);

                        auto det = Detection{score, x0, y0, x1, y1};
                        out_dets.push_back(det);
                    }
                }

            }
        }
    }

    NMS(out_dets, 0.2);


    float x_max = static_cast<float>(new_width) / inp_shape[3];
    float y_max = static_cast<float>(new_height) / inp_shape[2];
    for (auto &det:out_dets) {
        det.x0 = std::max(det.x0 / x_max, 0.f);
        det.y0 = std::max(det.y0 / y_max, 0.f);
        det.x1 = std::min(det.x1 / x_max, 1.f);
        det.y1 = std::min(det.y1 / y_max, 1.f);

        auto p0 = cv::Point2f{det.x0 * img.cols, det.y0 * img.rows};
        auto p1 = cv::Point2f{det.x1 * img.cols, det.y1 * img.rows};

//        auto p0 = cv::Point2f{det.x0, det.y0};
//        auto p1 = cv::Point2f{det.x1, det.y1};
        cv::rectangle(img, p0, p1, {0, 255, 255}, 2);
    }

    cv::imshow("Detector Out", img);
    cv::waitKey(0);
//    std::cout << std::endl;
}