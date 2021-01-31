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

void EfficientDet::Predict(const cv::Mat &img, Detections &out_dets) {
    out_dets.clear();
    cv::Mat inp;
    img.copyTo(inp);
    cv::cvtColor(inp, inp, cv::COLOR_RGB2BGR);

    auto inp_shape = net->GetInputShape();

    int h, w;
    if (inp.rows > inp.cols) {
        h = inp_shape[2];
        w = int(inp.cols * (inp_shape[3] * 1.f / inp.rows));
    } else {
        w = inp_shape[3];
        h = int(inp.rows * (inp_shape[2] * 1.f / inp.cols));
    }
    cv::resize(inp, inp, {w, h});

    cv::Mat padded(inp_shape[3], inp_shape[2], CV_8UC3);
    padded = 0;
    inp.copyTo(padded(cv::Rect(0, 0, w, h)));

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
        float stride_y = (inp_shape[2] * 1.f) / rows;
        float stride_x = (inp_shape[3] * 1.f) / cols;


        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {

                auto *reg_cell = reg_res + (rows * i + j) * reg_cell_size;
                auto *cls_cell = cls_res + (rows * i + j) * cls_cell_size;

                for (size_t sc_idx = 0; sc_idx < scales.size(); sc_idx++) {
                    for (size_t r_idx = 0; r_idx < ratios.size(); r_idx++) {
                        auto reg_vals = reg_cell + (sc_idx * scales.size() + r_idx) * box_reg_n;
                        auto cls_vals = cls_cell + (sc_idx * scales.size() + r_idx) * n_cls;

                        float score = 0.0f;
                        size_t cls_idx = 0;
                        for (size_t cls = 0; cls < n_cls; ++cls) {
                            float val = cls_vals[cls];
                            if (val > score) {
                                score = val;
                                cls_idx = cls;
                            }
                        }
                        if (score < 0.5) {
                            continue;
                        }

                        std::cout << idx2cls[cls_idx] << " " << score << std::endl;
//                        std::cout << "{";
//                        for (const auto &s: shape) {
//                            if (s != 1) {
//                                std::cout << ",";
//                            }
//                            std::cout << s;
//                        }
//                        std::cout << "}" << std::endl;

                        auto scale = scales[sc_idx];
                        auto ratio = ratios[r_idx];


                        float anchor_y = 4.f * scale * ratio[1] * stride_y;
                        float anchor_x = 4.f * scale * ratio[0] * stride_x;


                        float offset_y = (float(i) + 0.5f) * stride_y;
                        float offset_x = (float(j) + 0.5f) * stride_x;


                        float w = exp(reg_vals[3]) * anchor_x;
                        float h = exp(reg_vals[2]) * anchor_y;

                        float y_c = reg_vals[1] * anchor_y + offset_y;
                        float x_c = reg_vals[0] * anchor_x + offset_x;

                        float x0 = x_c - w / 2;
                        float y0 = y_c - h / 2;
                        float x1 = x_c + w / 2;
                        float y1 = y_c + h / 2;

                        auto det = Detection{score, x0, y0, x1, y1};
                        out_dets.push_back(det);
                    }
                }

            }
        }
    }


    std::cout << std::endl;
}