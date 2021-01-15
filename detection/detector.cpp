//
// Created by kenny on 15/01/2021.
//
#include "detector.h"

Detector::Detector(const std::string &model_name) {
    net = std::make_shared<IENetwork>(model_name + ".bin", model_name + ".xml");
    net->Build(netOptions);
}