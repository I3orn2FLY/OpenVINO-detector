//
// Created by Дмитрий Федоров on 18/05/2020.
//

#ifndef TEYE_COMMON_IENETWORK_H
#define TEYE_COMMON_IENETWORK_H

#include "network.h"
#include <inference_engine.hpp>


class IECore {
public:


    /**
     * @brief
     * @return
     */
    static IECore *GetSingletonPtr() {
        static IECore instance;
        return &instance;
    }


    /**
     * @brief
     * @return
     */
    InferenceEngine::Core &Get() { return core; }


    /**
     * @brief
     * @return
     */
    const InferenceEngine::Core &Get() const { return core; }

private:
    InferenceEngine::Core core;
};


class IENetwork : public INetwork {
public:


    /**
     * @brief
     * @param modelPath
     * @param weightsPath
     */
    IENetwork(const std::string &modelPath, const std::string &weightsPath);


    /**
     * @brief
     */
    void Build(const NetworkOptions &options) override;


    /**
     * @brief
     */
    void SetInput(const cv::Mat &pInput, size_t pInputIndex, float lower, float upper) override;


    /**
     * @brief
     */
    void SetInput(const std::vector<cv::Mat> &pInputs, size_t pInputIndex, float lower, float upper) override;

    /**
     * @brief
     * @param pOutputIndex
     * @return
     */
    const float *GetOutput(size_t pOutputIndex) override;


    /**
     * @brief
     */
    void Predict() override;


    /**
     * @brief
     */
    void Free() override;

private:


    /**
     * @brief
     * @param config
     */
    void CreateExecutableNetworkAndInferRequest(const std::map<std::string, std::string> &config);


    InferenceEngine::Core core;
    std::shared_ptr<InferenceEngine::CNNNetwork> cnnNet;
    std::shared_ptr<InferenceEngine::InferRequest> inferRequest;
    std::shared_ptr<InferenceEngine::ExecutableNetwork> executableNetwork;
    std::vector<InferenceEngine::Blob::Ptr> ieInputs, ieOutputs;

    const std::string deviceName = "CPU";
};

#endif //TEYE_COMMON_IENETWORK_H
