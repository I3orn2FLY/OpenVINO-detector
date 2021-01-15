//
// Created by Дмитрий Федоров on 18/05/2020.
//
#include "ie_network.h"


template<NetworkPrecisionFormat f>
struct IEPrecisionEnum {
    static const InferenceEngine::Precision::ePrecision value;
};
template<>
struct IEPrecisionEnum<NN_PRECISION_FP32> {
    static const InferenceEngine::Precision::ePrecision value = InferenceEngine::Precision::FP32;
};


IENetwork::IENetwork(const std::string &modelPath, const std::string &weightsPath) {
    cnnNet = std::make_shared<InferenceEngine::CNNNetwork>(core.ReadNetwork(modelPath, weightsPath));
}


void IENetwork::Build(const NetworkOptions &options) {
    ::setenv("OMP_NUM_THREADS", std::to_string(1).c_str(), 1);
    ::setenv("KMP_BLOCKTIME", std::to_string(1).c_str(), 1);
    ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);

    std::map<std::string, std::string> config;
    //config[InferenceEngine::PluginConfigParams::KEY_SINGLE_THREAD] = InferenceEngine::PluginConfigParams::YES;
    //config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(1);
    config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(1);
    config[InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD] = InferenceEngine::PluginConfigParams::NO;
    config[InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(2);
    //config[InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = InferenceEngine::PluginConfigParams::YES;
    //config[InferenceEngine::PluginConfigParams::KEY_PERF_COUNT] = InferenceEngine::PluginConfigParams::NO;

    inputNames.clear();
    InferenceEngine::InputsDataMap inputsInfo = cnnNet->getInputsInfo();
    for (auto &it : inputsInfo) {
        inputNames.push_back(it.first);
        const InferenceEngine::InputInfo::Ptr &info = it.second;

        info->setPrecision((InferenceEngine::Precision::ePrecision) options.input.precision);
        info->setLayout((InferenceEngine::Layout) options.input.layoutFormat);
        info->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::BGR);
        info->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);

        inputShapes.push_back(info->getTensorDesc().getDims());
    }

    outputNames.clear();
    InferenceEngine::OutputsDataMap outputsInfo = cnnNet->getOutputsInfo();
    for (auto &it : outputsInfo) {
        outputNames.push_back(it.first);
        const auto &info = it.second;
        info->setPrecision((InferenceEngine::Precision::ePrecision) options.output.precision);
        info->setLayout((InferenceEngine::Layout) options.output.layoutFormat);

        outputShapes.push_back(info->getTensorDesc().getDims());
    }

    batchSize = 1;
    if (options.batchSize > 1) {
        try {
            cnnNet->setBatchSize(options.batchSize);
            config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
            CreateExecutableNetworkAndInferRequest(config);
            inferRequest->SetBatch(options.batchSize);
        }
        catch (std::exception &e) {
            batchSize = options.batchSize;
            cnnNet->setBatchSize(1);
            config.erase(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED);
            CreateExecutableNetworkAndInferRequest(config);
        }
    } else
        CreateExecutableNetworkAndInferRequest(config);

    ieInputs.clear();
    for (const auto &inputName : inputNames)
        ieInputs.push_back(inferRequest->GetBlob(inputName));

    ieOutputs.clear();
    for (const auto &outputName : outputNames)
        ieOutputs.push_back(inferRequest->GetBlob(outputName));

}


void IENetwork::CreateExecutableNetworkAndInferRequest(const std::map<std::string, std::string> &config) {
    executableNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(
            core.LoadNetwork(*cnnNet, deviceName, config));
    inferRequest = std::make_shared<InferenceEngine::InferRequest>(executableNetwork->CreateInferRequest());
}


void
IENetwork::SetInput(const cv::Mat &pInput, size_t pInputIndex, float lower, float upper) // TODO: add lower and upper
{
    int batchIndex = pInputIndex;
    const InferenceEngine::Blob::Ptr &inputBlob = ieInputs[0];
    InferenceEngine::SizeVector blobSize = inputShapes[0];
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(ieInputs[0]);
    if (!mblob) {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                           << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    auto *blob_data = mblobHolder.as<uint8_t *>();

    cv::Mat resized_image;
    if (static_cast<int>(width) != pInput.size().width ||
        static_cast<int>(height) != pInput.size().height) {
        cv::resize(pInput, resized_image, cv::Size(width, height));
    } else {
        resized_image = pInput;
    }
    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + c * width * height + h * width + w] = (uint8_t) resized_image.at<cv::Vec3b>(h,
                                                                                                                    w)[c];
            }
        }
    }
}

void IENetwork::SetInput(const std::vector<cv::Mat> &pInputs, size_t pInputIndex, float lower,
                         float upper) // TODO: add lower and upper
{
    //TODO: FIX
}


void IENetwork::Predict() {
    try {
        inferRequest->Infer();
    } catch (std::exception &ex) {
        std::cout << ex.what() << std::endl;
        return;
    }

}


const float *IENetwork::GetOutput(size_t pOutputIndex) {
    const InferenceEngine::Blob::Ptr &outputBlob = ieOutputs[pOutputIndex];
    InferenceEngine::MemoryBlob::CPtr memoryBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob);
    if (!memoryBlob)
        throw std::logic_error("Error with reading the output blolb.");

    auto lockedMemory = memoryBlob->rmap();
    return lockedMemory.as<float *>();
}


void IENetwork::Free() {
    inputNames.clear();
    outputNames.clear();
    ieInputs.clear();
    ieOutputs.clear();
}