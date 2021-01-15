//
// Created by Дмитрий Федоров on 18/05/2020.
//

#ifndef TEYE_COMMON_NETWORK_H
#define TEYE_COMMON_NETWORK_H

#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @brief
 */
enum NetworkImageFormat {
    NN_IMAGE_RAW = 0u,  ///< Plain blob (default), no extra color processing required
    NN_IMAGE_RGB,       ///< RGB color format
    NN_IMAGE_BGR,       ///< BGR color format, default in DLDT
    NN_IMAGE_RGBX,      ///< RGBX color format with X ignored during inference
    NN_IMAGE_BGRX,      ///< BGRX color format with X ignored during inference
    NN_IMAGE_NV12,      ///< NV12 color format represented as compound Y+UV blob
    NN_IMAGE_I420,      ///< I420 color format represented as compound Y+U+V blob
};


/**
 * @brief
 */
enum NetworkLayoutFormat {
    NN_LAYOUT_ANY = 0,  //!< "any" layout

    // I/O data layouts
    NN_LAYOUT_NCHW = 1,  //!< NCHW layout for input / output blobs
    NN_LAYOUT_NHWC = 2,  //!< NHWC layout for input / output blobs
    NN_LAYOUT_NCDHW = 3,  //!< NCDHW layout for input / output blobs
    NN_LAYOUT_NDHWC = 4,  //!< NDHWC layout for input / output blobs

    // weight layouts
    NN_LAYOUT_OIHW = 64,  //!< NDHWC layout for operation weights
    NN_LAYOUT_GOIHW = 65,  //!< NDHWC layout for operation weights
    NN_LAYOUT_OIDHW = 66,  //!< NDHWC layout for operation weights
    NN_LAYOUT_GOIDHW = 67,  //!< NDHWC layout for operation weights

    // Scalar
    NN_LAYOUT_SCALAR = 95,  //!< A scalar layout

    // bias layouts
    NN_LAYOUT_C = 96,  //!< A bias layout for opearation

    // Single image layouts
    NN_LAYOUT_CHW = 128,  //!< A single image layout (e.g. for mean image)

    // 2D
    NN_LAYOUT_HW = 192,  //!< HW 2D layout
    NN_LAYOUT_NC = 193,  //!< HC 2D layout
    NN_LAYOUT_CN = 194,  //!< CN 2D layout

    NN_LAYOUT_BLOCKED = 200,  //!< A blocked layout
};


/**
 * @brief
 */
enum NetworkPrecisionFormat {
    NN_PRECISION_UNSPECIFIED = 255, /**< Unspecified value. Used by default */
    NN_PRECISION_MIXED = 0,         /**< Mixed value. Can be received from network. No applicable for tensors */
    NN_PRECISION_FP32 = 10,         /**< 32bit floating point value */
    NN_PRECISION_FP16 = 11,         /**< 16bit floating point value */
    NN_PRECISION_Q78 = 20,          /**< 16bit specific signed fixed point precision */
    NN_PRECISION_I16 = 30,          /**< 16bit signed integer value */
    NN_PRECISION_U8 = 40,           /**< 8bit unsigned integer value */
    NN_PRECISION_I8 = 50,           /**< 8bit signed integer value */
    NN_PRECISION_U16 = 60,          /**< 16bit unsigned integer value */
    NN_PRECISION_I32 = 70,          /**< 32bit signed integer value */
    NN_PRECISION_I64 = 72,          /**< 64bit signed integer value */
    NN_PRECISION_U64 = 73,          /**< 64bit unsigned integer value */
    NN_PRECISION_BIN = 71,          /**< 1bit integer value */
    NN_PRECISION_BOOL = 41,         /**< 8bit bool type */
    NN_PRECISION_CUSTOM = 80        /**< custom precision has it's own name and size of elements */
};


/**
 * @brief
 */
struct NetworkOptions {

    /**
     * @brief
     */
    struct Input {
        size_t width;
        size_t height;
        NetworkImageFormat imageFormat = NN_IMAGE_BGR;
        NetworkLayoutFormat layoutFormat = NN_LAYOUT_NCHW;
        NetworkPrecisionFormat precision = NN_PRECISION_U8;
    };


    /**
     * @brief
     */
    struct Output {
        NetworkImageFormat imageFormat = NN_IMAGE_RAW;
        NetworkLayoutFormat layoutFormat = NN_LAYOUT_ANY;
        NetworkPrecisionFormat precision = NN_PRECISION_FP32;
    };

    Input input;
    Output output;
    int tensorFormat;
    int batchSize = 1;
};


class INetwork {
public:


    /**
     * @brief
     */
    typedef std::shared_ptr<INetwork> Ptr;


    /**
     * @brief
     */
    virtual void Build(const NetworkOptions &options) = 0;


    /**
     * @brief
     */
    virtual void SetInput(const cv::Mat &pInput, size_t pInputIndex = 0, float lower = 0.0, float upper = 255.0) = 0;

    /**
     * @brief
     */
    virtual void
    SetInput(const std::vector<cv::Mat> &pInputs, size_t pInputIndex = 0, float lower = 0.0, float upper = 255.0) = 0;

    /**
     * @brief
     */
    virtual void Predict() = 0;


    /**
     * @brief
     * @param pOutputIndex
     * @return
     */
    virtual const float *GetOutput(size_t pOutputIndex = 0) = 0;


    /**
     * @brief
     * @param pOutputIndex
     * @return
     */
    virtual const std::vector<size_t> &GetInputShape(size_t pInputIndex = 0) { return inputShapes[pInputIndex]; }


    /**
     * @brief
     * @param pOutputIndex
     * @return
     */
    virtual const std::vector<size_t> &GetOutputShape(size_t pOutputIndex = 0) { return outputShapes[pOutputIndex]; }


    /**
     * @brief
     * @return
     */
    virtual const std::vector<std::vector<size_t>> &GetOutputShapes() { return outputShapes; }


    /**
    * @brief
    * @return
    */
    virtual const std::vector<std::vector<size_t>> &GetInputShapes() { return inputShapes; }


    /**
     * @brief
     * @return
     */

    virtual size_t GetCurrentNetWorkIndex() { return 0; };

    /**
     * @brief
     * @return
     */

    virtual size_t GetNetworksCount() { return 1; };

    /**
     * @brief
     */
    virtual void Free() = 0;


    virtual ~INetwork() = default;


protected:
    std::vector<std::vector<size_t>> inputShapes;
    std::vector<std::vector<size_t>> outputShapes;

    int batchSize = 1;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};

//std::shared_ptr<INetwork>   INetworkPtr;

#endif //TEYE_COMMON_NETWORK_H
