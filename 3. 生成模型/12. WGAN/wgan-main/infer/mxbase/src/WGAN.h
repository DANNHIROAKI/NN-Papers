#ifndef WGAN_H
#define WGAN_H

#include <string>
#include <vector>
#include <memory>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/DeviceManager/DeviceManager.h"

struct InitParam{
    uint32_t deviceId;
    std::string modelPath;
};

struct model_info{
    uint32_t noise_length;
    uint32_t nimages;
    uint32_t image_size;
};

class WGAN{
 public:
    APP_ERROR Generate_input_Tensor(const model_info &modelInfo, std::vector<MxBase::TensorBase> *input);
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &infer_outputs,
                           std::vector<float> *processed_result);
    APP_ERROR Process(const model_info &modelInfo, const std::string &resultPath);
    APP_ERROR WriteResult(const std::string &fileName, std::vector<float> *output_img_data);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};

#endif  // WGAN_H
