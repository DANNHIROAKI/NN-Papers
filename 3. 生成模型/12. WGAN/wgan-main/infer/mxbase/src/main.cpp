#include <iostream>
#include <fstream>
#include <vector>
#include "WGAN.h"
#include "MxBase/Log/Log.h"

void init_Param(InitParam *initParam, model_info *modelInfo) {
    initParam->deviceId = 0;
    initParam->modelPath = "../data/model/DCGAN/WGAN.om";

    modelInfo->noise_length = 100;
    modelInfo->nimages = 1;
    modelInfo->image_size = 64;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input output result path, such as '../data/mxbase_result/'.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    model_info modelInfo;
    init_Param(&initParam, &modelInfo);

    auto wgan = std::make_shared<WGAN>();
    APP_ERROR ret = wgan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "WGAN init failed, ret=" << ret << ".";
        return ret;
    }

    std::string resultPath = argv[1];
    ret = wgan->Process(modelInfo, resultPath);
    if (ret != APP_ERR_OK) {
        LogError << "WGAN process failed, ret=" << ret << ".";
        wgan->DeInit();
        return ret;
    }

    wgan->DeInit();
    return APP_ERR_OK;
}
