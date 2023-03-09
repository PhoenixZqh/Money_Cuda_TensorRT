#ifndef _CALIBRATOR_H_
#define _CALIBRATOR_H_

#include "macros.h"
#include <NvInfer.h>
#include <string>
#include <vector>

using namespace nvinfer1;

class Int8EntroyCalibrator2 : public IInt8EntropyCalibrator2 {
private:
    int batchsize_;
    int input_w_;
    int input_h_;
    int img_idx_;

    std::string img_dir_;
    std::vector<std::string> img_files_;
    size_t input_count_;
    std::string calib_table_name_;
    const char* input_blob_name_;

    bool read_cache_;
    void* device_input_;
    std::vector<char> calib_cache_;

public:
    Int8EntroyCalibrator2(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_trable_name, const char* input_blob_name, bool read_cache = true);
    virtual ~Int8EntroyCalibrator2();

    int getBatchSize() const TRT_NOEXCEPT override;
    bool getBatch(void* bindings[], const char* names[], int nBingdings) TRT_NOEXCEPT override;
    const void* readCalibrationCache(size_t& length) TRT_NOEXCEPT override;
    void writeCalibrationCache(const void* cache, size_t length) TRT_NOEXCEPT override;
};

#endif