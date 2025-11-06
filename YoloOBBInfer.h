#pragma once
#include "OnnxInferenceBase.h"
#include <vector>
#include <opencv2/opencv.hpp>

class YoloOBBInfer : public OnnxInferenceBase {
public:
    YoloOBBInfer(const std::wstring& model_path, float confidence_threshold, float nms_threshold)
        : OnnxInferenceBase(model_path), m_confidence_threshold(confidence_threshold), m_nms_threshold(nms_threshold) {}

    // 输出为 [xc, yc, w, h, angle_deg, cls, conf]
    std::vector<std::vector<float>> get_obb_boxes() const { return m_obb_result; }

    void setConfidenceThreshold(float confidence_threshold) { m_confidence_threshold = confidence_threshold; }
    void setNmsThreshold(float nms_threshold) { m_nms_threshold = nms_threshold; }

    bool inference() override;
    bool preprocess_data() override;
    bool postprocess_data() override { return true; }

private:
    void process_data(float* output_data_host, int num_det, int ch, int num_classes);
    float IoU_Rotated(const cv::RotatedRect& r1, const cv::RotatedRect& r2);
    void nms_rotated();

private:
    float m_confidence_threshold{0.3f};
    float m_nms_threshold{0.3f};
    std::vector<std::vector<float>> m_obb_result; // [xc, yc, w, h, angle_deg, cls, conf]
    float i2d[6]{0}, d2i[6]{0};
};