#pragma once
#include "OnnxInferenceBase.h"
#include <vector>

class Yolo_Inference :
	public OnnxInferenceBase
{
public:
	Yolo_Inference(const std::wstring& model_path, float confidence_threshold, float nms_threshold) :OnnxInferenceBase(model_path) {
		m_confidece_threshold = confidence_threshold;
		m_nms_threshold = nms_threshold;
	}

	std::vector < std::vector<float>> get_boxes() { return m_box_result; };
	std::vector<float> get_boxes_arr();

	void setConfidenceThreshold(float confidence_threshold) { m_confidece_threshold = confidence_threshold; };
	void setNmsThreshold(float nms_threshold) { m_nms_threshold = nms_threshold; };
private:
	void process_data(float* output_data_host, int output_numbox, int output_numprob, int num_classes);
public:
	bool inference() override;
private:
	bool preprocess_data() override;
	bool postprocess_data() override;

private:
	std::vector<std::vector<float>> bboxes;
	float m_confidece_threshold;
	float m_nms_threshold;
	std::vector<std::vector<float>> m_box_result;
	float i2d[6], d2i[6]; // warpaffine
};

