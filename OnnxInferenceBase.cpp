#include "OnnxInferenceBase.h"
using namespace cv;


bool OnnxInferenceBase::preprocess_data()
{
	return true;
}

bool OnnxInferenceBase::postprocess_data()
{
	return true;
}

bool OnnxInferenceBase::inference()
{
	return true;
}



void OnnxInferenceBase::setData(cv::Mat& input_data, uchar* out_data)
{
	m_input_img = input_data;

	// opencv行主序
	m_img_h = m_input_img.rows;
	m_img_w = m_input_img.cols;
	m_img_c = m_input_img.channels();

	m_img_size = m_img_h * m_img_c * m_img_w;

	m_output_data = out_data;
}

void OnnxInferenceBase::setInputData(const cv::Mat& input_data)
{
	m_input_img = input_data.clone();
}



void OnnxInferenceBase::GetSegData(std::vector<short>& outdata)
{
	outdata = m_out_seg_data;
}

void OnnxInferenceBase::GeDetectData(std::vector<float>& outdata)
{
	outdata = m_out_detect_data;
}

