#include "Yolo_Inference.h"
#include <fstream>
#include <iostream>
using namespace cv;

std::vector<float> Yolo_Inference::get_boxes_arr()
{
	std::vector<float> box_arr;
	for (const auto& row : m_box_result) {
		// 把行里的所有元素插入到vec1d的末尾
		box_arr.insert(box_arr.end(), row.begin(), row.end());
	}
	return box_arr;
}

void Yolo_Inference::process_data(float* output_data_host, int output_numbox, int output_numprob, int num_classes)
{
	
	for (int i = 0; i < output_numbox; i++) {

		float* ptr = output_data_host + i * output_numprob;
		//float objness_1 = ptr[4]; // confidence
		//float objness_2 = ptr[5];

		//float objness = (objness_2 > objness_1 ? objness_2 : objness_1);


		float objness = *std::max_element(ptr + 4, ptr + 4 + num_classes);



		if (objness < m_confidece_threshold)
			continue;



		float* pclass = ptr + 4;  // 向后挪5个距离
		// 找到在这一段中最大的位置
		int label = std::max_element(pclass, pclass + num_classes) - pclass;
		float prob = 1;
		float confidence = prob * objness;

		std::cout << "ptr[" << i << "] values: ";
		for (int k = 0; k < 6; k++) {  // 打印 ptr 到 ptr+5 共6个值
			std::cout << ptr[k] << " ";
		}
		std::cout << std::endl;


		float cx = ptr[0];
		float cy = ptr[1];
		float width = ptr[2];
		float height = ptr[3];

		float left = cx - width * 0.5;
		float top = cy - height * 0.5;
		float right = cx + width * 0.5;
		float bottom = cy + height * 0.5;

		// 数据坐标还原
		float image_base_left = d2i[0] * left + d2i[2];
		float image_base_right = d2i[0] * right + d2i[2];
		float image_base_top = d2i[0] * top + d2i[5];
		float image_base_bottom = d2i[0] * bottom + d2i[5];

		if (image_base_left >= 0 && image_base_right >= 0 && image_base_top >= 0 && image_base_bottom >= 0) {
			bboxes.push_back({ image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence });
			std::cout << label << std::endl;
		}




	}
	//printf_s("decoded bboxes.size = %zd\n", bboxes.size());

	// nms

	// 将confidence排序
	std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b) {return a[5] > b[5]; });
	// 记录什么bbox需要被删除
	std::vector<bool> remove_flags(bboxes.size());
	// 
	m_box_result.clear();
	// 预置空间，减少分配时间
	m_box_result.reserve(bboxes.size());


	// 函数：计算iou (lamdba写法)
	auto iou = [](const std::vector<float>& a, const std::vector<float>& b) {
		// 计算交集
		float cross_left = std::max(a[0], b[0]);
		float cross_top = std::max(a[1], b[1]);
		float cross_right = std::min(a[2], b[2]);
		float cross_bottom = std::min(a[3], b[3]);

		// 计算交集面积
		float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
		// 计算并集面积
		float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1])
			+ std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;

		if (cross_area == 0 || union_area == 0) return 0.0f;

		return cross_area / union_area;
		};


	for (int i = 0; i < bboxes.size(); ++i) {
		if (remove_flags[i])  continue;

		auto& ibox = bboxes[i];
		// 类似push_back
		m_box_result.emplace_back(ibox);
		for (int j = i + 1; j < bboxes.size(); ++j) {
			if (remove_flags[j]) continue;

			auto& jbox = bboxes[j];

			if (ibox[4] == jbox[4]) {
				if (iou(ibox, jbox) >= m_nms_threshold)
					remove_flags[j] = true;
			}
		}
	}
	
}

bool Yolo_Inference::preprocess_data()
{
	//// 缩放图像

	//float scale_x = m_re_w / (float)m_input_img.cols;
	//float scale_y = m_re_h / (float)m_input_img.rows;
	//float scale = std::min(scale_x, scale_y);

	//// 等比缩放系数确认
	//i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * m_input_img.cols + m_re_w + scale - 1) * 0.5;
	//i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * m_input_img.rows + m_re_h + scale - 1) * 0.5;

	//// 获取翻转矩阵
	//cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
	//cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
	//cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

	//cv::Mat tmp_image(m_re_h, m_re_w, CV_8UC3);
	//cv::warpAffine(m_input_img, tmp_image, m2x3_i2d, tmp_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
	//m_real_input_data.resize(m_re_c * m_re_h * m_re_w);

	//OnnxInferenceBase::converHWCtoCHW<uchar, float>(tmp_image, m_real_input_data.data());

	//for (int idx = 0; idx < m_real_input_data.size(); idx++) {
	//	m_real_input_data[idx] = m_real_input_data[idx] / 255.;
	//}
	//return true;

		// 目标尺寸
	int out_w = m_re_w;
	int out_h = m_re_h;
	int stride = 32;           // 与模型一致
	bool auto_mode = false;     // 与 Predictor.pre_transform 的 auto 行为一致
	bool scaleup = true;       // 需要完全一致时按你的实际设置
	bool center = true;
	int pad_value = 114;

	int in_w = m_input_img.cols;
	int in_h = m_input_img.rows;

	// 计算缩放比例 r
	float r = std::min(out_w / (float)in_w, out_h / (float)in_h);
	if (!scaleup) r = std::min(r, 1.0f);

	// 缩放后的整数尺寸
	int new_w_unpad = (int)std::round(in_w * r);
	int new_h_unpad = (int)std::round(in_h * r);

	// 计算需要的填充
	float dw = out_w - new_w_unpad;
	float dh = out_h - new_h_unpad;
	if (auto_mode) {  // stride 对齐
		dw = std::fmod(dw, (float)stride);
		dh = std::fmod(dh, (float)stride);
	}
	if (center) {
		dw *= 0.5f;
		dh *= 0.5f;
	}

	// 缩放
	cv::Mat resized;
	if (new_w_unpad != in_w || new_h_unpad != in_h) {
		cv::resize(m_input_img, resized, cv::Size(new_w_unpad, new_h_unpad), 0, 0, cv::INTER_LINEAR);
	}
	else {
		resized = m_input_img;
	}

	// 边框（与 Python 的 ±0.1 round 规则一致）
	int top = center ? (int)std::round(dh - 0.1f) : 0;
	int bottom = (int)std::round(dh + 0.1f);
	int left = center ? (int)std::round(dw - 0.1f) : 0;
	int right = (int)std::round(dw + 0.1f);

	cv::Mat tmp_image;
	cv::copyMakeBorder(resized, tmp_image, top, bottom, left, right,
		cv::BORDER_CONSTANT, cv::Scalar(pad_value, pad_value, pad_value));

	// BGR -> RGB（与 Predictor.preprocess一致）
	cv::cvtColor(tmp_image, tmp_image, cv::COLOR_BGR2RGB);

	// 保存 i2d / d2i（与 letterbox 的逻辑一致）
	i2d[0] = r;    i2d[1] = 0;    i2d[2] = (float)left;
	i2d[3] = 0;    i2d[4] = r;    i2d[5] = (float)top;
	cv::Mat M_i2d(2, 3, CV_32F, i2d);
	cv::Mat M_d2i(2, 3, CV_32F, d2i);
	cv::invertAffineTransform(M_i2d, M_d2i);

	// 转成 CHW 并归一化到 [0,1]
	m_real_input_data.resize(tmp_image.rows * tmp_image.cols * 3);
	OnnxInferenceBase::converHWCtoCHW<uchar, float>(tmp_image, m_real_input_data.data());
	for (size_t idx = 0; idx < m_real_input_data.size(); ++idx) {
		m_real_input_data[idx] = m_real_input_data[idx] / 255.0f;
	}
	return true;
}

bool Yolo_Inference::postprocess_data()
{
	return true;
}

bool Yolo_Inference::inference()
{
	bboxes.clear();

	auto input_dims = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	preprocess_data();

	std::array<int64_t, 4> input_shape = { 1, m_re_c, m_re_h, m_re_w };
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo, m_real_input_data.data(), m_real_input_data.size(), input_shape.data(), input_shape.size());

	// 定义输出维度
	int BatchSize = 1;
	int output_numbox = (int)output_dims[2];
	int output_numprob = (int)output_dims[1]; // 85 = 80类别+ 坐标
	int num_classes = (int)output_numprob - 4;
	//int output_numel = BatchSize * output_numbox * output_numprob;

	const char* input_names[] = { "images" };
	const char* output_names[] = { "output0" };
	std::vector <Ort::Value > outputTensor;
	try
	{
		outputTensor = session_->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
	}
	catch (const Ort::Exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	std::vector<float> output_data;
	if (outputTensor.size() > 0) {
		const auto& tensor_info = outputTensor[0].GetTensorTypeAndShapeInfo();
		std::vector<int64_t> shape = tensor_info.GetShape();
		int H = shape[1];
		int W = shape[2];
		std::vector<float> new_vector(W * H);

		size_t total_size = tensor_info.GetElementCount(); // 获取总元素数量  
		output_data.resize(total_size);

		// V8版本
		//memcpy(output_data.data(), outputTensor[0].GetTensorMutableData<float>(), total_size * sizeof(float));  // v8版本

		// v11 版本
		const float* tensor_data = outputTensor[0].GetTensorMutableData<float>();
#pragma omp parallel for
		for (int i = 0; i < H; ++i) {
			for (int j = 0; j < W; ++j) {

				output_data[j * H + i] = tensor_data[i * W + j];
			}
		}


		// output_data 维度是 B*BoxNum*class
		process_data(output_data.data(), output_numbox, output_numprob, num_classes);
	}


	return true;
}
