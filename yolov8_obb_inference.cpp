//#include "Yolo_Inference.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <string>
//
//using namespace std;
//using namespace cv;
//
//int main() {
//	std::wstring onnx_path = L"F:/Project/S-ShuoBeiDei/xizhu_data/test/SBD_YiXi.onnx";
//	Yolo_Inference detect_infer(onnx_path, 0.3, 0.3);
//
//
//	string img_folder = "F:/Project/S-ShuoBeiDei/xizhu_data/test/crop_2/";
//	string img_path = img_folder + "SRR_Temp_20250715095417177_Group-1-1_1_Src.bmp";
//	std::vector<string> img_list;
//	cv::glob(img_folder, img_list);
//
//	for (auto tmp_img_pth : img_list) {
//		std::vector<uchar>coord_index(4);
//		Mat img = cv::imread(tmp_img_pth, 1);
//		detect_infer.setResizeShape(1024, 1024, 3);
//		detect_infer.setData(img, coord_index.data());
//		detect_infer.inference();
//		std::vector < std::vector<float>> boxes = detect_infer.get_boxes();
//
//		//Mat img_cvt;
//		//cv::resize(img, img_cvt, Size(1024, 1024));
//		for (int idx = 0; idx < boxes.size(); idx++) {
//			std::vector<float> tmp_box = boxes[idx];
//			cv::Point topLeft(tmp_box[0], tmp_box[1]);
//			cv::Point bottomRight(tmp_box[2], tmp_box[3]);
//			cv::Scalar color(0, 255, 0);  // 绿色
//
//			// 定义线条厚度（如果 thickness < 0，表示填充矩形）
//			int thickness = 1;  // 线条厚度为 2
//
//
//			cv::rectangle(img, topLeft, bottomRight, color, thickness);
//		}
//
//		std::vector<std::vector<float>> pred_data = detect_infer.get_boxes();
//		std::vector<float> pred_data_arr = detect_infer.get_boxes_arr();
//		cv::imshow("Rectangle Example", img);
//		cv::waitKey(0);  // 等待按键
//	}
//	
//}



#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

struct RotatedBox {
	cv::RotatedRect rect;
	float conf;
	int cls;
};

float IoU_Rotated(const RotatedRect& r1, const RotatedRect& r2) {
	vector<Point2f> inter_pts;
	int result = cv::rotatedRectangleIntersection(r1, r2, inter_pts);
	if (result == INTERSECT_NONE) return 0.0f;
	if (result == INTERSECT_FULL) return 1.0f;
	double inter_area = contourArea(inter_pts);
	double union_area = r1.size.area() + r2.size.area() - inter_area;
	return (float)(inter_area / union_area);
}

vector<RotatedBox> NMS_Rotated(vector<RotatedBox>& boxes, float iou_thresh) {
	sort(boxes.begin(), boxes.end(),
		[](const RotatedBox& a, const RotatedBox& b) { return a.conf > b.conf; });

	vector<RotatedBox> result;
	vector<bool> removed(boxes.size(), false);

	for (size_t i = 0; i < boxes.size(); ++i) {
		if (removed[i]) continue;
		result.push_back(boxes[i]);
		for (size_t j = i + 1; j < boxes.size(); ++j) {
			if (removed[j]) continue;
			if (IoU_Rotated(boxes[i].rect, boxes[j].rect) > iou_thresh)
				removed[j] = true;
		}
	}
	return result;
}

cv::Mat preprocess(const cv::Mat& img, int new_w, int new_h, float& scale, int& dw, int& dh) {
	int w = img.cols, h = img.rows;
	scale = std::min((float)new_w / w, (float)new_h / h);
	int nw = int(w * scale), nh = int(h * scale);
	dw = (new_w - nw) / 2;
	dh = (new_h - nh) / 2;

	cv::Mat resized;
	cv::resize(img, resized, cv::Size(nw, nh));
	cv::Mat padded(new_h, new_w, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(padded(Rect(dw, dh, nw, nh)));

	cv::cvtColor(padded, padded, COLOR_BGR2RGB);
	padded.convertTo(padded, CV_32F, 1.0 / 255.0);
	return padded;
}

int main() {
	const wchar_t* model_path = L"F:/tmp/MJ.onnx";
	string img_path = "F:/tmp/MJ_1.bmp";

	// 1️⃣ 初始化 ONNX Runtime
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_obb");
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	Ort::Session session(env, model_path, session_options);




	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name = session.GetInputNameAllocated(0, allocator);
	auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	int input_h = input_shape[2];
	int input_w = input_shape[3];

	// 2️⃣ 读取与预处理图像
	cv::Mat img = cv::imread(img_path);

	float scale; int dw, dh;
	cv::Mat input = preprocess(img, input_w, input_h, scale, dw, dh);

	// 转为 NCHW
	std::vector<float> input_tensor_values;
	for (int c = 0; c < 3; ++c)
		for (int y = 0; y < input.rows; ++y)
			for (int x = 0; x < input.cols; ++x)
				input_tensor_values.push_back(input.at<Vec3f>(y, x)[c]);



	std::array<int64_t, 4> input_dims = { 1, 3, input_h, input_w };
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info, input_tensor_values.data(), input_tensor_values.size(),
		input_dims.data(), input_dims.size());

	// 3️⃣ 推理
	auto output_name = session.GetOutputNameAllocated(0, allocator);
	const char* input_names[] = { input_name.get() };
	const char* output_names[] = { output_name.get() };
	
	auto output_tensors = session.Run(
		Ort::RunOptions{ nullptr },
		input_names, &input_tensor, 1,
		output_names, 1);

	float* output = output_tensors.front().GetTensorMutableData<float>();
	auto out_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
	int64_t ch = out_shape[1];      // 7
	int64_t num_det = out_shape[2]; // 21504

	std::vector<float> detections(num_det * ch);

	// 转置: [ch, num_det] -> [num_det, ch]
	for (int c = 0; c < ch; ++c) {
		for (int i = 0; i < num_det; ++i) {
			detections[i * ch + c] = output[c * num_det + i];
		}
	}

	int det_dim = num_det;


	// 4️⃣ 解析检测结果
	std::vector<RotatedBox> boxes;
	int num_classes = ch-5; // 你的类别数
	for (int i = 0; i < num_det; ++i) {
		float* det = &detections[i * ch];
		float x = (det[0] - dw) / scale;
		float y = (det[1] - dh) / scale;
		float w = det[2] / scale;
		float h = det[3] / scale;

		// 找最大类别分数和类别索引
		float max_conf = det[4];
		int cls = 0;
		for (int k = 1; k < num_classes; ++k) {
			if (det[4 + k] > max_conf) {
				max_conf = det[4 + k];
				cls = k;
			}
		}

		if (max_conf < 0.3) continue; // 阈值过滤

		float angle = det[4 + num_classes]; // 最后一个是angle
		float angle_deg = angle * 180.0f / CV_PI;

		RotatedRect rect(Point2f(x, y), Size2f(w, h), angle_deg);
		boxes.push_back({ rect, max_conf, cls });
	}

	// 5️⃣ NMS（旋转框）
	auto nms_boxes = NMS_Rotated(boxes, 0.3f);

	// 6️⃣ 绘制结果
	for (auto& b : nms_boxes) {
		Point2f pts[4];
		b.rect.points(pts);
		for (int j = 0; j < 4; j++)
			line(img, pts[j], pts[(j + 1) % 4], Scalar(0, 255, 0), 2);
		putText(img, to_string(b.cls) + ":" + to_string(b.conf).substr(0, 4),
			pts[0], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
	}

	cv::imwrite("result.jpg", img);
	cout << "检测完成，共 " << nms_boxes.size() << " 个旋转目标。" << endl;
	return 0;
}
