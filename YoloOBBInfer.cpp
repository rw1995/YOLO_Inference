

#include "YoloOBBInfer.h"
#include <algorithm>
#include <iostream>
#include <numeric>

using namespace cv;

bool YoloOBBInfer::preprocess_data() {


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

float YoloOBBInfer::IoU_Rotated(const RotatedRect& r1, const RotatedRect& r2) {
    std::vector<Point2f> inter_pts;
    int result = rotatedRectangleIntersection(r1, r2, inter_pts);
    if (result == INTERSECT_NONE) return 0.0f;
    if (result == INTERSECT_FULL) return 1.0f;
    double inter_area = contourArea(inter_pts);
    double union_area = r1.size.area() + r2.size.area() - inter_area;
    if (union_area <= 0) return 0.0f;
    return static_cast<float>(inter_area / union_area);
}

void YoloOBBInfer::nms_rotated() {
    // 输入: m_obb_result = [xc, yc, w, h, angle_deg, cls, conf]
    std::vector<int> idxs(m_obb_result.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return m_obb_result[a][6] > m_obb_result[b][6]; });

    std::vector<bool> removed(m_obb_result.size(), false);
    std::vector<std::vector<float>> keep;

    for (size_t i = 0; i < idxs.size(); ++i) {
        int ai = idxs[i];
        if (removed[ai]) continue;
        keep.push_back(m_obb_result[ai]);

        RotatedRect ra(Point2f(m_obb_result[ai][0], m_obb_result[ai][1]),
                       Size2f(m_obb_result[ai][2], m_obb_result[ai][3]),
                       m_obb_result[ai][4]);
        for (size_t j = i + 1; j < idxs.size(); ++j) {
            int bj = idxs[j];
            if (removed[bj]) continue;
            if (static_cast<int>(m_obb_result[ai][5]) != static_cast<int>(m_obb_result[bj][5])) continue;
            RotatedRect rb(Point2f(m_obb_result[bj][0], m_obb_result[bj][1]),
                           Size2f(m_obb_result[bj][2], m_obb_result[bj][3]),
                           m_obb_result[bj][4]);
            if (IoU_Rotated(ra, rb) > m_nms_threshold) removed[bj] = true;
        }
    }
    m_obb_result.swap(keep);
}

void YoloOBBInfer::process_data(float* output_data_host, int num_det, int ch, int num_classes) {
    m_obb_result.clear();
    m_obb_result.reserve(num_det);

    for (int i = 0; i < num_det; ++i) {
        float* det = output_data_host + i * ch;
        float x = det[0];
        float y = det[1];
        float w = det[2];
        float h = det[3];

        int cls = 0;
        float max_conf = det[4];
        for (int k = 1; k < num_classes; ++k) {
            if (det[4 + k] > max_conf) {
                max_conf = det[4 + k];
                cls = k;
            }
        }
        if (max_conf < m_confidence_threshold) {
            continue;
        };

        float angle_rad = det[4 + num_classes];
        float angle_deg = angle_rad * 180.0f / CV_PI;

        // 坐标从网络输入空间映射回原图空间
        float x_img = d2i[0] * x + d2i[2];
        float y_img = d2i[4] * y + d2i[5];
        float w_img = d2i[0] * w; // 等比例缩放，d2i[0] == d2i[4]
        float h_img = d2i[4] * h;

        if (x_img >= 0 && y_img >= 0 && w_img > 0 && h_img > 0) {
            m_obb_result.push_back({ x_img, y_img, w_img, h_img, angle_deg, static_cast<float>(cls), max_conf });
        }
    }

    // 旋转框 NMS
    nms_rotated();
}

bool YoloOBBInfer::inference() {
    auto input_dims = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_dims = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    // 注意：必须先设置 m_re_h/m_re_w（通过 setResizeShape）
    preprocess_data();

    std::array<int64_t, 4> input_shape = { 1, m_re_c, m_re_h, m_re_w };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo, m_real_input_data.data(), (size_t)m_real_input_data.size(), input_shape.data(), input_shape.size());

    int output_numbox = static_cast<int>(output_dims[2]);
    int output_numprob = static_cast<int>(output_dims[1]); // ch
    int num_classes = output_numprob - 5; // 最后一个是 angle

    const char* input_names[] = { "images" };
    const char* output_names[] = { "output0" };
    std::vector<Ort::Value> outputTensor;
    try {
        outputTensor = session_->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
    } catch (const Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return false;
    }

    if (outputTensor.empty()) return false;

    const auto& tensor_info = outputTensor[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();
    int H = static_cast<int>(shape[1]);
    int W = static_cast<int>(shape[2]);

    m_real_output_data.resize((size_t)H * (size_t)W);
    const float* tensor_data = outputTensor[0].GetTensorMutableData<float>();

    // 转置为 [num_det, ch]
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            m_real_output_data[j * H + i] = tensor_data[i * W + j];
        }
    }

    process_data(m_real_output_data.data(), output_numbox, output_numprob, num_classes);
    return true;
}