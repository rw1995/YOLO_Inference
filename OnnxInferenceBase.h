#pragma once 
#include "onnxruntime_cxx_api.h" 
#include <memory>  
#include <vector>  
#include <codecvt>  
#include <iostream>
#include <locale>  


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/photo/photo.hpp"

typedef unsigned char uchar;


#pragma comment(lib, "onnxruntime.lib")
#pragma comment(lib, "onnxruntime_providers_cuda.lib")
#pragma comment(lib, "onnxruntime_providers_shared.lib")
#pragma comment(lib, "onnxruntime_providers_tensorrt.lib")


class OnnxInferenceBase
{
public:
	// 初始化模型
	OnnxInferenceBase(const std::wstring& model_path,
		const int re_height = 640,
		const int re_width = 640,
		const int re_channel = 3,
		std::string model_type="detect")
		: m_model_path(model_path),
		memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU)),
		session_(nullptr)
	{
		// 创建 ONNX Runtime 环境  
		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");

		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(1);
		session_options.SetInterOpNumThreads(1);

		OrtCUDAProviderOptions cuda_option;
		cuda_option.device_id = 0;
		cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda_option.gpu_mem_limit = SIZE_MAX;
		cuda_option.do_copy_in_default_stream = 0;
		cuda_option.has_user_compute_stream = 1;

		session_options.AppendExecutionProvider_CUDA(cuda_option);

		std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		std::wstring wstr = m_model_path/*converter.from_bytes(m_model_path)*/;

		try
		{
			session_ = std::make_unique<Ort::Session>(env, wstr.c_str(), session_options);
		}
		catch (std::exception& ex)
		{
			std::cout << "模型初始化失败:" << ex.what() << std::endl;
			return;
		}

		auto providers = Ort::GetAvailableProviders();
		for (const auto& provider : providers) {
			std::cout << "Available provider: " << provider << std::endl;
		}

		m_b_fix_mean_dev = false;
		m_re_c = re_channel;
		m_re_h = re_height;
		m_re_w = re_width;

		m_model_type = model_type;
	}





	virtual ~OnnxInferenceBase() {
		if (session_) {
			Ort::Session* temp = session_.release();  // Extracts the pointer, session_ becomes nullptr  
			// Custom release function based on ORT_DEFINE_RELEASE macro  
			Ort::GetApi().ReleaseSession(*temp);
		}


	}

	// 设置输入
	void setData(cv::Mat& input_data, uchar* out_data);		// 设置输入结果、输出结果
	virtual void setInputData(const cv::Mat& input_data);	// 设置输入结果

	// 获取输出
	void GetSegData(std::vector<short> &outdata);			// 获取分割结果
	void GeDetectData(std::vector<float>& outdata);		// 获取检测结果
	

	
	// 设置网络输入大小
	void setResizeShape(int re_h, int re_w, int re_c) { 
		m_re_h = re_h; m_re_w = re_w; m_re_c = re_c; };

	virtual bool preprocess_data();
	virtual bool postprocess_data();
	// 推理
	virtual bool inference();




public:

	void set_fix_mean_fix_dev(bool b_fix_mean_dev, std::vector<float> fix_mean_val, std::vector<float> fix_dev_std) {
		m_mean_val = fix_mean_val;
		m_dev_val = fix_dev_std;
		m_b_fix_mean_dev = b_fix_mean_dev;
	}

	std::wstring GetModelPath() const { return m_model_path; }

	// 模板 HWC 转 CHW
	template <typename T1, typename T2>
	void converHWCtoCHW(const cv::Mat& in_data, T2* out_data) {
		// 获取输入图像的高度、宽度和通道数  
		int tmp_h = in_data.rows;
		int tmp_w = in_data.cols;
		int tmp_c = in_data.channels();

		// 获取输入数据的指针  
		const T1* input_data_ptr = in_data.ptr<T1>(); // 使用指针访问数据  

		int img_width_times_channels = tmp_w * tmp_c;

		// 执行 HWC 到 CHW 的转换  
		for (int h = 0; h < tmp_h; ++h) {
			for (int w = 0; w < tmp_w; ++w) {
				for (int c = 0; c < tmp_c; ++c) {
					out_data[c * tmp_h * tmp_w + h * tmp_w + w] =
						static_cast<T2>(input_data_ptr[h * img_width_times_channels + w * tmp_c + c]);
				}
			}
		}
	}

	template <typename T1>
	void NormalizeData_By_mean_dev(std::vector<T1>& CHW_Data, std::vector<float> mean_vals, std::vector<float>dev_vals) {


		// 标准差存在0
		bool is_exist_zero = false;
		for (auto t_dev : dev_vals) {
			if (t_dev == 0)
				is_exist_zero = true;
		}
		if (is_exist_zero == true)
			return;

		// 正常流程
		for (int t_c = 0; t_c < m_re_c; ++t_c) {
			for (int t_c_idx = 0; t_c_idx < m_re_h * m_re_w; ++t_c_idx) {
				int t_idx = t_c * m_re_h * m_re_w + t_c_idx;
				CHW_Data[t_idx] = (CHW_Data[t_idx] - mean_vals[t_c]) / dev_vals[t_c];
			}
		}
	}



public:
	std::vector<const char*> m_input_names;
	std::vector<const char*> m_output_names;

	cv::Mat m_input_img;
	std::vector<short> m_out_seg_data;
	std::vector<float> m_out_detect_data;

	std::vector<float> m_real_input_data; // 推理输入指针
	std::vector<float> m_real_output_data;
	std::vector<float> m_tmp_data;

	uchar* m_output_data;

	int m_img_h=0, m_img_w=0, m_img_c=0;
	int m_img_size=0; // 图像输入长度
	int m_re_h=0, m_re_w=0, m_re_c=0;

	Ort::MemoryInfo memoryInfo;
	std::unique_ptr<Ort::Session> session_;

	
	std::vector<float> m_mean_val;
	std::vector<float> m_dev_val;
	bool m_b_fix_mean_dev = false;

	std::string m_model_type;
private:
	std::wstring m_model_path;
	std::unique_ptr<wchar_t[]> wchar_array_;
	Ort::Env env; // 将 Ort::Env 定义为类的成员  

};