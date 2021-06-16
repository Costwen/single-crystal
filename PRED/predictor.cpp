#include "predictor.h"

const float mean_value[] = { 0.485, 0.456, 0.406 };
const float std_value[] = { 0.229, 0.224, 0.225 };

std::shared_ptr<float> Pred::hwc2chw(cv::Mat & im)
{
	auto h = im.size().height, w = im.size().width;
	int b = 1;
	int c = im.channels();
	int nums = b * c * h * w;
	// nhwc -> nchw
	std::shared_ptr<float> input_data(new float[nums]);
	input_data.reset(new float[nums]);

	for (int i = 0; i < c; ++i) {
		cv::extractChannel(im, cv::Mat(h, w, CV_32FC1, input_data.get() + i * h * w), i);
	}
	return input_data;
}

void Pred::add(double precent)
{
	if (resultQueue.size() == queueSize) {
		resultQueue.pop_front();
	}
	resultQueue.push_back(precent);
}

Pred::Pred(const std::string modelDir, int numThread, bool mkldnnEnable, bool memoryEnable, int queueSize, int normal_h, int normal_w)
{

	this->queueSize = queueSize;
	this->normal_h = normal_h;
	this->normal_w = normal_w;


	paddle_infer::Config config;
	config.SetModel(modelDir + "/infer.pdmodel", modelDir + "/infer.pdparams");
	config.DisableGpu();

	/* mkldnn accelerate*/
	//if (mkldnnEnable) {
	//	std::unordered_set<std::string> op_list = { "softmax", "relu" };
	//	config.EnableMKLDNN();
	//	if (config.mkldnn_enabled()) std::cout << "mkldnn enabled\n";
	//	else std::cout << "mkldnn disabled\n";
	//}
	//memory accelerate
	if (memoryEnable) {
		config.SetMkldnnCacheCapacity(1);
		config.EnableMemoryOptim();

		if (config.enable_memory_optim()) std::cout << "memory enabled\n";
		else std::cout << "memory disabled\n";
	}
	// set cpu threads
	if (numThread == 0) {
		numThread = std::thread::hardware_concurrency();
	}
	config.SetCpuMathLibraryNumThreads(numThread);

	std::cout << numThread << "\n";
	predictor = paddle_infer::CreatePredictor(config);
}

Pred::~Pred()
{
	resultQueue.clear();
	resultQueue.swap(std::deque<double>());

	predictor.reset();
}

cv::Mat return_im;
cv::Mat Pred::getPredictIm(cv::Mat & im)
{
	return_im.release();

	auto input_data = this->hwc2chw(im);
	if (input_data == NULL)
	{
		input_data.reset();
		return return_im;
	}

	// create input_data
	auto input_names = predictor->GetInputNames();

	auto input_t = predictor->GetInputHandle(input_names[0]);

	std::vector<int> input_shape = { 1, im.channels(), im.size().height, im.size().width };

	input_t->Reshape(input_shape);
	input_t->CopyFromCpu(input_data.get());

	predictor->Run();
	//std::cout << 1 << "\n";
	//get out_data
	auto output_names = predictor->GetOutputNames();
	auto output_t = predictor->GetOutputHandle(output_names[0]);
	std::vector<int> output_shape = output_t->shape();

	int out_num = 1;

	for (auto i : output_shape) out_num = out_num * i;
	std::vector<float> out_data;
	out_data.resize(out_num);
	output_t->CopyToCpu(out_data.data());

	//std::cout << 1 << "\n";
	int height = im.size().height, width = im.size().width;
	return_im = cv::Mat::zeros(im.size(), CV_8UC3);
	int solidcnt = 0, liquidcnt = 0;
	for (int h = 0; h < height; h++)
		for (int w = 0; w < width; w++) {
			float b = out_data[0 * (height*width) + h*width + w];
			float g = out_data[1 * (height*width) + h*width + w];
			float r = out_data[2 * (height*width) + h*width + w];
			if (r >= g && r >= b) {
				return_im.data[h * (3 * width) + w * 3 + 0] = 255;
			}
			else if (g >= r && g >= b) {
				return_im.data[h * (3 * width) + w * 3 + 1] = 255;
				liquidcnt++;
			}
			else {
				return_im.data[h * (3 * width) + w * 3 + 2] = 255;
				solidcnt++;
			}
		}
	if (liquidcnt + solidcnt == 0) {
		add(1.0);
	}
	else {
		add(solidcnt*1.0 / (solidcnt + liquidcnt));
	}

	input_data.reset();
	input_shape.swap(std::vector<int>());
	output_shape.swap(std::vector<int>());
	out_data.swap(std::vector<float>());
	input_t.release();
	output_t.release();

	for (auto i = 0; i < input_names.size(); i++)
	{
		input_names[i].clear();
	}
	input_names.swap(std::vector<cv::String >());

	for (auto i = 0; i < output_names.size(); i++)
	{
		output_names[i].clear();
	}
	output_names.swap(std::vector<cv::String >());

	return return_im;
}

cv::Mat outs;
cv::Mat Pred::normalize(cv::Mat & im)
{
	// resize
	outs.release();

	auto size = cv::Size(this->normal_w, this->normal_h);
	outs = cv::Mat::zeros(size, CV_8UC3);
	cv::resize(im, outs, size, cv::INTER_LINEAR);
	cv::cvtColor(outs, outs, cv::COLOR_BGR2RGB);

	// normalize
	outs.convertTo(outs, CV_32F, 1.0 / 255, 0);
	std::vector<cv::Mat> rgbChannels(3);
	cv::split(outs, rgbChannels);

	for (auto i = 0; i < rgbChannels.size(); i++)
	{
		rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
	}
	cv::merge(rgbChannels, outs);

	//ÇåÀí
	for (auto i = 0; i < rgbChannels.size(); i++)
	{
		rgbChannels[i].release();
	}
	rgbChannels.swap(std::vector<cv::Mat>());

	return outs;
}

double Pred::getPercent()
{
	if (resultQueue.empty()) {
		return 1.0;
	}
	return resultQueue.back();
}

void Pred::setSize(int h, int w)
{
	this->normal_h = h;
	this->normal_w = w;
}

double Pred::compute()
{
	double total = 0; int cnt = 0;
	for (auto percent : resultQueue) {
		cnt++;
		total += percent;
	}
	if (!cnt) return 1.0;
	return total / cnt;
}
