/************************************************************************
*
*    Copyright:
*    File Name:   predictor.h
*    Description:  定义了Predictor 类,用于预测模型的载入,图像的预测和预测结果功能。
*
*    Version:        V1.00
*    Author:            文皓
*    Create Time:    2021-03-24
*
*************************************************************************/
#pragma once
#ifndef Predictor_H
#define Predictor_H

class __declspec(dllexport) Pred;

/*
简单用例:

Predictor *predictor = Predictor::getPredictorInstance(modelDir);
cv::Mat pim = predictor->predict(im);
double percent = predictor->getPercent(pim);

*/

#include <opencv2\opencv.hpp>
#include "paddle_inference_api.h"
#include <thread>

class Pred
{
private:

	std::shared_ptr<paddle_infer::Predictor> predictor; // 使用paddle的预测引擎
	std::deque<double> resultQueue;
	int queueSize;
	int normal_h, normal_w;

	/*
	用于图像的通道转化,从hwc转化到 chw

	用例

	std::shared_ptr<float> pim = hwc2chw(im);
	*/
	std::shared_ptr<float> hwc2chw(cv::Mat &im);

	void add(double precent);
public:

	/*
	构造函数

	用例:
	Predictor predictor = new Predictor(modelDir);
	*/

	explicit Pred(const std::string modelDir, int numThread = 0, bool mkldnnEnable = true, bool memoryEnable = true, int queueSize = 100, int normal_h = 512, int normal_w = 512);
	~Pred();

	/*
	用于获取预测图片 要调用此函数,你需要先实例化一个Predictor对象

	用例:
	cv::Mat pim = predictor->getPredictIm(im);
	*/

	cv::Mat getPredictIm(cv::Mat &im);

	/*
	用于图像的归一化和resize

	用例:
	cv::Mat normIm = normalize(im);
	*/
	cv::Mat normalize(cv::Mat &im);


	/*
	用于获取当前预测图片中固液占比 要调用此函数,你需要先实例化一个Predictor对象
	如果没有预测任何图片将会返回 1
	用例:
	double percent = predictor->getPercent(im);
	*/
	double getPercent();

	/*
	设置要处理的图片的大小
	*/
	void setSize(int h, int w);

	/*
	计算连续的queueSize大小的队列中平均固体占比。
	如果没有队列中没有任何元素，返回1
	用例:
	double percent = predictor->getPercent(im);
	*/
	double compute();

};

#endif // !Predictor_H
#pragma once
