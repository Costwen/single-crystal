/************************************************************************
*
*    Copyright:
*    File Name:   predictor.h
*    Description:  ������Predictor ��,����Ԥ��ģ�͵�����,ͼ���Ԥ���Ԥ�������ܡ�
*
*    Version:        V1.00
*    Author:            ���
*    Create Time:    2021-03-24
*
*************************************************************************/
#pragma once
#ifndef Predictor_H
#define Predictor_H

class __declspec(dllexport) Pred;

/*
������:

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

	std::shared_ptr<paddle_infer::Predictor> predictor; // ʹ��paddle��Ԥ������
	std::deque<double> resultQueue;
	int queueSize;
	int normal_h, normal_w;

	/*
	����ͼ���ͨ��ת��,��hwcת���� chw

	����

	std::shared_ptr<float> pim = hwc2chw(im);
	*/
	std::shared_ptr<float> hwc2chw(cv::Mat &im);

	void add(double precent);
public:

	/*
	���캯��

	����:
	Predictor predictor = new Predictor(modelDir);
	*/

	explicit Pred(const std::string modelDir, int numThread = 0, bool mkldnnEnable = true, bool memoryEnable = true, int queueSize = 100, int normal_h = 512, int normal_w = 512);
	~Pred();

	/*
	���ڻ�ȡԤ��ͼƬ Ҫ���ô˺���,����Ҫ��ʵ����һ��Predictor����

	����:
	cv::Mat pim = predictor->getPredictIm(im);
	*/

	cv::Mat getPredictIm(cv::Mat &im);

	/*
	����ͼ��Ĺ�һ����resize

	����:
	cv::Mat normIm = normalize(im);
	*/
	cv::Mat normalize(cv::Mat &im);


	/*
	���ڻ�ȡ��ǰԤ��ͼƬ�й�Һռ�� Ҫ���ô˺���,����Ҫ��ʵ����һ��Predictor����
	���û��Ԥ���κ�ͼƬ���᷵�� 1
	����:
	double percent = predictor->getPercent(im);
	*/
	double getPercent();

	/*
	����Ҫ�����ͼƬ�Ĵ�С
	*/
	void setSize(int h, int w);

	/*
	����������queueSize��С�Ķ�����ƽ������ռ�ȡ�
	���û�ж�����û���κ�Ԫ�أ�����1
	����:
	double percent = predictor->getPercent(im);
	*/
	double compute();

};

#endif // !Predictor_H
#pragma once
