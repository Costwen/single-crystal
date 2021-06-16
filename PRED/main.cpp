#include <opencv2/opencv.hpp>
#include "predictor.h"
#include "windows.h"

int main(int argc, char** argv) {

	//std::string src_image_path = argv[1];
	std::string src_image_path = "F:/ML/84810.png";
	//std::string predict_model_dir = argv[2];
	std::string predict_model_dir = "F:/ML/Project/Seg2/infer_model";


	LPVOID imBuffer;
	std::string strMapName("global_share_memory");
	HANDLE hMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, 0, strMapName.c_str());


	Pred* predictor = new Pred(predict_model_dir,0,true,true,100,512,512);

	for (int i = 1; i <= 1000; i++) {
		float start_time = clock();
		cv::Mat image = cv::imread(src_image_path);
		auto scaled_image = predictor->normalize(image);

		auto rgb_predict = predictor->getPredictIm(scaled_image);
		//
		printf("%lf\n", predictor->getPercent());
		float end_time = clock();
		if (i % 10 == 0) {
			printf("%lf\n", predictor->compute());
		}
		std::cout << end_time - start_time << "\n";
	}

}