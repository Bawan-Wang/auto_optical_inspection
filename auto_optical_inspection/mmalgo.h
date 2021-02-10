#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "main.h"

class MMAlgo
{
public:
	explicit MMAlgo(void);
	~MMAlgo();

	void setInput(Mat oriMainboard, Mat oriTemplate);
	void RGB2Gray();
	void Gray2BWImg();
	void GetCircleRegion();
	void GetROIRegion();
	void ToDoMatching();
	void getObjectPosition();
	void getMainAngle();

	Mat mainboardRGBImg;
	Mat mainboardGrayImg;
	Mat mainboardBWImg;
	Mat mainboardROIBWImg;
	Mat mainboardROIRGBImg;
	Mat mainboardROIGrayImg;

	Mat templateRGBImg;
	Mat templateGrayImg;
	double stdScale;
	Mat matchImg;
	Mat transMatrix;
	vector<Point2f>template_corner_in_scene;
	double mainAlgo;
};

#endif
