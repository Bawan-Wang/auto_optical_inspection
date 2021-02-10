#include "main.h"
#include "mmalgo.h"

int main()
{

	Mat locatedImg = imread(".\\scene.jpg");
	Mat tamplateImg = imread(".\\template.jpg");

	MMAlgo mmalgo = MMAlgo();
	mmalgo.setInput(locatedImg, tamplateImg);
	mmalgo.RGB2Gray();
	mmalgo.Gray2BWImg();
	mmalgo.GetCircleRegion();
	mmalgo.GetROIRegion();
	mmalgo.ToDoMatching();
	mmalgo.getObjectPosition();
	mmalgo.getMainAngle();

	waitKey(0);
	locatedImg.release();
	tamplateImg.release();

	return 0;
}