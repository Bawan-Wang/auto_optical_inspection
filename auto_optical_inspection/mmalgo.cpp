#include "mmalgo.h"

double squr(double num, int bb)
{
	double temp = 1;
	for (int i = 0; i < bb; i++)
		temp = temp * num;

	return temp;
}

void grayImgSlicing(Mat grayImg)
{
	int transArr[256] = { 0 };
	Scalar s;
	for (int i = 0; i < 256; i++)
	{
		transArr[i] = (int)(255 * squr(((double)(i + 1) / 256), 3));
	}
	for (int i = 0; i < grayImg.rows; i++)
	{
		for (int j = 0; j < grayImg.cols; j++)
		{
			s[0] = grayImg.at<unsigned char>(i, j);
			s[0] = transArr[(unsigned char)(s[0])];
			grayImg.at<unsigned char>(i, j) = (unsigned char)s[0];
		}
	}
}

void GetCircles(Mat inBWImg, Mat outBWImg)
{
	CvScalar s;

	IplImage iplBWImg = IplImage(inBWImg);
	
	CvSeq* contour = 0;
	CvMemStorage* storage = cvCreateMemStorage();
	//Get all contours
	cvFindContours(&iplBWImg, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	outBWImg.setTo(0);
	IplImage iplOutBWImg = IplImage(outBWImg);
	s.val[0] = 255;
	for (; contour != 0; contour = contour->h_next)
	{
		double area = fabs(cvContourArea(contour, CV_WHOLE_SEQ));        //Get area
		if (area<10 || area > squr(64, 2) / 4)                           //filtered small region
			continue;
		double length = contour->total;

		double ratio = squr(length, 2) / (4 * CV_PI*area);
		//cout<<ratio<<endl;  //ratio °¾¤j
		if (ratio >= 0.4 && ratio < 1)                                    //check whether the area is circle or not(from 0.4 to 1)
		{
			CvPoint *point = new CvPoint[contour->total];
			CvPoint *Point;
			for (int i = 0; i < contour->total; i++)
			{
				Point = (CvPoint*)cvGetSeqElem(contour, i);
				point[i].x = Point->x;
				point[i].y = Point->y;
			}
			int ptnum[1] = { contour->total };
			cvFillPoly(&iplOutBWImg, &point, ptnum, 1, CV_RGB(255, 255, 255));

		}

	}
}

//Get the strength of horizontal or vertical
vector<int> FindStrength(Mat pic, int type) //type = 0 : vertical;type = 1 : horizontal
{
	int temp;
	vector<int> rtnArr;
	if (type == 0)
	{
		rtnArr.resize(pic.cols);
		for (int i = 0; i < pic.cols; i++)
		{
			rtnArr[i] = 0;
			for (int j = 0; j < pic.rows; j++)
			{
				temp = pic.at<unsigned char>(j, i);
				if (temp == 255)
				{
					rtnArr[i]++;
				}

			}
		}
	}
	else
	{
		rtnArr.resize(pic.rows);
		for (int i = 0; i < pic.rows; i++)
		{
			rtnArr[i] = 0;
			for (int j = 0; j < pic.cols; j++)
			{
				temp = pic.at<unsigned char>(i, j);
				if (temp == 255)
				{
					rtnArr[i]++;
				}

			}
		}

	}


	return rtnArr;
}

//Calculate mean and variance fromm an array
int* calAvgVar(vector<int> arr, double stdscale)
{
	int* rtn = new int[4];
	double sum = 0;
	double mean = 0;
	double avg = 0;

	for (int i = 0; i < (int)arr.size(); i++)
	{
		sum += arr[i];
	}

	for (int i = 0; i < (int)arr.size(); i++)//Calculate mean
	{
		mean = mean + (i + 1)*(arr[i] / sum);
	}
	for (int i = 0; i < (int)arr.size(); i++)//Calculate variance
	{
		avg = avg + squr(((i + 1) - mean), 2)*(arr[i] / sum);
	}

	avg = pow(avg, 0.5);

	rtn[0] = (int)(mean + stdscale * avg);
	rtn[1] = (int)(mean - stdscale * avg);
	rtn[2] = (int)(mean + (stdscale + 0.5)*avg);
	rtn[3] = (int)(mean - (stdscale + 0.5)*avg);

	for (int i = 0; i < 4; i++)
	{
		if (i == 1 || i == 3)
		{
			if (rtn[i] <= 0)
			{
				rtn[i] = 1;
			}
		}
		else
		{
			if (rtn[i] > arr.size())
			{
				rtn[i] = (int)arr.size();
			}

		}
	}
	return rtn;
}

void findROIRegion(Mat inBWImg, Mat ROIBWImg, Mat ROIRGBImg, double stdscale)
{
	vector<int> vertical = FindStrength(inBWImg, 0);
	vector<int> horizon = FindStrength(inBWImg, 1);

	int* horRange = calAvgVar(vertical, stdscale);
	int* verRange = calAvgVar(horizon, stdscale);

	CvScalar s;
	s.val[0] = 0;
	// Delete everything if they are over ROI region
	for (int i = 0; i < inBWImg.rows; i++)
	{
		for (int j = 0; j < inBWImg.cols; j++)
		{
			if (*(verRange + 2) > i && i > *(verRange + 3))  // multiple 2
				if (*(horRange + 2) > j && j > *(horRange + 3))
					continue;
			ROIBWImg.at<unsigned char>(i, j) = 0;
			Vec3b noColor = Vec3b(0, 0, 0);
			ROIRGBImg.at<Vec3b>(i, j) = noColor;
			//cvSet2D(ROIBWImg, i, j, s);
			//cvSet2D(ROIRGBImg, i, j, cvScalar(0, 0, 0));
		}
	}
}

MMAlgo::MMAlgo()
{

	mainboardRGBImg = NULL;
	mainboardGrayImg = NULL;
}

MMAlgo::~MMAlgo()
{
	mainboardRGBImg.release();
	mainboardGrayImg.release();
}

void MMAlgo::setInput(Mat oriMainboard, Mat oriTemplate)
{
	oriMainboard.copyTo(mainboardRGBImg);
	oriTemplate.copyTo(templateRGBImg);
}

void MMAlgo::RGB2Gray()
{
	mainboardGrayImg.create(mainboardRGBImg.rows, mainboardRGBImg.cols, CV_8UC1);
	cvtColor(mainboardRGBImg, mainboardGrayImg, CV_BGR2GRAY);
	templateGrayImg.create(templateRGBImg.rows, templateRGBImg.cols, CV_8UC1);
	cvtColor(templateRGBImg, templateGrayImg, CV_BGR2GRAY);
#ifdef DEBUG_MODE
	namedWindow("Debug:Gray Image", 0);
	resizeWindow("Debug:Gray Image", mainboardGrayImg.cols / 4, mainboardGrayImg.rows / 4);
	imshow("Debug:Gray Image", mainboardGrayImg);
	namedWindow("Debug:Template Image", 0);
	resizeWindow("Debug:Template Image", templateGrayImg.cols / 4, templateGrayImg.rows / 4);
	imshow("Debug:Template Image", templateGrayImg);
#endif
}

void MMAlgo::Gray2BWImg()
{
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat hist;
	calcHist(&mainboardGrayImg, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	Scalar total = sum(hist);
#ifdef DEBUG_MODE
	cout << "total : "<< (int)total[0] << endl;
#endif
	Mat zero_array = Mat::zeros((histSize / 2), 1, CV_32FC1);
	Mat one_array = Mat::ones((histSize / 2), 1, CV_32FC1);
	Mat combine;
	vconcat(zero_array, one_array, combine);
	double half = hist.dot(combine);
#ifdef DEBUG_MODE
	cout << "half : " << half << endl;
#endif
	if (((double)(half) / (double)(total[0])) > 0.25) //if half > total, it means the picture of mainboard is overexposed
		grayImgSlicing(mainboardGrayImg);

	threshold(mainboardGrayImg, mainboardBWImg, 0, 255, THRESH_BINARY | THRESH_OTSU);

#ifdef DEBUG_MODE
	namedWindow("Debug:BW Image", 0);
	resizeWindow("Debug:BW Image", mainboardBWImg.cols / 4, mainboardBWImg.rows / 4);
	imshow("Debug:BW Image", mainboardBWImg);
#endif
}

void MMAlgo::GetCircleRegion()
{
	GetCircles(mainboardBWImg, mainboardBWImg);  //Get circles from binary image

#ifdef DEBUG_MODE
	namedWindow("Debug:Circle Region", 0);
	resizeWindow("Debug:Circle Region", mainboardBWImg.cols / 4, mainboardBWImg.rows / 4);
	imshow("Debug:Circle Region", mainboardBWImg);
#endif
}

void MMAlgo::GetROIRegion()
{
	mainboardBWImg.copyTo(mainboardROIBWImg);
	mainboardRGBImg.copyTo(mainboardROIRGBImg);
	
	findROIRegion(mainboardBWImg, mainboardROIBWImg, mainboardROIRGBImg, stdScale);
#ifdef DEBUG_MODE
	namedWindow("Debug:ROI BW Image", 0);
	resizeWindow("Debug:ROI BW Image", mainboardROIBWImg.cols / 4, mainboardROIBWImg.rows / 4);
	imshow("Debug:ROI BW Image", mainboardROIBWImg);
	namedWindow("Debug:ROI RGB Image", 0);
	resizeWindow("Debug:ROI RGB Image", mainboardROIRGBImg.cols / 4, mainboardROIRGBImg.rows / 4);
	imshow("Debug:ROI RGB Image", mainboardROIRGBImg);
#endif
}

void MMAlgo::ToDoMatching()
{
	//detect kp
	BRISK brisk_detector;
	vector<KeyPoint> template_kp, mainboard_kp;

	mainboardROIGrayImg.create(mainboardGrayImg.rows, mainboardGrayImg.cols, CV_8UC1);
	cvtColor(mainboardROIRGBImg, mainboardROIGrayImg, CV_BGR2GRAY);

	template_kp.reserve(5000);
	mainboard_kp.reserve(5000);
	brisk_detector.detect(templateGrayImg, template_kp);
	brisk_detector.detect(mainboardROIGrayImg, mainboard_kp);

	//extract BRISK descriptor
	Mat template_desc, mainboard_desc;
	brisk_detector.compute(templateGrayImg, template_kp, template_desc);
	brisk_detector.compute(mainboardROIGrayImg, mainboard_kp, mainboard_desc);

	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches;
	matches.reserve(5000);
	matcher.match(template_desc, mainboard_desc, matches);
#ifdef DEBUG_MODE
	cout << "number of matched points: " << matches.size() << endl;
#endif

	//find good matched points
	vector<DMatch> good_matches;
	double minDist = 1000, good_matches_th;
	for (int i = 0; i < template_desc.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;
		}
	}
	good_matches_th = max(2 * minDist, 50.0);
	for (int i = 0; i < template_desc.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < good_matches_th)
		{
			good_matches.push_back(matches[i]);
		}
	}

	vector<DMatch> zero_matches;
	drawMatches(templateRGBImg, template_kp, mainboardRGBImg, mainboard_kp, zero_matches, matchImg);

	vector<Point2f> template_match_kp;
	vector<Point2f> mainboard_match_kp;
	template_match_kp.reserve(3000);
	mainboard_match_kp.reserve(3000);
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		template_match_kp.push_back(template_kp[good_matches[i].queryIdx].pt);
		mainboard_match_kp.push_back(mainboard_kp[good_matches[i].trainIdx].pt);

	}
	//Generate transformation matrix(Homography matrix)
	transMatrix = findHomography(template_match_kp, mainboard_match_kp, RANSAC);

	vector<Point2f>().swap(template_match_kp);
	vector<Point2f>().swap(mainboard_match_kp);
	vector<DMatch>().swap(matches);
	template_desc.release();
	mainboard_desc.release();
	vector<KeyPoint>().swap(template_kp);
	vector<KeyPoint>().swap(mainboard_kp);
#ifdef DEBUG_MODE
	namedWindow("Debug:ROI Gray Image", 0);
	resizeWindow("Debug:ROI Gray Image", mainboardROIGrayImg.cols / 4, mainboardROIGrayImg.rows / 4);
	imshow("Debug:ROI Gray Image", mainboardROIGrayImg);
#endif
}

void MMAlgo::getObjectPosition() {
	vector<Point2f>template_corner(4);
	template_corner[0] = Point(0, 0);
	template_corner[1] = Point(templateGrayImg.cols, 0);
	template_corner[2] = Point(templateGrayImg.cols, templateGrayImg.rows);
	template_corner[3] = Point(0, templateGrayImg.rows);

	//Corners transform
	perspectiveTransform(template_corner, template_corner_in_scene, transMatrix);
	line(matchImg, template_corner_in_scene[0] + Point2f(templateGrayImg.cols, 0), template_corner_in_scene[1] + Point2f(templateGrayImg.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	line(matchImg, template_corner_in_scene[1] + Point2f(templateGrayImg.cols, 0), template_corner_in_scene[2] + Point2f(templateGrayImg.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	line(matchImg, template_corner_in_scene[2] + Point2f(templateGrayImg.cols, 0), template_corner_in_scene[3] + Point2f(templateGrayImg.cols, 0), Scalar(0, 0, 255), 2, 8, 0);
	line(matchImg, template_corner_in_scene[3] + Point2f(templateGrayImg.cols, 0), template_corner_in_scene[0] + Point2f(templateGrayImg.cols, 0), Scalar(0, 0, 255), 2, 8, 0);

	cout << "(" << template_corner_in_scene[0].x << ", " << template_corner_in_scene[0].y << ")" << " ";
	cout << "(" << template_corner_in_scene[1].x << ", " << template_corner_in_scene[1].y << ")" << endl;
	cout << "(" << template_corner_in_scene[2].x << ", " << template_corner_in_scene[2].y << ")" << " ";
	cout << "(" << template_corner_in_scene[3].x << ", " << template_corner_in_scene[3].y << ")" << endl;

	namedWindow("match demo", 0);
	resizeWindow("match demo", matchImg.cols / 4, matchImg.rows / 4);
	imshow("match demo", matchImg);
}

void PrintMat(Mat A)
{
	for (int i = 0; i < A.rows; i++)
	{
		for (int j = 0; j < A.cols; j++)
			cout << A.at<double>(i, j) << " ";
		cout << endl;
	}
	cout << endl;
}

void MMAlgo::getMainAngle() {
	double tmpAlg;

	//Get posibily angles
	if (transMatrix.at<double>(0, 0) <= 1) {
		tmpAlg = (acos(transMatrix.at<double>(0, 0)) * 180) / CV_PI;
	}
	else {
		tmpAlg = 0;
	}
#ifdef DEBUG_MODE
	PrintMat(transMatrix);
	cout << transMatrix.at<double>(0, 0) << " " << tmpAlg << endl;
#endif

	//Get center position in the scene
	Point2f template_center_in_scene;
    double min_v, max_v, min_h, max_h;
	min_v = min_h = 32767.0;
	max_v = max_h = -1.0;
	for (int i = 0; i < 4; i++) {
		if (template_corner_in_scene[i].x > max_h) { max_h = template_corner_in_scene[i].x; }
		if (template_corner_in_scene[i].y > max_v) { max_v = template_corner_in_scene[i].y; }
		if (template_corner_in_scene[i].x < min_h) { min_h = template_corner_in_scene[i].x; }
		if (template_corner_in_scene[i].y < min_v) { min_v = template_corner_in_scene[i].y; }
	}
	template_center_in_scene.x = (max_h + min_h) / 2;
	template_center_in_scene.y = (max_v + min_v) / 2;
	cout << "center_x :" << template_center_in_scene.x << "center_y :" << template_center_in_scene.y << endl;

	Mat zero_image = Mat::zeros(mainboardGrayImg.rows, mainboardGrayImg.cols, CV_8U);

	int thita, y, x, offset_x, offset_y, new_center_in_scene_x, new_center_in_scene_y, shift_x, shift_y;
	double angle[2];
	offset_x = template_center_in_scene.x - (templateGrayImg.cols / 2);
	offset_x = (offset_x >= 0) ? offset_x : 0;
	offset_y = template_center_in_scene.y - (templateGrayImg.rows / 2);
	offset_y = (offset_y >= 0) ? offset_y : 0;
	angle[0] = (tmpAlg * CV_PI) / 180.;
	angle[1] = ((360. - tmpAlg) * CV_PI) / 180.;
	thita = 0;

	int mae[2] = { 0 };
	for (thita = 0; thita < 2; thita++) {
		new_center_in_scene_x = template_center_in_scene.x * cos(angle[thita]) - template_center_in_scene.y * sin(angle[thita]);
		new_center_in_scene_y = template_center_in_scene.x * sin(angle[thita]) + template_center_in_scene.y * cos(angle[thita]);
		shift_x = template_center_in_scene.x - new_center_in_scene_x;
		shift_y = template_center_in_scene.y - new_center_in_scene_y;
		for (y = 0; y < templateGrayImg.rows; y++) {
			for (x = 0; x < templateGrayImg.cols; x++) {
				int x2, y2, x3, y3;
				x2 = x + offset_x;
				y2 = y + offset_y;
				x3 = x2 * cos(angle[thita]) - y2 * sin(angle[thita]) + shift_x;
				y3 = x2 * sin(angle[thita]) + y2 * cos(angle[thita]) + shift_y;
				zero_image.at<uchar>(y3, x3) = templateGrayImg.at<uchar>(y, x);
				mae[thita] += abs(mainboardGrayImg.at<uchar>(y3, x3) - templateGrayImg.at<uchar>(y, x));
			}
		}
#ifdef DEBUG_MODE
		cout << "mae[" << thita << "] : " << mae[thita] / (templateGrayImg.rows * templateGrayImg.cols) << endl;
#endif
	}

	if (mae[0] < mae[1]) {
		mainAlgo = tmpAlg;
	}
	else {
		mainAlgo = 360. - tmpAlg;
	}
	cout << "main angle :" << mainAlgo << endl;
}