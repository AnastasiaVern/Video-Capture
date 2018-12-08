#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"

#include <time.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <string>
#include <vector>
#include <Windows.h>
#include <stdio.h>
#include <ctype.h>

using namespace cv;
using namespace cv::motempl;

// Function for Face Detection 
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade, double scale, char* text);
// Function for Motion detection
static bool update_mhi(const Mat& img, Mat& dst, int diff_threshold);
//Function for Clicking with Mouse
void myMouseCallback(int event, int x, int y, int flags, void* param);
void putMask();
////////////////////////////////////////////////////////
//-------------------PARAMETERS-------------------------

// various tracking parameters (in seconds)
const double MHI_DURATION = 5;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
int region_coordinates[2][4]; //координаты регионов, в которых надо определять движение.

void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawRectangle(Mat& img, Rect box);

Rect g_rectangle;
bool g_bDrawingBox = false;

//Define frame
Mat frame;
//Define detection params
std::string cascadeName, nestedCascadeName;
// number of cyclic frame buffer used for motion detection
std::vector<Mat> buf;
int last = 0;
// temporary images
Mat mhi, orient, mask, segmask, zplane;
std::vector<Rect> regions;

int main(int argc, char** argv)
{
	const int id = 0;
	bool cam1Event, cam2Event;
	VideoCapture cam(id);
	std::vector<Mat> q1, q2;
	q1.reserve(60);
	q2.reserve(60);
	char* text1 = "Camera 1";
	char* text2 = "Camera 2";
	char file_name[80];
	bool cam_flag = true;
	bool startrec1 = false;
	bool startrec2 = false;
	Mat motion;
	Mat frame1_copy, frame2_copy, frame1_copy_mask, frame2_copy_mask;
	IplImage *image = new IplImage(frame);
	VideoWriter vd_file;

	// PreDefined trained XML classifiers with facial features 
	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	// Load classifiers from "opencv/data/haarcascades" directory 
	nestedCascade.load("C:/OpenCV/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

	// Change path before execution 
	cascade.load("C:/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalcatface.xml");

	if (!cam.isOpened())
	{
		printf("Could not initialize video capture\n");
		return 0;
	}
	buf.resize(2);

	cam.open(id);
	if (!cam.isOpened()) {
		std::cout << "Could not open the video camera for read." << std::endl;
		return -1;
	}

	cam.set(CAP_PROP_FRAME_WIDTH, 500.0);
	cam.set(CAP_PROP_FRAME_HEIGHT, 500.0);

	double fps = cam.get(CAP_PROP_FPS);
	Size S = Size((int)cam.get(CAP_PROP_FRAME_WIDTH), (int)cam.get(CAP_PROP_FRAME_HEIGHT));

	auto timestamp = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
	struct tm * now = localtime(&time);

	strftime(file_name, 80, "%Y-%m-%d.wmv", now);
	vd_file.open(file_name, VideoWriter::fourcc('W', 'M', 'V', '2'), fps, S, true);
	if (!vd_file.isOpened())
	{
		std::cout << "Could not open the video file for write." << std::endl;
	}
	// Capture frames from video, detect faces and detect motions
	std::cout << "Face Detection and Motion Detection Started...." << std::endl;
	namedWindow("CAM1_Motion", WINDOW_NORMAL);
	namedWindow("CAM2_Motion", WINDOW_NORMAL);

	while (true)
	{
		auto timestamp = std::chrono::system_clock::now();
		std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
		auto output = std::ctime(&time);

		setMouseCallback("Camera 1", myMouseCallback, NULL); //подпрограмма mMouseCallback при событиях, связанных с мышью в окне 
		if (cam_flag) {
			cam >> frame;
			if (frame.empty())
				break;

			if (q1.size() == 60)
				q1.clear();
			else
				q1.push_back(frame);

			cam1Event = update_mhi(frame, motion, 30);
			//	std::cout << "CAM EVENT" << cam1Event << std::endl;
			if (cam1Event == true && startrec1 == false) {
				for (int i = 0; i < q1.size(); i++)
				{
					vd_file << q1.at(i);
				}
				startrec1 = true;
			}
			imshow("CAM1_Motion", motion);
			resizeWindow("CAM1_Motion", 400, 250);
			putMask();

			frame2_copy = frame.clone();
			detectAndDraw(frame, cascade, nestedCascade, scale, text1);
			if (startrec1) { vd_file << frame; std::cout << "IM HERE" << std::endl; }
			cam_flag = false;

			char c = waitKey(33);
			if (c == 105) {
				putText(frame1_copy, "CASH IN", Point2f(100, (frame.rows - 10)), FONT_HERSHEY_DUPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 1", frame1_copy);
				imwrite("CASH_IN.jpg", frame1_copy);
				putText(frame2_copy, "CASH IN", Point2f(100, (frame.rows - 10)), FONT_HERSHEY_DUPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 2", frame2_copy);

			}
			if (c == 111) {
				putText(frame1_copy, "CASH OUT", Point2f(100, (frame.rows - 10)), FONT_HERSHEY_DUPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 1", frame1_copy);
				imwrite("CASH_OUT.jpg", frame1_copy);
				putText(frame2_copy, "CASH OUT", Point2f(100, (frame.rows - 10)), FONT_HERSHEY_DUPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 2", frame2_copy);
			}
		}
		else {
			cam >> frame;
			if (frame.empty())
				break;

			if (q2.size() == 60)
				q2.clear();
			else
				q2.push_back(frame);

			cam2Event = update_mhi(frame, motion, 30);
			if (cam2Event == true && startrec2 == false) {
				std::cout << "COOL CAM2";
				for (int i = 0; i < q2.size(); i++)
				{
					vd_file << q2.at(i);
				}
				startrec2 = true;
			}
			imshow("CAM2_Motion", motion);
			resizeWindow("CAM2_Motion", 400, 250);

			putText(frame, std::string(output), Point2f(250, (frame.rows - 50)), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0), 1);
			setMouseCallback("Camera 2", myMouseCallback, NULL); //подпрограмма myMouseCallback при событиях, связанных с мышью в окне 
			putMask();
			frame1_copy = frame.clone();
			detectAndDraw(frame, cascade, nestedCascade, scale, text2);
			if (startrec2) vd_file << frame;
			cam_flag = true;

			char c = waitKey(33);
			if (c == 105) {
				putText(frame1_copy, "CASH IN", Point2f(150, (frame.rows - 20)), FONT_HERSHEY_COMPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 1", frame1_copy);
				putText(frame2_copy, "CASH IN", Point2f(150, (frame.rows - 20)), FONT_HERSHEY_COMPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 2", frame2_copy);
				imwrite("CASH_IN.jpg", frame2_copy);
			}
			if (c == 111) {
				putText(frame1_copy, "CASH OUT", Point2f(150, (frame.rows - 20)), FONT_HERSHEY_COMPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 1", frame1_copy);
				putText(frame2_copy, "CASH OUT", Point2f(150, (frame.rows - 20)), FONT_HERSHEY_COMPLEX, 2, Scalar(253, 63, 0), 1);
				imshow("Camera 2", frame2_copy);
				imwrite("CASH_OUT.jpg", frame2_copy);
			}
		}
		char c = waitKey(33);

		if (c == 113 || c == 81) //коды кнопки "q" - в английской и русской раскладках. 
		{
			cam.release();
			destroyWindow("Camera 1");
			destroyWindow("Camera 2");
			destroyWindow("CAM1_Motion");
			destroyWindow("CAM2_Motion");

			break;
		}
	}

	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, char* text)
{
	std::vector<Rect> faces, faces2;
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY); // Convert to Gray Scale 
	double fx = 1 / scale;

	// Resize the Grayscale Image 
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	// Detect faces of different sizes using cascade classifier 
	cascade.detectMultiScale(smallImg, faces, 1.1,
		2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Draw circles around the faces 
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		std::vector<Rect> nestedObjects;
		Point center;
		Scalar color = Scalar(255, 0, 0); // Color for Drawing tool 
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
				cvPoint(cvRound((r.x + r.width - 1)*scale),
					cvRound((r.y + r.height - 1)*scale)), color, 3, 8, 0);
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(r);

		// Detection of eyes int the input image 
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	}

	// Show Processed Image with detected faces 
	imshow(text, img);
}

static bool  update_mhi(const Mat& img, Mat& dst, int diff_threshold)
{
	bool event_ = false;
	double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
	Size size = img.size();
	int i, idx1 = last;
	Rect comp_rect;
	double count;
	double angle;
	Point center;
	double magnitude;
	Scalar color;

	// allocate images at the beginning or
	// reallocate them if the frame size is changed
	if (mhi.size() != size)
	{
		mhi = Mat::zeros(size, CV_32F);
		zplane = Mat::zeros(size, CV_8U);

		buf[0] = Mat::zeros(size, CV_8U);
		buf[1] = Mat::zeros(size, CV_8U);
	}

	cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale

	int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
	last = idx2;

	Mat silh = buf[idx2];
	absdiff(buf[idx1], buf[idx2], silh); // get difference between frames

	threshold(silh, silh, diff_threshold, 1, THRESH_BINARY); // and threshold it
	updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI

															 // convert MHI to blue 8u image
	mhi.convertTo(mask, CV_8U, 255. / MHI_DURATION, (MHI_DURATION - timestamp)*255. / MHI_DURATION);

	Mat planes[] = { mask, zplane, zplane };
	merge(planes, 3, dst);

	// calculate motion gradient orientation and valid orientation mask
	calcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. It is not used further
	regions.clear();
	segmentMotion(mhi, segmask, regions, timestamp, MAX_TIME_DELTA);

	// iterate through the motion components,
	// One more iteration (i == -1) corresponds to the whole image (global motion)
	for (i = -1; i < (int)regions.size(); i++) {

		if (i < 0) { // case of the whole image
			comp_rect = Rect(0, 0, size.width, size.height);
			color = Scalar(255, 255, 255);
			magnitude = 100;
		}
		else { // i-th motion component
			comp_rect = regions[i];
			if (comp_rect.width + comp_rect.height < 100) // reject very small components
				continue;
			color = Scalar(0, 0, 255);
			magnitude = 30;
			event_ = true;
		}

		// select component ROI
		Mat silh_roi = silh(comp_rect);
		Mat mhi_roi = mhi(comp_rect);
		Mat orient_roi = orient(comp_rect);
		Mat mask_roi = mask(comp_rect);

		// calculate orientation
		angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
		angle = 360.0 - angle;  // adjust for images with top-left origin

		count = norm(silh_roi, NORM_L1);; // calculate number of points within silhouette ROI

										  // check for the case of little motion
		if (count < comp_rect.width*comp_rect.height * 0.05)
			continue;

		// draw a clock with arrow indicating the direction
		center = Point((comp_rect.x + comp_rect.width / 2),
			(comp_rect.y + comp_rect.height / 2));

		circle(img, center, cvRound(magnitude*1.2), color, 3, 16, 0);
		if (region_coordinates[0][0] != 0 && region_coordinates[0][1] != 0
			&& region_coordinates[0][2] == 0 && region_coordinates[0][3] == 0) //Рисуем прямоугольник. Если есть в переменной только одни координаты - рисуем точку по этим координатам.
		{
			rectangle(dst, cvPoint(region_coordinates[0][0], region_coordinates[0][1]), cvPoint(region_coordinates[0][0] + 1, region_coordinates[0][1] + 1), CV_RGB(0, 0, 0), CV_FILLED, CV_AA, 0);
		}
		if (region_coordinates[0][0] != 0 && region_coordinates[0][1] != 0
			&& region_coordinates[0][2] != 0 && region_coordinates[0][3] != 0) //А если в переменной двое наборов координат - рисуем полностью прямоугольник.
		{
			rectangle(dst, cvPoint(region_coordinates[0][0], region_coordinates[0][1]), cvPoint(region_coordinates[0][2], region_coordinates[0][3]), CV_RGB(0, 0, 0), CV_FILLED, CV_AA, 0);
		}
	}
	return event_;
}

void myMouseCallback(int event, int x, int y, int flags, void* param) //описываем что нам надо будет делать при событиях, связанных с мышью
{
	IplImage* img = (IplImage*)param;
	switch (event) { //вбираем действие в зависимости от событий
	case CV_EVENT_MOUSEMOVE:     break; //ничего не делаем при движении мыши. А можно, например, кидать в консоль координаты под курсором: printf("%d x %d\n", x, y);

	case CV_EVENT_LBUTTONDOWN: //при нажатии левой кнопки мыши

		if (region_coordinates[0][0] != 0 && region_coordinates[0][1] != 0 && region_coordinates[0][2] == 0 && region_coordinates[0][3] == 0) //если это второе нажатие(заполнена первая половина координат - х и у верхнего угла региона), то записываем в переменную вторую половину - х и у нижнего угла региона
		{
			region_coordinates[0][2] = x; //dig_key - определяет, какой регион устанавливается сейчас. А меняется он нажатием цифровых кнопок.
			region_coordinates[0][3] = y;
		}
		if (region_coordinates[0][0] == 0 && region_coordinates[0][1] == 0)//если это первое нажатие(не заполнена первая половина координат ), то записываем в переменную первую половину.
		{
			region_coordinates[0][0] = x;
			region_coordinates[0][1] = y;
		}

		break;
	}
}

void putMask() {
	if (region_coordinates[0][0] != 0 && region_coordinates[0][1] != 0
		&& region_coordinates[0][2] == 0 && region_coordinates[0][3] == 0) //Рисуем прямоугольник. Если есть в переменной только одни координаты - рисуем точку по этим координатам.
	{
		rectangle(frame, cvPoint(region_coordinates[0][0], region_coordinates[0][1]), cvPoint(region_coordinates[0][0] + 1, region_coordinates[0][1] + 1), CV_RGB(0, 0, 0), CV_FILLED, CV_AA, 0);
	}
	if (region_coordinates[0][0] != 0 && region_coordinates[0][1] != 0
		&& region_coordinates[0][2] != 0 && region_coordinates[0][3] != 0) //А если в переменной двое наборов координат - рисуем полностью прямоугольник.
	{
		rectangle(frame, cvPoint(region_coordinates[0][0], region_coordinates[0][1]), cvPoint(region_coordinates[0][2], region_coordinates[0][3]), CV_RGB(0, 0, 0), CV_FILLED, CV_AA, 0);
	}
}
