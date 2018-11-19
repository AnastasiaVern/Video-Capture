#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <ctime>
#include <chrono>
#include <vector>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Function for Face Detection 
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade, double scale, char* text);
std::string cascadeName, nestedCascadeName;

int main(int argc, const char** argv)
{
	// VideoCapture class for playing video for which faces to be detected 
	char* text1 = "Camera 1";
	char* text2 = "Camera 2";
	const int id = 1;
	bool cam_flag = true;
	VideoCapture cam;
	VideoWriter vd_file;
	Mat frame, image;

	// PreDefined trained XML classifiers with facial features 
	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	// Load classifiers from "opencv/data/haarcascades" directory 
	nestedCascade.load("C:/OpenCV/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

	// Change path before execution 
	cascade.load("C:/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalcatface.xml");

	// Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video 
	cam.open(id);
	if (!cam.isOpened()) {
		std::cout << "Could not open the video camera for read." << std::endl;
		return -1;
	}

	cam.set(CAP_PROP_FRAME_WIDTH, 500.0);
	cam.set(CAP_PROP_FRAME_HEIGHT, 500.0);

	const std::string file_name = "cam_record.avi";
	//int fourcc = static_cast<int>(cam.get(CAP_PROP_FOURCC));
	double fps = cam.get(CAP_PROP_FPS);
	Size S = Size((int)cam.get(CAP_PROP_FRAME_WIDTH), (int)cam.get(CAP_PROP_FRAME_HEIGHT));

	vd_file.open(file_name, VideoWriter::fourcc('M', 'P', '4', '2'), fps, S);

	/*if (!vd_file.isOpened())
	{
		std::cout << "Could not open the video file for write." << std::endl;
		return -1;
	} */
		// Capture frames from video and detect faces 
		std::cout << "Face Detection Started...." << std::endl;
		while (true) {
			auto timestamp = std::chrono::system_clock::now();
			std::time_t time = std::chrono::system_clock::to_time_t(timestamp);
			auto output = std::ctime(&time);
			if (cam_flag) {
				cam >> frame;
				if (frame.empty())
					break;
				putText(frame, std::string(output), Point2f(250, (frame.rows - 50)), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0), 1);
				Mat frame1 = frame.clone();
				detectAndDraw(frame1, cascade, nestedCascade, scale, text1);
				vd_file << frame;
				cam_flag = false;
			}
			else {
				cam >> frame;
				if (frame.empty())
					break;
				putText(frame, std::string(output), Point2f(250, (frame.rows - 50)), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0), 1);
				Mat frame1 = frame.clone();
				detectAndDraw(frame1, cascade, nestedCascade, scale, text2);
				cam_flag = true;
			}
			if(waitKey(1) >= 0)
				break;
		}
	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, char* text)
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
