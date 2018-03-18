//g++ -std=c++11 -Wall -o calibracion calibracion.cpp `pkg-config --cflags --libs opencv` && ./calibracion

#include <iostream> // for standard I/O


#include <ctime>


#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"

#include "ObtenerAnillos.cpp"

using namespace std;
using namespace cv;



int main()
{
	/*VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FPS,60);
	int fps = cap.get(cv::CAP_PROP_FPS);
	cout<<fps<<endl;

	cap.set(cv::CAP_PROP_BRIGHTNESS,0.5);
	float brillo = cap.get(cv::CAP_PROP_BRIGHTNESS);
	cout<<brillo<<endl;*/

	//VideoCapture cap("PS3_rings.avi");
	//VideoCapture cap("calibration_ps3eyecam.avi");

	//VideoCapture cap("PS3_Anillo.webm");
	VideoCapture cap("LifeCam_anillos.webm");

	if(!cap.isOpened())
		return -1;

	Mat view, imgCopia;
	int key;


	for(;;)
	{
		//view = imread("imgEr.png", 1);
		cap >> view;
		//imgCopia = view.clone();

		vector<Point2f> pointBuf;
		bool found = findRingGrid( view, Size(5,4), pointBuf);

		if(found)
		{
			drawChessboardCorners( view, Size(5,4), Mat(pointBuf), found );
		}

		imshow( "View", view );
		moveWindow("View", 50, 50);


		key = waitKey(1);
		switch (key)
		{
		case 'q':
		{
			return 0;
		}
		case 'p':
		{
			waitKey(0);
			break;
		}


		}
	}

	return 0;
}

