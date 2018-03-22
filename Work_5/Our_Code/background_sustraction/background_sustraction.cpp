#include <iostream>
#include "stdafx.h"

//#include "ImagesNames.h"
#include "Eigen\Dense"
//#include <Eigen\SVD>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\core\eigen.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;






#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>


using namespace std;
using namespace std::experimental::filesystem::v1; // for directories
const short FULLNAMES = 1;
const short ONLYNAMES = 2;

class ImagesNames
{
private:

	int size;
	vector<string> imagesFullNames;
	vector<string> imagesOnlyNames;
	string pathDirectory;

public:
	ImagesNames();
	ImagesNames(string);
	void SetPath(string);
	bool GetImagesNames(bool show = true);
	bool ShowImagesNames(short typeShow = FULLNAMES);
	vector<string> &FullNames() { return imagesFullNames; }
	vector<string> &OnlyNames() { return imagesOnlyNames; }
	inline int &Size() { return size; };
	virtual ~ImagesNames();
};

ImagesNames::ImagesNames() : pathDirectory(string(""))
{

}

ImagesNames::ImagesNames(string path) : pathDirectory(path)
{

}

void ImagesNames::SetPath(string path)
{
	pathDirectory = path;
}


bool ImagesNames::GetImagesNames(bool show)
{
	if (pathDirectory == "")
	{
		std::cout << "No Path Directory " << std::endl;
		return false;
	}

	if (show) std::cout << pathDirectory << std::endl;
	size = 0;
	for (auto p : directory_iterator(pathDirectory))
	{
		if (show) std::cout << p << endl;
		string filename = p.path().string(); //para obtener el nombre dela rchivo, ruta + stem + extension		
		imagesFullNames.push_back(filename);
		filename = p.path().stem().string(); // obtener solo el stem		
		imagesOnlyNames.push_back(filename);
		size++;
	}
	return true;
}

bool ImagesNames::ShowImagesNames(short typeShow)
{
	if (imagesFullNames.size() == 0)
	{
		cout << "No names to show....";
		return false;
	}

	if (typeShow == FULLNAMES)
		for (vector<string>::iterator it = imagesFullNames.begin(); it != imagesFullNames.end(); ++it)
			cout << *it << endl;
	else
		for (vector<string>::iterator it = imagesOnlyNames.begin(); it != imagesOnlyNames.end(); ++it)
			cout << *it << endl;
	return true;
}


ImagesNames::~ImagesNames()
{
}
















const double epsilon = 1e-9;
const double M_PI = 3.1415926535897932384626433832795;
// si es el valor es menors a 1e-9, se asume que es cero  
bool tolerance_zero(double x)
{
	return fabs(x) < 1e-9;
}

// retorna el valor del coseno 
double cosd(double degree)
{
	return cos(degree*M_PI / 180.0);
}

// retorna el valor del seno 
double sind(double degree)
{
	return sin(degree*M_PI / 180.0);
}

// retorna el angulo entre dos coordenadas y y x 
double fqatan(double y, double x)
{
	double t = atan2(y, x) / M_PI*180.0;
	//cout << "y: " << y << " x: " << x <<" atan: "<< atan2(y, x)<<endl;     
	if (t < 0.0)
		t += 360;
	//cout <<" t: "<< t<<endl; 
	return t;
}

//Calcular f7(x) = (x ^ 7 / (x ^ 7 + 25 ^ 7)) ^ 0.5 
double f7(double x)
{
	if (x < 1.0)
		return pow(x / 25.0, 3.5);
	return 1.0 / sqrt(1.0 + pow(25.0 / x, 7.0));
}

double CIEDE2000ColorDistance(Vec3d color1, Vec3d color2)
{
	// Calculate C1', C2', h1', and h2' 
	double  C1ab, C2ab;
	double  Cab, G;
	double  a1_, a2_;
	double  C1_, C2_;
	double  h1_, h2_;

	double L1 = color1[0];
	double a1 = color1[1];
	double b1 = color1[2];
	double L2 = color2[0];
	double a2 = color2[1];
	double b2 = color2[2];


	C1ab = sqrt(a1*a1 + b1*b1);
	C2ab = sqrt(a2*a2 + b2*b2);
	Cab = (C1ab + C2ab) / 2.0;
	G = 0.5*(1.0 - f7(Cab));
	a1_ = (1.0 + G)*a1;
	a2_ = (1.0 + G)*a2;
	C1_ = sqrt(a1_*a1_ + b1*b1);
	C2_ = sqrt(a2_*a2_ + b2*b2);
	if (tolerance_zero(a1_) && tolerance_zero(b1))
		h1_ = 0.0;
	else
		h1_ = fqatan(b1, a1_);
	if (tolerance_zero(a2_) && tolerance_zero(b2))
		h2_ = 0.0;
	else
		h2_ = fqatan(b2, a2_);

	// Calculate dL', dC', and dH' 
	double  dL_, dC_, dH_, dh_;
	double  C12;

	dL_ = L2 - L1;
	dC_ = C2_ - C1_;
	C12 = C1_*C2_;
	if (tolerance_zero(C12))
	{
		dh_ = 0.0;
	}
	else
	{
		double  tmp = h2_ - h1_;

		if (fabs(tmp) <= 180.0 + epsilon)
			dh_ = tmp;
		else if (tmp>180.0)
			dh_ = tmp - 360.0;
		else if (tmp<-180.0)
			dh_ = tmp + 360.0;
	}
	dH_ = 2.0*sqrt(C12)*sind(dh_ / 2.0);


	// Calculate L', C', h', T, and dTh 
	double  L_, C_, h_, T, dTh;

	L_ = (L1 + L2) / 2.0;
	C_ = (C1_ + C2_) / 2.0;
	if (tolerance_zero(C12))
	{
		h_ = h1_ + h2_;
	}
	else
	{
		double  tmp1 = fabs(h1_ - h2_);
		double  tmp2 = h1_ + h2_;

		if (tmp1 <= 180.0 + epsilon)
			h_ = tmp2 / 2.0;
		else if (tmp2<360.0)
			h_ = (tmp2 + 360.0) / 2.0;
		else if (tmp2 >= 360.0)
			h_ = (tmp2 - 360.0) / 2.0;
	}
	T = 1.0 - 0.17*cosd(h_ - 30.0) + 0.24*cosd(2.0*h_)
		+ 0.32*cosd(3.0*h_ + 6.0) - 0.2*cosd(4.0*h_ - 63.0);
	dTh = 30.0*exp(-pow((h_ - 275.0) / 25.0, 2.0));


	// Calculate RC, SL, SC, SH, and RT 
	double  RC, SL, SC, SH, RT;
	double  L_2 = (L_ - 50.0)*(L_ - 50.0);

	RC = 2.0*f7(C_);
	SL = 1.0 + 0.015*L_2 / sqrt(20.0 + L_2);
	//cout << "SL: " << SL << endl; 
	SC = 1.0 + 0.045*C_;
	//cout << "SC: " << SC << endl; 
	SH = 1.0 + 0.015*C_*T;
	//cout << "SH: " << SH << endl; 
	RT = -sind(2.0*dTh)*RC;
	//cout << "RT: " << RT << endl; 

	// Calculate dE00 
	const double kL = 1.0;    // These are proportionally coefficients 
	const double kC = 1.0;    // and vary according to the condition. 
	const double kH = 1.0;    // These mostly are 1. 

	double  LP = dL_ / (kL*SL);
	double  CP = dC_ / (kC*SC);
	double  HP = dH_ / (kH*SH);

	return sqrt(LP*LP + CP*CP + HP*HP + RT*CP*HP);
}









































































// para obtener el valor estructural invariante utilizando SVD
// la imagen de entrada es en escala de grises
// y las coordenandas del pixels a ser procesado
float LocalStructuralInvariant(const Eigen::MatrixXf &image, int x, int y)
{
	Eigen::MatrixXf b;
	//cout << x<<" - " << y << " : " << x - 1 << " - " << y - 1<<endl;
	b = image.block<3, 3>(x - 1, y - 1); // extraer el bloque cuyo centro esta en x,y	
	Eigen::BDCSVD<Eigen::MatrixXf> svd(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Vector3f sv = svd.singularValues();
	return ((sv[1] + sv[2]) / sv[0]);
}


// Para obtener el la Matriz con los coeficiente SVD normalizados
// la imagen de entrada es en escala de grises
// la imagen de salida es en escala de grise, pero en flotante

void NormalizedSVDCoefficient(Mat &image, Mat &out)
{
	Eigen::MatrixXf eigImage;
	cv2eigen(image, eigImage);

	Eigen::MatrixXf gImage = Eigen::MatrixXf::Zero(eigImage.rows(), eigImage.cols());

	for (int i = 1; i<gImage.rows() - 1; ++i)
		for (int j = 1; j<gImage.cols() - 1; ++j)
			gImage(i, j) = LocalStructuralInvariant(eigImage, i, j);

	eigen2cv(gImage, out);
}

float localLSBP(Mat &in, int cx, int cy) // solo para ocho vecinos
{
	int R = 1;
	int P = 8;
	float T = 0.05;

	// se recorren los vecino en orden en este caso lo hare antihorario
	vector<int> incX = { 1,0,-1,-1,0,0,1,1 };
	vector<int> incY = { 0,-1,0,0,1,1,0,0 };

	int px = cx, py = cy;

	float lsbp = 0;
	for (int p = 0; p < P; ++p)
	{
		px += incX[p];
		py += incY[p];
		if (in.at<float>(px, py) - in.at<float>(cx, cy) > T)
			lsbp += pow(2, p);
	}
	return lsbp;
}

void LSBP(Mat &in, Mat &out)
{
	out = Mat::zeros(in.size(), in.type());
	for (int i = 1; i < in.rows - 1; ++i)
		for (int j = 1; j < in.cols - 1; ++j)
			out.at<float>(i, j) = localLSBP(in, i, j);
}

int HammingDistance(int a, int b)
{
	short dist = 0;
	int val = a ^ b;// what's the meaning?
	while (val) {
		++dist;
		val &= val - 1; // why?
	}
	return dist;
}

int L1Distance(Vec3b a, Vec3b b)
{
	int sum = 0;
	sum += abs(a[0] - b[0]);
	sum += abs(a[1] - b[1]);
	sum += abs(a[2] - b[2]);
	return sum;
}

IplImage* imfill(IplImage* src)
{
	CvScalar white = CV_RGB(255, 255, 255);

	IplImage* dst = cvCreateImage(cvGetSize(src), 8, 3);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;

	cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	cvZero(dst);

	for (; contour != 0; contour = contour->h_next)
	{
		cvDrawContours(dst, contour, white, white, 0, CV_FILLED);
	}

	IplImage* bin_imgFilled = cvCreateImage(cvGetSize(src), 8, 1);
	cvInRangeS(dst, white, white, bin_imgFilled);

	return bin_imgFilled;
}





int main()
{

	ImagesNames images("highway/input");
	images.GetImagesNames();


	int NroFramesToInit = 5;

	vector<Mat> samples_int(NroFramesToInit);
	vector<Mat> samples_lsbp(NroFramesToInit);

	Mat frame;

	int frameCount = 0;

	int cdnet_count = 1;
	string name, name_out;
	string ruta = "overpass/";


	//SVD_init
	//La parte para inicializar
	frame = imread(images.FullNames()[frameCount]);

	while (frame.cols > 0 && frameCount < NroFramesToInit)
	{
		cout << images.FullNames()[frameCount] << endl;
		Mat frameGray;
		cvtColor(frame, frameGray, CV_BGR2GRAY);
		///cambio de espacio de color
		///cvtColor(frame, frame, CV_BGR2Lab);

		frame.copyTo(samples_int[frameCount]);	 // se almacena la imagen en color

		
		Mat in, lsbp;
		NormalizedSVDCoefficient(frameGray, in);
		LSBP(in, lsbp);
		lsbp.copyTo(samples_lsbp[frameCount]); // se almacena el LSBP de la imagen

		imshow("frame", frame);
		imshow("lsbp", lsbp);
		waitKey(10);
		frameCount++;
		frame = imread(images.FullNames()[frameCount]);
	}


	cv::waitKey();

	//SVD_step
	// se rebonina el video

	int min = 2;
	int H_LSBP = 4;

	//Rebobinar
	frameCount = 0;
	//La parte para inicializar
	frame = imread(images.FullNames()[frameCount++]);

	while (frame.cols > 0)
	{
		auto start = std::chrono::high_resolution_clock::now();

		imshow("frame", frame);
		
		Mat frameGray;
		cvtColor(frame, frameGray, CV_BGR2GRAY);
		///cambio de espacio de color
		///cvtColor(frame, frame, CV_BGR2Lab);

		//cv::GaussianBlur(frame, frame, Size(9, 9), 0, 0);
		

		Mat in, lsbp;
		NormalizedSVDCoefficient(frameGray, in);
		LSBP(in, lsbp);


		Mat result = Mat::zeros(frame.size(), CV_8UC1);

		for (int i = 1; i < frame.rows - 1; ++i)
			for (int j = 1; j < frame.cols - 1; ++j)
			{
				int matches = 0;
				int index = 0;

				while (index < NroFramesToInit && matches < min)
				{
					
					/*if (HammingDistance((int)samples_lsbp[index].at<float>(i, j), (int)lsbp.at<float>(i, j)) <= H_LSBP &&
						CIEDE2000ColorDistance(samples_int[index].at<Vec3b>(i, j), frame.at<Vec3b>(i, j)) < 15)
						matches++;*/
					if (HammingDistance((int)samples_lsbp[index].at<float>(i, j), (int)lsbp.at<float>(i, j)) <= H_LSBP &&
						L1Distance(samples_int[index].at<Vec3b>(i, j), frame.at<Vec3b>(i, j)) < 12)
						matches++;
					index++;
				}

				//cout <<"matches: "<< matches <<" index: "<< index << " Nro de frames: "<<NroFramesToInit<<endl;

				if (matches < min)
					result.at <unsigned char>(i, j) = 255;

			}


		
		imshow("lsbp", lsbp);
		imshow("result", result);
		Mat  kernel, eros;
		kernel = cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		cv::morphologyEx(result, eros, MORPH_OPEN, kernel);
		kernel = cv::getStructuringElement(MORPH_RECT, Size(9, 9));
		cv::morphologyEx(eros, eros, MORPH_CLOSE, kernel);
		kernel = cv::getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
		cv::morphologyEx(eros, eros, MORPH_OPEN, kernel);
		imshow("Erosion", eros);
		IplImage* image2 = cvCloneImage(&(IplImage)eros);
		IplImage* image3=imfill(image2);
		Mat fill = cv::cvarrToMat(image3);
		imshow("Fill", fill);

		stringstream ss;
		ss << cdnet_count;
		if (cdnet_count < 10)
		{			
			name = ruta + "output_lsbp/in00000" + ss.str() + ".jpg";
			imwrite(name, result);
			name = ruta + "output_ero/in00000" + ss.str() + ".jpg";
			imwrite(name, eros);
			name = ruta + "output_fill/in00000" + ss.str() + ".jpg";
			imwrite(name, fill);
		}
			
		else if (cdnet_count < 100)
		{
			name = ruta + "output_lsbp/in0000" + ss.str() + ".jpg";
			imwrite(name, result);
			name = ruta + "output_ero/in0000" + ss.str() + ".jpg";
			imwrite(name, eros);
			name = ruta + "output_fill/in0000" + ss.str() + ".jpg";
			imwrite(name, fill);
		}
		else if (cdnet_count < 1000)
		{
			name = ruta + "output_lsbp/in000" + ss.str() + ".jpg";
			imwrite(name, result);
			name = ruta + "output_ero/in000" + ss.str() + ".jpg";
			imwrite(name, eros);
			name = ruta + "output_fill/in000" + ss.str() + ".jpg";
			imwrite(name, fill);
		}
		else
		{
			name = ruta + "output_lsbp/in00" + ss.str() + ".jpg";
			imwrite(name, result);
			name = ruta + "output_ero/in00" + ss.str() + ".jpg";
			imwrite(name, eros);
			name = ruta + "output_fill/in00" + ss.str() + ".jpg";
			imwrite(name, fill);
		}
		cout << cdnet_count << "  " << name << endl;
		cdnet_count++;
		
		
		auto finish = std::chrono::high_resolution_clock::now();
		std::cout << "Tiempo: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1000000.0 << "ms por frame" << endl;
		waitKey(1);
		frame = imread(images.FullNames()[frameCount++]);
	}

	waitKey();
	return 0;
}































































































/*
int main_video()
{
	VideoCapture video;
	video.open("run.avi");

	int NroFramesToInit = 10;

	vector<Mat> samples_int(NroFramesToInit);
	vector<Mat> samples_lsbp(NroFramesToInit);

	Mat frame;

	int frameCount = 0;


	//SVD_init

	//La parte para inicializar
	video >> frame;
	while (frame.cols > 0 && frameCount < NroFramesToInit)
	{
		

		frame.copyTo(samples_int[frameCount]);	 // se almacena la imagen en color
		//cv::GaussianBlur(frame, frame, Size(9, 9), 0, 0);
		cvtColor(frame, frame, CV_BGR2GRAY);
		Mat in, lsbp;
		NormalizedSVDCoefficient(frame, in);
		LSBP(in, lsbp);
		lsbp.copyTo(samples_lsbp[frameCount]); // se almacena el LSBP de la imagen

		imshow("frame", frame);
		imshow("lsbp", lsbp);
		waitKey(10);
		video >> frame;
		frameCount++;
	}


	cv::waitKey();

	//SVD_step
	// se rebonina el video

	int min = 2;
	int H_LSBP = 4;

	video.set(CAP_PROP_POS_AVI_RATIO, 0);
	video >> frame;
	while (frame.cols > 0)
	{
		//cv::GaussianBlur(frame, frame, Size(9, 9), 0, 0);
		Mat frameGray;
		cvtColor(frame, frameGray, CV_BGR2GRAY);

		Mat in, lsbp;
		NormalizedSVDCoefficient(frameGray, in);
		LSBP(in, lsbp);


		Mat result = Mat::zeros(frame.size(), CV_8UC1);

		for (int i = 1; i < frame.rows - 1; ++i)
			for (int j = 1; j < frame.cols - 1; ++j)
			{
				int matches = 0;
				int index = 0;

				while (index < NroFramesToInit && matches < min)
				{
					if (HammingDistance((int)samples_lsbp[index].at<float>(i, j), (int)lsbp.at<float>(i, j)) <= H_LSBP &&
						L1Distance(samples_int[index].at<Vec3b>(i, j), frame.at<Vec3b>(i, j)) < 120)
						matches++;
					index++;
				}

				//cout <<"matches: "<< matches <<" index: "<< index << " Nro de frames: "<<NroFramesToInit<<endl;

				if (matches < min)
					result.at <unsigned char>(i, j) = 255;

			}


		imshow("frame", frame);
		imshow("lsbp", lsbp);
		Mat  kernel;
		kernel = cv::getStructuringElement(MORPH_ELLIPSE, Size(1, 1));
		cv::morphologyEx(result, result, MORPH_OPEN, kernel);
		kernel = cv::getStructuringElement(MORPH_RECT, Size(3, 3));
		cv::morphologyEx(result, result, MORPH_CLOSE, kernel);
		imshow("result", result);


		waitKey(1);
		video >> frame;
	}


	waitKey();
	return 0;
}


int main_image()
{
	Mat image = imread("run.avi", 0);
	imshow("image", image);

	Mat in;
	NormalizedSVDCoefficient(image, in);
	imshow("out", in);

	Mat lsbp;
	LSBP(in, lsbp);
	imshow("LSBP", lsbp);

	waitKey();

	return 0;
}*/





















/*#include "stdafx.h"

#include <iostream>

#include "Eigen\Dense"

//#include <Eigen\SVD>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\core\eigen.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

// para obtener el valor estructural invariante utilizando SVD
// la imagen de entrada es en escala de grises
// y las coordenandas del pixels a ser procesado
float LocalStructuralInvariant(const Eigen::MatrixXf &image, int x, int y)
{
	Eigen::MatrixXf b;
	//cout << x<<" - " << y << " : " << x - 1 << " - " << y - 1<<endl;
	b = image.block<3, 3>(x - 1, y - 1); // extraer el bloque cuyo centro esta en x,y	
	Eigen::BDCSVD<Eigen::MatrixXf> svd(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Vector3f sv = svd.singularValues();
	return ((sv[1] + sv[2]) / sv[0]);
}


// Para obtener el la Matriz con los coeficiente SVD normalizados
// la imagen de entrada es en escala de grises
// la imagen de salida es en escala de grise, pero en flotante

void NormalizedSVDCoefficient(Mat &image, Mat &out)
{
	Eigen::MatrixXf eigImage;
	cv2eigen(image, eigImage);

	Eigen::MatrixXf gImage = Eigen::MatrixXf::Zero(eigImage.rows(), eigImage.cols());

	for (int i = 1; i<gImage.rows() - 1; ++i)
		for (int j = 1; j<gImage.cols() - 1; ++j)
			gImage(i, j) = LocalStructuralInvariant(eigImage, i, j);

	eigen2cv(gImage, out);
}

float localLSBP(Mat &in, int cx, int cy) // solo para ocho vecinos
{
	int R = 1;
	int P = 8;
	float T = 0.05;

	// se recorren los vecino en orden en este caso lo hare antihorario
	vector<int> incX = { 1,0,-1,-1,0,0,1,1 };
	vector<int> incY = { 0,-1,0,0,1,1,0,0 };

	int px = cx, py = cy;

	float lsbp = 0;
	for (int p = 0; p < P; ++p)
	{
		px += incX[p];
		py += incY[p];
		if (in.at<float>(px, py) - in.at<float>(cx, cy) > T)
			lsbp += pow(2, p);
	}
	return lsbp;
}

void LSBP(Mat &in, Mat &out)
{
	out = Mat::zeros(in.size(), in.type());
	for (int i = 1; i < in.rows - 1; ++i)
		for (int j = 1; j < in.cols - 1; ++j)
			out.at<float>(i, j) = localLSBP(in, i, j);
}

int HammingDistance(int a, int b)
{
	short dist = 0;
	int val = a ^ b;// what's the meaning?
	while (val) {
		++dist;
		val &= val - 1; // why?
	}
	return dist;
}

int L1Distance(Vec3b a, Vec3b b)
{
	int sum = 0;
	sum += abs(a[0] - b[0]);
	sum += abs(a[1] - b[1]);
	sum += abs(a[2] - b[2]);
	return sum;
}*/

/*
int main()
{
	VideoCapture video;
	video.open("campus.avi");

	int NroFramesToInit = 10;

	vector<Mat> samples_int(NroFramesToInit);
	vector<Mat> samples_lsbp(NroFramesToInit);

	Mat frame;

	int frameCount = 0;


	//SVD_init

	//La parte para inicializar
	//video >> frame;
	//load images CDNET
	int cdnet_count = 1;
	string name, name_out; 
	string ruta = "overpass/";
	

	//while (frame.cols > 0 && frameCount < NroFramesToInit)
	while ( frameCount < NroFramesToInit)
	{
		stringstream ss;
		ss << cdnet_count;
		if (cdnet_count < 10)
			name = ruta+"input/in00000" + ss.str()+".jpg";
		else if (cdnet_count < 100)
			name = ruta + "input/in0000" + ss.str() + ".jpg";
		else if (cdnet_count < 1000)
			name = ruta + "input/in000" + ss.str() + ".jpg";
		else
			name = ruta + "input/in00" + ss.str() + ".jpg";

		cout<<cdnet_count<<"  " << name<<endl;

		frame=imread(name, CV_LOAD_IMAGE_COLOR);
		

		//cv::GaussianBlur(frame, frame, Size(9, 9), 0, 0);
		frame.copyTo(samples_int[frameCount]);	 // se almacena la imagen en color

		cvtColor(frame, frame, CV_BGR2GRAY);
		Mat in, lsbp;
		NormalizedSVDCoefficient(frame, in);
		LSBP(in, lsbp);
		lsbp.copyTo(samples_lsbp[frameCount]); // se almacena el LSBP de la imagen

		imshow("frame", frame);
		imshow("lsbp", lsbp);
		waitKey(1);
		//video >> frame;
		frameCount++;
		cdnet_count++;
	}

	//SVD_step
	// se rebonina el video

	int min = 4;


	//video.set(CAP_PROP_POS_AVI_RATIO, 0);
	//video >> frame;
	cdnet_count = 2000;
	//while (frame.cols > 0&& cdnet_count<1701)
	while ( cdnet_count<3000)
	{
		stringstream ss;
		ss << cdnet_count;
		if (cdnet_count < 10)
			name = ruta + "input/in00000" + ss.str() + ".jpg";
		else if (cdnet_count < 100)
			name = ruta + "input/in0000" + ss.str() + ".jpg";
		else if (cdnet_count < 1000)
			name = ruta + "input/in000" + ss.str() + ".jpg";
		else
			name = ruta + "input/in00" + ss.str() + ".jpg";

		cout << cdnet_count << "  " << name << endl;

		frame = imread(name, CV_LOAD_IMAGE_COLOR);


		//cv::GaussianBlur(frame, frame, Size(9, 9), 0, 0);
		Mat frameGray;
		cvtColor(frame, frameGray, CV_BGR2GRAY);
		Mat in, lsbp;
		NormalizedSVDCoefficient(frameGray, in);
		LSBP(in, lsbp);


		int min = 4;

		Mat result = Mat::zeros(frame.size(), CV_8UC1);

		for (int i = 1; i < frame.rows - 1; ++i)
			for (int j = 1; j < frame.cols - 1; ++j)
			{
				int matches = 0;
				int index = 0;

				while (index < NroFramesToInit && matches < min)
				{
					if (HammingDistance((int)samples_lsbp[index].at<float>(i, j), (int)lsbp.at<float>(i, j)) < min &&
						L1Distance(samples_int[index].at<Vec3b>(i, j), frame.at<Vec3b>(i, j)) <100)
						matches++;
					index++;
				}

				if (matches < min)
					result.at <unsigned char>(i, j) = 255;

			}


		imshow("frame", frame);
		imshow("lsbp", lsbp);
		
		//cv::GaussianBlur(result, result, Size(9, 9), 0, 0);
		Mat  kernel;
		kernel = cv::getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		cv::morphologyEx(result, result, MORPH_OPEN, kernel);

		//kernel = cv::getStructuringElement(MORPH_RECT, Size(3, 3));
		//cv::morphologyEx(result, result, MORPH_CLOSE, kernel);

		imshow("result", result);
		if (cdnet_count < 10)
			name_out = ruta + "output/in00000" + ss.str() + ".jpg";
		else if (cdnet_count < 100)
			name_out = ruta + "output/in0000" + ss.str() + ".jpg";
		else if (cdnet_count < 1000)
			name_out = ruta + "output/in000" + ss.str() + ".jpg";
		else
			name_out = ruta + "output/in00" + ss.str() + ".jpg";
		imwrite(name_out, result);



		waitKey(1);
		//video >> frame;
		cdnet_count++;
	}


	waitKey();
	return 0;
}*/



//leer un video y guardar cada frame como imagen
/*int main()
{
	int cdnet_count = 1;
	string name, name_out;
	string ruta = "my_video_auto/";
	VideoCapture cap("my_video_auto.mp4");
	Mat save_img;
	Size size(320, 240);
	while (1)
	{
		stringstream ss;
		ss << cdnet_count;
		if (cdnet_count < 10)
			name = ruta + "input/in00000" + ss.str() + ".jpg";
		else if (cdnet_count < 100)
			name = ruta + "input/in0000" + ss.str() + ".jpg";
		else if (cdnet_count < 1000)
			name = ruta + "input/in000" + ss.str() + ".jpg";
		else
			name = ruta + "input/in00" + ss.str() + ".jpg";

		cout << cdnet_count << "  " << name << endl;
		cap >> save_img;
		resize(save_img, save_img, size);
		imshow("image", save_img);
		imwrite(name, save_img);
		cdnet_count++;
	}

	return 0;
}*/




//resize images
/*
int main()
{
	int cdnet_count = 1;
	string name, name_out;
	string ruta = "PETS2006/";
	Mat frame;
	Size size(320, 240);



	while (1)
	{
		stringstream ss;

		ss << cdnet_count;
		if (cdnet_count < 10)
			name = ruta + "input/in00000" + ss.str() + ".jpg";
		else if (cdnet_count < 100)
			name = ruta + "input/in0000" + ss.str() + ".jpg";
		else if (cdnet_count < 1000)
			name = ruta + "input/in000" + ss.str() + ".jpg";
		else
			name = ruta + "input/in00" + ss.str() + ".jpg";

		cout << cdnet_count << "  " << name << endl;

		frame = imread(name, CV_LOAD_IMAGE_COLOR);


		if (cdnet_count < 10)
			name = ruta + "input_2/in00000" + ss.str() + ".jpg";
		else if (cdnet_count < 100)
			name = ruta + "input_2/in0000" + ss.str() + ".jpg";
		else if (cdnet_count < 1000)
			name = ruta + "input_2/in000" + ss.str() + ".jpg";
		else
			name = ruta + "input_2/in00" + ss.str() + ".jpg";

		cout << cdnet_count << "  " << name << endl;

		resize(frame, frame, size);
		imshow("image", frame);
		imwrite(name, frame);
		cdnet_count++;
	}

	return 0;
}*/








//Recall pressision F1
/*
int main()
{
	int cdnet_count = 1;
	string name, name_out;
	string ruta = "PETS2006/";
	Mat frame;

	while (1)
	{
	stringstream ss;

	ss << cdnet_count;
	if (cdnet_count < 10)
		name = ruta + "input/in00000" + ss.str() + ".jpg";
	else if (cdnet_count < 100)
		name = ruta + "input/in0000" + ss.str() + ".jpg";
	else if (cdnet_count < 1000)
		name = ruta + "input/in000" + ss.str() + ".jpg";
	else
		name = ruta + "input/in00" + ss.str() + ".jpg";
	cout << cdnet_count << "  " << name << endl;
	frame = imread(name, CV_LOAD_IMAGE_COLOR);




	return 0;
}*/