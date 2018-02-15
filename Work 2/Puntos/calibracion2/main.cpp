
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <stdio.h> 
#include <type_traits> 
#include <cmath> 
#include <time.h>
#include <fstream>
#include <chrono>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/calib3d.hpp>

typedef double T_DOUBLE;
typedef char T_CHAR;
typedef long T_LONG;
typedef float T_FLOAT;
typedef int T_INT;
typedef unsigned char T_BYTE;

const T_LONG BLOQUELINEA = 1024;
const T_INT NUMTHREADS = 8;

using namespace cv;
using namespace std;

//#define norm(x, y) (fabs(x) + fabs(y)) 
void getPointsWithMaxDistance(vector <Point>& points, vector<int> hull, Point& A, Point& B, int &, int&);
void getPointsWithMaxDistance2(vector <Point>& points, vector<int> hull, Point& A, Point& B, int iA, int iB, int& iC, int& iD, Point&, Point&);
Mat DibujarPatron(Mat imagen, vector<Point> puntos, Scalar color[], T_INT m, T_INT n);
vector<Point> GenerarPatron1(T_INT m, T_INT n, Point centro, T_INT espaciado);
vector<Point> GenerarPatron2(T_INT m, T_INT n, Point p1, Point p2, Point p3, Point p4);

Point2f FindCentroid(static vector< RotatedRect > &boxElegidos)
{
	Point2f centroide;
	centroide.x = 0;
	centroide.y = 0;
	int nbelegidos = boxElegidos.size();
	for (int i = 0; i < nbelegidos; i++)
	{
		centroide.x += boxElegidos[i].center.x;
		centroide.y += boxElegidos[i].center.y;
	}
	centroide.x /= nbelegidos;
	centroide.y /= nbelegidos;
	return centroide;
}

Point2f FindCentroid(static vector< Point > &points)
{
	Point2f centroide;
	centroide.x = 0;
	centroide.y = 0;
	int nbelegidos = points.size();
	for (int i = 0; i < nbelegidos; i++)
	{
		centroide.x += points[i].x;
		centroide.y += points[i].y;
	}
	centroide.x /= nbelegidos;
	centroide.y /= nbelegidos;
	return centroide;
}

void FindDistances(static vector< RotatedRect > &boxElegidos, static Point2f &centroide, vector< double > &distancias, double &distMin, double &distMax, double &distPromedio, double &distDesvEst, double &maxRadio)
{

	distMin = 10000;
	distMax = 0;
	distPromedio = 0;
	distDesvEst = 0;
	maxRadio = 0;

	int nbelegidos = boxElegidos.size();
	distancias.clear();

	Point2f p;
	double dist1;

	for (int i = 0; i < nbelegidos; i++)
	{

		double rBox = MAX(boxElegidos[i].size.height, boxElegidos[i].size.width);
		if (rBox > maxRadio)
			maxRadio = rBox;

		p = boxElegidos[i].center;
		dist1 = sqrt(pow((p.x - centroide.x), 2) + pow((p.y - centroide.y), 2));
		distPromedio += dist1;
		distDesvEst += pow(dist1, 2);

		if (dist1 < distMin)
			distMin = dist1;

		if (dist1 > distMax)
			distMax = dist1;

		distancias.push_back(dist1);
	}
	distPromedio /= nbelegidos;
	distDesvEst -= pow(distPromedio, 2);
	distDesvEst /= nbelegidos;
	distDesvEst = sqrt(distDesvEst);

	cout << "dist max: " << distMax << " dist Min:" << distMin << " promedio:" << distPromedio << " Desv: " << distDesvEst << endl;

}

void MovePattern(vector<Point> &pts, Point2f actualCenter, Point2f newCenter) {
	T_DOUBLE dx = newCenter.x - actualCenter.x;
	T_DOUBLE dy = newCenter.y - actualCenter.y;
	for (T_INT i = 0; i < pts.size(); i++) {
		pts[i].x += dx;
		pts[i].y += dy;
	}
}


Scalar colorTab[] =
{
	Scalar(0, 0, 255),
	Scalar(0, 255, 0),
	Scalar(255, 100, 100),
	Scalar(255, 0, 255),
	Scalar(0, 255, 255),
	Scalar(255, 255, 255),
	Scalar(0, 0, 255),
	Scalar(0, 255, 0),
	Scalar(255, 100, 100),
	Scalar(255, 0, 255),
	Scalar(0, 255, 255),
	Scalar(255, 255, 255)
};


//Variables globales

clock_t h_tIni, h_tFin, h_tTotal; //  Para calculo de tiempo en CPU


int main()
{


	//cv::VideoCapture cvideo("videos/PadronAnillos_01.avi");
	//cv::VideoCapture cvideo("videos/PadronAnillos_02.avi");
	//cv::VideoCapture cvideo("videos/PadronAnillos_03.avi");
	//cv::VideoCapture cvideo("videos/PadronCirculos_01.avi");
	//cv::VideoCapture cvideo("videos/PadronCirculos_02.avi");

	cv::VideoCapture cvideo("videos/calibration_ps3eyecam.avi");
	//cv::VideoCapture cvideo("videos/calibration_mslifecam.avi");

	int m = 4;
	int n = 5;
	T_INT NroPuntos = m*n;

	if (!cvideo.isOpened())
	{
		std::cout << "no se pudo abrir el video";
		getchar();
		return -1;
	}
	T_DOUBLE nf = cvideo.get(CV_CAP_PROP_FRAME_COUNT);// para video cargado
	T_DOUBLE contf = 0;
	Mat cvimagen, cvimagen1, cvimagen2;

	Mat src;
	int w, h;


	vector<Point> patronGenerado;

	while (contf++<nf)
	{

		cvideo >> cvimagen;
		auto start = std::chrono::high_resolution_clock::now();

		w = src.cols;
		h = src.rows;

		// para el video de baja resolucion
		//Size tmp = cvimagen.size();
		//tmp.height /= 2;
		//tmp.width /= 2;
		//resize(cvimagen, cvimagen, tmp);

		src = cvimagen.clone();

		/////////////////////////////////
		// Acciones a realizar
		/////////////////////////////////

		// Convertir Escala Grises
		cv::cvtColor(cvimagen, cvimagen1, CV_BGR2GRAY);

		// Gaussian BLUR OpenCV
		cv::GaussianBlur(cvimagen1, cvimagen1, Size(5, 5), 0, 0);
		//cv::medianBlur(cvimagen1, cvimagen1, 5);

		//Adaptative Threshold Open CV
		//cv::adaptiveThreshold(cvimagen1, cvimagen1, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,11,5);

		// Detector debordes Cany
		cv::Canny(cvimagen1, cvimagen2, 100, 200);

		vector< RotatedRect > boxElegidos;
		vector< vector<Point> > contours;



		findContours(cvimagen2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_TREE // CV_RETR_EXTERNAL

		Mat cimage = Mat::zeros(cvimagen2.size(), CV_8UC3);

		////////////////////////////////////////////////////////
		// Para elegir contornos candidatos que parecen elipses
		////////////////////////////////////////////////////////
		for (int i = 0; i < contours.size(); i++)
		{
			Mat ellipseTmp1 = Mat::zeros(cvimagen2.size(), CV_8UC1);
			Mat ellipseTmp2 = Mat::zeros(cvimagen2.size(), CV_8UC1);

			// ajustar contornos a elipses

			if (contours[i].size() < 10) // pàra fit Ellipse se requiere al menos 5 puntos
				continue;

			Mat pointsf;
			Mat(contours[i]).convertTo(pointsf, CV_32F);
			RotatedRect box = fitEllipse(pointsf);

			// si las elipses son muy achatadas se descartan
			if (MIN(box.size.width, box.size.height) / MAX(box.size.width, box.size.height) < 0.25)
				continue; // se descarta

						  // las elipses no pueden ser muy grandes
			if (MAX(box.size.width, box.size.height)>cvimagen2.size().width / 4)
				continue; // se descarta

			Rect roi = box.boundingRect();

			if (roi.x < 0 || roi.x + roi.width > cvimagen2.size().width - 1 || roi.y < 0 || roi.y + roi.height > cvimagen2.size().height - 1)
			{
				continue;
			}
			else
			{
				//cout << "box center: " << box.center << "image" << ellipseTmp1.size() << endl;
				Mat cropEllipseTmp1;
				Mat cropEllipseTmp2;

				drawContours(ellipseTmp1, contours, i, Scalar(255));
				cropEllipseTmp1 = ellipseTmp1(box.boundingRect());
				floodFill(ellipseTmp1, Point(roi.width / 2, roi.height / 2), Scalar(255));

				ellipse(ellipseTmp2, box, Scalar(255));
				cropEllipseTmp2 = ellipseTmp2(box.boundingRect());
				floodFill(ellipseTmp2, Point(roi.width / 2, roi.height / 2), Scalar(255));

				//imshow("tmp ellipse", cropEllipseTmp1);
				//imshow("tmp ellipse2", cropEllipseTmp2);
				Mat ellipseXOR;
				bitwise_xor(cropEllipseTmp1, cropEllipseTmp2, ellipseXOR);

				//imshow("Ellipse And", ellipseXOR);

				int white = 0;

				for (int ei = 0; ei < ellipseXOR.cols; ei++)
					for (int ej = 0; ej < ellipseXOR.rows; ej++)
						if (ellipseXOR.at<uchar>(ej, ei) == 255)
							white++;

				double area = CV_PI*box.size.width*box.size.height;

				// Se inserta box elegido
				if (white <= area*0.02)
				{
					boxElegidos.push_back(box);
					//cout << "blanco" << white << "are<" << area << endl;
				}

				//waitKey();
			}
		}

		///////////////////////////////////////////////
		// HALLAR SOLO LAS ELIPSES NECESARIAS
		//////////////////////////////////////////////

		//Hallar el centroide por medio de la media y varianza
		T_DOUBLE mediax = 0, mediay = 0;
		T_DOUBLE varianzax = 0, varianzay = 0, dex, dey;
		T_DOUBLE dist1, dist2, distt;
		Point p, p1, p2, c, pt;


		T_INT nbelegidos = boxElegidos.size();


		// Solo si hay los puntos necesarios, si son menos no se hace nada, se pierde el frame
		if (nbelegidos >= NroPuntos)
		{
			// Se eligen puntos que esten dentro de las 2 desviacion estandar
			T_DOUBLE distx, disty;

			//calculo todas las distancias
			vector<double> distancias;


			// Para hallar el centroide			

			Point2f centroide = FindCentroid(boxElegidos), centroide2;

			vector< RotatedRect > boxFinal;

			double distMin = 10000;
			double distMax = 0;
			double distPromedio = 0;
			double distDesvEst = 0;
			double maxRadio = 0;


			while (1)
			{
				Mat temp = cvimagen.clone();

				circle(temp, centroide, 2, Scalar(255, 15, 140), 2, 8);

				boxFinal.clear();
				// luego hallo las distancias de todos los puntos al centroide

				FindDistances(boxElegidos, centroide, distancias, distMin, distMax, distPromedio, distDesvEst, maxRadio);

				//cout <<"dist max: "<< distMax<<" dist Min:" << distMin<<" promedio:" << distPromedio<<" Desv: "<<distDesvEst<<endl;

				// Aca es donde descarto a los puntos que no me son utiles

				circle(temp, centroide, distPromedio + distDesvEst / 1.3, Scalar(0, 125, 255), 2, 8);

				T_INT insertable = 0;
				for (int i = 0; i < nbelegidos; i++)
				{
					if (distancias[i] <= distPromedio + distDesvEst / 1.5)
						boxFinal.push_back(boxElegidos[i]);
				}

				// dibujar los puntos preseleccionados
				for (int i = 0; i < boxFinal.size(); i++)
					circle(temp, boxFinal[i].center, 2, Scalar(0, 125, 255), 2, 8);

				imshow("los puntos que quedan", temp);
				//waitKey(300);		

				centroide2 = FindCentroid(boxFinal);
				cout << "centroide: " << centroide << " - centroide 2: " << centroide2 << endl;


				if (centroide == centroide2)
					break;
				else
				{
					boxElegidos = boxFinal;
					nbelegidos = boxElegidos.size();
					centroide = centroide2;
				}
			}

			cout << "box final puntos: " << boxFinal.size() << endl;



			if (boxFinal.size() == NroPuntos)
			{

				// Dibujar las lineas de contorno de los puntos seleccionados con  convex Hull

				vector<Point> puntos;
				vector<Point> corners;
				vector<Point> hullCorners;

				vector<int> hull;
				for (int i = 0; i < boxFinal.size(); i++)
					puntos.push_back(boxFinal[i].center);

				convexHull(puntos, hull, false);

				for (int i = 0; i < hull.size() - 1; i++)
					line(cvimagen, puntos[hull[i]], puntos[hull[i + 1]], Scalar(0, 255, 0), 1, 8);
				line(cvimagen, puntos[hull[0]], puntos[hull[hull.size() - 1]], Scalar(0, 255, 0), 1, 8);

				// se obtiene las esquinas de los bordes, con sus respectivos angulos

				T_DOUBLE ma, mb, theta;
				Point2f	p0, p1, p2;
				T_INT contAng = 0;
				for (T_INT i = 0; i < hull.size(); i++)
				{
					if (i == 0)					p1 = puntos[hull[hull.size() - 1]];	else p1 = puntos[hull[i - 1]];
					if (i == hull.size() - 1)	p2 = puntos[hull[0]];				else p2 = puntos[hull[i + 1]];

					p0 = puntos[hull[i]];

					circle(src, p1, 6, colorTab[0], 2, 8);
					circle(src, p2, 6, colorTab[1], 2, 8);

					double c1 = norm(p0 - p1);
					double c2 = norm(p0 - p2);
					double h = norm(p1 - p2);

					double th = (c1 + c2)*1.25;
					cout << "Punto1: " << p1 << " Punto0: " << p0 << " Punto2: " << p2;
					cout << "c1: " << c1 << " c2: " << c2 << " h: " << h << " = " << th;
					if (h>th)
					{
						cout << " insertado";
						hullCorners.push_back(p0);
					}
					cout << endl;
					circle(src, p0, 6, colorTab[2], 2, 8);
					imshow("corners", src);
					//waitKey();


				}

				cout << "corners" << hullCorners.size() << endl;

				//
				if (contf == 1)// solo en el primer frame
				{
					//Definir la plantilla inicial
					Point2f p1, p2, p3, p4;

					T_DOUBLE xx, yy;

					xx = (maxRadio + maxRadio / 2)*(n / 2);
					yy = (maxRadio + maxRadio / 2)*(m / 2);

					p1.x = centroide.x - xx;	p1.y = centroide.y - yy;
					p2.x = centroide.x + xx;	p2.y = centroide.y - yy;
					p3.x = centroide.x + xx;	p3.y = centroide.y + yy;
					p4.x = centroide.x - xx;	p4.y = centroide.y + yy;

					patronGenerado = GenerarPatron2(m, n, p1, p2, p3, p4);

					vector<Point> esquinasPatron;
					esquinasPatron.push_back(patronGenerado[0]);
					esquinasPatron.push_back(patronGenerado[n - 1]);
					esquinasPatron.push_back(patronGenerado[m*n - 1]);
					esquinasPatron.push_back(patronGenerado[(m - 1)*n]);

					// Ajustar el patron generado a los puntos del convex hull
					//aca debo buscar las esquinas mas cercanas del hull a las esquinas del patron
					T_DOUBLE dist, disttemp;
					T_INT pivote;
					for (T_INT i = 0; i < esquinasPatron.size(); i++) {
						dist = 10000;
						for (T_INT j = 0; j < hullCorners.size(); j++) {
							disttemp = norm(esquinasPatron[i] - hullCorners[j]);
							if (disttemp < dist) {
								dist = disttemp;
								pivote = j;
							}
						}
						esquinasPatron[i] = hullCorners[pivote];
					}
					patronGenerado = GenerarPatron2(m, n, esquinasPatron[0], esquinasPatron[1], esquinasPatron[2], esquinasPatron[3]);

				}
				else
				{
					Point2f ca, cn;

					ca = FindCentroid(patronGenerado);
					cn = centroide;
					cout << "centroide anterior:" << ca << " nuevo centroide:" << cn << endl;
					MovePattern(patronGenerado, ca, cn);

					vector<Point> esquinasPatron;
					esquinasPatron.push_back(patronGenerado[0]);
					esquinasPatron.push_back(patronGenerado[n - 1]);
					esquinasPatron.push_back(patronGenerado[m*n - 1]);
					esquinasPatron.push_back(patronGenerado[(m - 1)*n]);

					// Ajustar el patron generado a los puntos del convex hull
					//aca debo buscar las esquinas mas cercanas del hull a las esquinas del patron
					T_DOUBLE dist, disttemp;
					T_INT pivote;
					for (T_INT i = 0; i < esquinasPatron.size(); i++) {
						dist = 10000;
						for (T_INT j = 0; j < hullCorners.size(); j++) {
							disttemp = norm(esquinasPatron[i] - hullCorners[j]);
							if (disttemp < dist) {
								dist = disttemp;
								pivote = j;
							}
						}
						esquinasPatron[i] = hullCorners[pivote];
					}
					patronGenerado = GenerarPatron2(m, n, esquinasPatron[0], esquinasPatron[1], esquinasPatron[2], esquinasPatron[3]);

				}

				cout << "aca todo bien" << endl;

				DibujarPatron(src, patronGenerado, colorTab, 4, 5);
				imshow("patron generado", src);
				//waitKey();

				circle(src, c, 2, Scalar(120, 15, 140), 2, 8);

			}

		}

		imshow("result", cimage);
		imshow("4 esquinas", src);
		imshow("original", cvimagen);
		imshow("canny", cvimagen2);

		auto finish = std::chrono::high_resolution_clock::now();
		std::cout << "Tiempo: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1000000.0 << "ms por frame" << endl;
		uchar tec = cv::waitKey(1);
		if (tec == 27) break;
	}
	std::cout << std::endl << "termino el video";
	getchar();

}

void getPointsWithMaxDistance(vector <Point>& points, vector<int> hull, Point& A, Point& B, int &iA, int & iB) {

	double d = 0;

	for (int i = 0; i < hull.size(); i++) {

		Point2f max1 = points[hull[i]];
		for (int j = 0; j < hull.size(); j++) {

			Point2f max2 = points[hull[j]];
			double distance = cv::norm(Mat(max1), Mat(max2), NORM_L2);//cout << "distance: " << distance << endl;
			if (distance > d) {
				A = max1;
				B = max2;
				iA = i;
				iB = j;
				d = distance;
			}
		}
	}
}

void getPointsWithMaxDistance2(vector <Point>& points, vector<int> hull, Point& A, Point& B, int iA, int iB, int& iC, int& iD, Point& C, Point& D) {

	double d = 0;

	for (int i = 0; i < hull.size(); i++) {

		if (i != iA && i != iB) {
			Point2f max1 = points[hull[i]];
			for (int j = 0; j < hull.size(); j++) {
				if (j != iA && j != iB) {
					Point2f max2 = points[hull[j]];
					double distance = cv::norm(Mat(max1), Mat(max2), NORM_L2);//cout << "distance: " << distance << endl;
					if (distance > d) {
						C = max1;
						D = max2;
						iC = i;
						iD = j;
						d = distance;
					}
				}
			}
		}
	}
}


Mat DibujarPatron(Mat imagen, vector<Point> puntos, Scalar color[], T_INT m, T_INT n)
{

	for (T_INT i = 0; i < m; i++)
	{
		T_INT j;
		for (j = 0; j < n - 1; j++)
		{
			circle(imagen, puntos[i*n + j], 4, color[i], 1, 8);
			circle(imagen, puntos[i*n + j], 1, color[i], 1, 8);
			circle(imagen, puntos[i*n + j + 1], 4, color[i], 1, 8);
			circle(imagen, puntos[i*n + j + 1], 1, color[i], 1, 8);
			line(imagen, puntos[i*n + j], puntos[i*n + j + 1], color[i], 1, 8);
			cv::imshow("...", imagen);
			//cv::waitKey(50);
		}
		if (i < m - 1)
			line(imagen, puntos[i*n + j], puntos[(i + 1)*n], color[i], 1, 8);
	}


	circle(imagen, puntos[0], 2, color[1], 1, 8);
	circle(imagen, puntos[n - 1], 2, color[1], 1, 8);
	circle(imagen, puntos[m*n - 1], 2, color[1], 1, 8);
	circle(imagen, puntos[(m - 1)*n], 2, color[1], 1, 8);


	return imagen;
}


vector<Point> GenerarPatron1(T_INT m, T_INT n, Point centro, T_INT espaciado)
{
	vector<Point> puntos;
	Point p;
	T_INT dx = (espaciado*(n - 1)) / 2, dy = (espaciado*(m - 1)) / 2;
	for (T_INT i = 0; i < m; i++)
		for (T_INT j = 0; j < n; j++)
		{
			p.x = centro.x + j*espaciado - dx;
			p.y = centro.y + i*espaciado - dy;
			puntos.push_back(p);
		}
	return puntos;
}

vector<Point> GenerarPatron2(T_INT m, T_INT n, Point p1, Point p2, Point p3, Point p4)
{
	Point pt1, pt2, p;
	vector<Point> puntos;

	for (T_INT i = 0; i < m*n; i++)// inicializar los puntos
		puntos.push_back(p1);
	//deformar los puntos

	T_DOUBLE disx_x, disx_y, disy1_x, disy1_y, disy2_x, disy2_y;

	// Insertar los cuatro puntos
	puntos[0] = p1;
	puntos[n - 1] = p2;
	puntos[m*n - 1] = p3;
	puntos[(m - 1)*n] = p4;

	disy1_x = (p4.x - p1.x) / (m - 1);
	disy1_y = (p4.y - p1.y) / (m - 1);

	disy2_x = (p3.x - p2.x) / (m - 1);
	disy2_y = (p3.y - p2.y) / (m - 1);

	for (T_INT i = 0; i < m; i++)
	{
		if (i > 0) {
			puntos[i*n].x = puntos[(i - 1)*n].x + disy1_x;
			puntos[i*n].y = puntos[(i - 1)*n].y + disy1_y;
			puntos[i*n + n - 1].x = puntos[(i - 1)*n + n - 1].x + disy2_x;;
			puntos[i*n + n - 1].y = puntos[(i - 1)*n + n - 1].y + disy2_y;;
			pt1 = puntos[i*n];
			pt2 = puntos[i*n + n - 1];
		}
		else {
			pt1 = puntos[0];
			pt2 = puntos[n - 1];
		}

		disx_x = (pt2.x - pt1.x) / (n - 1);
		disx_y = (pt2.y - pt1.y) / (n - 1);

		for (T_INT j = 1; j < n - 1; j++) {
			p.x = pt1.x + j*disx_x;
			p.y = pt1.y + j*disx_y;
			puntos[i*n + j] = p;
		}
	}
	return puntos;
}
