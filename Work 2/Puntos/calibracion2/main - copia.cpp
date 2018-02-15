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
void getPointsWithMaxDistance(vector <Point>& points, vector<int> hull, Point& A, Point& B, int &, int&);
void getPointsWithMaxDistance2(vector <Point>& points, vector<int> hull, Point& A, Point& B, int iA, int iB, int& iC, int& iD, Point&, Point&);
Mat DibujarPatron(Mat imagen, vector<Point> puntos, Scalar color[], T_INT m, T_INT n);
vector<Point> GenerarPatron1(T_INT m, T_INT n, Point centro, T_INT espaciado);
vector<Point> GenerarPatron2(T_INT m, T_INT n, Point p1, Point p2, Point p3, Point p4);

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



template<class T>
class h_Matriz
{
public:
	T *ptr;
	size_t row, col;

	h_Matriz(){}
	h_Matriz(size_t n)				{ inicializar(n, n); }
	h_Matriz(size_t m, size_t n)	{ inicializar(m, n); }
	h_Matriz(cv::Mat img)
	{
		inicializar(img.rows, img.cols);
		for (int i = 0; i < row; i++)
			memcpy(&(ptr[i*col]), img.ptr<T>(i, 0), col * sizeof(T));
	}

	void inicializar(size_t m, size_t n)
	{
		row = m;
		col = n;
		ptr = new T[row * col];
	}

	inline T Get(size_t r, size_t c)			{ return *(ptr + r*(col)+c); }
	inline void Set(size_t r, size_t c, T val)	{ *(ptr + r*(col)+c) = val; }

	void Set_Matriz(h_Matriz<T> mat)
	{
		delete ptr;
		inicializar(mat.row, mat.col);
		memcpy(&(ptr[0]), &(mat.ptr[0]), row*col * sizeof(T));
	}

	void Get_Matriz(h_Matriz<T> *mat)
	{
		if (mat->row == row && mat->col == col)
			memcpy(&mat->ptr[0], &(ptr[0]), row*col * sizeof(T));
	}

	void h_Matriz2Mat(cv::Mat *img)
	{
		if (img->rows == row && img->cols == col)
			for (size_t i = 0; i < row; i++)
				memcpy(img->ptr<T>(i, 0), &(ptr[i*col]), col * sizeof(T));
	}

	void Imprimir()
	{
		for (size_t i = 0; i < row; i++)
		{
			for (size_t j = 0; j < col; j++)
				cout << ptr[i*col + j] << "  ";
			cout << endl;
		}
	}

	~h_Matriz()
	{
	}
};


//Variables globales

clock_t h_tIni, h_tFin, h_tTotal; //  Para calculo de tiempo en CPU


int main()
{


	//cv::VideoCapture cvideo("videos/PadronAnillos_01.avi");T_INT NroPuntos = 30;
	//cv::VideoCapture cvideo("videos/PadronAnillos_02.avi"); T_INT NroPuntos = 30;
	//cv::VideoCapture cvideo("videos/PadronAnillos_03.avi");T_INT NroPuntos = 30;
	//cv::VideoCapture cvideo("videos/PadronCirculos_01.avi");T_INT NroPuntos = 44;
	//cv::VideoCapture cvideo("videos/PadronCirculos_02.avi");T_INT NroPuntos = 44;
	//cv::VideoCapture cvideo("videos/PadronCirculos_03.avi");T_INT NroPuntos = 44;

	//cv::VideoCapture cvideo("videos/calibration_kinectv2.avi"); T_INT NroPuntos = 20;
	//cv::VideoCapture cvideo("videos/calibration_mslifecam.avi"); T_INT NroPuntos = 20;
	cv::VideoCapture cvideo("videos/calibration_ps3eyecam.avi"); T_INT NroPuntos = 20;
	//cv::VideoCapture cvideo("videos/realsense_Depth.avi"); T_INT NroPuntos = 20;
	
	//real time
	//cv::VideoCapture cvideo(0); T_INT NroPuntos = 20;
	
	
	
	


	if (!cvideo.isOpened())
	{
		std::cout << "no se pudo abrir el video";
		getchar();
		return -1;
	}
	//T_DOUBLE nf = 1000;
	T_DOUBLE nf = cvideo.get(CV_CAP_PROP_FRAME_COUNT);// para video cargado
	T_DOUBLE contf = 1.0;
	Mat cvimagen, cvimagen1, cvimagen2;
	
	Mat src;
	int w, h;
	
	while (contf++<nf)
	//while (1)
	{

		cvideo >> cvimagen;
		auto start = std::chrono::high_resolution_clock::now();
		src = cvimagen.clone();
		w = src.cols;
		h = src.rows;

		/////////////////////////////////
		// Acciones a realizar
		/////////////////////////////////

		// Convertir Escala Grises
		cv::cvtColor(cvimagen, cvimagen1, CV_BGR2GRAY);
		imshow("Gray", cvimagen1);


		// Gaussian BLUR OpenCV
		cv::GaussianBlur(cvimagen1, cvimagen1, Size(5, 5), 0, 0);
		imshow("Blur", cvimagen1);

		//Adaptative Threshold Open CV
		//cv::adaptiveThreshold(cvimagen1, cvimagen1, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,11,5);

		// Detector debordes Cany
		cv::Canny(cvimagen1, cvimagen2, 100, 200);

		vector< RotatedRect > boxElegidos;
		vector< vector<Point> > contours;
		//findContours(cvimagen2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_TREE
		findContours(cvimagen2, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//CV_RETR_TREE


		//Mat mask = Mat::zeros(cvimagen2.rows, cvimagen2.cols, CV_8UC1);
		//drawContours(mask, contours, -2, Scalar(255), 0);

		Mat cimage = Mat::zeros(cvimagen2.size(), CV_8UC3);

		////////////////////////////////////////////////////////
		// Para elegir contornos candidatos que parecen elipses
		////////////////////////////////////////////////////////

		vector< RotatedRect > boxElegidos_hijos;


		for (size_t i = 0; i < contours.size(); i++)
		{
			size_t count = contours[i].size();

			// Contornos muy pequenos, se eliminan
			if (count < 10)
				continue;// se descarta

			// ajustar contornos a elipses
			Mat pointsf;
			Mat(contours[i]).convertTo(pointsf, CV_32F);
			RotatedRect box = fitEllipse(pointsf);


			//esta parte es de acuerdo a la forma del contorno
			T_DOUBLE dist1, dist2;
			Point pini;
			Point pfin;

			pini = contours[i][count*0.25];	pfin = contours[i][count*0.75];
			dist1 = sqrt(pow(pini.x - pfin.x, 2) + pow(pini.y - pfin.y, 2));

			pini = contours[i][0];	pfin = contours[i][count*0.5];
			dist2 = sqrt(pow(pini.x - pfin.x, 2) + pow(pini.y - pfin.y, 2));

			if (dist1 < 4 || dist2 < 4)
				continue; // se descarta


			// si las elipses son muy achatadas se descartan
			if (MIN(box.size.width, box.size.height) / MAX(box.size.width, box.size.height) < 0.25)
				continue; // se descarta



			// debo encontrar los puntos que se intersectan las elipses con los rectangulos

			Point2f vtx[4];
			box.points(vtx);
			Point2f vtxq[4];
			T_INT v;
			for (v = 0; v < 3; v++)
			{
				vtxq[v].x = (vtx[v].x + vtx[v + 1].x) / 2;
				vtxq[v].y = (vtx[v].y + vtx[v + 1].y) / 2;
			}
			vtxq[v].x = (vtx[0].x + vtx[v].x) / 2;
			vtxq[v].y = (vtx[0].y + vtx[v].y) / 2;

			//dibujar los contornos
			drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);

			//verificar si alguno de los puntos se intersectan con el rectangulo
			T_INT nc = 5, // se busca en una grilla 3 x 3
				fnc, // badera que indica que encontro un punto blanco
				cn = 0,
				cm = nc / 2;
			for (T_INT j = 0; j < 4; j++)
			{
				//buscar un punto blanco en el contorno del punto que intersecta
				fnc = 0;

				if (vtxq[j].y - cm >= 0 &&
					vtxq[j].y + cm < cimage.rows &&
					vtxq[j].x - cm >= 0 &&
					vtxq[j].x + cm < cimage.cols) // comprobar qu esta dentro dela imagen
				{
					for (T_INT k = 0; k < nc; k++)
						for (T_INT l = 0; l < nc; l++)
							if (cimage.at<Vec3b>(vtxq[j].y + k - cm, vtxq[j].x + l - cm)[0] == 255)
							{
								fnc = 1;
								goto AQUI;
							}
				AQUI:
					if (fnc)
						cn++;
				}
				else
					break;
				//imshow("...", cimage);
				//cv::waitKey(10);

			}

			if (cn < 4)
				continue; // se desctarta


			/////////////////////////////////////////////
			// Hasta aca tengo las ellipses
			////////////////////////////////////////////

			// Se inserta box elegido
			boxElegidos.push_back(box);

		}




		


		///////////////////////////////////////////////
		// HALLAR SOLO LAS ELIPSES NECESARIAS
		//////////////////////////////////////////////

		//Hallar el centroide por medio de la media y varianza
		T_DOUBLE mediax = 0, mediay = 0;
		T_DOUBLE varianzax = 0, varianzay = 0, dex, dey;
		T_DOUBLE dist1, dist2, distt;
		Point p1, p2, c, pt;


		T_INT nbelegidos = boxElegidos.size();


		// Solo si hay los puntos necesarios, si son menos no se hace nada, se pierde el frame
		if (nbelegidos >= NroPuntos)
		{
			// Se eligen puntos que esten dentro de las 2 desviacion estandar
			T_DOUBLE distx, disty;


			//
			//calculo todas las distancias
			h_Matriz<T_DOUBLE> distancias(nbelegidos);
			h_Matriz<T_DOUBLE> distanciasmin(nbelegidos, 1); // un vector

			// variable para la distancia minima
			T_DOUBLE dmin = 10000;// las distancias siempre son positivas
			T_DOUBLE dmax = 0;
			for (int i = 0; i < nbelegidos; i++)
			{
				dmin = 10000;
				for (int j = i; j < nbelegidos; j++)
				{
					if (i == j)
						distancias.Set(i, j, 0.0);
					else
					{
						p1 = boxElegidos[i].center;
						p2 = boxElegidos[j].center;
						dist1 = sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
						distancias.Set(i, j, dist1);
						distancias.Set(j, i, dist1);
						if (dist1 < dmin)
							dmin = dist1;
					}
				}
				dmax += MAX(boxElegidos[i].size.width, boxElegidos[i].size.height);
				distanciasmin.Set(i, 0, dmin);
			}
			dmax /= nbelegidos; // es el grosor promedio de las esferas


			// Aca es donde descarto a los puntos que no me son utiles

			vector< RotatedRect > boxFinal;
			T_INT insertable = 0;
			for (int i = 0; i < nbelegidos; i++)
			{
				insertable = 0;
				for (int j = 0; j < nbelegidos; j++)
				{
					if (i != j)
						if (distancias.Get(i, j) <= dmax*3.5)
							insertable++;
					if (insertable == 3)
						break;
				}
				if (insertable == 3)
					boxFinal.push_back(boxElegidos[i]);
			}

			//cambios para hijos
			/*for (int ff = 0; ff < boxFinal.size(); ff++)
			{
				for (int rr = 0; rr < boxFinal.size(); rr++)
				{
					if (boxFinal[ff].center.x - boxFinal[rr].center.x < 150 && boxFinal[ff].center.y - boxFinal[rr].center.y < 150)
						boxElegidos_hijos.push_back(boxFinal[ff]);
				}

			}*/


			std::cout << " dmax: " << dmax << " tam: " << boxFinal.size() << std::endl;
			// dibujar los puntos preseleccionados
			for (int i = 0; i < boxFinal.size(); i++)
				circle(cvimagen, boxFinal[i].center, 2, Scalar(0, 125, 255), 2, 8);


			if (boxFinal.size() == NroPuntos)
			{

				// Dibujar las lineas de contorno de los puntos seleccionados con  convex Hull

				vector<Point> puntos;
				vector<int> hull;
				for (int i = 0; i < boxFinal.size(); i++)
					puntos.push_back(boxFinal[i].center);
				

			}

		}
				
		imshow("original", cvimagen);		
		imshow("Canny detection", cvimagen2);
		
		auto finish = std::chrono::high_resolution_clock::now();
		std::cout << "Tiempo: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1000000.0 << "ms por frame" << endl;
		uchar tec = cv::waitKey(1);
		if (tec == 27) break;
	}
	std::cout << std::endl << "fin del video";
	getchar();

}