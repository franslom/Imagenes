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

using namespace std;
using namespace cv;


struct Anillo
{
public:

	int idVector = -1;
	float cx=0;
	float cy=0;
	float radioX=1;
	float radioY=1;

	Anillo ()
	{

	}
	Anillo (float cx, float cy, float radioX, float radioY)
	{
		this->cx = cx;
		this->cy = cy;
		this->radioX = radioX;
		this->radioY = radioY;
	}

	void reemplazar (Anillo ani)
	{
		this->cx = ani.cx;
		this->cy = ani.cy;
		this->radioX = ani.radioX;
		this->radioY = ani.radioY;
	}

	Anillo clonar()
	{
		return  Anillo(cx, cy, radioX, radioY);
	}

};

void ordenarLinea(vector<Anillo>&lineaEncontrada)
{
	float xMen;
	for(uint i=0; i<lineaEncontrada.size(); i++)
	{
		xMen = lineaEncontrada[i].cx;
		for(uint j=i+1; j<lineaEncontrada.size(); j++)
		{
			if(xMen>lineaEncontrada[j].cx)
			{
				xMen = lineaEncontrada[j].cx;
				Anillo aux = lineaEncontrada[i].clonar();
				lineaEncontrada[i].reemplazar(lineaEncontrada[j]);
				lineaEncontrada[j].reemplazar(aux);
			}
		}
	}
}

vector<Anillo> puntosRectaInferior(Mat img, vector<Anillo>& ptsEncontrados, uint ancho, int co)
						{
	vector<Anillo> lineaEncontrada;
	float x1, y1, x2, y2, m;
	float yCal;
	uint cant=0;

	Anillo a1;
	Anillo a2;

	for(uint k=0; k<2; k++)
	{
		Anillo a1 = ptsEncontrados[k].clonar();
		Anillo a2 = ptsEncontrados[k+1].clonar();
		x1 = a1.cx;
		y1 = a1.cy;
		x2 = a2.cx;
		y2 = a2.cy;
		m = (y2-y1)/(x2-x1);

		for(uint j=0; j<ptsEncontrados.size(); j++)
		{
			yCal = m*(ptsEncontrados[j].cx-x1)+y1;
			if( abs(yCal-ptsEncontrados[j].cy)<5  )
			{
				cant++;
				lineaEncontrada.push_back(ptsEncontrados[j].clonar());
				ptsEncontrados[j].idVector = j;
			}
		}
		if(cant==ancho)
		{
			break;
		}
		else
		{
			for(uint j=0; j<lineaEncontrada.size(); j++)
			{
				ptsEncontrados[lineaEncontrada[j].idVector].idVector = -1;
			}
			cant = 0;
			lineaEncontrada.clear();

		}
	}

	/*Scalar color;
	if(co==0)
		color = Scalar(0,0,255);
	else if(co==1)
		color = Scalar(0,255,255);
	else if(co==2)
		color = Scalar(0,255,0);
	else
		color = Scalar(255,255,0);
	for(uint j=0; j<lineaEncontrada.size(); j++)
	{
		circle(img,Point(lineaEncontrada[j].cx, lineaEncontrada[j].cy),3,color,-1);
		putText(img, to_string(j), Point(lineaEncontrada[j].cx, lineaEncontrada[j].cy),
				FONT_HERSHEY_PLAIN, 1, Scalar(200,255,200) );
	}
imshow("Cirq", img);*/

	if(lineaEncontrada.size()!=ancho)
	{
		//cout<<"no encontro"<<endl;
		lineaEncontrada.clear();
		return lineaEncontrada;
	}

	for(int j=ptsEncontrados.size()-1; j>=0; j--)
	{
		if(ptsEncontrados[j].idVector!=-1)
		{
			ptsEncontrados.erase(ptsEncontrados.begin()+j);
		}
	}





	ordenarLinea(lineaEncontrada);

	/*for(uint j=0; j<lineaEncontrada.size(); j++)
	{
		putText(img, to_string(j), Point(lineaEncontrada[j].cx, lineaEncontrada[j].cy+20),
				FONT_HERSHEY_PLAIN, 1, Scalar(50,50,255) );
	}*/

	//imshow("Cir", img);
	return lineaEncontrada;
						}

bool ordenarPuntos(vector<Anillo> &ptsEncontrados, Mat img, Size size)
{
	if(size.height<2 || size.width<2)
		return false;

	vector<Anillo> ptsRestantes;
	for(uint i=0; i<ptsEncontrados.size(); i++)
	{
		ptsRestantes.push_back(ptsEncontrados[i].clonar());
	}

	vector<vector<Anillo>> lineas;
	for(int i=0; i<size.height; i++)
	{
		vector<Anillo> lineaEncontrada = puntosRectaInferior(img, ptsRestantes, size.width,i);
		if(lineaEncontrada.empty())
		{
			return false;
		}
		lineas.push_back(lineaEncontrada);
	}


	ptsEncontrados.clear();
	for(uint i=0; i<lineas.size(); i++)
	{
		for(uint j=0; j<lineas[i].size(); j++)
		{
			ptsEncontrados.push_back(lineas[i][j].clonar());
		}

	}
	return true;
}



int encontrarBlobElipse(int pos, vector<KeyPoint> bloquesPosibles, vector<RotatedRect> elipsesPosibles)
{
	int posElipse = -1;

	float cxB=bloquesPosibles[pos].pt.x, cyB=bloquesPosibles[pos].pt.y;
	float cxE, cyE;

	for (int i = elipsesPosibles.size()-1; i >=0 ; --i)
	{
		cxE =  elipsesPosibles[i].center.x;
		cyE =  elipsesPosibles[i].center.y;
		if( sqrt( pow(cxB-cxE,2)+pow(cyB-cyE,2) )<= 5 )
		{
			return i;
		}
	}

	return posElipse;
}

bool encontrarBlobCercano(vector<KeyPoint> keypoints, int pos)
{
	float x=0, y=0, ancho=1;
	x = keypoints[pos].pt.x;
	y = keypoints[pos].pt.y;
	ancho = keypoints[pos].size;
	for (int i = keypoints.size()-1; i >=0 ; --i)
	{
		if( i!=pos && sqrt( pow(x-keypoints[i].pt.x,2)+pow(y-keypoints[i].pt.y,2) )< ancho*5 &&
				sqrt( pow(x-keypoints[i].pt.x,2)+pow(y-keypoints[i].pt.y,2) )< keypoints[i].size*5)
		{
			return true;
		}
	}
	return false;
}

bool encontrarHijosElipse(int pos, vector<RotatedRect> misElipses)
{
	//ellipse( img, misElipses[pos], Scalar( 0,255,0), 1, 8 );
	float cx1, cx2, cy1, cy2;

	cx1 = misElipses[pos].center.x;
	cy1 = misElipses[pos].center.y;

	for( uint i = pos+1; i< misElipses.size(); i++ )
	{
		cx2 = misElipses[i].center.x;
		cy2 = misElipses[i].center.y;
		if( sqrt( pow(cx1-cx2,2)+pow(cy1-cy2,2) ) <= 5)
		{
			return true;
		}
	}
	return false;
}

bool findRingGrid(Mat imgE, Size size, vector<Point2f> &pointBuf)
{
	Mat img, imgGris, imgBlur, imgBinaria, imgCanny;
	img = imgE.clone();
	Mat imgCopia = imgE.clone();
	cvtColor( img, imgGris, CV_BGR2GRAY );
	blur( imgGris, imgBlur, Size(5,5) );
	adaptiveThreshold(imgBlur, imgBinaria,255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,11,3);
	Canny( imgBinaria, imgCanny, 3, 100, 3 );

	//imshow("Cirq", img);
	//Detector Elipses
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours( imgBinaria, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


	vector<RotatedRect> minRect( contours.size() );
	vector<RotatedRect> minEllipse, misElipses, elipsesPosibles;
	float radioInterno = img.cols/24;

	vector<Point2f> pointBufAux;
	for( uint i = 0; i < contours.size(); i++ )
	{
		minRect[i] = minAreaRect( Mat(contours[i]) );
		if( contours[i].size() > 5)
		{
			minEllipse.push_back(fitEllipse( Mat(contours[i]) ) );
		}
	}

	//Primera eliminacion de elipses (en tamaño y proporcion ancho y alto)
	for( uint i = 0; i< minEllipse.size(); i++ )
	{
		Size2f tam = minEllipse[i].size;
		//que los posibles radios no sean tan grandes, ni tan pequeños
		if(tam.height<radioInterno*5 && tam.width<radioInterno*5 &&
				tam.height>5 && tam.width>5)
		{
			//que ancho i alto tengan realcion
			if(tam.height>tam.width && (tam.height/tam.width)<5)
				misElipses.push_back(minEllipse[i]);
			else if( (tam.width/tam.height)<5)
				misElipses.push_back(minEllipse[i]);
		}
	}

	//Para ver si tiene algun hijo dentro;
	for( uint i = 0; i< misElipses.size(); i++ )
	{
		if(encontrarHijosElipse(i, misElipses))
		{
			//ellipse( img, misElipses[i], Scalar( 255,255,0), 1, 8 );
			elipsesPosibles.push_back( misElipses[i] );
		}
	}




	//Blob Detector
	SimpleBlobDetector::Params params;
	params.filterByArea = true;
	params.minArea = 100;
	params.filterByCircularity = true;
	params.maxCircularity = 1;
	params.minCircularity=0.5;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	vector<KeyPoint> keypoints, bloquesPosibles;
	detector->detect( imgCanny, keypoints);

	float x, y;
	//float ancho, alto;

	//Reducir los q estan muy lejos
	for (int i = keypoints.size()-1; i >=0 ; --i)
	{

		if( encontrarBlobCercano(keypoints, i) )
		{
			/*x = keypoints[i].pt.x;
			y = keypoints[i].pt.y;
			ancho = keypoints[i].size;
			alto = keypoints[i].size;*/

			bloquesPosibles.push_back(keypoints[i]);
			//rectangle(img, Rect(x-ancho/2, y-alto/2, ancho, alto), Scalar(255,255,0), 1, 8);
		}
	}

	vector<Anillo> patron;
	//Verifica que exista el bloque y la elipse
	int posElipse;
	for (int i = bloquesPosibles.size()-1; i >=0 ; --i)
	{
		posElipse = encontrarBlobElipse(i, bloquesPosibles, elipsesPosibles);
		if( posElipse!=-1 )
		{
			x = bloquesPosibles[i].pt.x;
			y = bloquesPosibles[i].pt.y;

			x = (x+elipsesPosibles[posElipse].center.x)/2;
			y = (y+elipsesPosibles[posElipse].center.y)/2;

			//circle(img, Point2f(x,y), 3, Scalar(0,255,0),-1);
			//ellipse( img, elipsesPosibles[posElipse], Scalar( 0,0,255), 1, 8 );
			patron.push_back(Anillo(x,y,elipsesPosibles[posElipse].size.width/4, elipsesPosibles[posElipse].size.height/4 ));
		}
	}
	//cout<<patron.size()<<endl;


	//imshow("Cir", img);


	if(patron.size()==20)
	{


		if(!ordenarPuntos(patron, imgCopia, size))
		{
			//cout<<"no Ordeno"<<endl;
			return false;
		}

		for(uint i=0; i<patron.size(); i++)
		{
			pointBuf.push_back(Point2f(patron[i].cx, patron[i].cy));
			//rectangle(img, Rect(patron[i].cx-patron[i].ancho/2, y-alto/2, ancho, alto), Scalar(255,0,0), 2, 8);
			//ellipse( img, elipsesPosibles[posElipse], Scalar( 0,0,255), 1, 8 );
			circle(img, Point(patron[i].cx, patron[i].cy), patron[i].radioX, Scalar(0,0,255), 2, 8 );
		}
		//imshow("Cir", img);
		return true;
	}
	else
	{
		//imshow("Cir", img);
		//cout<<"no hay 20"<<endl;
	}


	return false;

}

