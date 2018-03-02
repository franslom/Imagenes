//g++ -std=c++11 -Wall -o calibracionOpencv CalibracionOpencv.cpp `pkg-config --cflags --libs opencv`
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

struct Anillo
{
public:
	float usado=false;
	int id=-1;
	int idVector = -1;
	float cx=0;
	float cy=0;
	float radioX=1;
	float radioY=1;
	int vecinos = 0;

	Anillo ()
	{

	}
	Anillo (float cx, float cy, float radioX, float radioY, int id)
	{
		this->id = id;
		this->cx = cx;
		this->cy = cy;
		this->radioX = radioX;
		this->radioY = radioY;
	}
	Anillo (float cx, float cy, float radioX, float radioY)
	{
		this->cx = cx;
		this->cy = cy;
		this->radioX = radioX;
		this->radioY = radioY;
	}

	Anillo clonar()
	{
		Anillo nuevo =  Anillo(cx, cy, radioX, radioY, id);
		nuevo.usado = usado;
		nuevo.vecinos = vecinos;
		nuevo.idVector = idVector;
		return nuevo;
	}

};

vector<Anillo> patronAntiguo;
vector<Anillo> patronNuevo;
Rect areqPequenia;

Mat getImgPequenia(Mat img, Size size)
{
	SimpleBlobDetector::Params params;
	params.filterByArea = true;
	params.minArea = 100;
	params.filterByCircularity = true;
	params.maxCircularity = 1;
	params.minCircularity=0.5;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	vector<KeyPoint> keypoints;


	Mat imgFinal, imgGris, imgBlur, imgCanny, imgBinaria;
	imgFinal = img.clone();
	cvtColor( img, imgGris, CV_BGR2GRAY );
	blur( imgGris, imgBlur, Size(5,5) );
	adaptiveThreshold(imgBlur, imgBinaria,255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,11,3);
	Canny( imgBinaria, imgCanny, 3, 100, 3 );



	detector->detect( imgCanny, keypoints);

	vector<Anillo> patron;

	for (size_t i = 0; i < keypoints.size(); ++i)
	{
		float cx = keypoints[i].pt.x;
		float cy = keypoints[i].pt.y;
		float radio = keypoints[i].size;
		Rect roiRec(cx-radio,cy-radio,radio*2,radio*2);

		if(roiRec.x>0 && roiRec.y>0 && (roiRec.x+roiRec.width)<img.cols && (roiRec.y+roiRec.height)<img.rows &&
				roiRec.width<100 && roiRec.height<100)
		{
			Mat roiImage1 = imgCanny(roiRec);
			Mat roiImage = roiImage1.clone();
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours( roiImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

			vector<RotatedRect> minEllipse( contours.size() );

			for( uint i = 0; i < contours.size(); i++ )
			{
				if( contours[i].size() > 5 )
				{
					minEllipse[i] = fitEllipse( Mat(contours[i]) );
				}
			}

			int contador=0;
			for( uint i = 0; i< contours.size(); i++ )
			{
				if(minEllipse[i].center.x-minEllipse[i].size.width/2>0 && minEllipse[i].center.y-minEllipse[i].size.height/2>0 &&
						minEllipse[i].center.x+minEllipse[i].size.width/2<roiRec.width &&
						minEllipse[i].center.y+minEllipse[i].size.height/2<roiRec.height)
				{
					minEllipse[i].center.x += roiRec.x;
					minEllipse[i].center.y += roiRec.y;

					if( abs(minEllipse[i].center.x<cx)<2 && abs(minEllipse[i].center.y<cy)<2 )
					{
						contador++;
					}
				}
			}
			if(contador>0)
			{
				rectangle(imgFinal, roiRec, Scalar(0,0,255), 1, 8, 0);
				circle(imgFinal, Point(cx, cy), radio, Scalar(0,0,255), 1, 8, 0);
				patron.push_back(Anillo(cx, cy, roiRec.width, roiRec.height));
			}
		}
	}

	if(patron.size()==20)
	{
		int minX=100000, minY=100000;
		int maxX=0, maxY=0;
		for(uint i=0; i<patron.size(); i++)
		{
			if(minX > patron[i].cx)
				minX = patron[i].cx;
			if(minY>patron[i].cy)
				minY = patron[i].cy;
			if((maxX) < patron[i].cx)
				maxX = patron[i].cx;
			if((maxY) < patron[i].cy)
				maxY = patron[i].cy;
		}

		minX += -patron[0].radioX*2;
		minY += -patron[0].radioY/2;

		maxX += patron[0].radioX*2;
		maxY += patron[0].radioY/2;

		if(minX<0)
			minX = 0;
		if(minY<0)
			minY = 0;
		if(maxX >= img.cols)
			maxX = img.cols -1;
		if(maxY >= img.rows)
			maxY = img.rows -1 ;

		maxX = maxX-minX;
		maxY = maxY-minY;
		areqPequenia.x = minX;
		areqPequenia.y = minY;
		areqPequenia.width = maxX;
		areqPequenia.height = maxY;

		Mat imgPequenia = img(areqPequenia);
		return imgPequenia;
	}
	else
	{
		return img.clone();
	}
}

bool actualizarPatron(vector<Anillo> &patronAntiguo, vector<Anillo> &patronNuevo)
{

	bool noCoicidioUno = false;
	float paddingAncho, paddingAlto;
	for(uint i=0; i<patronAntiguo.size(); i++)
	{
		bool tieneCercano = false;
		uint j=0;
		for(; j<patronNuevo.size(); j++)
		{
			paddingAncho = 10;
			paddingAlto = 10;
			if(patronNuevo[j].usado==false &&
					patronNuevo[j].cx-paddingAncho < patronAntiguo[i].cx && patronAntiguo[i].cx < patronNuevo[j].cx+paddingAncho &&
					patronNuevo[j].cy-paddingAlto < patronAntiguo[i].cy && patronAntiguo[i].cy < patronNuevo[j].cy+paddingAlto)
			{
				patronAntiguo[i].cx = patronNuevo[j].cx;
				patronAntiguo[i].cy = patronNuevo[j].cy;
				patronAntiguo[i].radioX = patronNuevo[j].radioX;
				patronAntiguo[i].radioY = patronNuevo[j].radioY;
				patronNuevo[j].usado = true;
				tieneCercano = true;
				break;
			}

		}
		if(!tieneCercano)
		{

			noCoicidioUno = true;
			break;
		}
	}

	if(noCoicidioUno)
	{
		//cout<<"nada q ver"<<endl;
		patronAntiguo.clear();
		for(uint i=0; i<patronNuevo.size(); i++)
		{
			patronAntiguo.push_back(Anillo(patronNuevo[i].cx, patronNuevo[i].cy, patronNuevo[i].radioX, patronNuevo[i].radioY));
		}
		return false;
	}
	return true;
}

int encontrarAnillo(vector<Anillo> &patronAntiguo, int id)
{
	for(uint i=0; i<patronAntiguo.size(); i++)
	{
		if(patronAntiguo[i].id==id)
		{
			return i;
		}
	}
	return 0;
}

int encontrarCercano(vector<Anillo> &patronAntiguo, float x, float y)
{
	int menor=100000;
	int id=0;
	for(uint i=0; i<patronAntiguo.size(); i++)
	{
		if(patronAntiguo[i].id==-1 && sqrt(pow(x-patronAntiguo[i].cx,2)+pow(y-patronAntiguo[i].cy,2))<menor)
		{
			menor = sqrt(pow(x-patronAntiguo[i].cx,2)+pow(y-patronAntiguo[i].cy,2));
			id = i;
		}
	}
	return id;
}

bool ordenarPuntos(vector<Anillo> &patronAntiguo, Mat &img)
{

	vector<Point> contours;
	float xA, xV, yA, yV, radioA;


	for(uint i=0; i<patronAntiguo.size(); i++)
	{
		patronAntiguo[i].idVector = i;
		xA = patronAntiguo[i].cx;
		yA = patronAntiguo[i].cy;
		radioA = patronAntiguo[i].radioX;
		for(uint j=0; j<patronAntiguo.size(); j++)
		{
			xV = patronAntiguo[j].cx;
			yV = patronAntiguo[j].cy;
			if( sqrt(pow(xA-xV,2)+pow(yA-yV,2))<radioA*5  )
			{
				patronAntiguo[i].vecinos++;
				line(img, Point(xA,yA), Point(xV,yV), Scalar(255,0,0), 1, 8, 0);
			}
		}
	}

	//Buscando esquinas
	vector<Anillo> esquinas;
	for(uint i=0; i<patronAntiguo.size(); i++)
	{
		xA = patronAntiguo[i].cx;
		yA = patronAntiguo[i].cy;
		if(patronAntiguo[i].vecinos==4)
		{
			esquinas.push_back(patronAntiguo[i].clonar());

		}
	}

	if(esquinas.size()==4)
	{
		int yMenor = 100000;
		Anillo e1;
		for(uint i=0; i<esquinas.size(); i++)
		{
			if(esquinas[i].cy<yMenor)
			{
				yMenor = esquinas[i].cy;
				e1 = esquinas[i].clonar();
			}
		}

		yMenor = 100000;
		Anillo e2;
		for(uint i=0; i<esquinas.size(); i++)
		{
			if(esquinas[i].cy<yMenor && e1.cy!=esquinas[i].cy && e1.cx!=esquinas[i].cx)
			{
				yMenor = esquinas[i].cy;
				e2 = esquinas[i].clonar();
			}

		}
		if(e1.cx>e2.cx)
		{
			Anillo aux = e1.clonar();
			e1 = e2.clonar();
			e2 = aux.clonar();
		}

		float x1 = e1.cx;
		float x2 = e2.cx;
		float y1 = e1.cy;
		float y2 = e2.cy;
		float m = (y2-y1)/(x2-(x1+0.0001));
		float ycal;

		vector<Anillo> intermedios;
		int anillisIntermedios = 0;
		for(uint i=0; i<patronAntiguo.size(); i++)
		{
			ycal = m*(patronAntiguo[i].cx-x1)+y1;
			if( abs(ycal-patronAntiguo[i].cy)<patronAntiguo[i].radioX )
			{
				anillisIntermedios++;
				if(!(patronAntiguo[i].cx==x1 || patronAntiguo[i].cx==x2))
				{
					intermedios.push_back(patronAntiguo[i].clonar());
					circle(img, Point(patronAntiguo[i].cx, patronAntiguo[i].cy), 10, Scalar(0,255,0), 4, 8, 0);
				}

			}
		}
		//cout<<"canI"<<intermedios.size()<<endl;

		if(anillisIntermedios==5)
		{
			//cout<<"caso1"<<endl;
			patronAntiguo[e1.idVector].id = 0;
			patronAntiguo[e2.idVector].id = 4;
			//Primero los que esten entre e1 y e2
			//vecino de e1 en la recta enrte e1 y e2
			float disMenor= 100000;
			int pos1=0;
			for(uint i=0; i<intermedios.size(); i++)
			{
				if(sqrt(pow(e1.cx-intermedios[i].cx,2)+pow(e1.cy-intermedios[i].cy,2))<disMenor)
				{
					disMenor = sqrt(pow(e1.cx-intermedios[i].cx,2)+pow(e1.cy-intermedios[i].cy,2));
					pos1 = i;
				}

			}
			patronAntiguo[intermedios[pos1].idVector].id = 1;



			//vecino de e2 en la recta enrte e1 y e2
			disMenor= 100000;
			int pos2=0;
			for(uint i=0; i<intermedios.size(); i++)
			{
				if(sqrt(pow(e2.cx-intermedios[i].cx,2)+pow(e2.cy-intermedios[i].cy,2))<disMenor)
				{
					disMenor = sqrt(pow(e2.cx-intermedios[i].cx,2)+pow(e2.cy-intermedios[i].cy,2));
					pos2 = i;
				}

			}
			patronAntiguo[intermedios[pos2].idVector].id = 3;


			//vecino del medio
			int pos3 = 0;
			for(int i=intermedios.size()-1; i>=0; i--)
			{
				if(i!=pos1 && i!=pos2)
				{
					pos3=i;
					break;
				}
			}
			patronAntiguo[intermedios[pos3].idVector].id = 2;



			int aux = encontrarAnillo(patronAntiguo, 0);
			int cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 5;

			aux = encontrarAnillo(patronAntiguo, 1);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 6;

			aux = encontrarAnillo(patronAntiguo, 2);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 7;

			aux = encontrarAnillo(patronAntiguo, 3);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 8;

			aux = encontrarAnillo(patronAntiguo, 4);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 9;

			aux = encontrarAnillo(patronAntiguo, 5);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 10;

			aux = encontrarAnillo(patronAntiguo, 6);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 11;

			aux = encontrarAnillo(patronAntiguo, 7);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 12;

			aux = encontrarAnillo(patronAntiguo, 8);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 13;

			aux = encontrarAnillo(patronAntiguo, 9);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 14;

			aux = encontrarAnillo(patronAntiguo, 10);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 15;

			aux = encontrarAnillo(patronAntiguo, 11);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 16;

			aux = encontrarAnillo(patronAntiguo, 12);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 17;

			aux = encontrarAnillo(patronAntiguo, 13);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 18;

			aux = encontrarAnillo(patronAntiguo, 14);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 19;


		}
		else if(e1.cy>e2.cy)
		{
			//cout<<"caso2"<<endl;
			patronAntiguo[e1.idVector].id = 15;
			patronAntiguo[e2.idVector].id = 0;


			//Primero los que esten entre e1 y e2
			//vecino de e1 en la recta enrte e1 y e2
			float disMenor= 100000;
			int pos1=0;
			for(uint i=0; i<intermedios.size(); i++)
			{
				if(sqrt(pow(e1.cx-intermedios[i].cx,2)+pow(e1.cy-intermedios[i].cy,2))<disMenor)
				{
					disMenor = sqrt(pow(e1.cx-intermedios[i].cx,2)+pow(e1.cy-intermedios[i].cy,2));
					pos1 = i;
				}

			}
			patronAntiguo[intermedios[pos1].idVector].id = 10;


			//vecino de e2 en la recta enrte e1 y e2
			disMenor= 100000;
			int pos2=0;
			for(uint i=0; i<intermedios.size(); i++)
			{
				if(sqrt(pow(e2.cx-intermedios[i].cx,2)+pow(e2.cy-intermedios[i].cy,2))<disMenor)
				{
					disMenor = sqrt(pow(e2.cx-intermedios[i].cx,2)+pow(e2.cy-intermedios[i].cy,2));
					pos2 = i;
				}

			}
			patronAntiguo[intermedios[pos2].idVector].id = 5;


			int aux = encontrarAnillo(patronAntiguo, 0);
			int cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 1;

			aux = encontrarAnillo(patronAntiguo, 5);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 6;

			aux = encontrarAnillo(patronAntiguo, 10);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 11;

			aux = encontrarAnillo(patronAntiguo, 15);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 16;

			aux = encontrarAnillo(patronAntiguo, 1);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 2;

			aux = encontrarAnillo(patronAntiguo, 6);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 7;

			aux = encontrarAnillo(patronAntiguo, 11);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 12;

			aux = encontrarAnillo(patronAntiguo, 16);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 17;

			aux = encontrarAnillo(patronAntiguo, 2);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 3;

			aux = encontrarAnillo(patronAntiguo, 7);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = 8;

			int val;
			val = 12;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val+1;

			val = 17;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val+1;

			val = 3;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val+1;

			val = 8;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val+1;

			val = 13;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val+1;

			val = 18;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val+1;
		}
		else if(e1.cy<e2.cy)
		{
			//cout<<"caso3"<<endl;
			patronAntiguo[e1.idVector].id = 4;
			patronAntiguo[e2.idVector].id = 19;

			//Primero los que esten entre e1 y e2
			//vecino de e1 en la recta enrte e1 y e2
			float disMenor= 100000;
			int pos1=0;
			for(uint i=0; i<intermedios.size(); i++)
			{
				if(sqrt(pow(e1.cx-intermedios[i].cx,2)+pow(e1.cy-intermedios[i].cy,2))<disMenor)
				{
					disMenor = sqrt(pow(e1.cx-intermedios[i].cx,2)+pow(e1.cy-intermedios[i].cy,2));
					pos1 = i;
				}

			}
			patronAntiguo[intermedios[pos1].idVector].id = 9;

			//vecino de e2 en la recta enrte e1 y e2
			disMenor= 100000;
			int pos2=0;
			for(uint i=0; i<intermedios.size(); i++)
			{
				if(sqrt(pow(e2.cx-intermedios[i].cx,2)+pow(e2.cy-intermedios[i].cy,2))<disMenor)
				{
					disMenor = sqrt(pow(e2.cx-intermedios[i].cx,2)+pow(e2.cy-intermedios[i].cy,2));
					pos2 = i;
				}

			}
			patronAntiguo[intermedios[pos2].idVector].id = 14;

			int val;
			val = 4;
			int aux = encontrarAnillo(patronAntiguo, val);
			int cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 9;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 14;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 19;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 3;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 8;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 13;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 18;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 2;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 7;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 12;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 17;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 1;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 6;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 11;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

			val = 16;
			aux = encontrarAnillo(patronAntiguo, val);
			cer = encontrarCercano(patronAntiguo, patronAntiguo[aux].cx, patronAntiguo[aux].cy);
			patronAntiguo[cer].id = val-1;

		}


		int aux;
		vector<Anillo> patronAux;
		for(uint i=0; i<patronAntiguo.size(); i++)
		{
			aux = encontrarAnillo(patronAntiguo, i);
			patronAux.push_back(patronAntiguo[aux].clonar());
		}

		patronAntiguo.clear();
		for(uint i=0; i<patronAux.size(); i++)
		{
			putText(img, to_string(i), Point(patronAux[i].cx,patronAux[i].cy), FONT_HERSHEY_SCRIPT_COMPLEX, 2, CV_RGB(125,12,145), 2);
			patronAntiguo.push_back(patronAux[i].clonar());
		}


		patronAntiguo[0].id = 15;
		patronAntiguo[1].id = 16;
		patronAntiguo[2].id = 17;
		patronAntiguo[3].id = 18;
		patronAntiguo[4].id = 19;

		patronAntiguo[5].id = 10;
		patronAntiguo[6].id = 11;
		patronAntiguo[7].id = 12;
		patronAntiguo[8].id = 13;
		patronAntiguo[9].id = 14;

		patronAntiguo[10].id = 5;
		patronAntiguo[11].id = 6;
		patronAntiguo[12].id = 7;
		patronAntiguo[13].id = 8;
		patronAntiguo[14].id = 9;

		patronAntiguo[15].id = 0;
		patronAntiguo[16].id = 1;
		patronAntiguo[17].id = 2;
		patronAntiguo[18].id = 3;
		patronAntiguo[19].id = 4;






		patronAux.clear();
		for(uint i=0; i<patronAntiguo.size(); i++)
		{
			aux = encontrarAnillo(patronAntiguo, i);
			patronAux.push_back(patronAntiguo[aux].clonar());
		}

		patronAntiguo.clear();
		for(uint i=0; i<patronAux.size(); i++)
		{
			putText(img, to_string(i), Point(patronAux[i].cx,patronAux[i].cy), FONT_HERSHEY_SCRIPT_COMPLEX, 2, CV_RGB(125,12,145), 2);
			patronAntiguo.push_back(patronAux[i].clonar());
		}

		//imshow("hill", img);
		//waitKey(0);
		return true;
	}


	return false;




}

bool boolAreaGrande = true;
int contNoEncontrado = 0;
bool boolPatronAntiguo = false;
bool findRingGrid(Mat img, Size size, vector<Point2f> &pointBuf)
{
	Mat imgPequenia;
	if(boolAreaGrande)
	{
		imgPequenia = getImgPequenia(img, size);
		if(imgPequenia.size == img.size)
		{
			return false;
		}
		boolAreaGrande = false;
	}
	else
	{
		imgPequenia = img(areqPequenia);
	}
	//imshow("imgPequenia", imgPequenia);
	//A partir de aqui trabajo con el area pequeña
	SimpleBlobDetector::Params params;
	params.filterByArea = true;
	params.minArea = 100;
	params.filterByCircularity = true;
	params.maxCircularity = 1;
	params.minCircularity=0.5;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	vector<KeyPoint> keypoints;


	Mat imgFinal, imgGris, imgBlur, imgCanny, imgBinaria;
	imgFinal = img.clone();
	cvtColor( imgPequenia, imgGris, CV_BGR2GRAY );
	blur( imgGris, imgBlur, Size(5,5) );
	adaptiveThreshold(imgBlur, imgBinaria,255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,11,3);
	Canny( imgBinaria, imgCanny, 3, 100, 3 );

	detector->detect( imgCanny, keypoints);
	if(keypoints.size()==20)
	{
		contNoEncontrado = 0;
		patronNuevo.clear();

		int minX=100000, minY=100000;
		int maxX=0, maxY=0;
		vector<Anillo> patronAux;
		for(int i=keypoints.size()-1; i>=0; i--)
		{
			circle(imgFinal, Point(keypoints[i].pt.x+areqPequenia.x, keypoints[i].pt.y+areqPequenia.y), keypoints[i].size, Scalar(0,0,255), 2, 8 );
			patronNuevo.push_back(Anillo(keypoints[i].pt.x+areqPequenia.x, keypoints[i].pt.y+areqPequenia.y, keypoints[i].size, keypoints[i].size));

			if(minX > keypoints[i].pt.x)
				minX = keypoints[i].pt.x;
			if(minY>keypoints[i].pt.y)
				minY = keypoints[i].pt.y;
			if((maxX) < keypoints[i].pt.x)
				maxX = keypoints[i].pt.x;
			if((maxY) < keypoints[i].pt.y)
				maxY = keypoints[i].pt.y;
		}

		///Para mover el area pequeña de busqueda
		areqPequenia.x = areqPequenia.x+minX-keypoints[0].size*2;
		areqPequenia.y = areqPequenia.y+minY-keypoints[0].size*2;
		if(areqPequenia.x<0)
			areqPequenia.x = 0;
		if(areqPequenia.y<0)
			areqPequenia.y = 0;

		areqPequenia.width = maxX-minX+keypoints[0].size*4;
		areqPequenia.height = maxY-minY+keypoints[0].size*4;


		if(areqPequenia.x+ areqPequenia.width >= img.cols)
			areqPequenia.width = img.cols-areqPequenia.x;
		if(areqPequenia.y+ areqPequenia.height >= img.rows)
			areqPequenia.height = img.rows-areqPequenia.y;



		if(boolPatronAntiguo)
		{
			//Actualizar puntos antiguos con los nuevos
			bool coiciden = actualizarPatron(patronAntiguo, patronNuevo);


			if(!coiciden)
			{
				if(!ordenarPuntos(patronAntiguo, imgFinal))
					return false;
			}

			for(uint i=0; i<patronAntiguo.size(); i++)
			{
				pointBuf.push_back(Point2f(patronAntiguo[i].cx, patronAntiguo[i].cy));
			}

		}
		else
		{
			patronAntiguo.clear();
			//patronA = patron Nuevo
			for(uint i=0; i<patronNuevo.size(); i++)
			{
				patronAntiguo.push_back(Anillo(patronNuevo[i].cx, patronNuevo[i].cy, patronNuevo[i].radioX, patronNuevo[i].radioY));
			}
			//ordenar
			if(!ordenarPuntos(patronAntiguo, imgFinal))
				return false;

			for(uint i=0; i<patronAntiguo.size(); i++)
			{
				pointBuf.push_back(Point2f(patronAntiguo[i].cx, patronAntiguo[i].cy));
			}
			boolPatronAntiguo = true;
		}
		//imshow("imgFinal", imgFinal);
		return true;
	}
	else
	{
		contNoEncontrado++;
		if(contNoEncontrado>10)
		{
			boolAreaGrande = true;
		}
		//cout<<"Se perdio"<<endl;
	}
	//imshow("imgFinal", imgFinal);

	return false;

}

/*static void help()
{
    cout <<  "This is a camera calibration sample." << endl
         <<  "Usage: camera_calibration [configuration_file -- default ./default.xml]"  << endl
         <<  "Near the sample file you'll find the configuration file, which has detailed help of "
             "how to edit it.  It may be any OpenCV supported file format XML/YAML." << endl;
}*/
class Settings
{
public:
	Settings() : goodInput(false) {}
	enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID, RING_GRID };
	enum InputType { INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST };

	void write(FileStorage& fs) const                        //Write serialization for this class
	{
		fs << "{"
				<< "BoardSize_Width"  << boardSize.width
				<< "BoardSize_Height" << boardSize.height
				<< "Square_Size"         << squareSize
				<< "Calibrate_Pattern" << patternToUse
				<< "Calibrate_NrOfFrameToUse" << nrFrames
				<< "Calibrate_FixAspectRatio" << aspectRatio
				<< "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
				<< "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

				<< "Write_DetectedFeaturePoints" << writePoints
				<< "Write_extrinsicParameters"   << writeExtrinsics
				<< "Write_outputFileName"  << outputFileName

				<< "Show_UndistortedImage" << showUndistorsed

				<< "Input_FlipAroundHorizontalAxis" << flipVertical
				<< "Input_Delay" << delay
				<< "Input" << input
				<< "}";
	}
	void read(const FileNode& node)                          //Read serialization for this class
	{
		node["BoardSize_Width" ] >> boardSize.width;
		node["BoardSize_Height"] >> boardSize.height;
		node["Calibrate_Pattern"] >> patternToUse;
		node["Square_Size"]  >> squareSize;
		node["Calibrate_NrOfFrameToUse"] >> nrFrames;
		node["Calibrate_FixAspectRatio"] >> aspectRatio;
		node["Write_DetectedFeaturePoints"] >> writePoints;
		node["Write_extrinsicParameters"] >> writeExtrinsics;
		node["Write_outputFileName"] >> outputFileName;
		node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
		node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
		node["Calibrate_UseFisheyeModel"] >> useFisheye;
		node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
		node["Show_UndistortedImage"] >> showUndistorsed;
		node["Input"] >> input;
		node["Input_Delay"] >> delay;
		node["Fix_K1"] >> fixK1;
		node["Fix_K2"] >> fixK2;
		node["Fix_K3"] >> fixK3;
		node["Fix_K4"] >> fixK4;
		node["Fix_K5"] >> fixK5;

		validate();
	}
	void validate()
	{
		goodInput = true;
		if (boardSize.width <= 0 || boardSize.height <= 0)
		{
			cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
			goodInput = false;
		}
		if (squareSize <= 10e-6)
		{
			cerr << "Invalid square size " << squareSize << endl;
			goodInput = false;
		}
		if (nrFrames <= 0)
		{
			cerr << "Invalid number of frames " << nrFrames << endl;
			goodInput = false;
		}

		if (input.empty())      // Check for valid input
			inputType = INVALID;
		else
		{
			if (input[0] >= '0' && input[0] <= '9')
			{
				stringstream ss(input);
				ss >> cameraID;
				inputType = CAMERA;
			}
			else
			{
				if (isListOfImages(input) && readStringList(input, imageList))
				{
					inputType = IMAGE_LIST;
					nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
				}
				else
					inputType = VIDEO_FILE;
			}
			if (inputType == CAMERA)
				inputCapture.open(cameraID);
			if (inputType == VIDEO_FILE)
				inputCapture.open(input);
			if (inputType != IMAGE_LIST && !inputCapture.isOpened())
				inputType = INVALID;
		}
		if (inputType == INVALID)
		{
			cerr << " Input does not exist: " << input;
			goodInput = false;
		}

		flag = 0;
		if(calibFixPrincipalPoint) flag |= CALIB_FIX_PRINCIPAL_POINT;
		if(calibZeroTangentDist)   flag |= CALIB_ZERO_TANGENT_DIST;
		if(aspectRatio)            flag |= CALIB_FIX_ASPECT_RATIO;
		if(fixK1)                  flag |= CALIB_FIX_K1;
		if(fixK2)                  flag |= CALIB_FIX_K2;
		if(fixK3)                  flag |= CALIB_FIX_K3;
		if(fixK4)                  flag |= CALIB_FIX_K4;
		if(fixK5)                  flag |= CALIB_FIX_K5;

		if (useFisheye) {
			// the fisheye model has its own enum, so overwrite the flags
			flag = fisheye::CALIB_FIX_SKEW | fisheye::CALIB_RECOMPUTE_EXTRINSIC;
			if(fixK1)                   flag |= fisheye::CALIB_FIX_K1;
			if(fixK2)                   flag |= fisheye::CALIB_FIX_K2;
			if(fixK3)                   flag |= fisheye::CALIB_FIX_K3;
			if(fixK4)                   flag |= fisheye::CALIB_FIX_K4;
			if (calibFixPrincipalPoint) flag |= fisheye::CALIB_FIX_PRINCIPAL_POINT;
		}

		calibrationPattern = NOT_EXISTING;
		if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
		if (!patternToUse.compare("CIRCLES_GRID")) calibrationPattern = CIRCLES_GRID;
		if (!patternToUse.compare("ASYMMETRIC_CIRCLES_GRID")) calibrationPattern = ASYMMETRIC_CIRCLES_GRID;
		if (!patternToUse.compare("RING_GRID")) calibrationPattern = RING_GRID;
		if (calibrationPattern == NOT_EXISTING)
		{
			cerr << " Camera calibration mode does not exist: " << patternToUse << endl;
			goodInput = false;
		}
		atImageList = 0;

	}
	Mat nextImage()
	{
		Mat result;
		if( inputCapture.isOpened() )
		{
			Mat view0;
			inputCapture >> view0;
			view0.copyTo(result);
		}
		else if( atImageList < imageList.size() )
			result = imread(imageList[atImageList++], IMREAD_COLOR);

		return result;
	}

	static bool readStringList( const string& filename, vector<string>& l )
	{
		l.clear();
		FileStorage fs(filename, FileStorage::READ);
		if( !fs.isOpened() )
			return false;
		FileNode n = fs.getFirstTopLevelNode();
		if( n.type() != FileNode::SEQ )
			return false;
		FileNodeIterator it = n.begin(), it_end = n.end();
		for( ; it != it_end; ++it )
			l.push_back((string)*it);
		return true;
	}

	static bool isListOfImages( const string& filename)
	{
		string s(filename);
		// Look for file extension
		if( s.find(".xml") == string::npos && s.find(".yaml") == string::npos && s.find(".yml") == string::npos )
			return false;
		else
			return true;
	}
public:
	Size boardSize;              // The size of the board -> Number of items by width and height
	Pattern calibrationPattern;  // One of the Chessboard, circles, or asymmetric circle pattern
	float squareSize;            // The size of a square in your defined unit (point, millimeter,etc).
	int nrFrames;                // The number of frames to use from the input for calibration
	float aspectRatio;           // The aspect ratio
	int delay;                   // In case of a video input
	bool writePoints;            // Write detected feature points
	bool writeExtrinsics;        // Write extrinsic parameters
	bool calibZeroTangentDist;   // Assume zero tangential distortion
	bool calibFixPrincipalPoint; // Fix the principal point at the center
	bool flipVertical;           // Flip the captured images around the horizontal axis
	string outputFileName;       // The name of the file where to write
	bool showUndistorsed;        // Show undistorted images after calibration
	string input;                // The input ->
	bool useFisheye;             // use fisheye camera model for calibration
	bool fixK1;                  // fix K1 distortion coefficient
	bool fixK2;                  // fix K2 distortion coefficient
	bool fixK3;                  // fix K3 distortion coefficient
	bool fixK4;                  // fix K4 distortion coefficient
	bool fixK5;                  // fix K5 distortion coefficient

	int cameraID;
	vector<string> imageList;
	size_t atImageList;
	VideoCapture inputCapture;
	InputType inputType;
	bool goodInput;
	int flag;

private:
	string patternToUse;


};

static inline void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
	if(node.empty())
		x = default_value;
	else
		x.read(node);
}

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

bool runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,
		vector<vector<Point2f> > imagePoints );

int main(int argc, char* argv[])
{
	//help();

	//! [file_read]
	Settings s;
	const string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
	FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
		return -1;
	}
	fs["Settings"] >> s;
	fs.release();                                         // close Settings file
	//! [file_read]

	//FileStorage fout("settings.yml", FileStorage::WRITE); // write config as YAML
	//fout << "Settings" << s;

	if (!s.goodInput)
	{
		cout << "Invalid input detected. Application stopping. " << endl;
		return -1;
	}

	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;
	int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
	clock_t prevTimestamp = 0;
	const Scalar RED(0,0,255), GREEN(0,255,0);
	const char ESC_KEY = 27;
	bool capturar = false;
	bool usoImagen = false;

	//! [get_input]
	for(;;)
	{
		Mat view;
		bool blinkOutput = false;

		view = s.nextImage();

		//-----  If no more image, or got enough, then stop calibration and show result -------------
		if( mode == CAPTURING && imagePoints.size() >= (size_t)s.nrFrames )
		{
			if( runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints))
				mode = CALIBRATED;
			else
				mode = DETECTION;
		}
		if(view.empty())          // If there are no more images stop the loop
		{
			// if calibration threshold was not reached yet, calibrate now
			if( mode != CALIBRATED && !imagePoints.empty() )
				runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints);
			break;
		}
		//! [get_input]

		imageSize = view.size();  // Format input image.
		if( s.flipVertical )    flip( view, view, 0 );

		//! [find_pattern]
		vector<Point2f> pointBuf;

		bool found;

		int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;

		if(!s.useFisheye) {
			// fast check erroneously fails with high distortions like fisheye
			chessBoardFlags |= CALIB_CB_FAST_CHECK;
		}

		switch( s.calibrationPattern ) // Find feature points on the input format
		{
		case Settings::CHESSBOARD:
			found = findChessboardCorners( view, s.boardSize, pointBuf, chessBoardFlags);
			break;
		case Settings::CIRCLES_GRID:
			found = findCirclesGrid( view, s.boardSize, pointBuf );
			break;
		case Settings::ASYMMETRIC_CIRCLES_GRID:
			found = findCirclesGrid( view, s.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID );
			break;
		case Settings::RING_GRID:
			found = findRingGrid( view, s.boardSize, pointBuf);
			//found = findCirclesGrid( view, s.boardSize, pointBuf);
			break;
		default:
			found = false;
			break;
		}



		//! [find_pattern]
		//! [pattern_found]
		if ( found )                // If done with success,
		{
			if(capturar)
			{
				string nombre = to_string((int)imagePoints.size());
				if( s.calibrationPattern == Settings::CHESSBOARD)
					nombre = "img/CHESSBOARD_r"  + nombre + ".jpg";
				else if( s.calibrationPattern == Settings::CIRCLES_GRID)
					nombre = "img/CIRCLES_GRID_r"  + nombre + ".jpg";
				else if( s.calibrationPattern == Settings::ASYMMETRIC_CIRCLES_GRID)
					nombre = "img/ASYMMETRIC_CIRCLES_GRID_r"  + nombre + ".jpg";
				else if( s.calibrationPattern == Settings::RING_GRID)
					nombre = "img/RING_GRID_r"  + nombre + ".jpg";
				else
					nombre = "img/Otros_r"  + nombre + ".jpg";
				imwrite(nombre,view);

				usoImagen = false;
				// improve the found corners' coordinate accuracy for chessboard
				if( s.calibrationPattern == Settings::CHESSBOARD)
				{
					Mat viewGray;
					cvtColor(view, viewGray, COLOR_BGR2GRAY);
					cornerSubPix( viewGray, pointBuf, Size(11,11),
							Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ));
				}

				if( mode == CAPTURING &&  // For camera only take new samples after delay time
						(!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) )
				{
					imagePoints.push_back(pointBuf);
					prevTimestamp = clock();
					blinkOutput = s.inputCapture.isOpened();
				}
				capturar = false;
				usoImagen = true;
			}
			// Draw the corners.
			drawChessboardCorners( view, s.boardSize, Mat(pointBuf), found );

		}
		//! [pattern_found]
		//----------------------------- Output Text ------------------------------------------------
		//! [output_text]
		string msg = (mode == CAPTURING) ? "100/100" :
				mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

		if( mode == CAPTURING )
		{
			if(s.showUndistorsed)
				msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames );
			else
				msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames );
		}

		putText( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);
		if(usoImagen)
		{
			string nombre = to_string((int)imagePoints.size());
			if( s.calibrationPattern == Settings::CHESSBOARD)
				nombre = "img/CHESSBOARD_"  + nombre + ".jpg";
			else if( s.calibrationPattern == Settings::CIRCLES_GRID)
				nombre = "img/CIRCLES_GRID_"  + nombre + ".jpg";
			else if( s.calibrationPattern == Settings::ASYMMETRIC_CIRCLES_GRID)
				nombre = "img/ASYMMETRIC_CIRCLES_GRID_"  + nombre + ".jpg";
			else if( s.calibrationPattern == Settings::RING_GRID)
				nombre = "img/RING_GRID_"  + nombre + ".jpg";
			else
				nombre = "img/Otros_"  + nombre + ".jpg";
			imwrite(nombre,view);
			usoImagen = false;
		}


		if( blinkOutput )
			bitwise_not(view, view);
		//! [output_text]
		//------------------------- Video capture  output  undistorted ------------------------------
		//! [output_undistorted]
		if( mode == CALIBRATED && s.showUndistorsed )
		{
			Mat temp = view.clone();
			if (s.useFisheye)
				cv::fisheye::undistortImage(temp, view, cameraMatrix, distCoeffs);
			else
				undistort(temp, view, cameraMatrix, distCoeffs);
		}
		//! [output_undistorted]
		//------------------------------ Show image and check for input commands -------------------
		//! [await_input]
		imshow("Image View", view);
		char key = (char)waitKey(s.inputCapture.isOpened() ? 1 : s.delay);

		if( key  == ESC_KEY || key=='q' )
			break;

		if( key == 'u' && mode == CALIBRATED )
			s.showUndistorsed = !s.showUndistorsed;

		if( s.inputCapture.isOpened() && key == 'g' )
		{
			mode = CAPTURING;
			imagePoints.clear();
		}
		if(key=='c')
		{
			capturar = true;
		}

		//! [await_input]
	}

	// -----------------------Show the undistorted image for the image list ------------------------
	//! [show_results]
	if( s.inputType == Settings::IMAGE_LIST && s.showUndistorsed )
	{
		Mat view, rview, map1, map2;

		if (s.useFisheye)
		{
			Mat newCamMat;
			fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,
					Matx33d::eye(), newCamMat, 1);
			fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), newCamMat, imageSize,
					CV_16SC2, map1, map2);
		}
		else
		{
			initUndistortRectifyMap(
					cameraMatrix, distCoeffs, Mat(),
					getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
					CV_16SC2, map1, map2);
		}

		for(size_t i = 0; i < s.imageList.size(); i++ )
		{
			view = imread(s.imageList[i], IMREAD_COLOR);
			if(view.empty())
				continue;
			remap(view, rview, map1, map2, INTER_LINEAR);
			imshow("Image View", rview);
			char c = (char)waitKey();
			if( c  == ESC_KEY || c == 'q' || c == 'Q' )
				break;
		}
	}
	//! [show_results]

	return 0;
}

//! [compute_errors]
static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
		const vector<vector<Point2f> >& imagePoints,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const Mat& cameraMatrix , const Mat& distCoeffs,
		vector<float>& perViewErrors, bool fisheye)
{
	vector<Point2f> imagePoints2;
	size_t totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for(size_t i = 0; i < objectPoints.size(); ++i )
	{
		if (fisheye)
		{
			fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix,
					distCoeffs);
		}
		else
		{
			projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		}
		err = norm(imagePoints[i], imagePoints2, NORM_L2);

		size_t n = objectPoints[i].size();
		perViewErrors[i] = (float) std::sqrt(err*err/n);
		totalErr        += err*err;
		totalPoints     += n;
	}

	return std::sqrt(totalErr/totalPoints);
}
//! [compute_errors]
//! [board_corners]
static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
		Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
	corners.clear();

	switch(patternType)
	{
	case Settings::CHESSBOARD:
	case Settings::RING_GRID:
	case Settings::CIRCLES_GRID:
		for( int i = 0; i < boardSize.height; ++i )
			for( int j = 0; j < boardSize.width; ++j )
				corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
		break;

	case Settings::ASYMMETRIC_CIRCLES_GRID:
		for( int i = 0; i < boardSize.height; i++ )
			for( int j = 0; j < boardSize.width; j++ )
				corners.push_back(Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));
		break;
	default:
		break;
	}
}
//! [board_corners]
static bool runCalibration( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
		vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
		vector<float>& reprojErrs,  double& totalAvgErr)
{
	//! [fixed_aspect]
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if( s.flag & CALIB_FIX_ASPECT_RATIO )
		cameraMatrix.at<double>(0,0) = s.aspectRatio;
	//! [fixed_aspect]
	if (s.useFisheye) {
		distCoeffs = Mat::zeros(4, 1, CV_64F);
	} else {
		distCoeffs = Mat::zeros(8, 1, CV_64F);
	}

	vector<vector<Point3f> > objectPoints(1);
	calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

	objectPoints.resize(imagePoints.size(),objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	double rms;

	if (s.useFisheye) {
		Mat _rvecs, _tvecs;
		rms = fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
				_tvecs, s.flag);

		rvecs.reserve(_rvecs.rows);
		tvecs.reserve(_tvecs.rows);
		for(int i = 0; i < int(objectPoints.size()); i++){
			rvecs.push_back(_rvecs.row(i));
			tvecs.push_back(_tvecs.row(i));
		}
	} else {
		rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
				s.flag);
	}

	cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
			distCoeffs, reprojErrs, s.useFisheye);

	return ok;
}

// Print camera parameters to the output file
static void saveCameraParams( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
		double totalAvgErr )
{
	FileStorage fs( s.outputFileName, FileStorage::WRITE );

	time_t tm;
	time( &tm );
	struct tm *t2 = localtime( &tm );
	char buf[1024];
	strftime( buf, sizeof(buf), "%c", t2 );

	fs << "calibration_time" << buf;

	if( !rvecs.empty() || !reprojErrs.empty() )
		fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << s.boardSize.width;
	fs << "board_height" << s.boardSize.height;
	fs << "square_size" << s.squareSize;

	if( s.flag & CALIB_FIX_ASPECT_RATIO )
		fs << "fix_aspect_ratio" << s.aspectRatio;

	if (s.flag)
	{
		std::stringstream flagsStringStream;
		if (s.useFisheye)
		{
			flagsStringStream << "flags:"
					<< (s.flag & fisheye::CALIB_FIX_SKEW ? " +fix_skew" : "")
					<< (s.flag & fisheye::CALIB_FIX_K1 ? " +fix_k1" : "")
					<< (s.flag & fisheye::CALIB_FIX_K2 ? " +fix_k2" : "")
					<< (s.flag & fisheye::CALIB_FIX_K3 ? " +fix_k3" : "")
					<< (s.flag & fisheye::CALIB_FIX_K4 ? " +fix_k4" : "")
					<< (s.flag & fisheye::CALIB_RECOMPUTE_EXTRINSIC ? " +recompute_extrinsic" : "");
		}
		else
		{
			flagsStringStream << "flags:"
					<< (s.flag & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "")
					<< (s.flag & CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "")
					<< (s.flag & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "")
					<< (s.flag & CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "")
					<< (s.flag & CALIB_FIX_K1 ? " +fix_k1" : "")
					<< (s.flag & CALIB_FIX_K2 ? " +fix_k2" : "")
					<< (s.flag & CALIB_FIX_K3 ? " +fix_k3" : "")
					<< (s.flag & CALIB_FIX_K4 ? " +fix_k4" : "")
					<< (s.flag & CALIB_FIX_K5 ? " +fix_k5" : "");
		}
		fs.writeComment(flagsStringStream.str());
	}

	fs << "flags" << s.flag;

	fs << "fisheye_model" << s.useFisheye;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (s.writeExtrinsics && !reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	if(s.writeExtrinsics && !rvecs.empty() && !tvecs.empty() )
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
		bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
		bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

		for( size_t i = 0; i < rvecs.size(); i++ )
		{
			Mat r = bigmat(Range(int(i), int(i+1)), Range(0,3));
			Mat t = bigmat(Range(int(i), int(i+1)), Range(3,6));

			if(needReshapeR)
				rvecs[i].reshape(1, 1).copyTo(r);
			else
			{
				//*.t() is MatExpr (not Mat) so we can use assignment operator
				CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
				r = rvecs[i].t();
			}

			if(needReshapeT)
				tvecs[i].reshape(1, 1).copyTo(t);
			else
			{
				CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
				t = tvecs[i].t();
			}
		}
		fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
		fs << "extrinsic_parameters" << bigmat;
	}

	if(s.writePoints && !imagePoints.empty() )
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for( size_t i = 0; i < imagePoints.size(); i++ )
		{
			Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}
}

//! [run_and_save]
bool runCalibrationAndSave(Settings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
		vector<vector<Point2f> > imagePoints)
{
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,
			totalAvgErr);
	cout << (ok ? "Calibration succeeded" : "Calibration failed")
        										 << ". avg re projection error = " << totalAvgErr << endl;

	if (ok)
		saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints,
				totalAvgErr);
	return ok;
}
