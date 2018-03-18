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

vector<Point2f>calcularPuntosHoja(Mat view, vector<Point2f> pointBuf, vector<Point2f> dst_vertices, int offset)
																								{

	vector<Point2f> src_vertices;
	src_vertices.push_back( Point(pointBuf[15].x, pointBuf[15].y ) );
	src_vertices.push_back( Point(pointBuf[19].x, pointBuf[19].y ) );
	src_vertices.push_back( Point(pointBuf[0 ].x, pointBuf[0 ].y ) );
	src_vertices.push_back( Point(pointBuf[4 ].x, pointBuf[4 ].y ) );



	Mat H = findHomography(src_vertices, dst_vertices);
	//Matx33f H = getPerspectiveTransform(src_vertices, dst_vertices);

	cv::Mat rotated;
	warpPerspective(view, rotated, H, rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

	vector<Point2f> nuevos_vertices;
	int ancho = view.cols, alto = view.rows;
	nuevos_vertices.push_back( Point(0-offset, 0-offset) );
	nuevos_vertices.push_back( Point(ancho+offset, 0-offset) );
	nuevos_vertices.push_back( Point(0-offset, alto+offset) );
	nuevos_vertices.push_back( Point(ancho+offset, alto+offset) );


	perspectiveTransform( nuevos_vertices, nuevos_vertices, H.inv());


	//imshow("Fronto-Parallel-Corto", rotated);
	//waitKey(0);
	return nuevos_vertices;
																								}

bool reducirPuntos(vector<Point2f> pointBufAux, vector<Point2f> &pointBuf, float radioInt)
{
	float x, y;
	float xSum=0, ySum=0;
	int cant=0;
	for( uint i = 0; i< pointBuf.size(); i++ )
	{
		x = pointBuf[i].x;
		y = pointBuf[i].y;

		for( uint j = 0; j< pointBufAux.size(); j++ )
		{
			if( sqrt( pow(x-pointBufAux[j].x,2)+pow(y-pointBufAux[j].y,2)  ) < radioInt    )
			{
				xSum += pointBufAux[j].x;
				ySum += pointBufAux[j].y;
				cant++;
			}
		}
		xSum /= cant;
		ySum /= cant;
		cant = 0;
		pointBuf[i].x = xSum;
		pointBuf[i].y = ySum;
		xSum = 0;
		ySum = 0;
	}

	if(pointBuf.size()!=20)
		return false;
	return true;
}
bool findRingGrid2(Mat img, Size size, vector<Point2f> &pointBuf, float radioInterno)
{
	Mat imgFinal, imgGris, imgBlur, imgCanny, imgBinaria;
	imgFinal = img.clone();
	cvtColor( imgFinal, imgGris, CV_BGR2GRAY );
	blur( imgGris, imgBlur, Size(5,5) );
	threshold(imgBlur, imgBinaria,127,255,THRESH_BINARY+THRESH_OTSU);
	Canny( imgBinaria, imgCanny, 3, 100, 3 );

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours( imgBinaria, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Find the rotated rectangles and ellipses for each contour
	vector<RotatedRect> minRect( contours.size() );
	vector<RotatedRect> minEllipse, minEllipseReducidas;

	vector<Point2f> pointBufAux;
	for( uint i = 0; i < contours.size(); i++ )
	{
		minRect[i] = minAreaRect( Mat(contours[i]) );
		if( contours[i].size() > 5 && contours[i].size()<radioInterno*7 )
		{
			minEllipse.push_back(fitEllipse( Mat(contours[i]) ) );
		}
	}

	for( uint i = 0; i< minEllipse.size(); i++ )
	{
		Size2f tam = minEllipse[i].size;
		//que los posibles radios no sean tan grandes, ni tan pequeños
		if(tam.height<radioInterno*5 && tam.width<radioInterno*5 &&
				tam.height>5 && tam.width>5)
		{
			//que ancho i alto tengan realcion
			if(tam.height>tam.width && (tam.height/tam.width)<1.3)
				minEllipseReducidas.push_back(minEllipse[i]);
			else if( (tam.width/tam.height)<1.3)
				minEllipseReducidas.push_back(minEllipse[i]);
		}
	}

	if(minEllipseReducidas.size()>40)
		return false;

	for( uint i = 0; i< minEllipseReducidas.size(); i++ )
	{
		pointBufAux.push_back(Point2f(minEllipseReducidas[i].center.x, minEllipseReducidas[i].center.y));
		ellipse( imgFinal, minEllipseReducidas[i], Scalar( 0,255,0), 2, 8 );
	}


	if(!reducirPuntos(pointBufAux, pointBuf, radioInterno))
		return false;

	for( uint i = 0; i< pointBuf.size(); i++ )
	{
		circle( imgFinal, pointBuf[i], 2, Scalar( 0,255,0), 2, 8 );

		putText(imgFinal, to_string(i), pointBuf[i], FONT_HERSHEY_SCRIPT_COMPLEX, 2, CV_RGB(125,12,145), 2);

	}

	//cout<<pointBuf.size()<<endl;

	//imshow("adap", imgBinaria);
	//imshow("canny", imgCanny);
	//imshow("cir", imgFinal);

	if(pointBuf.size()==20)
		return true;
	return false;
}

void corregirAnillos(Mat imgE, Size size, vector<Point2f> &pointBufO, Mat &cameraMatrix, Mat &distCoeffs)
{
	Mat view = imgE.clone();
	vector<Point2f> pointBuf;
	for(uint i=0; i<pointBufO.size(); i++)
	{
		pointBuf.push_back(Point2f(pointBufO[i].x, pointBufO[i].y));

	}

	//-----------Aplicando undistort------------
	Mat temp = view.clone();
	undistort(temp, view, cameraMatrix, distCoeffs);
	undistortPoints(pointBuf, pointBuf, cameraMatrix, distCoeffs);
	for(uint i=0; i<pointBuf.size(); i++)
	{
		pointBuf[i].x = cameraMatrix.at<double>(0,2) + cameraMatrix.at<double>(0,0)*pointBuf[i].x;
		pointBuf[i].y = cameraMatrix.at<double>(1,2) + cameraMatrix.at<double>(1,1)*pointBuf[i].y;
		//circle(view, pointBuf[i], 3, Scalar(0,255,255), -1);
	}



	//---------Obteniendo Fronto-Parallel-------------
	float radioPeque = view.cols/24;
	vector<Point2f> dst_vertices;
	dst_vertices.push_back( Point(0, 0) );
	dst_vertices.push_back( Point(view.cols, 0) );
	dst_vertices.push_back( Point(0, view.rows) );
	dst_vertices.push_back( Point(view.cols, view.rows) );

	vector<Point2f> src_vertices = calcularPuntosHoja(view, pointBuf, dst_vertices, radioPeque*3);

	radioPeque = view.cols/30;

	cv::Mat rotated;
	Mat H = findHomography(src_vertices, dst_vertices);
	warpPerspective(view, rotated, H, rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

	perspectiveTransform( pointBuf, pointBuf, H);
	//for(uint i=0; i<pointBuf.size(); i++)
	{
		//circle(rotated, pointBuf[i], 3, Scalar(0,255,255), -1);
	}

	//----------Encontrando anillos en el fronto-parallel

	vector<Point2f> pointBufCopia;
	for(uint i=0; i<pointBuf.size(); i++)
	{
		pointBufCopia.push_back(pointBuf[i]);
	}

	bool foundUnd = findRingGrid2( rotated, Size(5,4), pointBuf, radioPeque);
	if(!foundUnd)
	{
		for(uint i=0; i<pointBufCopia.size(); i++)
		{
			pointBuf[i].x = pointBufCopia[i].x;
			pointBuf[i].y = pointBufCopia[i].y;
		}
	}
	drawChessboardCorners( rotated, Size(5,4), Mat(pointBuf), true );

	//--------Retornando los punto a la imagen sin distorcion-------
	perspectiveTransform( pointBuf, pointBuf, H.inv());
	drawChessboardCorners( view, Size(5,4), Mat(pointBuf), true );


	imshow("Corregida", view);
	moveWindow("Corregida", 650,20);

	imshow("F-P", rotated);
	moveWindow("F-P", 1300,20);

	//--------Regresando los puntos a la imagen con distorcion, osea a la capturada por la camara
	Mat imgRegreso = imgE.clone();
	float xDist, yDist, x, y;
	float k1 = distCoeffs.at<double>(0,0);
	float k2 = distCoeffs.at<double>(0,1);
	//float p1 = distCoeffs.at<double>(0,2);
	//float p2 = distCoeffs.at<double>(0,3);
	float k3 = distCoeffs.at<double>(0,4);
	float cx = cameraMatrix.at<double>(0,2);
	float cy = cameraMatrix.at<double>(1,2);
	float fx = cameraMatrix.at<double>(0,0);
	float fy = cameraMatrix.at<double>(1,1);
	float r;

	for(uint i=0; i<pointBuf.size(); i++)
	{

		x = (pointBuf[i].x-cx)/fx;
		y = (pointBuf[i].y-cy)/fy;
		r = sqrt(x*x+y*y);

		xDist = x*( 1 + k1*pow(r,2) + k2*pow(r,4) + k3*pow(r,6) );
		yDist = y*( 1 + k1*pow(r,2) + k2*pow(r,4) + k3*pow(r,6) );

		//xDist += x + (2*p1*y + p2*(pow(r,2) + 2*pow(x,2)) );
		//yDist += y + ( p1*(pow(r,2) + 2*pow(y,2))  +  2*p2*y);

		xDist = xDist * fx + cx;
		yDist = yDist * fy + cy;

		pointBuf[i].x = xDist;
		pointBuf[i].y = yDist;
	}

	//-----Promedieando los puntos anteriores, con el nuevo---
	for(uint i=0; i<pointBufO.size(); i++)
	{
		pointBufO[i].x = (pointBuf[i].x + pointBufO[i].x)/2;
		pointBufO[i].y = (pointBuf[i].y + pointBufO[i].y)/2;
	}

	drawChessboardCorners( imgRegreso, Size(5,4), Mat(pointBufO), true );

	imshow("imgRegreso", imgRegreso);
	moveWindow("imgRegreso", 650,500);



}
