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

vector<Point2f>calcularPuntosHoja2(Mat view, vector<Point2f> pointBuf, vector<Point2f> dst_vertices, int offset)
																														{

	vector<Point2f> src_vertices;
	src_vertices.push_back( Point(pointBuf[0 ].x, pointBuf[0 ].y ) );
	src_vertices.push_back( Point(pointBuf[6 ].x, pointBuf[6 ].y ) );
	src_vertices.push_back( Point(pointBuf[28].x, pointBuf[28].y ) );
	src_vertices.push_back( Point(pointBuf[34].x, pointBuf[34].y ) );




	Mat H = findHomography(src_vertices, dst_vertices);
	//Matx33f H = getPerspectiveTransform(src_vertices, dst_vertices);

	cv::Mat rotated;
	warpPerspective(view, rotated, H, rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

	vector<Point2f> nuevos_vertices;
	int ancho = view.cols, alto = view.rows;
	offset = offset/2;
	nuevos_vertices.push_back( Point(0-offset, 0-offset) );
	nuevos_vertices.push_back( Point(ancho+offset, 0-offset) );
	nuevos_vertices.push_back( Point(0-offset, alto+offset) );
	nuevos_vertices.push_back( Point(ancho+offset, alto+offset) );


	perspectiveTransform( nuevos_vertices, nuevos_vertices, H.inv());


	//imshow("Fronto-Parallel-Corto", rotated);
	//waitKey(0);
	return nuevos_vertices;
																														}

void corregirChessboard(Mat imgE, Size size, vector<Point2f> &pointBufO, Mat &cameraMatrix, Mat &distCoeffs, int chessBoardFlags)
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
	float radioPeque = view.cols/(size.width-1);
	vector<Point2f> dst_vertices;
	dst_vertices.push_back( Point(0, 0) );
	dst_vertices.push_back( Point(view.cols, 0) );
	dst_vertices.push_back( Point(0, view.rows) );
	dst_vertices.push_back( Point(view.cols, view.rows) );

	vector<Point2f> src_vertices = calcularPuntosHoja2(view, pointBuf, dst_vertices, radioPeque*3);

	radioPeque = view.cols/(size.width+1);

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
		pointBufCopia.push_back(Point(pointBuf[i].x, pointBuf[i].y));
	}

	bool foundUnd = findChessboardCorners( rotated, size, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
	if(!foundUnd)
	{
		pointBuf.clear();
		for(uint i=0; i<pointBufCopia.size(); i++)
		{
			pointBuf.push_back(pointBufCopia[i]);
		}
		//cout<<"no Encontro"<<endl;
	}

	drawChessboardCorners( rotated, size, Mat(pointBuf), true );

	//--------Retornando los punto a la imagen sin distorcion-------
	perspectiveTransform( pointBuf, pointBuf, H.inv());
	drawChessboardCorners( view, size, Mat(pointBuf), true );


	imshow("Corregida", view);
	moveWindow("Corregida", 650,20);

	imshow("F-P", rotated);
	moveWindow("F-P", 1300,20);
	//waitKey(0);

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

	drawChessboardCorners( imgRegreso, size, Mat(pointBufO), true );

	imshow("imgRegreso", imgRegreso);
	moveWindow("imgRegreso", 650,500);
	//waitKey(0);


}



vector<Point2f>calcularPuntosHoja3(Mat view, vector<Point2f> pointBuf, vector<Point2f> dst_vertices, int offset)
																														{

	vector<Point2f> src_vertices;
	src_vertices.push_back( Point(pointBuf[40].x, pointBuf[40].y ) );
	src_vertices.push_back( Point(pointBuf[0 ].x, pointBuf[0 ].y ) );
	src_vertices.push_back( Point(pointBuf[43].x, pointBuf[43].y ) );
	src_vertices.push_back( Point(pointBuf[3 ].x, pointBuf[3 ].y ) );





	Mat H = findHomography(src_vertices, dst_vertices);
	//Matx33f H = getPerspectiveTransform(src_vertices, dst_vertices);

	cv::Mat rotated;
	warpPerspective(view, rotated, H, rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

	vector<Point2f> nuevos_vertices;
	int ancho = view.cols, alto = view.rows;

	nuevos_vertices.push_back( Point(0-offset, 0-offset) );
	nuevos_vertices.push_back( Point(ancho+offset, 0-offset) );
	nuevos_vertices.push_back( Point(0-offset, alto+offset*2) );
	nuevos_vertices.push_back( Point(ancho+offset, alto+offset*2) );


	perspectiveTransform( nuevos_vertices, nuevos_vertices, H.inv());


	//imshow("Fronto-Parallel-Corto", rotated);
	//waitKey(0);
	return nuevos_vertices;
																														}




void corregirCirclesGrid(Mat imgE, Size size, vector<Point2f> &pointBufO, Mat &cameraMatrix, Mat &distCoeffs)
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
		float radioPeque = view.cols/10;
		vector<Point2f> dst_vertices;
		dst_vertices.push_back( Point(0, 0) );
		dst_vertices.push_back( Point(view.cols, 0) );
		dst_vertices.push_back( Point(0, view.rows) );
		dst_vertices.push_back( Point(view.cols, view.rows) );

		vector<Point2f> src_vertices = calcularPuntosHoja3(view, pointBuf, dst_vertices, radioPeque);

		radioPeque = view.cols/12;

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
			pointBufCopia.push_back(Point(pointBuf[i].x, pointBuf[i].y));
		}

		bool foundUnd = findCirclesGrid( rotated, size, pointBuf, CALIB_CB_ASYMMETRIC_GRID );
		if(!foundUnd)
		{
			pointBuf.clear();
			for(uint i=0; i<pointBufCopia.size(); i++)
			{
				pointBuf.push_back(pointBufCopia[i]);
			}
			//cout<<"no Encontro"<<endl;
		}

		drawChessboardCorners( rotated, size, Mat(pointBuf), true );

		//--------Retornando los punto a la imagen sin distorcion-------
		perspectiveTransform( pointBuf, pointBuf, H.inv());
		drawChessboardCorners( view, size, Mat(pointBuf), true );


		imshow("Corregida", view);
		moveWindow("Corregida", 650,20);

		imshow("F-P", rotated);
		moveWindow("F-P", 1300,20);
		//waitKey(0);

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

		drawChessboardCorners( imgRegreso, size, Mat(pointBufO), true );

		imshow("imgRegreso", imgRegreso);
		moveWindow("imgRegreso", 650,500);
		//waitKey(0);


}
