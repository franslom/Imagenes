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
}