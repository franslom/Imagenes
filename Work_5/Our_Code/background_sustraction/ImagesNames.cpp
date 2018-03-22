
#include "ImagesNames.h"


ImagesNames::ImagesNames() : pathDirectory(string(""))
{

}

ImagesNames::ImagesNames(string path): pathDirectory(path)
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
		std::cout <<"No Path Directory "<< std::endl;
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
