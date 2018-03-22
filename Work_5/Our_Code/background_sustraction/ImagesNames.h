#pragma once

// To load Images names from a path directory

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

