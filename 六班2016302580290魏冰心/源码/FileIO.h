#pragma once

#include <io.h>
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace std;
//using namespace cv;

class FilesIO {
private:
	
	vector<cv::Mat> images;
	bool getFilesName(const string FileDirectory, const string FileType, vector<string>&FilesName)
	{
		string buffer = FileDirectory + "\\*" + FileType;
		_finddata_t c_file;   // ����ļ����Ľṹ��
		intptr_t hFile;
		hFile = _findfirst(buffer.c_str(), &c_file);   //�ҵ�һ���ļ���
		if (hFile == -1L)   // ����ļ���Ŀ¼�´�����Ҫ���ҵ��ļ�
			cout << "No" << FileType << "files in current directory!\n" << endl;
		else
		{
			string fullFilePath;
			do
			{
				fullFilePath.clear();
				//����
				fullFilePath = FileDirectory + c_file.name;
				FilesName.push_back(fullFilePath);
			} while (_findnext(hFile, &c_file) == 0);  //����ҵ��¸��ļ������ֳɹ��Ļ��ͷ���0,���򷵻�-1  
			_findclose(hFile);
		}
		return FilesName.size() != 0 ? true : false;
	}
	void readImage() {
		for (size_t i = 0; i < FilesName.size(); i++)
		{
			cv::Mat temp = cv::imread(FilesName[i]);
			images.push_back(temp);
		}
	}
public:
	vector<string> FilesName;
	FilesIO() {
		cout << "lose parameters��FileDirectory, FileType��" << endl;
		return;
	}
	FilesIO(cv::String FilesDirectory, cv::String FileType)
	{
		if (getFilesName(FilesDirectory, FileType, FilesName))
		{
			readImage();
		}
		else
		{
			cout << "��ȡʧ�ܣ�" << endl;
		}
	}
	~FilesIO() {
		FilesName.shrink_to_fit();
		images.shrink_to_fit();
	}
	vector<cv::Mat> getImages() {
		return images;
	}
};
