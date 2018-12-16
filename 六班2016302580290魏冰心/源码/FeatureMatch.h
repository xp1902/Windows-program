#pragma once
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <iostream>
#include <vector>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
//using namespace cv;

class FeatureMatch {

public:

	static void FeatureObtatinAndMatchForAll(vector<cv::Mat>& images, vector<vector<cv::KeyPoint>>& key_points_for_all, vector<cv::Mat>& descriptor_for_all, vector<vector<cv::DMatch>>& matches_for_all);

	static void SiftFeatureMatch(vector<cv::Mat>& images, vector<vector<cv::KeyPoint>>& key_points_for_all, vector<cv::Mat>& descriptor_for_all);

	static void OrbFeatureMatch(vector<cv::Mat>& images, vector<vector<cv::KeyPoint>>& key_points_for_all, vector<cv::Mat>& descriptor_for_all);

	static void matchFeatures(cv::Mat & descriptors1, cv::Mat & descriptors2, vector<cv::DMatch>& matches);

	static void refine(vector<cv::DMatch>& matched, vector<cv::KeyPoint>& keypoinst1, vector<cv::KeyPoint>& keypoints2);

	static void refineMatchesWithHomography(vector<cv::DMatch>& matches, double repro, cv::Mat & homography, vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2);

	static void refineMatchesWithFundmentalMatrix(vector<cv::DMatch>& matches, cv::Mat & F, vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2);

	static void matchFeatures(cv::Mat & descriptors1, cv::Mat & descriptors2, vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& matches);

	static void matchFeatures(vector<cv::Mat>& descriptor_for_all, vector<vector<cv::DMatch>>& matches_for_all, vector<vector<cv::KeyPoint>>& key_points_for_all);

};