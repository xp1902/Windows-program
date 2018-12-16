#include "pch.h"
#include "FeatureMatch.h"


void FeatureMatch::FeatureObtatinAndMatchForAll(vector<cv::Mat>& images,
	vector<vector<cv::KeyPoint>>& key_points_for_all,
	vector<cv::Mat>& descriptor_for_all,
	vector<vector<cv::DMatch>>& matches_for_all) 
{
	SiftFeatureMatch(images, key_points_for_all, descriptor_for_all);
	//OrbFeatureMatch(images, key_points_for_all, descriptor_for_all, colors_for_all);
	matchFeatures(
		descriptor_for_all,
		matches_for_all,
		key_points_for_all
	);

}

void FeatureMatch::SiftFeatureMatch(
	vector<cv::Mat>& images,
	vector<vector<cv::KeyPoint>>& key_points_for_all,
	vector<cv::Mat>& descriptor_for_all
	//vector<vector<cv::Vec3b>>& colors_for_all
)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	cv::Mat image;

	//读取图像，获取图像特征点，并保存
	cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = images.begin(); it != images.end(); ++it)
	{
		cv::Mat img = *it;
		if (img.empty()) continue;

		vector<cv::KeyPoint> key_points;
		cv::Mat descriptor;
		//偶尔出现内存分配失败的错误
		sift->detectAndCompute(img, cv::noArray(), key_points, descriptor);

		//特征点过少，则排除该图像
		if (key_points.size() <= 10) continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);
	}
}


void FeatureMatch::OrbFeatureMatch(
	vector<cv::Mat>& images,
	vector<vector<cv::KeyPoint>>& key_points_for_all,
	vector<cv::Mat>& descriptor_for_all
)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();

	cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);//(500, 1.1f, 10,  31, 0,  2, ORB::HARRIS_SCORE, 31);orb参数设置研究中

	cv::Mat image;
	for (auto it = images.begin(); it != images.end(); ++it)
	{
		image = *it;
		if (image.empty()) continue;

		//cout << "Extracing features: " << *it << endl;

		vector<cv::KeyPoint> key_points;
		cv::Mat descriptor;

		//偶尔出现内存分配失败的错误
		orb->detectAndCompute(image, cv::Mat(), key_points, descriptor);

		//特征点过少，则排除该图像
		if (key_points.size() <= 10) continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);
	}
}

void 
FeatureMatch::matchFeatures
(
	cv::Mat& descriptors1,
	cv::Mat& descriptors2,
	vector<cv::DMatch>& matches)
{
	//Ptr<FeatureDetector> detector = ORB::create();
	//Ptr<DescriptorExtractor> descriptor = ORB::create();
	//Ptr<DescriptorMatcher> matcher_1 = DescriptorMatcher::create("BruteForce-Hamming");
	////检测 Oriented FAST 角点位置
	//vector<cv::KeyPoint> l_keyPoints;
	//vector<cv::KeyPoint> r_keyPoints;
	//detector->detect(query, l_keyPoints);
	//detector->detect(train, r_keyPoints);
	////根据角点位置计算 BRIEF 描述子
	//descriptor->compute(query, l_keyPoints, descriptors1);
	//descriptor->compute(query, r_keyPoints, descriptors2);
	////对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	//vector<cv::DMatch> matches_1;
	////BFMatcher matcher ( NORM_HAMMING );
	//matcher_1->match(descriptors1, descriptors2, matches_1);
	////匹配点对筛选
	//double min_dist = 1000, max_dist = 10;
	////找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
	//for (int i = 0; i < descriptors1.rows; i++)
	//{
	//	double dist = matches_1[i].distance;
	//	if (dist < min_dist) min_dist = dist;
	//	if (dist > max_dist) max_dist = dist;
	//}
	////筛选
	//for (int i = 0; i < descriptors1.rows; i++)
	//{
	//	if (matches_1[i].distance <= max(2 * min_dist, 20.0))
	//	{
	//		matches.push_back(matches_1[i]);
	//	}
	//}
	vector<vector<cv::DMatch>> knn_matches;
	cv::BFMatcher matcher(cv::NORM_L2);
	matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

	//获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;
		//保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
}

void 
FeatureMatch::refine
(
	vector<cv::DMatch>& matches,
	vector<cv::KeyPoint>& keypoints1,
	vector<cv::KeyPoint>& keypoints2
) {
	cv::Mat h, f;

	refineMatchesWithFundmentalMatrix(matches, f, keypoints1, keypoints2);
	//refineMatchesWithHomography(matches, 3.0, h, keypoints1, keypoints2);
	//vector<cv::Point2f> srcPoints(matches.size());
	//vector<cv::Point2f> dstPoints(matches.size());
	//for (size_t i = 0; i < matches.size(); i++) {
	//	srcPoints[i] = keypoints1[matches[i].queryIdx].pt;
	//	dstPoints[i] = keypoints2[matches[i].trainIdx].pt;
	//}

	//vector<uchar> inliersMask(srcPoints.size());
	//h = findHomography(srcPoints, dstPoints, CV_FM_RANSAC, 3.0f, inliersMask);
	//vector<cv::DMatch> inliers;
	//for (size_t i = 0; i < inliersMask.size(); i++) {
	//	if (inliersMask[i])
	//		inliers.push_back(matches[i]);
	//}
	//matches.swap(inliers);
	//----------------------------------------------------------------
	vector<cv::KeyPoint> srcPoints, dstPoints;
	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints.push_back(keypoints1[matches[i].queryIdx]);
		dstPoints.push_back(keypoints2[matches[i].trainIdx]);
	}

	//Keypoints to points
	//vector<cv::Point2f> ps1, ps2;
	//for (unsigned i = 0; i < srcPoints.size(); i++)
	//	ps1.push_back(srcPoints[i].pt);

	//for (unsigned i = 0; i < dstPoints.size(); i++)
	//	ps2.push_back(dstPoints[i].pt);

	////Compute fundmental matrix
	//vector<uchar> status;
	//f = findFundamentalMat(ps1, ps2, status, FM_RANSAC);

	////优化匹配结果
	////vector<cv::KeyPoint> leftInlier;
	////vector<cv::KeyPoint> rightInlier;
	//vector<cv::DMatch> inlierMatch;

	//int index = 0;
	//for (unsigned i = 0; i < matches.size(); i++) {
	//	if (status[i] != 0) {
	//		//leftInlier.push_back(alignedKps1[i]);
	//		//rightInlier.push_back(alignedKps2[i]);
	//		matches[i].trainIdx = index;
	//		matches[i].queryIdx = index;
	//		inlierMatch.push_back(matches[i]);
	//		index++;
	//	}
	//}
	////keypoints1 = leftInlier;
	////keypoints2 = rightInlier;
	//matches = inlierMatch;
}

void 
FeatureMatch::refineMatchesWithHomography
(
	vector<cv::DMatch>& matches, 
	double repro, 
	cv::Mat& homography,
	vector<cv::KeyPoint>& keypoints1,
	vector<cv::KeyPoint>& keypoints2
) {
	vector<cv::Point2f> srcPoints(matches.size());
	vector<cv::Point2f> dstPoints(matches.size());
	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = keypoints1[matches[i].trainIdx].pt;
		dstPoints[i] = keypoints2[matches[i].queryIdx].pt;
	}

	vector<uchar> inliersMask(srcPoints.size());
	homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC, repro, inliersMask);
	vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++) {
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);
}

void FeatureMatch::refineMatchesWithFundmentalMatrix(
	vector<cv::DMatch>& matches, 
	cv::Mat& F,
	vector<cv::KeyPoint>& keypoints1,
	vector<cv::KeyPoint>& keypoints2

) 
{
	//Align all points
	vector<cv::KeyPoint> srcPoints, dstPoints;
	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints.push_back(keypoints1[matches[i].queryIdx]);
		dstPoints.push_back(keypoints2[matches[i].trainIdx]);
	}

	//Keypoints to points
	vector<cv::Point2f> ps1, ps2;
	for (unsigned i = 0; i < srcPoints.size(); i++)
		ps1.push_back(srcPoints[i].pt);

	for (unsigned i = 0; i < dstPoints.size(); i++)
		ps2.push_back(dstPoints[i].pt);

	//Compute fundmental matrix
	vector<uchar> status;
	F = findFundamentalMat(ps1, ps2, status, cv::FM_RANSAC);

	//优化匹配结果
	//vector<cv::KeyPoint> leftInlier;
	//vector<cv::KeyPoint> rightInlier;
	vector<cv::DMatch> inlierMatch;

	int index = 0;
	for (unsigned i = 0; i < matches.size(); i++) {
		if (status[i] != 0) {
			//leftInlier.push_back(alignedKps1[i]);
			//rightInlier.push_back(alignedKps2[i]);
			matches[i].trainIdx = index;
			matches[i].queryIdx = index;
			inlierMatch.push_back(matches[i]);
			index++;
		}
	}
	//keypoints1 = leftInlier;
	//keypoints2 = rightInlier;
	matches = inlierMatch;
}


void//unused
FeatureMatch::matchFeatures
(
	cv::Mat& descriptors1, 
	cv::Mat& descriptors2, 
	vector<cv::KeyPoint>& keypoints1,
	vector<cv::KeyPoint>& keypoints2,
	vector<cv::DMatch>& matches
) {
	cv::flann::Index flannIndex(descriptors1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
	cv::Mat matchindex(descriptors2.rows, 2, CV_32SC1);
	cv::Mat matchdistance(descriptors2.rows, 2, CV_32FC1);
	flannIndex.knnSearch(descriptors2, matchindex, matchdistance, 2, cv::flann::SearchParams());
	for (int i = 0; i < matchdistance.rows; i++)
	{
		if (matchdistance.at<float>(i, 0) < 0.6*matchdistance.at<float>(i, 1))
		{
			cv::DMatch dmatches(matchindex.at<int>(i, 0), i, matchdistance.at<float>(i, 0));
			matches.push_back(dmatches);
		}
	}
}

void 
FeatureMatch::matchFeatures
(
	vector<cv::Mat>& descriptor_for_all,
	vector<vector<cv::DMatch>>& matches_for_all,
	vector<vector<cv::KeyPoint>>& key_points_for_all
) 
{
	matches_for_all.clear();
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;

		vector<cv::DMatch> matches;

		matchFeatures(
			descriptor_for_all[i], 
			descriptor_for_all[i + 1],
			matches
		);
		//储存n-1个matches结果
		//if(!matches.empty())
			//refine(matches, key_points_for_all[i], key_points_for_all[i + 1]);

		matches_for_all.push_back(matches);
	}
}