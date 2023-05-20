#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <direct.h>


Mat_<uchar> convertToGrayscale(Mat_<Vec3b> img) {
	Mat_<uchar> grayScale(img.rows, img.cols);
	cvtColor(img, grayScale, cv::COLOR_BGR2GRAY);
	return grayScale;
}

int get_scale_percent(Mat_<Vec3b> img) {
	int scale_percent = 0; 
	int preferredHeight = 900; 
	int preferredWidth = 600; 
	int minDiffHeight = INT_MAX;
	int minDiffWidth = INT_MAX;
	for (int i = 1; i <= 100; i++) {
		int scaledWidth = img.cols * i / 100;
		int scaledHeight = img.rows * i / 100;
		if (abs(scaledWidth - preferredWidth) < minDiffWidth && abs(scaledHeight - preferredHeight) < minDiffHeight) {
			scale_percent = i;
 			minDiffHeight = abs(scaledHeight - preferredHeight);
 			minDiffWidth = abs(scaledWidth - preferredWidth);
 		}
 	}
 	return scale_percent;
}

std::vector<int> compute_horizontal_histogram(Mat_<uchar> img) {
	std::vector<int> pc(img.rows,0);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				pc.at(i)++;
			}
		}
	}
	return pc;
}

std::vector<int> compute_vertical_histogram(Mat_<uchar> img) {
	std::vector<int> pc(img.cols, 0);
	for (int j = 0; j < img.cols; j++) {
		for (int i = 0; i < img.rows; i++) {
			if (img(i, j) == 0) {
				pc.at(j)++;
			}
		}
	}
	return pc;
}

Mat_<uchar> vertical_histogram(Mat_<uchar> img) {
	std::vector<int> pc = compute_vertical_histogram(img);
	Mat_<uchar> dst(img.rows, img.cols, 255);
	for (int j = 0; j < dst.cols; j++) {
		for (int i = 0; i < pc.at(j); i++) {
			dst(i, j) = 0;
		}
	}
	return dst;
}

Mat_<uchar> horizontal_histogram(Mat_<uchar> img) {
	std::vector<int> pc = compute_horizontal_histogram(img);
	Mat_<uchar> dst(img.rows, img.cols, 255);
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < pc.at(i); j++) {
			dst(i, j) = 0;
		}
	}
	return dst;
}

Mat_<uchar> rotate_image(Mat_<uchar> img, double angle) {
	Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	Mat rotation_matrix = getRotationMatrix2D(center, angle, 1.0);
	Rect2f bbox = RotatedRect(Point2f(), img.size(), angle).boundingRect2f();
	rotation_matrix.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rotation_matrix.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

	Mat rotated_img;
	warpAffine(img,rotated_img,rotation_matrix,bbox.size(),1,0,255);
	return rotated_img;
}

Mat_<Vec3b> rotate_image(Mat_<Vec3b> img, double angle) {
	Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	Mat rotation_matrix = getRotationMatrix2D(center, angle, 1.0);
	Rect2f bbox = RotatedRect(Point2f(), img.size(), angle).boundingRect2f();
	rotation_matrix.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rotation_matrix.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

	Mat rotated_img;
	warpAffine(img,rotated_img,rotation_matrix,bbox.size(),1,0,Scalar(255,255,255));
	return rotated_img;
}

Mat_<uchar> crop_white_borders(Mat_<uchar> img) {
	int top = 0, left = 0, bot = 0, right = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) != 255) {
				top = i;
				i = img.rows;
				j = img.cols;
			}
		}
	}
	Mat ROI(img, Rect(0, top, img.cols, img.rows - top));
	Mat_<uchar> cropped;
	ROI.copyTo(cropped);
	for (int j = 0; j < cropped.cols; j++) {
		for (int i = 0; i < cropped.rows; i++) {
			if (cropped(i, j) != 255) {
				left = j;
				i = cropped.rows;
				j = cropped.cols;
			}
		}
	}
	ROI = Mat(cropped, Rect(left, 0, cropped.cols - left, cropped.rows));
	ROI.copyTo(cropped);
	for (int i = cropped.rows - 1; i >= 0; i--) {
		for (int j = cropped.cols - 1; j >= 0; j--) {
			if (cropped(i, j) != 255) {
				bot = i;
				i = 0;
				j = 0;
			}
		}
	}
	ROI = Mat(cropped, Rect(0, 0, cropped.cols, bot));
	ROI.copyTo(cropped);
	for (int j = cropped.cols - 1; j >= 0; j--) {
		for (int i = cropped.rows - 1; i >= 0; i--) {
			if (cropped(i, j) != 255) {
				right = j;
				i = 0;
				j = 0;
			}
		}
	}
	ROI = Mat(cropped, Rect(0, 0, right, cropped.rows));
	ROI.copyTo(cropped);
	return cropped;
}

Mat_<Vec3b> crop_white_borders(Mat_<Vec3b> img) {
	int top = 0, left = 0, bot = 0, right = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) != Vec3b(255, 255, 255)) {
				top = i;
				i = img.rows;
				j = img.cols;
			}
		}
	}
	Mat ROI(img, Rect(0, top, img.cols, img.rows - top));
	Mat_<Vec3b> cropped;
	ROI.copyTo(cropped);
	for (int j = 0; j < cropped.cols; j++) {
		for (int i = 0; i < cropped.rows; i++) {
			if (cropped(i, j) != Vec3b(255, 255, 255)) {
				left = j;
				i = cropped.rows;
				j = cropped.cols;
			}
		}
	}
	ROI = Mat(cropped, Rect(left, 0, cropped.cols - left, cropped.rows));
	ROI.copyTo(cropped);
	for (int i = cropped.rows - 1; i >= 0; i--) {
		for (int j = cropped.cols-1; j >= 0; j--) {
			if (cropped(i, j) != Vec3b(255, 255, 255)) {
				bot = i;
				i = 0;
				j = 0;
			}
		}
	}
	ROI = Mat(cropped, Rect(0, 0, cropped.cols, bot));
	ROI.copyTo(cropped);
	for (int j = cropped.cols - 1; j >= 0; j--) {
		for (int i = cropped.rows - 1; i >= 0; i--) {
			if (cropped(i, j) != Vec3b(255, 255, 255)) {
				right = j;
				i = 0;
				j = 0;
			}
		}
	}
	ROI = Mat(cropped, Rect(0, 0, right, cropped.rows));
	ROI.copyTo(cropped);
	return cropped;
}

int getLargestContourIndex(std::vector<std::vector<Point>> contours) {
	int maxArea = 0, maxIndex = 0;
	for (int i = 0; i < contours.size(); i++) {
		int area = contourArea(contours.at(i));
		if (area > maxArea) {
			maxArea = area;
			maxIndex = i;
		}
	}
	return maxIndex;
}

//counts the nr of values in the histogram that are equal to the height of the image
int get_nr_of_eq_to_height(std::vector<int> verticalHisto, int height) {
	int nr = 0;
	int compHeight = height * 40 / 100;
	for (int i = 0; i < verticalHisto.size(); i++) {
		if (abs(verticalHisto.at(i) - height) < compHeight) nr++;
	}
	return nr;
}

int get_nr_of_eq_to_width(std::vector<int> horizHisto, int width) {
	int nr = 0;
	int compWidth = width * 40 / 100;
	for (int i = 0; i < horizHisto.size(); i++) {
		if (abs(horizHisto.at(i) - width) < compWidth) nr++;
	}
	return nr;
}

double getRotateAngle(Mat_<Vec3b> img) {
	int scale_percent = get_scale_percent(img);
	int width = img.cols * scale_percent / 100;
	int height = img.rows * scale_percent / 100;
	Mat_<Vec3b> resized = img.clone();
	resize(resized, resized, Size(width, height));
	Mat_<uchar> gray = convertToGrayscale(resized);
	//imshow("gray", gray);
	Mat_<uchar> blackSheet = gray.clone();
	for (int i = 0; i < blackSheet.rows; i++) {
		for (int j = 0; j < blackSheet.cols; j++) {
			if (gray(i, j) != 255) {
				blackSheet(i,j) = 0;
			}
		}
	}
	//imshow("blackSheet", blackSheet);
	std::vector<int> horizontal = compute_horizontal_histogram(blackSheet);
	std::vector<int> vertical = compute_vertical_histogram(blackSheet);
	double angle=0;
	int maxVertEqual = -1;
	int maxHorizEqual = -1;
	for (double i = -90; i < 90; i++) {
		Mat_<uchar> rotatedImg = rotate_image(blackSheet, i); 
		//imshow("rotatedImg", rotatedImg); 
		rotatedImg = crop_white_borders(rotatedImg);
		horizontal = compute_horizontal_histogram(rotatedImg); 
		vertical = compute_vertical_histogram(rotatedImg);
		Mat_<uchar> horizHisto = horizontal_histogram(rotatedImg); 
		Mat_<uchar> verticalHisto = vertical_histogram(rotatedImg);
		/*imshow("horizHisto", horizHisto);  
		imshow("verticalHisto", verticalHisto);*/

		int nrVertEqual = get_nr_of_eq_to_height(vertical, rotatedImg.rows);
		int nrHorizEqual = get_nr_of_eq_to_width(horizontal, rotatedImg.cols);
		//std::cout << "vertEqual: " << nrVertEqual << ", horizEqual: " << nrHorizEqual << std::endl;
		if (nrHorizEqual > nrVertEqual && nrVertEqual >= maxVertEqual && nrHorizEqual >= maxHorizEqual) {
			angle = i;
			maxVertEqual = nrVertEqual;
			maxHorizEqual = nrHorizEqual;
			std::cout<<"angle: " << angle<< ", maxHorizEqual: " << maxHorizEqual << ", maxVertEqual: " << maxVertEqual << std::endl;
		}
		//waitKey();
	}
	std::cout << "Done :D" << std::endl;
	return angle;
}

Mat_<Vec3b> convert_borders_to_white(Mat_<Vec3b> img) {
	int scale_percent = get_scale_percent(img);
	int width = img.cols * scale_percent / 100;
	int height = img.rows * scale_percent / 100;
	Mat_<Vec3b> resized = img.clone();
	resize(resized, resized, Size(width, height));
	//convert img to grayscale
	Mat_<uchar> greyscale = convertToGrayscale(img);
	//median blur to enhange edges
	medianBlur(greyscale, greyscale, 3);

	//dilate and erode to remove noise
	dilate(greyscale, greyscale, Mat());
	erode(greyscale, greyscale, Mat());
	//create binary img using otsu
	threshold(greyscale,greyscale, 0, 255, THRESH_OTSU);
	medianBlur(greyscale, greyscale, 7);
	//create inverted img
	Mat_<uchar> inverted = greyscale.clone();
	bitwise_not(inverted, inverted);

	//floodfill corners to black to keep only the elements inside the piece of paper that were not removed by otsu
	for (int i = 0; i < inverted.rows; i++) {
		if (inverted(i,0) != 0) {
			floodFill(inverted, Point(0, i), 0);
		}
		if (inverted(i, inverted.cols - 1) != 0) {
			floodFill(inverted, Point(inverted.cols - 1, i), 0);
		}
	}
	for (int j = 0; j < inverted.cols; j++) {
		if (inverted(0, j) != 0) {
			floodFill(inverted, Point(j, 0), 0);
		}
		if (inverted(inverted.rows - 1, j) != 0) {
			floodFill(inverted, Point(j, inverted.rows - 1), 0);
		}
	}

	//bitwise or between the binary img and the inverted img to get the page mask
	bitwise_or(greyscale, inverted, greyscale);
	
	std::vector<std::vector<Point>> contours;
	findContours(greyscale, contours, RETR_TREE, CHAIN_APPROX_NONE);
	int largestContourIndex = getLargestContourIndex(contours);
	Mat_<uchar> blackPageWithContours = greyscale.clone();
	
	for (int i = 0; i < contours.size(); i++) {
		if (i != largestContourIndex) {
			fillPoly(blackPageWithContours, contours.at(i), Vec3b(0, 0, 0));
		}
	}

	Mat_<uchar> blackSheet;
	bitwise_not(blackPageWithContours, blackSheet);
	bool white = true;
	for (int i = 0; i < greyscale.rows; i++) {
		for (int j = 0; j < greyscale.cols; j++) {
			if (greyscale(i, j) != 255) white = false;
		}
	}
	//if the image already had white borders, the grayscale image will be all white,
	//so we need to get the page mask from the greyscale image by converting all non white pixels to black
	if (white) {
		blackSheet = convertToGrayscale(img);
		medianBlur(blackSheet,blackSheet,15);
		for (int i = 0; i < blackSheet.rows; i++) {
			for (int j = 0; j < blackSheet.cols; j++) {
				if (blackSheet(i, j) != 255)blackSheet(i, j) = 0;
			}
		}
	}

	Mat_<Vec3b> whiteBorders;
	Mat_<Vec3b> mask(img.rows, img.cols);
	cvtColor(blackSheet, mask, COLOR_GRAY2BGR);
	bitwise_or(img, mask, whiteBorders);
	return whiteBorders;
}

Mat_<Vec3b> make_page_vertical(Mat_<Vec3b> img) {
	Mat_<Vec3b> whiteBorders = convert_borders_to_white(img);
	whiteBorders = crop_white_borders(whiteBorders);

	float angle = getRotateAngle(whiteBorders);
	Mat_<Vec3b> rotatedImg = rotate_image(whiteBorders, angle);
	
	rotatedImg = crop_white_borders(rotatedImg);
	Mat_<Vec3b> resizedRotatedImg = rotatedImg.clone();
	return rotatedImg;
}

std::vector<Rect> detect_letters(Mat_<Vec3b> img) {
	std::vector<Rect> boundRect;
	Mat gray, sobel, element;
	Mat_<uchar> thresh;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	Sobel(gray, sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	imshow("sobel", sobel);
	threshold(sobel, thresh, 0, 255, THRESH_OTSU + THRESH_BINARY);
	imshow("thresh", thresh);
	element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(thresh, thresh, MORPH_CLOSE, element);
	imshow("thresh2", thresh);
	std::vector<std::vector<Point>> contours;
	findContours(thresh, contours, 0, 1);
	std::vector<std::vector<Point>> contours_poly(contours.size());
	int sum = 0, nr = 0;

	for (int i = 0; i < contours.size(); i++) {
		if (contours.at(i).size() > 0) {
			approxPolyDP(Mat(contours.at(i)), contours_poly.at(i), 2, true);
			Rect appRect(boundingRect(Mat(contours_poly.at(i))));
			if (appRect.width * 3 >= appRect.height && appRect.height > 10) {
				if (appRect.y > img.rows * 20 / 100) {
					boundRect.push_back(appRect);
					sum += appRect.height;
					nr++;
				}
			}
		}
	}
	float avgHeight = sum / ((float)nr);
	for (int i = 0; i < boundRect.size(); i++) {
		if (boundRect.at(i).y > img.rows * 15 / 100) {
			if (boundRect.at(i).height > avgHeight * 1.4) {
				Rect doubleRect = boundRect.at(i);
				for (int k = 0; k < thresh.cols; k++) {
					if (thresh(doubleRect.y + doubleRect.height / 2, k) == 255) {
						thresh(doubleRect.y + doubleRect.height / 2, k) = 0;
						thresh(doubleRect.y + doubleRect.height / 2 + 1, k) = 0;
						thresh(doubleRect.y + doubleRect.height / 2 - 1, k) = 0;
					}
				}
			}
		}
	}

	contours.clear();
	findContours(thresh, contours, 0, 1);
	boundRect.clear();
	contours_poly = std::vector<std::vector<Point>>(contours.size());
	for (int i = 0; i < contours.size(); i++) {  
		if (contours.at(i).size() > 0) {  
			approxPolyDP(Mat(contours.at(i)), contours_poly.at(i), 2, true);  
			Rect appRect(boundingRect(Mat(contours_poly.at(i))));  
			if (appRect.width * 3 >= appRect.height && appRect.height > 10 && appRect.x > 20) {  
				boundRect.push_back(appRect);   
				//UNCOMMENT THIS AND COMMENT LINE 432 IF YOU WANT HEADER REMOVED 
				/*if (appRect.y > img.rows * 20 / 100) {   
					boundRect.push_back(appRect);
				} */
			} 
		} 
	}
	return boundRect;
}

void extract_text(Mat_<Vec3b> img, std::vector<Rect> letterBoxes) {
	for (int i = 0; i < letterBoxes.size(); i++) {
		int x = letterBoxes.at(i).x - 5 >= 0 ? letterBoxes.at(i).x - 5 : 0;
		int y = letterBoxes.at(i).y - 5 >= 0 ? letterBoxes.at(i).y - 5 : 0;
		Mat ROI(img, Rect(x, y, letterBoxes.at(i).width+10, letterBoxes.at(i).height+10));
		Mat_<Vec3b> cropped;
		ROI.copyTo(cropped);
		std::string filename = "Extracted Text/text" + std::to_string(i) + ".jpg";
		imshow(filename, cropped);
		imwrite(filename, cropped); 
	}
}

Mat_<Vec3b> print_bounding_boxes(Mat_<Vec3b> img, std::vector<Rect> letterBoxes) {
	Mat_<Vec3b> newImg = img.clone();
	for (int i = 0; i < letterBoxes.size(); i++) {
		rectangle(newImg, letterBoxes.at(i), Vec3b(0, 255, 0), 2);
		std::cout <<"box x: " << letterBoxes.at(i).x << ", box y: " << letterBoxes.at(i).y << ", box width: " << letterBoxes.at(i).width << ", box height: " << letterBoxes.at(i).height << std::endl;
	}
	return newImg;
}

int main()
{
	Mat_<Vec3b> img = imread("Images/NoBackground/Problematice/scan11.jpg", IMREAD_COLOR);
	int scale_percent = get_scale_percent(img);
	int width = img.cols * scale_percent / 100;
	int height = img.rows * scale_percent / 100;
	Mat_<Vec3b> resized = img.clone();
	resize(resized, resized, Size(width, height));
	imshow("resized", resized);
	
	Mat_<Vec3b> rotatedImg = make_page_vertical(img); 
	Mat_<Vec3b> resizedRotatedImg = rotatedImg.clone();
	scale_percent = get_scale_percent(rotatedImg);
	width = rotatedImg.cols * scale_percent / 100;
	height = rotatedImg.rows * scale_percent / 100;
	resize(resizedRotatedImg, resizedRotatedImg, Size(width, height));
	imshow("resizedRotatedImg", resizedRotatedImg);
	std::vector<Rect> letterBoxes = detect_letters(rotatedImg);
	
	if (_mkdir("Extracted text") == 0) {
		std::cout << "Directory created" << std::endl;
	}
	else {
		std::cout << "Directory already created" << std::endl;
	}

	extract_text(rotatedImg, letterBoxes);

	Mat_<Vec3b> letterBoxesImg = print_bounding_boxes(rotatedImg, letterBoxes);
	Mat_<Vec3b> resizedLetterBoxes = letterBoxesImg.clone();
	resize(resizedLetterBoxes, resizedLetterBoxes, Size(width, height));
	imshow("resizedLetterBoxes", resizedLetterBoxes);
	waitKey();
	destroyAllWindows();
	return 0;
}