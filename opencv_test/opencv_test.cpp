#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>


using namespace cv;
using namespace std;

//int erosion_elem = 0;
//int erosion_size = 0;
//int dilation_elem = 0;
//int dilation_size = 0;
//int const max_elem = 2;
//int const max_kernel_size = 21;
//void Erosion(int, void*);
//void Dilation(int, void*);
//Mat src, erosion_dst, dilation_dst;
//
//
//int main()
//{
//	/*Mat image = Mat::zeros(300, 600, CV_8UC3);
//	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
//	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
//	imshow("Display Window", image);
//	waitKey(0);
//	return 0;*/
//    
//    //std::string image_path = samples::findFile("starry_night.jpg");
//    //Mat img = imread(image_path, IMREAD_COLOR);
//    //if (img.empty())
//    //{
//    //    std::cout << "Could not read the image: " << image_path << std::endl;
//    //    return 1;
//    //}
//    //imshow("Display window", img);
//    //int k = waitKey(0); // Wait for a keystroke in the window
//    //if (k == 's')
//    //{
//    //    imwrite("starry_night.png", img);
//    //}
//    //return 0;
//
//    // Lettura immagine
//    Mat image = imread("C:\\Users\\ArticaWareD\\Downloads\\castelli-casentino-big.jpg", IMREAD_COLOR);
//    src = imread("C:\\Users\\ArticaWareD\\Downloads\\castelli-casentino-big.jpg", IMREAD_COLOR);
//
//
//    // Controllo
//    if (image.empty())
//    {
//        cout << "Immagine vuota" << endl;
//        cin.get(); 
//        return -1;
//    }
//
//    //cout << "Dimensioni:" << endl;
//    //cout << "Colonne " << image.cols << endl;
//    //cout << "Righe " << image.rows << endl;
//    //cout << "Canali " << image.channels() << endl;
//
//
//    String windowName = "Test"; 
//    namedWindow(windowName); 
//    imshow(windowName, image); 
//    waitKey(0); 
//
//    //resizeWindow(windowName, 30, 30);
//    //moveWindow(windowName, 1000, 400);
//    //waitKey(0);
//
//    //Mat cropped_img = image.clone();
//    //cropped_img = cropped_img(Range(80, 280), Range(150, 330));
//    //imshow("Crop", cropped_img);
//
//    ////resizeWindow(windowName, 100, 100);
//    //waitKey(0);
//
//    //destroyWindow(windowName); 
//
//    /*Mat dst = image.clone();
//    blur(image, dst, Size(5, 5), Point(-1, -1));
//    String windowName1 = "Blur";
//    namedWindow(windowName1);
//    imshow(windowName1, dst);
//    waitKey(0);
//
//    Mat dst_2 = image.clone();
//    GaussianBlur(image, dst_2, Size(5, 5), 0, 0);
//    String windowName2 = "Gaussian";
//    namedWindow(windowName2);
//    imshow(windowName2, dst_2);
//    waitKey(0);
//
//    Mat dst_3 = image.clone();
//    medianBlur(image, dst_3, 5);
//    String windowName3 = "Median";
//    namedWindow(windowName3);
//    imshow(windowName3, dst_3);
//    waitKey(0);
//
//    Mat dst_4 = image.clone();
//    bilateralFilter(image, dst_4, 5, 5 * 2, 5 / 2);
//    String windowName4 = "Bilateral";
//    namedWindow(windowName4);
//    imshow(windowName4, dst_4);
//    waitKey(0);*/
//
//
//    namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
//    namedWindow("Dilation Demo", WINDOW_AUTOSIZE);
//    moveWindow("Dilation Demo", src.cols, 0);
//
//    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
//        &erosion_elem, max_elem,
//        Erosion);
//    createTrackbar("Kernel size:\n 2n +1", "Erosion Demo",
//        &erosion_size, max_kernel_size,
//        Erosion);
//    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
//        &dilation_elem, max_elem,
//        Dilation);
//    createTrackbar("Kernel size:\n 2n +1", "Dilation Demo",
//        &dilation_size, max_kernel_size,
//        Dilation);
//    Erosion(0, 0);
//    Dilation(0, 0);
//
//    waitKey(0);
//
//    return 0;
//}
//
//void Erosion(int, void*)
//{
//    int erosion_type = 0;
//    if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
//    else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
//    else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
//    Mat element = getStructuringElement(erosion_type,
//        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
//        Point(erosion_size, erosion_size));
//    erode(src, erosion_dst, element);
//    imshow("Erosion Demo", erosion_dst);
//}
//void Dilation(int, void*)
//{
//    int dilation_type = 0;
//    if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
//    else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
//    else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
//    Mat element = getStructuringElement(dilation_type,
//        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
//        Point(dilation_size, dilation_size));
//    dilate(src, dilation_dst, element);
//    imshow("Dilation Demo", dilation_dst);
//}

//Mat src, src_gray;
//Mat dst, detected_edges;
//int lowThreshold = 0;
//const int max_lowThreshold = 100;
//const int ratio2 = 3;
//const int kernel_size = 3;
//const char* window_name = "Edge Map";
//static void CannyThreshold(int, void*)
//{
//    blur(src_gray, detected_edges, Size(3, 3));
//    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio2, kernel_size);
//    dst = Scalar::all(0);
//    src.copyTo(dst, detected_edges);
//    imshow(window_name, dst);
//}
//int main(int argc, char** argv)
//{
//    CommandLineParser parser(argc, argv, "{@input | fruits.jpg | input image}");
//    src = imread("C:\\Users\\ArticaWareD\\Downloads\\castelli-casentino-big.jpg", IMREAD_COLOR);
//    if (src.empty())
//    {
//        std::cout << "Could not open or find the image!\n" << std::endl;
//        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
//        return -1;
//    }
//    dst.create(src.size(), src.type());
//    cvtColor(src, src_gray, COLOR_BGR2GRAY);
//    namedWindow(window_name, WINDOW_AUTOSIZE);
//    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
//    CannyThreshold(0, 0);
//    waitKey(0);
//    return 0;
//}

// Parametri
float Amin = 100;		// Minima area threshold
float Amax = 2000;		// Massima area threshold
float Cmin = 0.65;		// Circolarità threshold
float Smin = 120;		// Media saturazione threshold
float Vmin = 200;		// Media valore threshold
int expected_lights = 1;

// Calcolo la circolarità degli oggetti trovati
float circularity(std::vector <cv::Point>& object)
{
	float A = cv::contourArea(object);
	float p = cv::arcLength(object, true);
	return (4 * M_PI * A) / (p * p);
}

// Calcolo la media dell'intensità dei canali trovati nell'oggetto
float averageWithinContour(cv::Mat& img_channel, std::vector <cv::Point>& object, double dist = 0)
{
	cv::Rect bbox = cv::boundingRect(object);
	float sum = 0;
	int num_pixels = 0;
	for (int y = bbox.y; y < bbox.y + bbox.height; y++)
	{
		unsigned char* yThRow = img_channel.ptr<unsigned char>(y);
		for (int x = bbox.x; x < bbox.x + bbox.width; x++)
			if (cv::pointPolygonTest(object, cv::Point2f(x, y), true) >= dist)
			{
				sum += yThRow[x];
				num_pixels++;
			}
	}
	return sum / num_pixels;
}


int main()
{
	try
	{
		// Set dei parametri
		Amin = 100;		// Area threshold
		Amax = 2000;	// Massima area threshold
		Cmin = 0.65;	// Circolarità threshold
		Smin = 120;		// Media saturazione threshold
		Vmin = 180;		// Media valore threshold

		// Lettura immagine
	    Mat image = imread("C:\\Users\\ArticaWareD\\Downloads\\Semafori\\image.jpg", IMREAD_COLOR);
	
	    // Controllo
	    if (image.empty())
	    {
	        cout << "Immagine vuota" << endl;
	        cin.get(); 
	        return -1;
	    }

		// Originale
		imshow("Originale", image);
		waitKey(0);

		// Converto l'immagine a colori nello spazio dei colori HSV
		cv::Mat frame_HSV;
		cv::cvtColor(image, frame_HSV, cv::COLOR_BGR2HSV);
		std::vector <cv::Mat> hsv;
		cv::split(frame_HSV, hsv);

		// Visualizzo HSV
		imshow("HSV", frame_HSV);
		waitKey(0);

		// Binarizzo l'immagine
		cv::Mat frame_bin;
		cv::threshold(hsv[2], frame_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		// Visualizzo immagine binarizzata
		imshow("Binarize", frame_bin);
		waitKey(0);

		// Operazione morfologica di tipo Open
		cv::morphologyEx(frame_bin, frame_bin, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

		// Visualizzo l'operazione morfologica
		imshow("Morphology", frame_bin);
		waitKey(0);

		// Estraggo le componenti connesse (trovo gli oggetti)
		std::vector < std::vector <cv::Point> > objects;
		cv::findContours(frame_bin, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		// Disegno gli oggetti trovati su uno sfondo vuoto
		vector<Vec4i> hierarchy;
		RNG rng(12345);
		Mat drawing = Mat::zeros(frame_bin.size(), CV_8UC3);
		for (size_t i = 0; i < objects.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			drawContours(drawing, objects, (int)i, color, 2, LINE_8, hierarchy, 0);
		}
		
		// Visualizzo i contorni trovati
		imshow("Contours", drawing);
		waitKey(0);

		// Copio l'immagine in modo da disegnarci sopra
		cv::Mat processed_frame = image.clone();

		// Troviamo le luci del traffico
		int detected_lights = 0;
		for (int k = 0; k < objects.size(); k++)
		{
			float A = cv::contourArea(objects[k]);
			float C = circularity(objects[k]);

			// Primo criterio di decisione
			// basato sulla circonferenza minima e area minima
			if (C >= Cmin && A >= Amin && A <= Amax)
			{
				float avgS = averageWithinContour(hsv[1], objects[k]);
				float avgV = averageWithinContour(hsv[2], objects[k]);

				// Secondo criterio di decisione
				// basato sulla media della saturazione e sul valore nell'interno del contorno
				if (avgS > Smin && avgV > Vmin)
				{
					// Se ok allora abbiamo trovato la luce del semaforo
					detected_lights++;

					// Determino il colore del semaforo
					float avgH = averageWithinContour(hsv[0], objects[k], 3);

					float distance_to_yellow = std::abs(15 - avgH);
					float distance_to_green = std::abs(65 - avgH);
					float distance_to_red = std::min(std::abs(avgH), std::abs(120 - avgH));

					cv::Scalar light_color;
					if (distance_to_yellow < distance_to_green &&
						distance_to_yellow < distance_to_red)
						light_color = cv::Scalar(0, 255, 255);
					else if (distance_to_green < distance_to_yellow &&
						distance_to_green < distance_to_red)
						light_color = cv::Scalar(0, 255, 0);
					else
						light_color = cv::Scalar(0, 0, 255);

					cv::drawContours(processed_frame, objects, k, light_color, 4, cv::LINE_AA);


					cv::Rect bbox = cv::boundingRect(objects[k]);
				}
			}
		}	

		imshow("Immagine finale con semafori trovati", processed_frame);
		waitKey(0);

		return 1;
	}
	catch (Exception ex)
	{
		std::cout << ex.msg << endl;
	}
	
}