#include "emotionsimage.h"

EmotionsImage::EmotionsImage()
{
    this->load_face();
}


double EmotionsImage::get_pleasure_feature(cv::Mat *image) {

    double pleasure = 0;
    cv::Mat HSV;
    cv::cvtColor(*image, HSV,CV_BGR2HSV);
    double sizeMatrix = HSV.rows * HSV.cols;

    cv::MatIterator_<cv::Vec3b> it = HSV.begin<cv::Vec3b>(), it_end = HSV.end<cv::Vec3b>();
    for(; it != it_end; ++it) {
        pleasure = pleasure + (0.69) * (*it)[2] + 0.22 * (*it)[1];
    };

    return pleasure/sizeMatrix;
}

double EmotionsImage::get_arousal_feature(cv::Mat *image) {
    double arousal = 0;
    cv::Mat HSV;
    cv::cvtColor(*image, HSV,CV_BGR2HSV);
    double sizeMatrix = HSV.rows * HSV.cols;

    cv::MatIterator_<cv::Vec3b> it = HSV.begin<cv::Vec3b>(), it_end = HSV.end<cv::Vec3b>();
    for(; it != it_end; ++it) {
        arousal = arousal + (-0.31) * (*it)[2] + 0.6 * (*it)[1];
    };

    return arousal/sizeMatrix;

}

double EmotionsImage::get_dominance_feature(cv::Mat *image) {
    double dominance = 0;
    cv::Mat HSV;
    cv::cvtColor(*image, HSV,CV_BGR2HSV);
    double sizeMatrix = HSV.rows * HSV.cols;

    cv::MatIterator_<cv::Vec3b> it = HSV.begin<cv::Vec3b>(), it_end = HSV.end<cv::Vec3b>();
    for(; it != it_end; ++it) {
        dominance = dominance + (0.76) * (*it)[2] + 0.32 * (*it)[1];
    };

    return dominance/sizeMatrix;

}

double EmotionsImage::get_face_search(cv::Mat *image) {

    vector<cv::Rect> faces;
    cv::Mat image_gray;
    cv::cvtColor( *image, image_gray, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( image_gray, image_gray );

    this->face_cascade.detectMultiScale(
                image_gray, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20) );

    double faceArea = 0;
    for(std::vector<cv::Rect>::iterator it = faces.begin(); it != faces.end(); ++it) {
        faceArea += (*it).height * (*it).width;
    }
    return faceArea;
}
