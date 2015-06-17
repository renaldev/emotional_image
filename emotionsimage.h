#ifndef EMOTIONSIMAGE_H
#define EMOTIONSIMAGE_H

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

using namespace std;

class EmotionsImage
{
    cv::CascadeClassifier face_cascade;
    void load_face()
        {
            this->face_cascade.load( "/home/ubuntu/lf_serv/root/haarcascade_frontalface_alt000.xml" );
        }

public:
    EmotionsImage();

    double get_pleasure_feature(cv::Mat *image);
    double get_arousal_feature(cv::Mat *image);
    double get_dominance_feature(cv::Mat *image);
    double get_face_search(cv::Mat *image);
};

#endif // EMOTIONSIMAGE_H
