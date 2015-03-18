/*
 Eric Peterson & Christoph Leuze 2015
 This is our code to visualize brains on a video of our heads
 
 TODO ETP:
 1. Cascade detector improvement
    a. Filter the locations based on how they move around over time?
    b. Maybe also a spatial-temporal filter? Kalman of the head pose?
 3. The checkerboard undistortion crashes when it isn't detected in a frame, why?
 4. Test the image undistortion
 5. Use the checkerboard calibration rather than the head one. Why doesn't it look good though?
 6. Change to using the internal framesize rather than function inputs (and maybe put it in the constructor?)
 
 To improve matching:
 1. Initial pose with face detection
 2. Find matching points simliar to model registration by finding locations on the 3D data of the image markers from ORB detectors
    a. make sure the head pose is the correct one, not the static one
    b. verify that the head pose is valid in the first place.
    c. modify get_nearest_3D_point so it works with many points rather than just 2!
    d. bound the points to the face plus a buffer (this could later be the head plus a buffer if we change our method)
    e. crop the model down to as few verticies as possible, maybe even lower resolution of the surface or something?
    f. do a ratio test of the current matched detected points
    g. save the output points from the object interaction
    h. make sure the coordinates are valid for the 3D interaction code
    i. speed up the frame detection by computing the descriptors in the matching section when needed rather than dynamically
    j. display the matched detected points!
    k. make sure the points don't cross link to each other!
 3. Find the pose using matched points rather than face detection
    a. Could compare to the pose from face detection?
 4. Alternative functions that could be of help
    a. findHomography - this finds the transformation between two point sets and finds outliers. I would just need to filter out the background points initially somehow and match the points, at least between frames
    b. kmeans - clustering would be good, but I need to know the number of initial clusters! This could work by sending in the SURF descriptors I think...
    c. flann hierarchical clustering - I think this could be similar to kmeans except you don't need to know the number of clusters
    d. my own matching and rejection algorithm...
 5. Implement some kind of thread pool for proj_2D_to_3D
 
 
 
 I created the ply file with surfaces in paraview
 
 
 OPENCV 2D/3D pose with a mesh!
 http://docs.opencv.org/trunk/doc/tutorials/calib3d/real_time_pose/real_time_pose.html
 
 line and triangle intersection
 http://geomalgorithms.com/a07-_distance.html
 http://geomalgorithms.com/a06-_intersect-2.html
 http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
 
 The chessboard I used
 http://docs.opencv.org/_downloads/pattern.png
 
 We're using OpenCV, but PCL could be another option
 http://pointclouds.org/
 http://opencv.org/
 
 OpenCV documentation about matching and features
 http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
 
 */


#include <iostream>
#include <thread>
//#include <functional>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/core_c.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/objdetect/objdetect.hpp>

//local includes
#include "Mesh.h"

//#include "featureroi.h"

//using namespace cv;

/** Global variables */
//cv::String face_cascade_name = "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml";
//cv::String eyes_cascade_name = "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_big.xml";

int CALIBRATE_FACE=0, CALIBRATE_CHESSBOARD=1;

cv::Point2f pt(-1,-1);
cv::Point2f xyloc(-1,-1);
bool wasclick=false,wasrelease=true;



//function definitions
void bound_rect(cv::Rect &rect,cv::Point2i size); //downsize rects that exceed the frame size




bool rect_largest_to_smallest (const cv::Rect i,const cv::Rect j) { return (i.area()>j.area()); } //rect sorting helper function (largest to smallest)




void image_callback(int event, int x, int y, int flags, void* image)
{
    //cv::Point* xylocptr = (cv::Point*) xyloc;
    //cv::Mat* imageptr = (cv::Mat*) image;
    //cv::Point2i point(x,y);
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
        pt.x=x;
        pt.y=y;
        //xylocptr->x=x;
        //xylocptr->y=y;
    }
    else if  ( event == cv::EVENT_RBUTTONDOWN )
    {
        std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
    else if  ( event == cv::EVENT_MBUTTONDOWN )
    {
        std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
    else if ( event == cv::EVENT_MOUSEMOVE )
    {
        std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
        std::cout << "Original clock location (" << pt.x << ", " << pt.y << ")" << std::endl;
        xyloc.x=x;
        xyloc.y=y;
        //point.x=x;
        //point.y=y;
        //cv::line(*imageptr,pt,point,CV_RGB(255,255,255));
        
    }
    
}




//the squash court class
class vis3D {
protected:
    //points
    int min_cal_pts=4; //minimum points needed in a calibration frame
    int npoints=6; //number of points we know (4 in the face, 54 in the chessboard)
    cv::Point2i framesize;
    //cv::Point2i court_img_verts[22]; //all the court verticies
    std::vector< std::vector<cv::Point2f> > image_points_cal; //must be floats otherwise we get strange issues from
    std::vector<cv::Point2f> image_points;
    //cv::Point3f court_verts[22]; //the actual 3D locations of the court verticies
    std::vector< std::vector<cv::Point3f> > volume_points_cal; //all the court verticies in actual locations (filled in camera_calibration)
    //std::vector<cv::Point3f> volume_points = {cv::Point3f(125,50,190),cv::Point3f(125,25,160),cv::Point3f(100,60,130),cv::Point3f(155,60,130),cv::Point3f(55,160,140),cv::Point3f(90,160,140)}; //initialize the points in the volume
    std::vector<cv::Point3f> volume_points; // = {cv::Point3f(0.0,-118.0,-120.0),cv::Point3f(4.0,-130.0,-85.0),cv::Point3f(-25.0,-95.0,-50.0),cv::Point3f(30.0,-95.0,-50.0),cv::Point3f(-70.0,-5.0,-70.0),cv::Point3f(-70,0.0,-75.0)}; //points in the volume in mm
    std::vector<cv::Point3f> MRI_points = {cv::Point3f(0.0,-118.0,-120.0),cv::Point3f(4.0,-130.0,-85.0),cv::Point3f(-25.0,-95.0,-50.0),cv::Point3f(30.0,-95.0,-50.0),cv::Point3f(-70.0,-5.0,-70.0),cv::Point3f(-70,0.0,-75.0)}; //points in the volume in mm
    std::vector<cv::Point3f> chessboard_points; //chessboard points in mm
    //std::vector<cv::Point3f> court_verts;
    //let's define the back left as (0,0,0), and everything can be positive from there in meters
    //so x is across the court, y is front to back, and z is vertical
    
    //chessboard
    cv::Size2i board_size = cv::Size2i(9,6); //9x6 grid
    float square_size = 23.7; //mm IS THIS CORRECT?
    
    //classifier
    cv::CascadeClassifier face_cascade, eyes_cascade, mouth_cascade, nose_cascade, righteye_cascade, lefteye_cascade, profile_cascade, rightear_cascade, leftear_cascade;
    std::vector<cv::Rect> objects; //all objects detected
    
    
    const std::string cascade_names[9] = {"face_cascade", "eyes_cascade", "mouth_cascade", "nose_cascade", "righteye_cascade", "lefteye_cascade", "profile_cascade", "rightear_cascade", "leftear_cascade"};
    const std::string point_names[6] = {"mouth_cascade", "nose_cascade", "righteye_cascade", "lefteye_cascade", "rightear_cascade", "leftear_cascade"}; //are there any more cascades?


    
    //feature detection
    cv::ORB orbdet{5000,1.2f,8,11,0,2,cv::ORB::HARRIS_SCORE,11}; //only use the 100 best features
    cv::SURF surfdet{200,4,2,true,true}; //don't bother with the orientation of the features
    //std::vector<cv::KeyPoint> kp;
    std::vector<std::vector<cv::KeyPoint>> kp; //key points found by surf
    std::vector<std::vector<cv::Point3f>> pt3D; //points corresponding to the key points
    std::vector<std::vector<bool>> validpoint; //true if valid (for whatever reason) false if not
    //cv::Mat desc;
    std::vector<cv::Mat> desc;
    cv::BFMatcher bfmatch{cv::NORM_L1,false}; //brute force matcher
    std::vector<std::vector<std::vector<cv::DMatch>>> surfmatch;
    struct correspondence{ //a structure to hold information about matching locations in the video
        std::vector<cv::Point2f> image_point;
        std::vector<cv::Point3f> volume_point;
        std::vector<float> distance;
        std::vector<int> frameidx;
        std::vector<cv::KeyPoint> kp;
        std::vector<cv::Mat> desc;
    };
    
    //background detector
    cv::BackgroundSubtractorMOG2 mog2{180,50,true}; //(200,16,false)
    
    //camera section
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs_cal, tvecs_cal;
    cv::Mat rvecs, tvecs;
    cv::Mat rmat, tmat;
    double rms;
    
    //functions
    bool run_cascade_detector(cv::CascadeClassifier,cv::Mat &, std::vector<cv::Point2f> &, int,cv::Point2i); //a helper function for the cascade detection
    //bool run_cascade_detector(cv::CascadeClassifier classifier,cv::Mat &frame, std::vector<cv::Point2f> &pts, int loc,cv::Point2i offset{0,0});
    
    //definitions
    int CAMERA_CALIBRATION_FLAGS=CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_PRINCIPAL_POINT |  CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3;
    
public:
    void calibrate_camera(cv::Mat); //the main camera calibration function
    static void vis_call_back(int , int , int , int , void *); //mouse handler callback function
    void setup_camera_matrix(cv::Point2i); //camera calibration helper function
    void setup_rvec(); //camera calibration helper function
    void setup_tvec(); //camera calibration helper function
    void camera_calibration(cv::Point2i); //camera calibration helper function
    //void get_pose();
    void detect_anatomy(cv::Mat &, int); //detect anatomy using cascade detectors
    void render_image_points_cal(cv::Mat &,int); //draw image_points_cal on an image
    void render_image_points(cv::Mat &); //draw image_points on an image
    void render_objects(cv::Mat &,std::string); //draw all the detected objects on the given frame
    void project_points(); //fill image_points with locations calculated from the rotation, translation, camera matrix, and the object points
    void find_pose(); //find the rotation and translation vectors based on the object and image points
    void detect_chessboard(cv::Mat &frame, int framenum); //detect the chessboard features
    void undistort_frame(cv::Mat &frame); //undistort the output image
    void setup_calibration(int caltype); //set the variables correctly for calibrating with either a face or a chessboard
    void detect_keypoints(cv::Mat frame); //find the keypoints in the frame
    void render_detected(cv::Mat &frame); //draw the detected points on the frame
    void render_matched_detected(cv::Mat &frame); //draw the detected and matched points on the frames
    void match_keypoints(const Mesh &head); //filter the image keypoints and match them to the object
    bool backproject2DPoint(const Mesh *mesh, const cv::Point2f &point2d, cv::Point3f &point3d,cv::Mat *rmat, cv::Mat *tmat); //find if a point on the displayed image intersects the mesh at any point
    bool intersect_ray_triangle(Ray &Ray, Triangle &Triangle, double *out); //find if the ray interacts the triangle, if so return the position
    void proj_2D_to_3D(const Mesh &head, const int framenum,cv::Mat rmat, cv::Mat tmat); //project 2D points to the 3D surface
    void setframesize(cv::Point2i szin){framesize=szin;}; //sets the internal variable framesize
    
    
    vis3D();

    
};

vis3D::vis3D(){
    //I think this could also be moved up to the class definition to make it cleaner
    
    
    //camera section
    //cameraMatrix.eye(3, 3, CV_64F);
    //distCoeffs.eye(8, 1, CV_64F);
    //cameraMatrix=cv::Mat::eye
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F); //we need to tweak this later once the image size is known
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F); //only a 5 parameter fit
    rms=0;
    rvecs_cal.resize(1);
    rvecs_cal[0] = cv::Mat::zeros(3, 1, CV_64F); //Try initializing zeros
    tvecs_cal.resize(1);
    tvecs_cal[0] = cv::Mat::zeros(3, 1, CV_64F); //Try initializing zeros and fill in later
    rmat = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
    tmat = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix
    //std::vector<cv::Mat> rvecs, tvecs;
    
    //volume_points=&MRI_points;
    //image_points.resize(volume_points.size()); //allocate image_points
    
    //classifier (could put these names as variables in the class!)
    if( !face_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml" ) ){ std::cout<<"Error loading face classifier"<<std::endl; }
    if( !eyes_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_big.xml" ) ){ std::cout<<"Error loading eyes classifier"<<std::endl; }
    if( !mouth_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml" ) ){ std::cout<<"Error loading mouth classifier"<<std::endl; }
    if( !nose_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml" ) ){ std::cout<<"Error loading nose classifier"<<std::endl; }
    if( !righteye_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_righteye.xml" ) ){ std::cout<<"Error loading right eye classifier"<<std::endl; }
    if( !lefteye_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_lefteye.xml" ) ){ std::cout<<"Error loading left eye classifier"<<std::endl; }
    if( !profile_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_profileface.xml" ) ){ std::cout<<"Error loading profile classifier"<<std::endl; }
    if( !rightear_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_rightear.xml" ) ){ std::cout<<"Error loading profile classifier"<<std::endl; }
    if( !leftear_cascade.load( "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_leftear.xml" ) ){ std::cout<<"Error loading profile classifier"<<std::endl; }


    
    //create the chessboard points
    for( int i = 0; i < board_size.height; ++i ){
        for( int j = 0; j < board_size.width; ++j ){
            chessboard_points.push_back(cv::Point3f(static_cast<float>(j)*square_size , static_cast<float>(i)*square_size , 0.0));
        }
    }

}



//the call back for the squash court definition
void vis3D::vis_call_back(int event, int x, int y, int flags, void* image){
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        //std::cout << "CCB Left button of the mouse is down - position (" << x << ", " << y << ")" << std::endl;
        pt.x=x;
        pt.y=y;
        wasclick=true;
        wasrelease=false;
        //xylocptr->x=x;
        //xylocptr->y=y;
    }else if ( event == cv::EVENT_MOUSEMOVE ){
        //std::cout << "CCB Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
        //std::cout << "CCB Original clock location (" << pt.x << ", " << pt.y << ")" << std::endl;
        xyloc.x=x;
        xyloc.y=y;
        //point.x=x;
        //point.y=y;
        //cv::line(*imageptr,pt,point,CV_RGB(255,255,255));
    }else if ( event == cv::EVENT_LBUTTONUP){
        //std::cout << "CCB Left button of the mouse is up" << std::endl;
        wasrelease=true;
        wasclick=false; //this is set elsewhere, but we can do it again
    }
    
    //}else if ( event == cv::EVENT_LBUTTONDBLCLK){
    //    std::cout << "CCB Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;

    //}

}

//the get corners function
/*void vis3D::calibrate_camera(cv::Mat frame_raw){
    
    cv::Mat frame_corners;
    int keyresponse=-1;
    cv::Point2i p2i_zeros(0,0);
    int court_vert_idx=0;
    cv::Point2i framesize=frame_raw.size();
    float mindist=1.0e12,dprod=0.0;
    int minidx=0;
    
    //set up the camera matrix (it's kind of complicated)
    setup_camera_matrix(framesize);

    
    //set up the tvec
    setup_tvec();

    
    //set up the rvec
    setup_rvec();
    
    //I need to add a while loop and some other key processing to keep it in here!
    std::cout<<"Press b to begin setting points, s to set the points, and u to undo the previous point"<<std::endl;
    
    //we took a blind guess at the beginning, so let's just project the points!
    //cv::projectPoints(court_verts[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, court_img_verts[0]);
    
    
    
    // this section lets you define the court by dragging the verticies around, this probably should just replace the click based method above
    std::cout<<"Drag the corners you want to move and press c to recalibrate the camera"<<std::endl;
    keyresponse=-1;
    while(keyresponse!='q'){
        frame_raw.copyTo(frame_corners);
        keyresponse=cv::waitKey(1);
        if (keyresponse=='c'){ //if we want to calibrate
            camera_calibration(framesize);
            //cv::projectPoints(court_verts[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, court_img_verts[0]);
            std::cout<<"RMS error = "<<rms<<std::endl;
            std::cout<<"camera matrix"<<std::endl<<cameraMatrix<<std::endl;  //this doesn't seem to change
            std::cout<<"rotation vector"<<std::endl<<rvecs[0]<<std::endl;
            std::cout<<"translation vectors"<<std::endl<<tvecs[0]<<std::endl;
        }
        if (wasrelease==false) {
            //court_img_verts[0][minidx]=xyloc;
            //std::cout<<"setting points"<<std::endl;
        }
        imshow("frame callback", frame_corners);
    }
    cv::destroyWindow("frame callback"); //close the window now that we're done calibrating
}*/

void vis3D::camera_calibration(cv::Point2i framesize)  //should I pass by reference? currently the 2nd argument isn't used
//This function calibrates the camera using image_points_cal and volume_points_cal
{
    //replicate volume_points_cal as many times as image_points_cal. not ideal, I know
    //volume_points_cal=chessboard_points; //copy
    //volume_points=
    
    //volume_points_cal.resize(image_points_cal.size(), volume_points);
    //volume_points_cal.resize(image_points_cal.size());
    volume_points_cal.resize(image_points_cal.size(), volume_points);
    //crop each individual frame result to only the known matches
    
    std::cout<<"total frames "<<image_points_cal.size()<<std::endl;
    //std::vector<cv::Point2f>::iterator j;
    //std::vector<std::vector<cv::Point2f>>::iterator i;
    int i=0,j;
    //i=image_points_cal.begin();
    while(i<image_points_cal.size()){
        j=0;
        while(j<image_points_cal[i].size()){
            if (image_points_cal[i][j].x==-1 || image_points_cal[i][j].y==-1) { //if any part of it is -1, it's not matched from image to MR, so we have to get rid of it
                image_points_cal[i].erase(image_points_cal[i].begin()+j); //erase that element
                volume_points_cal[i].erase(volume_points_cal[i].begin()+j); //erase that element
            }else{
                j++;
            }
        }
        if (image_points_cal[i].size()<min_cal_pts) { //if we don't have enough elements in the frame (4), we have to get rid of the whole frame!
            image_points_cal.erase(image_points_cal.begin()+i); //erase the whole frame
            volume_points_cal.erase(volume_points_cal.begin()+i); //erase the whole frame
        }else{
            i++;
        }
    }
    std::cout<<"valid frames "<<image_points_cal.size()<<std::endl;
    /*for (int i=0; i<volume_points_cal.size(); i++) { //trying to figure out why I get an error when frames are dropped with the chessboard
    std::cout<<"volume_points_cal["<<i<<"][1] "<<volume_points_cal[i][1]<<std::endl;
    std::cout<<"image_points_cal["<<i<<"][1] "<<image_points_cal[i][1]<<std::endl;
    }*/
    if (image_points_cal.size()>0) {
        rms=cv::calibrateCamera(volume_points_cal, image_points_cal, framesize, cameraMatrix, distCoeffs, rvecs_cal, tvecs_cal, CAMERA_CALIBRATION_FLAGS);
        std::cout<<"RMS error = "<<rms<<std::endl;
        std::cout<<"camera matrix"<<std::endl<<cameraMatrix<<std::endl;  //this doesn't seem to change
        std::cout<<"distortion coefficients"<<std::endl<<distCoeffs<<std::endl;
        for (int i=0; i<rvecs_cal.size(); i++) {
            std::cout<<"rotation vector "<<i<<std::endl<<rvecs_cal[i]<<std::endl;
            std::cout<<"translation vector "<<i<<std::endl<<tvecs_cal[i]<<std::endl;
        }
        //std::cout<<"rotation vector"<<std::endl<<rvecs_cal[0]<<std::endl;
        //std::cout<<"translation vectors"<<std::endl<<tvecs_cal[0]<<std::endl;
    }else{
        std::cout<<"No valid frames in the calibration set! Not calibrating!"<<std::endl;
        
    }
    //CV_CALIB_FIX_ASPECT_RATIO | //we shouldn't be fixing the aspect ratio, it distorted the result
    //fix principal point could be removed, especially when finding the distortion matrix
    //note that if we let the distortions correct then the camera matrix also gets a little crazy
    //I think we can fairly safely assume that the camera is basically distortionless
    //all the above arguments to calibrateCamera mean that it's essentially not optimizing the distortion matrix, only the camera matrix, rotation and translation vectors
    
    //and lets clear all the points we used so that we can recalibrate later
    volume_points_cal.clear();
    image_points_cal.clear();
    rvecs_cal.clear();
    tvecs_cal.clear();
}

void vis3D::setup_camera_matrix(cv::Point2i framesize) //should I pass by reference?
{
    double f = 0.5; // focal length in mm (0.25 seems about right?)
    double sx = 2.4, sy = 1.8;             // sensor size in mm (full frame is 36x24, but this one is 1/4"?)
    std::cout<<"Setting up the camera matrix vector"<<std::endl;
    cameraMatrix.at<double>(0,2)=static_cast<double>(framesize.x)/2.0; //finish setting the cameraMatrix because calibrateCamera can't do it for me
    cameraMatrix.at<double>(1,2)=static_cast<double>(framesize.y)/2.0;  //finish setting cameraMatrix
    cameraMatrix.at<double>(0,0)=static_cast<double>(framesize.x)*f/sx; //fx
    cameraMatrix.at<double>(1,1)=static_cast<double>(framesize.y)*f/sy; //fy
    std::cout<<"camera matrix"<<std::endl<<cameraMatrix<<std::endl;  //this doesn't seem to change
    /*camera matrix (from the chessboard)
     [1007.119462405858, 0, 639.5;
     0, 1017.599265807175, 359.5;
     0, 0, 1]
     
     distortion coefficients (from the chessboard)
     [-0.1180841255846889;
     0.4235977079784225;
     0;
     0;
     -0.5166247165838871]*/
}

void vis3D::setup_tvec()
{
    //set up the tvec
    std::cout<<"Setting up the translation vector"<<std::endl;
    //think of these translations as happening to the court, not the camera!
    tvecs_cal[0].at<double>(0)=0.0; //3cm in x (centered)
    tvecs_cal[0].at<double>(1)=115.0; //4cm down (maybe)
    tvecs_cal[0].at<double>(2)=-45.0; //16.5cm back (maybe)
    std::cout<<"translation vectors"<<std::endl<<tvecs_cal[0]<<std::endl;
    /* this is an example tvec, which isn't too different than the input, but still isn't correct
     [-3.181589803420535, 2.668127841422967, 13.74342575277559]*/
    
}

void vis3D::setup_rvec()
{
    //set up the rvec
    std::cout<<"Setting up the rotation vector"<<std::endl;
    rvecs_cal[0].at<double>(0)=0.25; //not sure what this is
    rvecs_cal[0].at<double>(1)=2.0; //not sure what this is
    rvecs_cal[0].at<double>(2)=2.0; //not sure what this is
    std::cout<<"rotation vector"<<std::endl<<rvecs_cal[0]<<std::endl;
    /* This is based off of an output rvec which put the court just about in the middle of the screen
     [1.818635848296208, -0.008306380664320337, 0.005670786435876811]*/
}

/*void vis3D::get_pose()
{
    //Eric I need to find rvec and tvec
}*/

void vis3D::detect_anatomy(cv::Mat &frame, int framenum)
//This function uses cascade filters to detect faces and other facial structures
//It limits the detection region to the face area and saves the largest rectangle
{
    bool success;
    
    //this probably doesn't need to exist, we could just add to the size
    std::vector<cv::Point2f> *pts; //assign it to the correct vector below!
    
    if(framenum<0){ //if we're not calibrating!
        //volume_points=MRI_points;
        image_points.resize(volume_points.size());
        pts=&image_points;
    }else{ //if we're calibrating
        if(image_points_cal.size()<=framenum){ //if it's too small, we need to allocate more!
            image_points_cal.resize(framenum+1);
            image_points_cal[framenum].resize(npoints,cv::Point2f(-1,-1));
        }else{ //if it's already allocated, just empty it of previous points
            image_points_cal[framenum].assign(npoints, cv::Point2f(-1,-1));
        }
        pts=&image_points_cal[framenum];
    }
    //std::vector<cv::Point2f> *pts=&image_points_cal[framenum];
    /*std::vector<cv::Rect> faces;
    std::vector<cv::Rect> rightear;
    std::vector<cv::Rect> leftear;
    std::vector<cv::Rect> righteye;
    std::vector<cv::Rect> lefteye;
    std::vector<cv::Rect> mouth;
    std::vector<cv::Rect> nose;
    std::vector<cv::Rect> profile;*/
    
    cv::Mat frame_tmp, frame_gray; //ETP not sure if I need tmp or if equalizeHist can take the same input in both spots
    
    cvtColor( frame, frame_tmp, CV_BGR2GRAY );
    equalizeHist( frame_tmp, frame_gray );
    
    //"face_cascade", "eyes_cascade", "mouth_cascade", "nose_cascade", "righteye_cascade", "lefteye_cascade", "profile_cascade", "rightear_cascade", "leftear_cascade"}

    //something like this
    //what I need to do is to give run_cascade_detector the indicies for image_points_cal for each call and fill it with {-1,-1} here before
    success=run_cascade_detector(face_cascade,frame_gray,*pts,-1,cv::Point2i(0,0)); //detect the face
    if(framenum>=0){render_objects(frame, "face");}
    
    /*if(!success){  //once in profile, the other cascades don't do well
        success=run_cascade_detector(profile_cascade,frame,*pts,-1,cv::Point2i(0,0));
        render_objects(frame, "profile");
    }*/
    /*success=run_cascade_detector(rightear_cascade,frame_gray,*pts,-1,cv::Point2i(0,0));
    render_objects(frame, "right ear");
    success=run_cascade_detector(leftear_cascade,frame_gray,*pts,-1,cv::Point2i(0,0));
    render_objects(frame, "left ear");*/

    
    //image_points_cal.end() is supposed to return a vector to the end, or I need to figure this out
    if(success){ //if we're looking at the face
        //below we crop the face into 4 parts, the left and right eyes, and the nose and mouth for a more accurate detection
        bound_rect(objects[0], frame_gray.size()); //this needs to be inside the success check in case a face isn't found
        cv::Rect left_eye_obj(objects[0]), right_eye_obj(objects[0]), nose_obj(objects[0]), mouth_obj(objects[0]),left_ear_obj(objects[0]),right_ear_obj(objects[0]);
        left_eye_obj.height/=2.0; //from top to halfway down and the left side
        left_eye_obj.width/=2.0;
        left_eye_obj.x+=left_eye_obj.width;
        //bound_rect(left_eye_obj, frame_gray.size()); //no need to bind it because we bound the face
        right_eye_obj.height/=2.0; //we're already at the top right of the face, so just the top quadrant
        right_eye_obj.width/=2.0;
        //bound_rect(right_eye_obj, frame_gray.size()); //no need to bind it because we bound the face
        nose_obj.y+=nose_obj.height/2.0;  //from halfway down to 3/4 of the way down
        nose_obj.height/=4.0;
        //bound_rect(nose_obj, frame_gray.size()); //no need to bind it because we bound the face
        //std::cout<<" mouth "<<mouth_obj<<std::endl;
        mouth_obj.y+=mouth_obj.height*3.0/4.0;  //from 3/4 of the way down to all the way to the bottom of the face
        mouth_obj.height/=4.0;
        //bound_rect(mouth_obj, frame_gray.size()); //no need to bind it because we bound the face
        //std::cout<<" mouth "<<mouth_obj<<std::endl;
        left_ear_obj.x+=left_ear_obj.width/2; //move it over halfway to cover half the face and off the left side
        bound_rect(left_ear_obj, frame_gray.size());
        right_ear_obj.x-=right_ear_obj.width/2; //move it over halfway to cover half the face and off the right side
        bound_rect(right_ear_obj, frame_gray.size());
        
        cv::Mat frame_left_eye = frame_gray(left_eye_obj); //constrain the search to the face region only.
        cv::Mat frame_right_eye = frame_gray(right_eye_obj); //constrain the search to the face region only.
        cv::Mat frame_nose = frame_gray(nose_obj); //constrain the search to the face region only.
        cv::Mat frame_mouth = frame_gray(mouth_obj); //constrain the search to the face region only.
        cv::Mat frame_left_ear = frame_gray(left_ear_obj); //constrain the search to the left half of the face and off the side
        cv::Mat frame_right_ear = frame_gray(right_ear_obj); //constrain the search to the right half of the face and off the side
        //done with separating the face into its parts
        
        //std::cout<<"We're looking at a face!"<<std::endl;
        //success=run_cascade_detector(eyes_cascade,frame,pts);
        success=run_cascade_detector(mouth_cascade,frame_mouth,*pts,0,mouth_obj.tl());
        if(framenum>=0){render_objects(frame, "mouth");}
        success=run_cascade_detector(nose_cascade,frame_nose,*pts,1,nose_obj.tl());
        if(framenum>=0){render_objects(frame, "nose");}
        success=run_cascade_detector(righteye_cascade,frame_right_eye,*pts,2,right_eye_obj.tl());
        if(framenum>=0){render_objects(frame, "right eye");}
        success=run_cascade_detector(lefteye_cascade,frame_left_eye,*pts,3,left_eye_obj.tl());
        if(framenum>=0){render_objects(frame, "left eye");}
        success=run_cascade_detector(rightear_cascade,frame_right_ear,*pts,4,right_ear_obj.tl());
        if(framenum>=0){render_objects(frame, "right ear");}
        success=run_cascade_detector(leftear_cascade,frame_left_ear,*pts,5,left_ear_obj.tl());
        if(framenum>=0){render_objects(frame, "left ear");}
    }else{ //if we don't find a face (this really doesn't do anything anymore because it doesn't give us enough points to calibrate)
        //note that this is not well done, we could split it up like we did with the face above
        success=run_cascade_detector(profile_cascade,frame,*pts,-1,cv::Point2i(0,0));
        if(framenum>=0){render_objects(frame, "profile");}
        if(success){ //if we found a profile
            cv::Rect face_obj(objects[0]);
            cv::Mat frame_face = frame(objects[0]); //constrain the search to the face only
            //std::cout<<"We're looking at a profile!"<<std::endl;
            success=run_cascade_detector(rightear_cascade,frame_face,*pts,4,face_obj.tl());
            if(framenum>=0){render_objects(frame, "right ear");}
            if(!success){ //if we didn't find a right ear
                success=run_cascade_detector(leftear_cascade,frame_face,*pts,5,face_obj.tl());
                if(framenum>=0){render_objects(frame, "left ear");}
            }
        }
        
    }

    
}

bool vis3D::run_cascade_detector(cv::CascadeClassifier classifier,cv::Mat &frame, std::vector<cv::Point2f> &pts, int loc,cv::Point2i offset) //how to do default args?
{
    //std::vector<cv::Rect> objects;
    //objects.clear(); //I don't think this is required, but it may be!
    classifier.detectMultiScale( frame, objects, 1.1, 4, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    //std::cout<<"number of objects found "<<objects.size()<<std::endl;
    if(!objects.empty()){ //if we found a face! now we need to look for
        cv::groupRectangles(objects, 0, 0.2); //0 means minimum of 1
        //std::cout<<"offset "<<offset<<" objects[0] "<<objects[0]<<std::endl;
        std::sort(objects.begin(),objects.end(),rect_largest_to_smallest); //sort the outputs by size
        for(int i=0;i<objects.size();i++){objects[i]+=offset;} //add obbset to all objects
        //std::cout<<"objects ";
        //for(int i=0;i<objects.size();i++){std::cout<<objects[i].area()<<" ";}
        //std::cout<<std::endl;
        //std::cout<<"objects[0] "<<objects[0]<<std::endl;
        //objects[0].x+=offset.x;
        //objects[0].y+=offset.y;
        //std::cout<<"number of objects grouped "<<objects.size()<<std::endl;
        if (loc!=-1) { //if we have a location
            pts[loc]=cv::Point2f(objects[0].x+objects[0].width*0.5,objects[0].y+objects[0].height*0.5);
            //std::cout<<"object location "<<pts[loc]<<std::endl;
        }
        return true;
    }
    return false;

}

void vis3D::render_image_points_cal(cv::Mat &frame,int loc)
{
    for (int i=0; i<image_points_cal[loc].size(); i++) {
        cv::circle(frame, image_points_cal[loc][i], 2, cv::Scalar(0,255,0),-2);
        cv::putText(frame, point_names[i], image_points_cal[loc][i],CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0));
    }
}

void vis3D::render_image_points(cv::Mat &frame)
{
    for (int i=0; i<image_points.size(); i++) {
        cv::circle(frame, image_points[i], 2, cv::Scalar(0,255,0),-2);
        cv::putText(frame, point_names[i], image_points[i],CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0));
    }
}


void vis3D::render_objects(cv::Mat &frame,const std::string name)
{
    for (int i=0; i<objects.size(); i++) {
        if (i==0) {
            cv::rectangle(frame, objects[i].tl(), objects[i].br(), CV_RGB(255,0,0));
            cv::putText(frame, name, cv::Point2i(objects[i].x+objects[i].width*0.5,objects[i].y+objects[i].height*0.5),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
        }else{
            cv::rectangle(frame, objects[i].tl(), objects[i].br(), CV_RGB(0,255,0));
            cv::putText(frame, name, cv::Point2i(objects[i].x+objects[i].width*0.5,objects[i].y+objects[i].height*0.5),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0));
        }
    }
}

void vis3D::project_points()
{
    cv::projectPoints(volume_points, rvecs, tvecs, cameraMatrix, distCoeffs, image_points);
}

void vis3D::find_pose()
{
    cv::solvePnP(volume_points, image_points, cameraMatrix, distCoeffs, rvecs, tvecs,true,CV_ITERATIVE);
    //std::cout<<"rotation vector "<<std::endl<<rvecs<<std::endl;
    //std::cout<<"translation vector "<<std::endl<<tvecs<<std::endl;
}

void vis3D::detect_chessboard(cv::Mat &frame,const int framenum)
{
    bool found;
    std::vector<cv::Point2f> *pts; //assign it to the correct vector below!
    
    
    if(image_points_cal.size()<=framenum){ //if it's too small, we need to allocate more!
        image_points_cal.resize(framenum+1);
        image_points_cal[framenum].resize(npoints,cv::Point2f(-1,-1));
    }else{ //if it's already allocated, just empty it of previous points
        image_points_cal[framenum].assign(npoints, cv::Point2f(-1,-1));
    }
    pts=&image_points_cal[framenum];
    
    
    found = cv::findChessboardCorners( frame, board_size, *pts, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
    if (found) {
        cv::Mat frame_gray;
        
        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        
        cv::cornerSubPix( frame_gray, *pts, cv::Size(11,11),cv::Size(-1,-1), cv::TermCriteria( CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 30, 0.1 ));
        
        //image_points_cal[framenum]=pointBuf;
        //image_points.push_back(pointBuf);
        //prevTimestamp = clock();
        //blinkOutput = s.inputCapture.isOpened();
        
        drawChessboardCorners( frame, board_size, *pts, found );
    }
}

void vis3D::undistort_frame(cv::Mat &frame)
{
    cv::Mat temp = frame.clone();
    cv::undistort(temp, frame, cameraMatrix, distCoeffs);
}

void vis3D::setup_calibration(int caltype)
{
    if (caltype==CALIBRATE_FACE) {
        volume_points=MRI_points;
        npoints=6; //the number of anatomical points
        min_cal_pts=4; //minimum number of points to use
        CAMERA_CALIBRATION_FLAGS=CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_PRINCIPAL_POINT |  CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3;
    }else if (caltype==CALIBRATE_CHESSBOARD){
        volume_points=chessboard_points;
        npoints=54; //the number of chessboard points
        min_cal_pts=npoints; //this needs to be the same when using a chessboard
        CAMERA_CALIBRATION_FLAGS=CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_ZERO_TANGENT_DIST;
    }
}

void vis3D::detect_keypoints(cv::Mat frame)
{
    //detect the keypoints using an ORB detector
    //also need to find where they are on the object!
    //probably some filtering too
    
    cv::Mat mask, frame_tmp;
    
    cvtColor( frame, frame_tmp, CV_BGR2GRAY );
    equalizeHist( frame_tmp, frame );
    
    //allocate space in our keypoint and descriptor vectors
    kp.push_back(std::vector<cv::KeyPoint>());
    desc.push_back(cv::Mat());
    validpoint.push_back(std::vector<bool>()); //allocate trues later down
    
    //save the pose (which was computed previously!)
    rvecs_cal.push_back(rvecs);
    tvecs_cal.push_back(tvecs);
    
    //find the foreground
    //mog2(frame,mask);
    
    //detect features (I think SURF is the best, actually. But FAST detects a LOT of points!)
    //cv::FAST(frame, kp, 10,true);
    //orbdet(frame,mask,kp,desc); //ORB detects fairly well, but mostly around the edges of the head
    //orbdet.detect(frame, kp);
    //orbdet.detect(frame, kp, mask);
    //surfdet.detect(frame, kp.back()); //find the keypoints in the image
    //surfdet.compute(frame,kp,desc);
    surfdet(frame,cv::Mat::ones(frame.size(),CV_8U),kp.back(),desc.back()); //detect points in current image and calculates the descriptors
    //std::cout<<kp.back()[0].pt<<" "<<kp.back()[0].response<<std::endl;
    //std::cout<<desc.back().size()<<std::endl;
    //std::cout<<" kp size "<<kp.back().size()<<" desc size "<<desc.back().size();
    
    //call them all valid points for the moment
    validpoint.back().resize(kp.back().size(),true);
    
    //clean the points a little bit before doing the heavy processing!
    //remove them if they're too close together (1 pixel)
    //maybe sorting would do the trick?
    
    
    //the below is to speed up further calculations by removing points too close together, but it really doesn't remove many points and slows down the detection
    //not sure why I moved this to this function, but it probably should be in match_keypoints
    /*surfdet.detect(frame, kp.back());
    //for (long i=0; i<kp.size(); i++) { //frame loop
        std::cout<<"there are "<<kp.back().size()<<" points in this frame"<<std::endl;
        for (long j=kp.back().size()-2; j>=0; j--) { //point loop from 2nd to last point to the beginning
            for (long k=kp.back().size()-1; k>j; k--) { //partial loop over remaining values
                if((abs(kp.back()[j].pt.x-kp.back()[k].pt.x)+abs(kp.back()[j].pt.y-kp.back()[k].pt.y))<1){
                    //std::cout<<j<<" "<<k<<" "<<kp.back()[j].pt<<" "<<kp.back()[k].pt<<std::endl;
                    kp.back().erase(kp.back().begin()+k);
                }
            }
        }
        std::cout<<"there are now "<<kp.back().size()<<" points in this frame"<<std::endl;
    //}
    surfdet.compute(frame,kp.back(),desc.back());*/
    
    
    //match the features to the previous frame (if any)
    //if (desc.size()>1) {
    //    surfmatch.push_back(std::vector<cv::DMatch>());
    //    bfmatch.match(desc[desc.size()-1], desc.back(), surfmatch.back()); //match points to previous image
    //}
    
    
}



void vis3D::render_detected(cv::Mat &frame)
{
    //render the detected points and the head pose in some way
    cv::drawKeypoints(frame, kp.back(), frame);
}

void vis3D::render_matched_detected(cv::Mat &frame) //not actually working
{
    cv::Mat frame_new;
    //render the detected and matched keypoints from the current and previous frame
    //if (surfmatch.size()>0) { //make sure we have at least 2 frames
    //    cv::drawMatches(frame, kp[kp.size()-1], frame, kp.back(), surfmatch.back(), frame_new);
    //}
    
}

void vis3D::match_keypoints(const Mesh &head)
{
    //filter the keypoints and match them in time and to the object itself!
    //TODO: use a knn match to better filter the keypoints
    time_t tbegin, tend;
    
    //start by assuming all points are valid
    
    
    
    
    
    //matching the points between all frames using a radiusMatch to return all matches less than a specified distance
    std::cout<<"Matching a total of "<<desc.size()<<" frames"<<std::endl;
    //std::vector<cv::Mat> matchmat;
    std::vector<std::vector<cv::DMatch>> tmpmatch;
    time(&tbegin);
    bfmatch.add(desc); //add all training frames to the matcher
    surfmatch.resize(desc.size()); //create the base arrays for the matcher to use
    //matchmat.resize(desc.size()); //create the base size
    for (int i=0; i<desc.size(); i++) {
        /*for (int k=0; k<desc.size(); k++) {  //masks are impossible, I don't know what size they're supposed to be and there's no documentation anywhere!
            if (i==k) { //if it's the same frame we need to mask it out
                matchmat[k]=cv::Mat::zeros(desc[k].rows, desc[k].cols, CV_8UC1);
                //matchmat[k]=cv::Mat::zeros(desc[k].size(), CV_8UC1);
                std::cout<<matchmat[k].size()<<" "<<desc[k].size()<<std::endl;
            }else{
                matchmat[k]=cv::Mat::ones(desc[k].rows, desc[i].cols, CV_8UC1);
                //matchmat[k]=cv::Mat::ones(desc[k].size(), CV_8UC1);
            }
        }
        std::cout<<matchmat.size()<<" "<<matchmat[i].size()<<" "<<matchmat[i].type()<<" "<<CV_8UC1<<" rows "<<matchmat[i].rows<<" cols "<<matchmat[i].cols<<std::endl;
        std::cout<<"0,0 "<<matchmat[i].at<uchar>(0, 0)<<std::endl;*/
        //bfmatch.radiusMatch(desc[i], tmpmatch, 1.0);
        bfmatch.knnMatch(desc[i], tmpmatch, 3*static_cast<int>(desc.size()));  //lets assume that 3*desc.size() will give us enough matches so that we have at least one in each frame
        //surfmatch[i].resize(tmpmatch.size());
        std::cout<<"desc["<<i<<"].size() "<<desc[i].size()<<" tmpmatch.size() "<<tmpmatch.size()<<std::endl;
        /* OK, so this code goes through and picks the best matches and puts them into the final match vector.
         It could definitely be improved, like through a ratio test or something for example.
         At the moment it only grabs the top match for each frame and uses that.
         The next step is to put the cleaned points into the 2D to 3D matching code*/
        for (int j=0; j<tmpmatch.size(); j++) {
            //surfmatch[i][j].resize(desc.size()); //allocate one match for each image
            
            if (tmpmatch[j][1].distance<0.7*tmpmatch[j][2].distance) { //if it's good according to Lowe's ratio test
                surfmatch[i].push_back(std::vector<cv::DMatch>());
                surfmatch[i].back().resize(desc.size());
                for (int k=0; k<tmpmatch[j].size(); k++) { //faster would be converting this to a while loop!
                    if (tmpmatch[j][k].distance>0 && surfmatch[i].back()[tmpmatch[j][k].imgIdx].imgIdx==-1) { //if it's not a self match and we haven't put anything there yet. note that these are sorted by distance, so we don't have to worry about that, the first valid one we hit is the right one
                        //if (tmpmatch[j][k].distance>0 && surfmatch[i][j][tmpmatch[j][k].imgIdx].imgIdx==-1) { ORIGINAL IF
                        //surfmatch[i][j][tmpmatch[j][k].imgIdx]=tmpmatch[j][k];
                        surfmatch[i].back()[tmpmatch[j][k].imgIdx]=tmpmatch[j][k];
                    }
                }
            
            } else{ //remove the point from kp or set it to be invalid. or maybe this needs to be separate because of back and forth correspondances?
                validpoint[i][j]=false;
            }
        }
        for (int k=0; k<tmpmatch[0].size(); k++) {
            std::cout<<"match "<<k<<" distance "<<tmpmatch[0][k].distance<<" imgIdx "<<tmpmatch[0][k].imgIdx<<" queryIdx "<<tmpmatch[0][k].queryIdx<<" trainIdx "<<tmpmatch[0][k].trainIdx<<std::endl;
        }
        for (int k=0; k<surfmatch[i][0].size(); k++) {
            std::cout<<"match "<<k<<" distance "<<surfmatch[i][0][k].distance<<" imgIdx "<<surfmatch[i][0][k].imgIdx<<" queryIdx "<<surfmatch[i][0][k].queryIdx<<" trainIdx "<<surfmatch[i][0][k].trainIdx<<std::endl;
        }
        std::cout<<"size reduced from "<<tmpmatch.size()<<" to "<<surfmatch[i].size()<<" which is "<<100*surfmatch[i].size()/tmpmatch.size()<<"%"<<std::endl;
        /*for (int k=0; k<surfmatch[i][0].size(); k++) {
            std::cout<<"match "<<k<<" distance "<<surfmatch[i][0][k].distance<<" imgIdx "<<surfmatch[i][0][k].imgIdx<<" queryIdx "<<surfmatch[i][0][k].queryIdx<<" trainIdx "<<surfmatch[i][0][k].trainIdx<<std::endl;
        }*/
    }
    
    
    std::cout<<"keypoint matching complete!"<<std::endl;
    time(&tend);
    std::cout<<"all keypoint matching took "<<difftime(tend,tbegin)<<" s "<<std::endl;
    
    
    /* ETP 20150314 moving from not matching the same frame to matching the current frame to all frames
     note that this matches the frames to themselves as well, so we need to use a knnMatch approach
    //matching the points between all frames
    std::cout<<"Matching a total of "<<desc.size()<<" frames"<<std::endl;
    time(&tbegin);
    for (int i=0; i<desc.size()-1; i++) { //this just matches, I probably should do some position and distance matching as well because not every keypoint is in every frame!
        surfmatch.push_back(std::vector<std::vector<cv::DMatch>>());
        for (int j=i+1; j<desc.size(); j++) {
            std::cout<<"matching frame "<<i<<" to frame "<<j<<std::endl;
            surfmatch[i].push_back(std::vector<cv::DMatch>());
            bfmatch.match(desc[i], desc[j], surfmatch[i][j-i-1]); //match points to previous image
            std::cout<<"frame 1 "<<desc[i].size()<<" frame2 "<<desc[j].size()<<" surfmatch size "<<surfmatch[i][j-i-1].size()<<std::endl;
        }
        
    }
    std::cout<<"keypoint matching complete!"<<std::endl;
    time(&tend);
    std::cout<<"all keypoint matching took "<<difftime(tend,tbegin)<<" s "<<std::endl;
    
    
    //cleaning the points
    std::cout<<"Cleaning the keypoints from a total of "<<desc.size()<<" frames"<<std::endl;
    time(&tbegin);
    for (int i=0; i<desc.size()-1; i++) { //this just matches, I probably should do some position and distance matching as well because not every keypoint is in every frame!
        for (int j=i+1; j<desc.size(); j++) {
            int ii=j-i-1;
            std::cout<<"i "<<i<<" ii "<<ii<<" j "<<j<<std::endl;
            std::cout<<"initial # of matches "<<surfmatch[i][ii].size()<<std::endl;
            for (long int k=surfmatch[i][ii].size(); k>=0; k--) {
                if (surfmatch[i][ii][k].distance>1) {
                    surfmatch[i][ii].erase(surfmatch[i][ii].begin()+k); //erase
                }
            }
            std::cout<<"final # of matches "<<surfmatch[i][ii].size()<<std::endl;
            //for (int k=0;k<surfmatch[i][ii].size();k++){
            //    std::cout<<" distance "<<surfmatch[i][ii][k].distance<<" imgIdx "<<surfmatch[i][ii][k].imgIdx<<" queryIdx "<<surfmatch[i][ii][k].queryIdx<<" trainIdx "<<surfmatch[i][ii][k].trainIdx<<std::endl;
            //}
        }
        
    }
    std::cout<<"cleaning complete!"<<std::endl;
    time(&tend);
    std::cout<<"all keypoint cleaning took "<<difftime(tend,tbegin)<<" s "<<std::endl;
    ETP moving to knnMatch 20150315 end*/
    
    
    
    //finding correspondences between the volume and image
    std::vector<std::thread> threads; //array of threads
    std::cout<<"Calculating correspondences from "<<kp.size()<<" frames"<<std::endl;
     time(&tbegin);
    pt3D.resize(kp.size()); //resize here due to threading
     for (int i=0; i<kp.size(); i++) {
         std::cout<<"Corresponding frame "<<i<<std::endl;
         cv::Rodrigues(rvecs_cal[i], rmat); //convert vector to matrix
         tvecs_cal[i].copyTo(tmat); //still a vector?
         //proj_2D_to_3D(head,i); //This takes forever!
         //proj_2D_to_3D(head, i, tmat,rmat); //threaded version
         threads.push_back(std::thread(&vis3D::proj_2D_to_3D,*this,head, i, tmat,rmat));
     }
    for (int i=0;i<kp.size();i++){
        threads[i].join();
    }
     time(&tend);
     std::cout<<"all 2D to 3D correspondances took "<<difftime(tend,tbegin)<<" s "<<std::endl;

}


/* Functions for Möller–Trumbore intersection algorithm */

cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2)
{
    cv::Point3f tmp_p;
    tmp_p.x =  v1.y*v2.z - v1.z*v2.y;
    tmp_p.y =  v1.z*v2.x - v1.x*v2.z;
    tmp_p.z =  v1.x*v2.y - v1.y*v2.x;
    return tmp_p;
}

double DOT(cv::Point3f v1, cv::Point3f v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2)
{
    cv::Point3f tmp_p;
    tmp_p.x =  v1.x - v2.x;
    tmp_p.y =  v1.y - v2.y;
    tmp_p.z =  v1.z - v2.z;
    return tmp_p;
}

/* End functions for Möller–Trumbore intersection algorithm
 *  */

// Function to get the nearest 3D point to the Ray origin, thanks to Edgar Riba
cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin)
{
    cv::Point3f p1 = points_list[0];
    cv::Point3f p2 = points_list[1];
    
    double d1 = std::sqrt( std::pow(p1.x-origin.x, 2) + std::pow(p1.y-origin.y, 2) + std::pow(p1.z-origin.z, 2) );
    double d2 = std::sqrt( std::pow(p2.x-origin.x, 2) + std::pow(p2.y-origin.y, 2) + std::pow(p2.z-origin.z, 2) );
    
    if(d1 < d2)
    {
        return p1;
    }
    else
    {
        return p2;
    }
}

void vis3D::proj_2D_to_3D(const Mesh &head,const int framenum,cv::Mat tmat_loc, cv::Mat rmat_loc)
{
    //pt3D.push_back(std::vector<cv::Point3f>()); //moved outide the function to make it paralleizable
    pt3D[framenum].resize(kp[framenum].size(),cv::Point3f(0,0,0));
    int skipctr=0,hitctr=0,missctr=0;
    cv::Point2i ptround;
    time_t tbegin, tend;
    
    //let's try to speed this process up with a dirty probability map
    //cv::Mat precise(framesize.y,framesize.x,CV_32F,0.5), probability(framesize.y,framesize.x,CV_32F,0.5); //initially a 50/50 probability everywhere
    //cv::namedWindow("probability",5); //set up a window
    //std::cout<<"precise size "<<precise.size()<<std::endl;
    
    time(&tbegin);
    std::cout<<"There are less than "<<kp[framenum].size()<<" points to correspond"<<std::endl;
    for (int i=0; i<kp[framenum].size(); i++) {
        ptround=cv::Point2i(cvRound(kp[framenum][i].pt.x),cvRound(kp[framenum][i].pt.y));
        /*if ((i+1)%25==0) { //every 100 searches calculate the new probability. ETP commented after moving to threading and improved rejection with Lowe's ratio
            //std::cout<<"calculating the new probability"<<std::endl;
            cv::GaussianBlur(precise, probability, cv::Size(11,11), 2.0,2.0,cv::BORDER_REFLECT);
            //imshow("probability", probability);
            //cv::waitKey(100);
        }*/
        if (validpoint[framenum][i] ){//&& probability.at<float>(ptround.y,ptround.x)>=0.5){
            std::cout<<"Corresponding point "<<i<<" at "<<kp[framenum][i].pt;
            validpoint[framenum][i]=backproject2DPoint(&head, kp[framenum][i].pt, pt3D[framenum][i],&tmat_loc,&rmat_loc);
            if (validpoint[framenum][i]) {
                std::cout<<" hit!"<<std::endl;
                //precise.at<float>(ptround.y,ptround.x)=1.0; //what about rounding? I can't see the opencv website so I'm guessing here
                hitctr++;
            }else{
                std::cout<<" miss!"<<std::endl;
                //precise.at<float>(ptround.y,ptround.x)=0.0;
                missctr++;
            }
        }else{
            //std::cout<<"skipped point "<<i<<" at "<<kp[framenum][i].pt<<" validity "<<validpoint[framenum][i]<<" probability "<<probability.at<float>(ptround.y,ptround.x)<<std::endl;
            skipctr++;
        }
    }
    std::cout<<"hit a total of "<<hitctr<<" points"<<std::endl;
    std::cout<<"missed a total of "<<missctr<<" points"<<std::endl;
    std::cout<<"skipped a total of "<<skipctr<<" points"<<std::endl;
    time(&tend);
    std::cout<<"this frame took "<<difftime(tend,tbegin)<<" s "<<std::endl;
}


// Back project a 2D point to 3D and returns if it's on the object surface, thanks to Edgar Riba
bool vis3D::backproject2DPoint(const Mesh *mesh, const cv::Point2f &point2d, cv::Point3f &point3d,cv::Mat *tmat_loc, cv::Mat *rmat_loc)
{
    // Triangles list of the object mesh
    std::vector<std::vector<int> > triangles_list = mesh->getTrianglesList();
    
    double lambda = 8;
    double u = point2d.x;
    double v = point2d.y;
    
    // Point in vector form
    cv::Mat point2d_vec = cv::Mat::ones(3, 1, CV_64F); // 3x1
    point2d_vec.at<double>(0) = u * lambda;
    point2d_vec.at<double>(1) = v * lambda;
    point2d_vec.at<double>(2) = lambda;
    
    // Point in camera coordinates
    cv::Mat X_c = cameraMatrix.inv() * point2d_vec ; // 3x1
    
    // Point in world coordinates
    cv::Mat X_w = rmat_loc->inv() * ( X_c - *tmat_loc ); // 3x1
    
    // Center of projection
    cv::Mat C_op = cv::Mat(rmat_loc->inv()).mul(-1) * *tmat_loc; // 3x1
    
    // Ray direction vector
    cv::Mat ray = X_w - C_op; // 3x1
    ray = ray / cv::norm(ray); // 3x1
    
    // Set up Ray
    Ray R((cv::Point3f)C_op, (cv::Point3f)ray);
    
    // A vector to store the intersections found
    std::vector<cv::Point3f> intersections_list;
    
    // Loop for all the triangles and check the intersection
    for (unsigned int i = 0; i < triangles_list.size(); i++)
    {
        cv::Point3f V0 = mesh->getVertex(triangles_list[i][0]);
        cv::Point3f V1 = mesh->getVertex(triangles_list[i][1]);
        cv::Point3f V2 = mesh->getVertex(triangles_list[i][2]);
        
        Triangle T(i, V0, V1, V2);
        
        double out;
        if(this->intersect_ray_triangle(R, T, &out))
        {
            cv::Point3f tmp_pt = R.getP0() + out*R.getP1(); // P = O + t*D
            intersections_list.push_back(tmp_pt);
        }
    }
    
    // If there are intersection, find the nearest one
    if (!intersections_list.empty())
    {
        std::cout<<"found "<<intersections_list.size()<<" interactions!"<<std::endl;
        point3d = get_nearest_3D_point(intersections_list, R.getP0()); //this only works for 2 points, not many!
        return true;
    }
    else
    {
        return false;
    }
}



// Möller–Trumbore intersection algorithm, thanks to Edgar Riba
bool vis3D::intersect_ray_triangle(Ray &Ray, Triangle &Triangle, double *out)
{
    const double EPSILON = 0.000001;
    
    cv::Point3f e1, e2;
    cv::Point3f P, Q, T;
    double det, inv_det, u, v;
    double t;
    
    cv::Point3f V1 = Triangle.getV0();  // Triangle vertices
    cv::Point3f V2 = Triangle.getV1();
    cv::Point3f V3 = Triangle.getV2();
    
    cv::Point3f O = Ray.getP0(); // Ray origin
    cv::Point3f D = Ray.getP1(); // Ray direction
    
    //Find vectors for two edges sharing V1
    e1 = SUB(V2, V1);
    e2 = SUB(V3, V1);
    
    // Begin calculation determinant - also used to calculate U parameter
    P = CROSS(D, e2);
    
    // If determinant is near zero, ray lie in plane of triangle
    det = DOT(e1, P);
    
    //NOT CULLING
    if(det > -EPSILON && det < EPSILON) return false;
    inv_det = 1.f / det;
    
    //calculate distance from V1 to ray origin
    T = SUB(O, V1);
    
    //Calculate u parameter and test bound
    u = DOT(T, P) * inv_det;
    
    //The intersection lies outside of the triangle
    if(u < 0.f || u > 1.f) return false;
    
    //Prepare to test v parameter
    Q = CROSS(T, e1);
    
    //Calculate V parameter and test bound
    v = DOT(D, Q) * inv_det;
    
    //The intersection lies outside of the triangle
    if(v < 0.f || u + v  > 1.f) return false;
    
    t = DOT(e2, Q) * inv_det;
    
    if(t > EPSILON) { //ray intersection
        *out = t;
        return true;
    }
    
    // No hit, no win
    return false;
}



void bound_rect(cv::Rect &rect, cv::Point2i size)
{
    //we just shrink the rect to make it fit on the frame, we don't move the edges that are legit
    if(rect.x<0){ //if it's off the frame to the left
        rect.width+=rect.x;
        rect.x=0;
    }
    if(rect.y<0){ //if it's off the frame to the right
        rect.height+=rect.y;
        rect.y=0;
    }
    if(rect.x+rect.width>=size.x){ //off the frame to the right
        rect.width=size.x-rect.x-1;
    }
    if(rect.y+rect.height>=size.y){ //off the frame to the bottom
        rect.height=size.y-rect.y-1;
    }
}

void help_on_screen(cv::Mat frame,const int flags)
{
    //std::cout<<"flags "<<flags<<std::endl;
    if (flags&1) { //this help
        cv::putText(frame, "(h)elp on", cv::Point2i(5,15),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
    }else{ //never shown, but I'll put it here for completeness
        cv::putText(frame, "(h)elp off", cv::Point2i(5,15),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
    }
    if (flags&2) { //visualizing
        cv::putText(frame, "(v)isualizing on", cv::Point2i(5,35),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
    }else{
        cv::putText(frame, "(v)isualizing off", cv::Point2i(5,35),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
    }
    if (flags&4) { //calibrating
        cv::putText(frame, "face calib(r)ating on", cv::Point2i(5,55),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
    }else{
        cv::putText(frame, "face calib(r)ating off", cv::Point2i(5,55),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
    }
    if (flags&8) { //chessboard calibrating
        cv::putText(frame, "chess(b)oard calibrating on", cv::Point2i(5,75),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
    }else{
        cv::putText(frame, "chess(b)oard calibrating off", cv::Point2i(5,75),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
    }
    if (flags&16) { //undistorting
        cv::putText(frame, "(u)ndistorting on", cv::Point2i(5,95),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
    }else{
        cv::putText(frame, "(u)ndistorting off", cv::Point2i(5,95),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
    }
    if (flags&32) { //finding keypoints
        cv::putText(frame, "(f)inding keypoints on", cv::Point2i(5,115),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
    }else{
        cv::putText(frame, "(f)inding keypoints off", cv::Point2i(5,115),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
    }
    cv::putText(frame, "(c)alibrate", cv::Point2i(5,135),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
    cv::putText(frame, "(m)atch keypoints", cv::Point2i(5,155), CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 255));
    //cv::putText(frame, point_names[i], image_points_cal[loc][i],CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0));
}


//the original callback for mouse events
void CallBackFunc(int event, int x, int y, int flags, void* image)
{
    //cv::Point* xylocptr = (cv::Point*) xyloc;
    //cv::Mat* imageptr = (cv::Mat*) image;
    //cv::Point2i point(x,y);
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        //std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
        pt.x=x;
        pt.y=y;
        //xylocptr->x=x;
        //xylocptr->y=y;
    }
    /*else if  ( event == cv::EVENT_RBUTTONDOWN )
    {
        std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
    else if  ( event == cv::EVENT_MBUTTONDOWN )
    {
        std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }*/
    else if ( event == cv::EVENT_MOUSEMOVE )
    {
        //std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
        //std::cout << "Original clock location (" << pt.x << ", " << pt.y << ")" << std::endl;
        xyloc.x=x;
        xyloc.y=y;
        //point.x=x;
        //point.y=y;
        //cv::line(*imageptr,pt,point,CV_RGB(255,255,255));
        
    }
    
}

int main(int argc, char* argv[])
{
    cv::Point2i test;
    //set up some variables
    int keyresponse;
    int waittime=1,helpflags=1;
    int iscalibrating=0,isvisualizing=0,ischessboardcal=0,isundistorting=0,isdetecting=0;//,ishelp=1;
    //int bgframestart=0, bgframeend=-1;
    int calframenum=0;
    //int xyloc[2];
    //cv::Point2i xyloc(100,100);
    unsigned concurentThreadsSupported = std::thread::hardware_concurrency();  //number of possible threads
    Mesh head;                // instantiate Mesh object
    
    //mesh file input and read
    //currently fixed to /Users/etpeters/Documents/xcode/visualize3D/visualize3D/opencv_test/data/surface_ETP.ply
    //how do I do a relative path?
    if (argc<2) {
        std::cout<<"No input mesh file specified!"<<std::endl;
    }else{
        std::cout<<"Input mesh file "<<argv[1]<<std::endl;
        
        head.load(argv[1]); // load an object mesh
        std::cout<<"Loaded the mesh file which contains "<<head.getNumVertices()<<" verticies"<<std::endl;
    }
    
    cv::VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()){  // check if we succeeded
        std::cout<<"Sorry, we couldn't open the camera"<<std::endl;
        return -1;
    }
    //std::cout<<"intial FPS "<<cap.get(CV_CAP_PROP_FPS)<<std::endl;
    //cap.set(CV_CAP_PROP_FPS, 30); //set 30 FPS
    
    //resize the video stream
    //cap.set(CV_CAP_PROP_FRAME_WIDTH,cap.get(CV_CAP_PROP_FRAME_WIDTH)/2);
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT,cap.get(CV_CAP_PROP_FRAME_HEIGHT)/2);
    
    std::cout<<"Number of supported threads "<<concurentThreadsSupported<<std::endl;
    
    cv::Mat frame; //create a frame
    cv::Mat frame_raw, frame_court_define;
    cv::namedWindow("frame",1); //create the window
    //cv::setMouseCallback("frame", CallBackFunc, (void*)&xyloc); //create the mouse callback for point passback
    cv::setMouseCallback("frame",CallBackFunc,(void*) &frame);
    
    cap.read(frame); //read the frame for setup purposes
    std::cout<<"framesize "<<frame.size()<<std::endl;
    
    /* define the visualizer */
    vis3D vis; //initialize the visualizer
    vis.setframesize(frame.size()); //for internal use
    vis.setup_camera_matrix(frame.size()); //setup the initial guess camera matrix
    vis.setup_rvec(); //setup the initial guess rotation
    vis.setup_tvec(); //setup the initial guess translation

    
    
    while(cap.isOpened()){
        cap.read(frame); //read the frame
        
        //some basic movie controls
        keyresponse=cv::waitKey(1); //no keyresponse returns -1
        waittime=1;
        if (keyresponse=='q'){ //q
            std::cout<<"Exiting the initial loop"<<std::endl;
            break;
        }else if (keyresponse=='h'){
            if (helpflags&1) {
                std::cout<<"Turing on help"<<std::endl;
                //ishelp=1;
                helpflags=helpflags^1<<0;
            }else{
                std::cout<<"Turning off help"<<std::endl;
                //ishelp=0;
                helpflags=helpflags^1<<0;
            }
        }else if (keyresponse=='u'){
            if (isundistorting==0) {
                std::cout<<"Turning undistorting on"<<std::endl;
                isundistorting=1;
                helpflags=helpflags^1<<4;
            }else{
                std::cout<<"Turning undistorting off"<<std::endl;
                isundistorting=0;
                helpflags=helpflags^1<<4;
            }
        
        }else if (keyresponse=='r'){ //pause (p or space bar)
            //waittime=0; //wait forever next time
            if (iscalibrating==0){ //start recording for calibration
                std::cout<<"Recording for calibration"<<std::endl;
                iscalibrating=1;
                helpflags=helpflags^1<<2;
                vis.setup_calibration(CALIBRATE_FACE); //a little setup so we don't do it a lot in other functions
            }else{ //end calibrating
                std::cout<<"Ending recodring for calibration"<<std::endl;
                iscalibrating=0;
                helpflags=helpflags^1<<2;
                //vis.camera_calibration(frame.size()); //actually calibrate
            }
            
        }else if(keyresponse=='c'){ //start calibrating
            //Eric calibrate the camera
            vis.camera_calibration(frame.size()); //calibrate using recorded positions
            calframenum=0; //reset our calibration frames (also done in camera_calibration)
        }else if(keyresponse=='v'){ //start visualizing
            if(isvisualizing==0){ //start visualizing
                std::cout<<"Starting visualization"<<std::endl;
                isvisualizing=1;
                helpflags=helpflags^1<<1;
                vis.setup_calibration(CALIBRATE_FACE); //a little setup so we don't do it a lot in other functions
            }else{
                std::cout<<"Ending visualization"<<std::endl;
                isvisualizing=0;
                helpflags=helpflags^1<<1;
            }
        }else if(keyresponse=='b'){ //chessboard calibration
            if (ischessboardcal==0) { //start a chessboard calibration
                std::cout<<"Recording for chessboard calibration"<<std::endl;
                ischessboardcal=1;
                helpflags=helpflags^1<<3;
                vis.setup_calibration(CALIBRATE_CHESSBOARD); //a little setup so we don't do it a lot in other functions
            }else{
                std::cout<<"Ending recodring for chessboard calibration"<<std::endl;
                ischessboardcal=0;
                helpflags=helpflags^1<<3;
            }
        }else if (keyresponse=='f'){
            if (isdetecting==0) { //turn the point detection on
                std::cout<<"Recording for keypoint detection"<<std::endl;
                isdetecting=1;
                helpflags=helpflags^1<<5;
                vis.setup_calibration(CALIBRATE_FACE); //a little setup so we don't do it a lot in other functions
            }else{ //turn point detection off
                std::cout<<"Recording for keypoint detection"<<std::endl;
                isdetecting=0;
                helpflags=helpflags^1<<5;
            }
        }else if (keyresponse=='m'){
            std::cout<<"Matching keypoints to the object"<<std::endl;
            vis.match_keypoints(head);
        }
        
        
        
        if(iscalibrating==1){
            //Christoph
            //here we need to detect the interesting points in the image and add them to image_points
            //if nothing is detected for that object, put a point at (-1,-1)
            vis.detect_anatomy(frame, calframenum);
            //vis.render_image_points_cal(frame,calframenum); //this should be the final render step (once debugging is done)
            calframenum++;
        }else if(isvisualizing==1){ //we can't calibrate and visualize at the same time!
            //Eric
            //Run solvePnP on the current frame
            vis.detect_anatomy(frame, -1); //don't give it a frame number because that causes calibration
            vis.find_pose(); //calculate the pose
            vis.project_points(); //using the pose, project the points on the image
            vis.render_image_points(frame); //render the image
        }else if(ischessboardcal==1){
            vis.detect_chessboard(frame, calframenum);
            calframenum++;
        }else if(isdetecting==1){
            //detect keypoints on the image
            vis.detect_anatomy(frame, -1);
            vis.find_pose();
            
            vis.detect_keypoints(frame);
            vis.render_detected(frame);
            //vis.render_matched_detected(frame); //not actually working
        }
        
        if (isundistorting==1) {
            vis.undistort_frame(frame);
        }
        if (helpflags&1) {
            help_on_screen(frame,helpflags);
        }
        
        //frame=frame_raw;
        //frame_raw.copyTo(frame);
        

        imshow("frame", frame);
        

    }
    std::cout<<"Goodbye"<<std::endl;
    
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
