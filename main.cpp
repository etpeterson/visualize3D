/*
 Eric Peterson & Christoph Leuze 2015
 This is our code to visualize brains on a video of our heads
 
 TODO ETP:
 1. Cascade detector improvement
    a. Filter the locations based on how they move around over time?
    b. Maybe also a spatial-temporal filter?
 3. The checkerboard undistortion crashes when it isn't detected in a frame, why?
 4. Try ORB rather than SURF, it's supposedly faster and more accurate as well as being free!
 5. Test the image undistortion
 
 To improve matching:
 1. Need ply files of the MRI data
 2. Initial pose with face detection
 3. Find matching points simliar to model registration by finding locations on the 3D data of the image markers from ORB detectors
    a. Use a single image to find matches but later use video and filtering to find the best matches
 4. Find the pose using matched points rather than face detection
    a. Could compare to the pose from face detection?
 
 
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
 
 */


#include <iostream>
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




bool rect_largest_to_smallest (cv::Rect i,cv::Rect j) { return (i.area()>j.area()); } //rect sorting helper function (largest to smallest)




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


    
    //camera section
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs_cal, tvecs_cal;
    cv::Mat rvecs, tvecs;
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
    void render_image_points_cal(cv::Mat,int); //draw image_points_cal on an image
    void render_image_points(cv::Mat); //draw image_points on an image
    void render_objects(cv::Mat,std::string); //draw all the detected objects on the given frame
    void project_points(); //fill image_points with locations calculated from the rotation, translation, camera matrix, and the object points
    void find_pose(); //find the rotation and translation vectors based on the object and image points
    void detect_chessboard(cv::Mat &frame, int framenum); //detect the chessboard features
    void undistort_frame(cv::Mat &frame); //undistort the output image
    void setup_calibration(int caltype); //set the variables correctly for calibrating with either a face or a chessboard
    void detect_keypoints(cv::Mat frame);
    void render_detected(cv::Mat frame);
    
    
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
    rvecs_cal[0] = cv::Mat::zeros(1, 3, CV_64F); //Try initializing zeros
    tvecs_cal.resize(1);
    tvecs_cal[0] = cv::Mat::zeros(1, 3, CV_64F); //Try initializing zeros and fill in later
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

void vis3D::render_image_points_cal(cv::Mat frame,int loc)
{
    for (int i=0; i<image_points_cal[loc].size(); i++) {
        cv::circle(frame, image_points_cal[loc][i], 2, cv::Scalar(0,255,0),-2);
        cv::putText(frame, point_names[i], image_points_cal[loc][i],CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0));
    }
}

void vis3D::render_image_points(cv::Mat frame)
{
    for (int i=0; i<image_points.size(); i++) {
        cv::circle(frame, image_points[i], 2, cv::Scalar(0,255,0),-2);
        cv::putText(frame, point_names[i], image_points[i],CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0));
    }
}


void vis3D::render_objects(cv::Mat frame,std::string name)
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

void vis3D::detect_chessboard(cv::Mat &frame, int framenum)
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
}



void render_detected(cv::Mat frame)
{
    //render the detected points and the head pose in some way
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

void help_on_screen(cv::Mat frame,int flags)
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
    cv::putText(frame, "(c)alibrate", cv::Point2i(5,115),CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255));
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
    
    cv::Mat frame; //create a frame
    cv::Mat frame_raw, frame_court_define;
    cv::namedWindow("frame",1); //create the window
    //cv::setMouseCallback("frame", CallBackFunc, (void*)&xyloc); //create the mouse callback for point passback
    cv::setMouseCallback("frame",CallBackFunc,(void*) &frame);
    
    cap.read(frame); //read the frame for setup purposes
    
    /* define the visualizer */
    vis3D vis; //initialize the visualizer
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
            }else{ //turn point detection off
                std::cout<<"Recording for keypoint detection"<<std::endl;
                isdetecting=0;
                helpflags=helpflags^1<<5;
            }
        
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
            //vis.detect_keypoints(frame);
            //vis.render_detected(frame);
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
