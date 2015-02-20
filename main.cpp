/*
 Eric Peterson & Christoph Leuze 2015
 This is our code to visualize brains on a video of our heads
 
 TODO ETP:
 1. Output the full vector of rotation and translation components when it's calibrated
 2. Initialize the rotation and translation for get_pose nicely for the first call
 3. Implement get_pose
 
 */


#include <iostream>
//#include <functional>
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/features2d.hpp"
//#include <opencv2/core/core_c.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/objdetect/objdetect.hpp>

//using namespace cv;

cv::Point2f pt(-1,-1);
cv::Point2f xyloc(-1,-1);
bool wasclick=false,wasrelease=true;




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
    int npoints; //number of points we know
    //cv::Point2i court_img_verts[22]; //all the court verticies
    std::vector< std::vector<cv::Point2f> > face_points_cal; //must be floats otherwise we get strange issues from
    std::vector<cv::Point2f> face_points;
    //cv::Point3f court_verts[22]; //the actual 3D locations of the court verticies
    std::vector< std::vector<cv::Point3f> > volume_points_cal; //all the court verticies in actual locations (sized in the constructor)
    std::vector<cv::Point3f> volume_points;
    //std::vector<cv::Point3f> court_verts;
    //let's define the back left as (0,0,0), and everything can be positive from there in meters
    //so x is across the court, y is front to back, and z is vertical

    
    //camera section
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs_cal, tvecs_cal;
    cv::Mat rvecs, tvecs;
    double rms;
    
public:
    void calibrate_camera(cv::Mat); //the main camera calibration function
    static void vis_call_back(int , int , int , int , void *); //mouse handler callback function
    void setup_camera_matrix(cv::Point2i); //camera calibration helper function
    void setup_rvec(); //camera calibration helper function
    void setup_tvec(); //camera calibration helper function
    void camera_calibration(cv::Point2i); //camera calibration helper function
    void get_pose();
    
    
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
void vis3D::calibrate_camera(cv::Mat frame_raw){
    /* this function gets the corners by asking the user to click on them.
     I should really let the users drag the corners around, but that would take some reworking of the code */
    
    cv::Mat frame_corners;
    int keyresponse=-1;
    cv::Point2i p2i_zeros(0,0);
    int court_vert_idx=0;
    cv::Point2i framesize=frame_raw.size();
    float mindist=1.0e12,dprod=0.0;
    int minidx=0;
    
    //set up the camera matrix (it's kind of complicated)
    setup_camera_matrix(framesize);
    /*double f = 55.0; // focal length in mm (55 is about what it finds, but I don't know if that's right
    double sx = 36.0, sy = 24.0;             // sensor size (total guess that it's full frame)
    cameraMatrix.at<double>(0,2)=static_cast<double>(framesize.x)/2.0; //finish setting the cameraMatrix because calibrateCamera can't do it for me
    cameraMatrix.at<double>(1,2)=static_cast<double>(framesize.y)/2.0;  //finish setting cameraMatrix
    cameraMatrix.at<double>(0,0)=static_cast<double>(framesize.x)*f/sx; //fx
    cameraMatrix.at<double>(1,1)=static_cast<double>(framesize.y)*f/sy; //fy
    std::cout<<"camera matrix"<<std::endl<<cameraMatrix<<std::endl;  //this doesn't seem to change*/
    /* This seems to be a decent camera matrix, which is not too different than what I am sending in
     [2207.302662623425, 0, 640;
     0, 1241.607747725677, 360;
     0, 0, 1]*/
    //cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    
    //set up the tvec
    setup_tvec();
    /*tvecs[0].at<double>(0)=3.2; //3.2m in x (centered)
    tvecs[0].at<double>(1)=-10.0; //5m back
    tvecs[0].at<double>(2)=5.0; //5m up */
    /* this is an example tvec, which isn't too different than the input, but still isn't correct
     [-3.186344372084879, 4.024008286339618, 16.0116542958002]*/
    
    //set up the rvec
    setup_rvec();
    /*rvecs[0].at<double>(0)=2.0; //not sure what this is
    rvecs[0].at<double>(1)=0.0; //not sure what this is
    rvecs[0].at<double>(2)=0.0; //not sure what this is */
    /* This is based off of an output rvec which put the court just about in the middle of the screen
     [2.011442578743535, -0.009058716752575486, 0.0001397269699944912]*/
    
    //I need to add a while loop and some other key processing to keep it in here!
    std::cout<<"Press b to begin setting points, s to set the points, and u to undo the previous point"<<std::endl;
    
    //we took a blind guess at the beginning, so let's just project the points!
    //cv::projectPoints(court_verts[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs, court_img_verts[0]);
    
    
    
    /* this section lets you define the court by dragging the verticies around, this probably should just replace the click based method above*/
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
}

void vis3D::camera_calibration(cv::Point2i framesize)  //should I pass by reference? currently the 2nd argument isn't used
{
    //crop each individual frame result to only the known matches
    int j;
    for (int i=0; i<volume_points_cal.size(); i++) {
        j=0;
        while (j<volume_points_cal[i].size()) {
            if (face_points_cal[i][j].x==-1 || face_points_cal[i][j].y==-1) { //if any part of it is -1, it's not matched!
                face_points_cal[i].erase(face_points_cal[i].begin()+j);
                volume_points_cal[i].erase(volume_points_cal[i].begin()+j);
            }
            j++;
        }
    }
    rms=cv::calibrateCamera(volume_points_cal, face_points_cal, framesize, cameraMatrix, distCoeffs, rvecs_cal, tvecs_cal, CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_PRINCIPAL_POINT |  CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3);
    std::cout<<"RMS error = "<<rms<<std::endl;
    std::cout<<"camera matrix"<<std::endl<<cameraMatrix<<std::endl;  //this doesn't seem to change
    std::cout<<"rotation vector"<<std::endl<<rvecs_cal[0]<<std::endl;
    std::cout<<"translation vectors"<<std::endl<<tvecs_cal[0]<<std::endl;
    //CV_CALIB_FIX_ASPECT_RATIO | //we shouldn't be fixing the aspect ratio, it distorted the result
    //fix principal point could be removed, especially when finding the distortion matrix
    //note that if we let the distortions correct then the camera matrix also gets a little crazy
    //I think we can fairly safely assume that the camera is basically distortionless
    //all the above arguments to calibrateCamera mean that it's essentially not optimizing the distortion matrix, only the camera matrix, rotation and translation vectors
}

void vis3D::setup_camera_matrix(cv::Point2i framesize) //should I pass by reference?
{
    double f = 55.0; // focal length in mm (55 is about what it finds, but I don't know if that's right
    double sx = 36.0, sy = 21.0;             // sensor size, full frame is 36x24, but this one seems less square?
    std::cout<<"Setting up the camera matrix vector"<<std::endl;
    cameraMatrix.at<double>(0,2)=static_cast<double>(framesize.x)/2.0; //finish setting the cameraMatrix because calibrateCamera can't do it for me
    cameraMatrix.at<double>(1,2)=static_cast<double>(framesize.y)/2.0;  //finish setting cameraMatrix
    cameraMatrix.at<double>(0,0)=static_cast<double>(framesize.x)*f/sx; //fx
    cameraMatrix.at<double>(1,1)=static_cast<double>(framesize.y)*f/sy; //fy
    std::cout<<"camera matrix"<<std::endl<<cameraMatrix<<std::endl;  //this doesn't seem to change
    /*camera matrix
     [1900.143582407142, 0, 640;
     0, 1829.372842887497, 360;
     0, 0, 1]*/
}

void vis3D::setup_tvec()
{
    //set up the tvec
    std::cout<<"Setting up the translation vector"<<std::endl;
    //think of these translations as happening to the court, not the camera!
    tvecs_cal[0].at<double>(0)=-3.2; //-3.2m in x (centered)
    tvecs_cal[0].at<double>(1)=2.5; //2.5m up (maybe)
    tvecs_cal[0].at<double>(2)=14.0; //14m back (maybe)
    std::cout<<"translation vectors"<<std::endl<<tvecs_cal[0]<<std::endl;
    /* this is an example tvec, which isn't too different than the input, but still isn't correct
     [-3.181589803420535, 2.668127841422967, 13.74342575277559]*/
    
}

void vis3D::setup_rvec()
{
    //set up the rvec
    std::cout<<"Setting up the rotation vector"<<std::endl;
    rvecs_cal[0].at<double>(0)=1.8; //not sure what this is
    rvecs_cal[0].at<double>(1)=0.0; //not sure what this is
    rvecs_cal[0].at<double>(2)=0.0; //not sure what this is
    std::cout<<"rotation vector"<<std::endl<<rvecs_cal[0]<<std::endl;
    /* This is based off of an output rvec which put the court just about in the middle of the screen
     [1.818635848296208, -0.008306380664320337, 0.005670786435876811]*/
}

void vis3D::get_pose()
{
    //Eric I need to find rvec and tvec
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
    int waittime=1;
    int iscalibrating=0,isvisualizing=0;
    int bgframestart=0, bgframeend=-1;
    //int xyloc[2];
    //cv::Point2i xyloc(100,100);
    
    cv::VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()){  // check if we succeeded
        std::cout<<"Sorry, we couldn't open the camera"<<std::endl;
        return -1;
    }
    
    
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
        keyresponse=cv::waitKey(1); //no keyresponse is -1
        waittime=1;
        if (keyresponse=='q'){ //q
            std::cout<<"Exiting the initial loop"<<std::endl;
            break;
        }else if (keyresponse=='r'){ //pause (p or space bar)
            //waittime=0; //wait forever next time
            if (iscalibrating==0){ //start recording for calibration
                std::cout<<"Recording for calibration"<<std::endl;
                iscalibrating=1;
            }else{ //end calibrating
                std::cout<<"Ending recodring for calibration"<<std::endl;
                iscalibrating=0;
            }
            
        }else if(keyresponse=='c'){ //start calibrating
            //Eric calibrate the camera
            vis.camera_calibration(frame.size());
        }else if(keyresponse=='v'){ //start visualizing
            if(isvisualizing==0){ //start visualizing
                std::cout<<"Starting visualization"<<std::endl;
                isvisualizing=1;
            }else{
                std::cout<<"Ending visualization"<<std::endl;
                isvisualizing=0;
            }
        }
        if(iscalibrating==1){
            //Christoph
            //here we need to detect the interesting points in the image and add them to face_points
            //if nothing is detected for that object, put a point at (-1,-1)

        }else if(isvisualizing==1){ //we can't calibrate and visualize at the same time!
            //Eric
            //Run solvePnP on the current frame
        }
        
        
        //frame=frame_raw;
        //frame_raw.copyTo(frame);
        

        imshow("frame", frame);
        

    }
    std::cout<<"Goodbye"<<std::endl;
    
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
