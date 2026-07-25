// Live camera demo for CvTracker.
//
// Uses OpenCV (preinstalled with JetPack) ONLY for camera capture, display
// and mouse input - the tracker itself consumes raw frames and has no
// OpenCV dependency.
//
// Usage:   ./camera_demo [device] [width] [height]
// Example: ./camera_demo 0 1280 720
//
// Controls:
//   left click  - capture object at cursor
//   right click - reset tracking
//   +/-         - grow / shrink tracking rectangle
//   a           - toggle rectangle auto-size
//   c           - toggle correlation surface window
//   q / ESC     - quit
#include <chrono>
#include <cstdio>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cvtracker/CvTracker.h>

using namespace cr::vtracker;

static CvTracker g_tracker;

static void onMouse(int event, int x, int y, int, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN)
        g_tracker.executeCommand(VTrackerCommand::CAPTURE, (float)x, (float)y,
                                 -1.0f);
    else if (event == cv::EVENT_RBUTTONDOWN)
        g_tracker.executeCommand(VTrackerCommand::RESET);
}

int main(int argc, char** argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const int reqW = argc > 2 ? std::atoi(argv[2]) : 1280;
    const int reqH = argc > 3 ? std::atoi(argv[3]) : 720;

    cv::VideoCapture cap(device, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        std::printf("ERROR: can't open /dev/video%d\n", device);
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, reqW);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, reqH);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 2);

    // Tracker init.
    VTrackerParams p;
    p.rectWidth = 72;
    p.rectHeight = 72;
    p.searchWindowWidth = 256;
    p.searchWindowHeight = 256;
    p.frameBufferSize = 8; // no stop-frame needed in this demo
    p.maxFramesInLostMode = 90;
    p.lostModeOption = 1;
    p.multipleThreads = true;
    if (!g_tracker.initVTracker(p))
    {
        std::printf("ERROR: initVTracker failed\n");
        return 1;
    }
    std::printf("CvTracker %s, camera /dev/video%d\n",
                CvTracker::getVersion().c_str(), device);

    cv::namedWindow("CvTracker", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("CvTracker", onMouse);

    cv::Mat bgr, gray;
    cr::video::Frame frame;
    cr::video::Frame surf;
    VTrackerParams r;
    int32_t frameId = 0;
    bool showSurface = false;
    double fps = 0.0;
    auto tPrev = std::chrono::steady_clock::now();

    while (true)
    {
        if (!cap.read(bgr) || bgr.empty())
        {
            std::printf("ERROR: capture failed\n");
            break;
        }
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        // Wrap into tracker frame (GRAY) - allocate once.
        if (frame.width != gray.cols || frame.height != gray.rows)
            frame.init(gray.cols, gray.rows, cr::video::Fourcc::GRAY);
        if (gray.isContinuous())
            std::memcpy(frame.data, gray.data, frame.size);
        else
            for (int y = 0; y < gray.rows; ++y)
                std::memcpy(frame.data + (size_t)y * gray.cols,
                            gray.ptr(y), gray.cols);
        frame.frameId = frameId++;

        g_tracker.processFrame(frame);
        g_tracker.getParams(r);

        // FPS.
        const auto tNow = std::chrono::steady_clock::now();
        const double dt =
            std::chrono::duration<double>(tNow - tPrev).count();
        tPrev = tNow;
        fps = 0.9 * fps + 0.1 * (dt > 0 ? 1.0 / dt : 0.0);

        // Overlay.
        const char* modeNames[] = {"FREE", "TRACKING", "LOST", "INERTIAL",
                                   "STATIC"};
        cv::Scalar rectColor(255, 255, 255); // FREE: white
        if (r.mode == 1)
            rectColor = cv::Scalar(0, 220, 0); // TRACKING: green
        else if (r.mode == 2)
            rectColor = cv::Scalar(0, 200, 255); // LOST: yellow
        else if (r.mode == 3)
            rectColor = cv::Scalar(255, 120, 0); // INERTIAL: blue

        if (r.mode != 0)
        {
            cv::rectangle(bgr,
                          cv::Rect(r.rectX - r.rectWidth / 2,
                                   r.rectY - r.rectHeight / 2, r.rectWidth,
                                   r.rectHeight),
                          rectColor, 2);
            // Estimated object rectangle (thin).
            cv::rectangle(bgr,
                          cv::Rect(r.objectX - r.objectWidth / 2,
                                   r.objectY - r.objectHeight / 2,
                                   r.objectWidth, r.objectHeight),
                          cv::Scalar(180, 180, 180), 1);
            // Velocity vector.
            cv::arrowedLine(bgr, cv::Point(r.rectX, r.rectY),
                            cv::Point(r.rectX + (int)(r.velX * 10),
                                      r.rectY + (int)(r.velY * 10)),
                            rectColor, 1);
        }
        char text[160];
        std::snprintf(text, sizeof(text),
                      "%s  prob %.2f  vel (%+.1f, %+.1f)  %d us  %.0f FPS",
                      modeNames[r.mode % 5], r.detectionProbability, r.velX,
                      r.velY, r.processingTimeMks, fps);
        cv::putText(bgr, text, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 0, 0), 3);
        cv::putText(bgr, text, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 255), 1);
        cv::imshow("CvTracker", bgr);

        if (showSurface)
        {
            g_tracker.getImage(2, surf);
            cv::Mat s(surf.height, surf.width, CV_8UC1, surf.data);
            cv::imshow("Correlation surface", s);
        }

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
            break;
        else if (key == '+' || key == '=')
            g_tracker.executeCommand(VTrackerCommand::CHANGE_RECT_SIZE, 8, 8);
        else if (key == '-')
            g_tracker.executeCommand(VTrackerCommand::CHANGE_RECT_SIZE, -8, -8);
        else if (key == 'a')
            g_tracker.setParam(VTrackerParam::RECT_AUTO_SIZE,
                               g_tracker.getParam(VTrackerParam::RECT_AUTO_SIZE) != 0.0f
                                   ? 0.0f : 1.0f);
        else if (key == 'c')
        {
            showSurface = !showSurface;
            if (!showSurface)
                cv::destroyWindow("Correlation surface");
        }
    }
    return 0;
}
