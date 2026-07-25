#pragma once
#include <cstdint>
#include "Frame.h"

namespace cr {
namespace vtracker {

/// Video tracker commands.
enum class VTrackerCommand
{
    /// Object capture. Arguments:
    /// arg1 - Capture rectangle center X coordinate. -1 - default coordinate.
    /// arg2 - Capture rectangle center Y coordinate. -1 - default coordinate.
    /// arg3 - frame ID. -1 - Capture on last frame.
    CAPTURE = 1,
    /// Object capture command. Arguments:
    /// arg1 - Capture rectangle center X coordinate, percent of frame width.
    /// arg2 - Capture rectangle center Y coordinate, percent of frame height.
    CAPTURE_PERCENTS,
    /// Reset command. No arguments.
    RESET,
    /// Set INERTIAL mode. No arguments.
    SET_INERTIAL_MODE,
    /// Set LOST mode. No arguments.
    SET_LOST_MODE,
    /// Set STATIC mode. No arguments.
    SET_STATIC_MODE,
    /// Adjust tracking rectangle size automatically once. No arguments.
    ADJUST_RECT_SIZE,
    /// Adjust tracking rectangle position automatically once. No arguments.
    ADJUST_RECT_POSITION,
    /// Move tracking rectangle (change position). Arguments:
    /// arg1 - add to X coordinate, pixels. 0 - no change.
    /// arg2 - add to Y coordinate, pixels. 0 - no change.
    MOVE_RECT,
    /// Set tracking rectangle position in FREE mode. Arguments:
    /// arg1 - Rectangle center X coordinate.
    /// arg2 - Rectangle center Y coordinate.
    SET_RECT_POSITION,
    /// Set tracking rectangle position in FREE mode in percents of frame size.
    SET_RECT_POSITION_PERCENTS,
    /// Move search window (change position). Arguments:
    /// arg1 - add to X coordinate, pixels. arg2 - add to Y coordinate, pixels.
    MOVE_SEARCH_WINDOW,
    /// Set search window position. Arguments:
    /// arg1 - Search window center X coordinate.
    /// arg2 - Search window center Y coordinate.
    SET_SEARCH_WINDOW_POSITION,
    /// Set search window position in percent of frame size.
    SET_SEARCH_WINDOW_POSITION_PERCENTS,
    /// Change tracking rectangle size. Arguments:
    /// arg1 - horizontal size add, pixels. arg2 - vertical size add, pixels.
    CHANGE_RECT_SIZE
};

/// Video tracker parameters. Total 23 elements.
enum class VTrackerParam
{
    SEARCH_WINDOW_WIDTH = 1,
    SEARCH_WINDOW_HEIGHT,
    RECT_WIDTH,
    RECT_HEIGHT,
    LOST_MODE_OPTION,
    FRAME_BUFFER_SIZE,
    MAX_FRAMES_IN_LOST_MODE,
    RECT_AUTO_SIZE,
    RECT_AUTO_POSITION,
    MULTIPLE_THREADS,
    NUM_CHANNELS,
    TYPE,
    CUSTOM_1,               // 13
    CUSTOM_2,               // 14
    CUSTOM_3,               // 15
    
    /// EKF process noise -- linear acceleration std deviation (px/frame^2).
    EKF_SIGMA_A,            // 16
    /// EKF process noise -- angular acceleration std deviation (rad/frame^2).
    EKF_SIGMA_ALPHA,        // 17
    /// EKF measurement noise base -- position measurement std deviation (px).
    EKF_R_BASE,             // 18

    /// Activation Switches
    ENABLE_EKF,             // 19
    ENABLE_DNN_VERIFIER,    // 20

    /// Verifier cadence in TRACKING mode, frames between verifications
    /// (default 6). Lower = faster drift detection, more inference load.
    DNN_VERIFY_INTERVAL,    // 21
    /// Cosine similarity below which the verifier votes "not the captured
    /// object" (default 0.45). Sustained mismatch blocks pattern learning
    /// and forces LOST mode.
    DNN_VETO_THRESHOLD,     // 22
    /// Cosine similarity required to accept a LOST-mode re-capture and to
    /// add new templates to the bank (default 0.60).
    DNN_ACCEPT_THRESHOLD,   // 23
};

struct VTrackerParamsMask;

/// Video tracker params and tracking results.
class VTrackerParams
{
public:
    /// Tracker mode index: 0 - FREE, 1 - TRACKING, 2 - LOST, 3 - INERTIAL,
    /// 4 - STATIC. Set by video tracker.
    int mode{0};
    /// Tracking rectangle horizontal center position.
    int rectX{0};
    /// Tracking rectangle vertical center position.
    int rectY{0};
    /// Tracking rectangle width, pixels.
    int rectWidth{72};
    /// Tracking rectangle height, pixels.
    int rectHeight{72};
    /// Estimated horizontal position of object center, pixels.
    int objectX{0};
    /// Estimated vertical position of object center, pixels.
    int objectY{0};
    /// Estimated object width, pixels.
    int objectWidth{72};
    /// Estimated object height, pixels.
    int objectHeight{72};
    /// Frame counter in LOST mode.
    int lostModeFrameCounter{0};
    /// Counter for processed frames after capture object.
    int frameCounter{0};
    /// Width of processed video frame.
    int frameWidth{0};
    /// Height of processed video frame.
    int frameHeight{0};
    /// Width of search window, pixels. Set by user.
    int searchWindowWidth{256};
    /// Height of search window, pixels. Set by user.
    int searchWindowHeight{256};
    /// Horizontal position of search window center (for next video frame).
    int searchWindowX{0};
    /// Vertical position of search window center (for next video frame).
    int searchWindowY{0};
    /// Option for LOST mode (see VTrackerParam::LOST_MODE_OPTION).
    int lostModeOption{0};
    /// Size of frame buffer (number of frames to store).
    int frameBufferSize{128};
    /// Maximum number of frames in LOST mode to auto reset of algorithm.
    int maxFramesInLostMode{128};
    /// ID of last processed frame in frame buffer.
    int processedFrameId{0};
    /// ID of last added frame to frame buffer.
    int frameId{0};
    /// Horizontal velocity of object on video frames (pixel/frame).
    float velX{0.0f};
    /// Vertical velocity of object on video frames (pixel/frame).
    float velY{0.0f};
    /// Estimated probability of object detection [0..1].
    float detectionProbability{0.0f};
    /// Use tracking rectangle auto size flag.
    bool rectAutoSize{false};
    /// Use tracking rectangle auto position flag.
    bool rectAutoPosition{false};
    /// Use multiple threads for calculations.
    bool multipleThreads{false};
    /// Number of channels for processing.
    int numChannels{2};
    /// Camera type: 0 - daylight, 1 - thermal.
    int type{0};
    /// Processing time for last frame, mks.
    int processingTimeMks{0};
    /// Custom parameter 1 (loss threshold, <= 0 - default 0.1).
    float custom1{0.0f};
    /// Custom parameter 2 (re-capture threshold, <= 0 - default 0.4).
    float custom2{0.0f};
    /// Custom parameter 3 (learning rate, <= 0 - default 0.075).
    float custom3{0.0f};
    // ── ADDED LIVE FIELD STORAGE ────────────────────────────────
    /// EKF process noise -- linear acceleration standard deviation.
    float ekfSigmaA{1.5f};     // Mapped to ID 16
    /// EKF process noise -- angular acceleration standard deviation.
    float ekfSigmaAlpha{0.15f}; // Mapped to ID 17
    /// EKF measurement noise base standard deviation.
    float ekfRBase{4.0f};       // Mapped to ID 18
    // ────────────────────────────────────────────────────────────
    /// EKF (CTRV) predictive filter enable. false - raw correlation peak
    /// (default), true - EKF-smoothed output and EKF-driven LOST-mode
    /// extrapolation. See VTrackerParam::ENABLE_EKF.
    bool ekfEnabled{false};
    /// DNN embedding verifier enable (see VTrackerParam::ENABLE_DNN_VERIFIER).
    /// Also requires CvTracker::setFeatureExtractor().
    bool dnnVerifierEnabled{false};
    /// Verifier cadence, frames between verifications in TRACKING mode.
    int dnnVerifyInterval{6};
    /// Similarity veto threshold [0..1] (below = appearance mismatch).
    float dnnVetoThreshold{0.45f};
    /// Similarity accept threshold [0..1] (re-capture / bank add).
    float dnnAcceptThreshold{0.60f};
    /// Last embedding similarity measured by the verifier [0..1].
    /// Set by video tracker (0 until the first verification).
    float dnnSimilarity{0.0f};

    /// Encode (serialize) params. data - buffer, bufferSize - buffer size,
    /// size - output encoded size, mask - optional fields mask.
    bool encode(uint8_t* data, int bufferSize, int& size,
                VTrackerParamsMask* mask = nullptr);

    /// Decode (deserialize) params.
    bool decode(uint8_t* data, int dataSize);
};

/// Fields mask for VTrackerParams encoding.
typedef struct VTrackerParamsMask
{
    bool mode{true};
    bool rectX{true};
    bool rectY{true};
    bool rectWidth{true};
    bool rectHeight{true};
    bool objectX{true};
    bool objectY{true};
    bool objectWidth{true};
    bool objectHeight{true};
    bool lostModeFrameCounter{true};
    bool frameCounter{true};
    bool frameWidth{true};
    bool frameHeight{true};
    bool searchWindowWidth{true};
    bool searchWindowHeight{true};
    bool searchWindowX{true};
    bool searchWindowY{true};
    bool lostModeOption{true};
    bool frameBufferSize{true};
    bool maxFramesInLostMode{true};
    bool processedFrameId{true};
    bool frameId{true};
    bool velX{true};
    bool velY{true};
    bool detectionProbability{true};
    bool rectAutoSize{true};
    bool rectAutoPosition{true};
    bool multipleThreads{true};
    bool numChannels{true};
    bool type{true};
    bool processingTimeMks{true};
    bool custom1{true};
    bool custom2{true};
    bool custom3{true};
    // ── ADDED MASK FIELDS ───────────────────────────────────────
    bool ekfSigmaA{true};
    bool ekfSigmaAlpha{true};
    bool ekfRBase{true};
    // ────────────────────────────────────────────────────────────
    bool ekfEnabled{true};
    bool dnnVerifierEnabled{true};
    bool dnnVerifyInterval{true};
    bool dnnVetoThreshold{true};
    bool dnnAcceptThreshold{true};
    bool dnnSimilarity{true};
} VTrackerParamsMask;

/// Video tracker interface class.
class VTracker
{
public:
    virtual ~VTracker() = default;

    /// Initialize the video tracker with a set of parameters.
    virtual bool initVTracker(VTrackerParams& params) = 0;

    /// Set a new video tracker parameter value. Thread-safe. The new value
    /// is applied when processing the next video frame.
    virtual bool setParam(VTrackerParam id, float value) = 0;

    /// Get a video tracker parameter value. Thread-safe.
    virtual float getParam(VTrackerParam id) = 0;

    /// Get all params and tracking results. Thread-safe.
    virtual void getParams(VTrackerParams& params) = 0;

    /// Execute video tracker command. Thread-safe.
    virtual bool executeCommand(VTrackerCommand id,
                                float arg1 = 0.0f,
                                float arg2 = 0.0f,
                                float arg3 = 0.0f) = 0;

    /// Process video frame. To get actual tracking results use getParams().
    virtual bool processFrame(cr::video::Frame& frame) = 0;

    /// Get image of internal matrices: 0 - reference image of object,
    /// 1 - object mask image, 2 - correlation surface image.
    virtual void getImage(int type, cr::video::Frame& image) = 0;

    /// Decode and execute command encoded by encodeSetParamCommand() or
    /// encodeCommand(). Thread-safe.
    virtual bool decodeAndExecuteCommand(uint8_t* data, int size);

    /// Encode SET_PARAM command. data - buffer (>= 11 bytes), size - output
    /// encoded size.
    static void encodeSetParamCommand(uint8_t* data, int& size,
                                      VTrackerParam id, float value);

    /// Encode action command. data - buffer (>= 19 bytes), size - output size.
    static void encodeCommand(uint8_t* data, int& size, VTrackerCommand id,
                              float arg1 = 0.0f, float arg2 = 0.0f,
                              float arg3 = 0.0f);

    /// Decode command. Returns: 0 - action command decoded, 1 - set param
    /// command decoded, -1 - error.
    static int decodeCommand(uint8_t* data, int size,
                             VTrackerParam& paramId,
                             VTrackerCommand& commandId,
                             float& value1, float& value2, float& value3);
};

} // namespace vtracker
} // namespace cr
