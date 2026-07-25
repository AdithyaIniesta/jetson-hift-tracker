#pragma once
#include <mutex>
#include <string>
#include "VTracker.h"



namespace cr
{
namespace vtracker
{

/// CSRM tracker class.
class CsrmTracker;

/**
 * @brief Correlation Video Tracker Class.
 */
class CvTracker : public VTracker
{
public:

    /**
     * @brief Get string of the library version.
     * @return String of current library version. Format "Major.Minor.Patch".
     */
    static std::string getVersion();

    /**
     * @brief Class constructor.
     */
    CvTracker();

    /**
     * @brief Class destructor.
     */
    ~CvTracker();

    /**
     * @brief Init video tracker. All params will be set.
     * @param params Parameters class.
     * @return TRUE if the video tracker is initialized or FALSE if not.
     */
    bool initVTracker(VTrackerParams& params) override;

    /**
     * @brief Set video tracker param.
     * @param id Parameter ID.
     * @param value Parameter value to set.
     * @return TRUE if the parameter was set or FALSE if not.
     */
    bool setParam(VTrackerParam id, float value) override;

    /**
     * @brief Get video tracker parameter value.
     * @param id Parameter ID.
     * @return Param value or -1 if the parameter not supported.
     */
    float getParam(VTrackerParam id) override;

    /**
     * @brief Get video tracker params (results).
     * @param Output video tracker params structure.
     */
    void getParams(VTrackerParams& params) override;

    /**
     * @brief Execute command.
     * @param id Command ID.
     * @param arg1 First argument. Value depends on command ID.
     * @param arg2 Second argument. Value depends on command ID.
     * @param arg3 Third argument. Value depends on command ID.
     * @return TRUE if the command accepted or FALSE if not.
     */
    bool executeCommand(VTrackerCommand id,
                        float arg1 = 0,
                        float arg2 = 0,
                        float arg3 = 0) override;

    /**
     * @brief Process frame. Must be used for each input video frame.
     * @param frame Source video frame.
     * @return TRUE if video frame was processed or FALSE if not.
     */
    bool processFrame(cr::video::Frame& frame) override;

    /**
     * @brief Get image of internal surfaces.
     * @param type Type of image:
     *             0 - pattern image,
     *             1 - object mask image,
     *             2 - correlation surface image.
     * @param image Output image. Will have size 256x256 pixels and GRAY format.
     */
    void getImage(int type, cr::video::Frame& image) override;

    /**
     * @brief Decode and execute command.
     * @param data Pointer to command data.
     * @param size Size of data.
     * @return TRUE if command decoded and executed or FALSE if not.
     */
    bool decodeAndExecuteCommand(uint8_t* data, int size) override;

private:

    /// Pointer to CSRM tracker object.
    CsrmTracker* m_tracker;
};
}
}
