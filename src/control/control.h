#pragma once
#include <linux/videodev2.h>

#include "common/globals.h"

using namespace cr::vtracker;

int uartOpen(const char *device, int baudrate);
void uartSendAngle(int fd, int mode, float angle_x, float angle_y,
                   float det_prob, int pixel_x, int pixel_y, int confirmed);
// ── Camera parameter control (V4L2 ioctl, no subprocess) ─────
// WHY: device path parameter means these work for either camera —
// caller passes leftDev or rightDev depending on which needs tuning.
bool setCameraControl(const char *device, uint32_t ctrl_id, int32_t value);
int32_t getCameraControl(const char *device, uint32_t ctrl_id);

void sendAck(uint32_t cmdType, uint32_t paramId, float value, bool success);
void controlThread(int port, int camera_id, const char *device_path);
bool saveParams();
bool loadParams();
