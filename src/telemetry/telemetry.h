#pragma once
#include "common/globals.h"

void sendTelemetry(const cr::vtracker::VTrackerParams &p, int frameId,
                   float angleX, float angleY, int pixOffX, int pixOffY,
                   int frameW, int frameH, int telemPort);