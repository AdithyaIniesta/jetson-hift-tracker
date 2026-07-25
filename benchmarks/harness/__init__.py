"""Harness — the plumbing that drives the tracker binary.

Modules:
    control          — CmdPacket / UDP sender (fires CMD_CAPTURE)
    telemetry        — TelemetryPacket / UDP listener (records per-frame output)
    feeder           — pipes JPEG frames into /dev/video20 via gst-launch
    tracker_process  — spawns and stops the tracker binary as a subprocess
    runner           — orchestrates one sequence: feeder + tracker + control +
                       telemetry + result collection
"""
