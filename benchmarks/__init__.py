"""Benchmark harness for the Jetson tracker.

Runs the unmodified tracker binary against public SOT datasets by feeding
frames through v4l2loopback and reading its telemetry over UDP. Metrics
computed offline from the recorded predictions.
"""
