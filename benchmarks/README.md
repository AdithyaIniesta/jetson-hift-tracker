# Benchmark harness (residual)

The v4l2loopback feeder path and the scoring frontends (`run_benchmark.py`,
`manual_score.py`) have been removed. What remains under `benchmarks/`:

- `datasets/` — dataset adapters (UAV123 loader). Useful for enumerating
  sequences and their init bboxes.
- `harness/runner.py` — a Python driver that spawns the tracker with
  `TRACKER_FILE_SRC_LEFT` pointing at a JPG folder, sends the init
  CAPTURE, and collects telemetry into `predictions.jsonl`. Not
  currently wired to a CLI; import it if you want to script runs.
- `harness/control.py`, `telemetry.py`, `tracker_process.py` — the
  UDP + subprocess plumbing used by `runner.py`.

## The recommended way to replay a dataset

Use `run_dataset.sh` at the repo root. It launches the tracker binary
directly with the file-source env var set, streams annotated video to
the GUI, and lets the operator click the template. Simpler than
scripting, matches production behaviour.

```bash
./run_dataset.sh \
    /home/nvidia/Downloads/Dataset_UAV123/UAV123/data_seq/UAV123/bike1 \
    192.168.0.100
```

## Adding a new dataset (for the Python driver)

Drop an adapter under `benchmarks/datasets/<name>.py` inheriting from
`benchmarks.datasets.base.DatasetAdapter`, and set its `direct_source`
tuple `(frames_dir, "%06d.jpg", start_index)`.
