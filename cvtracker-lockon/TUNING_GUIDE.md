# CvTracker Handbook — Working Principles, Mathematics, and Tuning

**Applies to build:** `verifier-stable` (commit `6babe8f` — correlation core + EKF layers + DNN appearance verifier). The `feature/vision-robust-lockon` branch adds three opt-in aids on top — see Part 8; with all three flags off it behaves identically except for the two limitation closures noted there.
**Audience:** engineers with partial computer-vision background. Every concept is defined before it is used. Every instruction is exact.

---

## Table of Contents

- [Part 0 — The Big Picture](#part-0--the-big-picture)
- [Part 1 — Layer 1: The Template Matcher (Correlation Core)](#part-1--layer-1-the-template-matcher-correlation-core)
- [Part 2 — Layer 2: The EKF Motion Filter and Mahalanobis Gate](#part-2--layer-2-the-ekf-motion-filter-and-mahalanobis-gate)
- [Part 3 — Layer 3: The DNN Appearance Verifier](#part-3--layer-3-the-dnn-appearance-verifier)
- [Part 4 — State Diagrams: How the Three Layers Cooperate](#part-4--state-diagrams-how-the-three-layers-cooperate)
- [Part 5 — The Tuning Guide](#part-5--the-tuning-guide)
- [Part 6 — Diagnostics and Telemetry](#part-6--diagnostics-and-telemetry)
- [Part 7 — Known Limitations of This Build](#part-7--known-limitations-of-this-build)
- [Glossary](#glossary)

---

## Part 0 — The Big Picture

A visual tracker answers one question every frame: **"Where is my object now?"**
This tracker answers it with three independent subsystems, each using a different kind of evidence:

| Layer | Subsystem | Question it answers | Evidence used |
|---|---|---|---|
| 1 | **Template matcher** (correlation filter) | "Which patch of this frame *looks like* what I was tracking?" | Pixel appearance, short-term memory |
| 2 | **EKF motion filter** + Mahalanobis gate | "Where *can* the object physically be, given how it was moving?" | Motion physics |
| 3 | **DNN verifier** (embedding template bank) | "Is this really *my* object, or a look-alike?" | Learned appearance identity, long-term memory |

**Why three layers?** Because each one fails in different situations, and the failures do not overlap:

- The template matcher drifts on straight edges, learns occluders, and cannot tell two similar cars apart.
- The motion filter is blind to identity — it will smoothly track the wrong object forever.
- The DNN verifier knows *what* the object looks like but nothing about *where* it can be.

Layer 1 **proposes** a position. Layer 2 **filters** it through physics. Layer 3 **audits** the identity. A wrong lock has to fool all three at once.

The tracker also has an operating-mode state machine (FREE / TRACKING / LOST / INERTIAL / STATIC) that decides which layers run each frame. Part 4 diagrams all of it.

---

## Part 1 — Layer 1: The Template Matcher (Correlation Core)

### 1.1 Plain-English Working Principle

When you capture an object, the tracker memorizes a small image of it — the **template**. Every following frame, it slides that template over a region around the last known position (the **search window**) and asks at each offset: "how well does the template match here?" The answer at every offset forms a **response surface** — a heat map whose brightest point (the **peak**) is the object's new position.

Doing this comparison directly, pixel by pixel at every offset, would be very slow. The tracker instead uses the **Fast Fourier Transform (FFT)**: a mathematical tool that converts "compare at every offset" into a single fast multiplication. This is why the tracker is called a *correlation* tracker — correlation is the mathematical name for "slide-and-compare."

The template is not frozen: it is updated a little bit every frame (a **running average**), so it follows gradual appearance change — lighting, slow rotation. That adaptivity is also its greatest danger: if something else enters the box, the template slowly *becomes* that something else. This is called **drift**, and Layers 2 and 3 exist mainly to prevent it.

### 1.2 The Processing Pipeline (Per Frame)

```
 camera frame (any supported YUV/gray format)
        │  take luma (brightness) plane only
        ▼
 crop search window around predicted position, resample to FFT size
        │  (bilinear interpolation; FFT size = next power of 2, 64..512)
        ▼
 PREPROCESS:  log → zero-mean/unit-norm → cosine window
        ▼
 FFT → multiply with learned filter → inverse FFT
        ▼
 response surface  →  find peak  →  sub-pixel refinement
        ▼
 probability  =  f(peak height, peak sharpness)
        ▼
 update template (learning rate scaled by confidence)
```

### 1.3 The Mathematics

**Notation used throughout:** capital letters ($F, G, H$) are FFT-domain quantities; $\odot$ is element-wise multiplication; $\bar{F}$ is the complex conjugate.

#### (a) Preprocessing — why the tracker survives lighting changes

Each cropped window $I$ (pixel values 0–255) is transformed:

1. **Log transform:** $p = \ln(1 + I)$.
   *Why:* compresses bright regions. A shadow that halves all pixel values becomes an additive shift in log space, which the next step removes.
2. **Normalization:** $\hat{p} = (p - \mu)/\sigma$, where $\mu$ and $\sigma$ are the mean and standard deviation over the window.
   *Why:* removes global brightness (offset) and contrast (gain) entirely. After this step, a camera gain change has *no effect* on the features.
3. **Cosine (Hann) window:** $x = \hat{p} \odot h$, where $h$ smoothly falls from 1 at the center to 0 at the edge.
   *Why (two reasons):* the FFT assumes the image wraps around at its edges — the taper prevents false edge artifacts; and it keeps the filter focused on the object rather than the background.

⚠️ **Critical detail with tuning consequences:** the cosine window $h$ does not span the whole search window. It spans only **2.5 × the tracking-rectangle size**, centered in the window. Everything further from the center is multiplied by **zero** — the matcher is mathematically blind to it. Think of it as a *sight bubble* of radius ≈ 1.25 × rect around the search center. During TRACKING this is harmless (the object is re-centered every frame). During LOST it defines how far away the object can be re-found. See Part 7, Limitation 1.

#### (b) The MOSSE filter — what "learning the template" means

The tracker learns a filter $H$ that, when correlated with the object, produces a clean single spike. Formally it minimizes, over all training samples $F_i$ (FFTs of preprocessed windows):

$$\min_H \sum_i \left| F_i \odot \bar{H} - G \right|^2$$

where $G$ is the FFT of the **desired response**: a small 2-D Gaussian bump centered on the object,

$$g(x,y) = \exp\!\left(-\tfrac{1}{2}\left[\tfrac{x^2}{\sigma_x^2}+\tfrac{y^2}{\sigma_y^2}\right]\right),\qquad \sigma = \frac{\text{rect size}}{10}.$$

In words: *"find the filter that answers 'spike here' on the object and 'silence' everywhere else."* The closed-form solution is a simple ratio, maintained as two running sums:

$$H = \frac{A}{B + \lambda}, \qquad A = \sum_i G \odot \bar{F_i}, \qquad B = \sum_i F_i \odot \bar{F_i}$$

$\lambda = 0.01$ is a regularizer that prevents division by ~0 in frequencies where the image has no energy.

**At capture**, the filter is trained on the captured patch **plus 8 perturbed copies** (rotated ±4°, ±8°; scaled ±5%, each at half weight). This bakes in tolerance to small orientation and scale changes from frame one.

**Detection** on a new window with FFT $F$:

$$\text{response} = \text{IFFT}\!\left( F \odot \frac{A}{B+\lambda} \right)$$

The brightest point of the response is the object. Its position is refined to sub-pixel accuracy by fitting a parabola through the peak and its neighbors.

#### (c) Template adaptation — the running average

Every TRACKING frame, the filter sums are blended toward the newest frame:

$$A \leftarrow (1-\eta)A + \eta\,(G \odot \bar{F}), \qquad B \leftarrow (1-\eta)B + \eta\,(F \odot \bar{F})$$

$\eta$ (eta) is the **learning rate** — the single most important tuning number in the tracker. The effective rate is confidence-weighted:

| Detection confidence (prob) | Learning rate actually used |
|---|---|
| prob ≥ re-capture threshold (`CUSTOM_2`) | full $\eta$ (`CUSTOM_3`, default 0.075) |
| loss threshold < prob < `CUSTOM_2` | $\eta \times \text{prob}$ (reduced) |
| prob < loss threshold (`CUSTOM_1`) | 0 — tracker goes LOST, template frozen |
| DNN verifier mismatch pending (Layer 3) | 0 — learning blocked |

*Memory length intuition:* with $\eta = 0.075$, a frame's influence decays by half in $\ln 2 / \eta \approx 9$ frames. The template effectively remembers only the last ~0.5 s at 30 FPS. That short memory is why the DNN verifier (long memory) exists.

#### (d) Detection probability — how the tracker knows it's still locked

Two independent measurements of the response surface are combined:

- **Peak height** — the trained filter answers ≈ 1.0 on the object, ≈ 0 on background:
  $\text{peakTerm} = \text{clamp}(1.6 \cdot \text{peak},\ 0,\ 1)$
- **Peak-to-Sidelobe Ratio (PSR)** — how much the peak stands out from the rest of the surface:
  $\text{PSR} = \dfrac{\text{peak} - \mu_{side}}{\sigma_{side}}$, computed excluding an 11×11 box around the peak;
  $\text{psrTerm} = \text{clamp}\!\left(\frac{\text{PSR}-2}{10},\ 0,\ 1\right)$

$$\boxed{\ \text{detectionProbability} = \text{peakTerm} \times \text{psrTerm}\ }$$

A sharp strong spike → probability near 1. A flat surface (occlusion) or a ridge (ambiguous edge) → probability near 0. This number, compared against `CUSTOM_1` and `CUSTOM_2`, drives every mode transition.

*(Build note: the 11×11 exclusion box is smaller than the response bump of objects larger than ~50 px, which biases PSR — and therefore probability — slightly low for large objects. See Part 7, Limitation 4.)*

#### (e) Scale adaptation

Every 4th TRACKING frame, the tracker re-detects at scales ×0.98 and ×1.02 and keeps whichever gives the strongest peak (with a 2% bias toward "no change" to prevent random walking). The cumulative scale is clamped to [0.2×, 5×] of the capture size. This follows objects that grow/shrink as camera distance changes — slowly, ~0.5%/frame at most.

#### (f) Object-size estimation (for auto-size and telemetry)

A binary mask of the object is estimated by comparing the reference image against background statistics sampled from the window border. The mask's **moments** give the object's center offset and extent:

$$\text{width} \approx \sqrt{12}\,\cdot \text{std}_x \qquad(\text{exact for a uniform block})$$

The estimate is smoothed with an exponential moving average (coefficients 0.85–0.9) and only updated on confident frames — so it moves slowly and never jumps on a single noisy mask.

### 1.4 What Layer 1 Cannot Do (Why Layers 2 and 3 Exist)

1. **The aperture problem:** on a straight edge (road marking, roofline), the response is a *ridge*, not a spike — sharp across the edge, flat along it. The peak can slide along the ridge freely: fast drift.
2. **Distractor blindness:** MOSSE templates use grayscale texture at one scale. Two same-class objects (two white cars) produce nearly equal peaks.
3. **Short memory:** ~9-frame half-life. Anything that sits in the box for a second becomes the new template.

---

## Part 2 — Layer 2: The EKF Motion Filter and Mahalanobis Gate

*Optional. Enabled with `params.ekfEnabled = true` or `setParam(VTrackerParam::ENABLE_EKF, 1.0f)`. Default OFF. Read Part 7, Limitation 2 before enabling in this build.*

### 2.1 Plain-English Working Principle

A **Kalman filter** maintains two things about the object's motion: a **best guess** (position, speed, heading, turn rate) and an **uncertainty** about that guess. Every frame it does two steps:

1. **Predict:** "Given how the object was moving, where should it be now?" — the guess advances, the uncertainty *grows* (the future is less certain).
2. **Update:** "The template matcher says it's here." — the guess moves toward the measurement, the uncertainty *shrinks*.

How far the guess moves toward the measurement depends on the *ratio of the two uncertainties*. A confident measurement pulls hard; a noisy one barely pulls. This ratio-based blending is the entire soul of the filter — every tuning constant just feeds one of the two uncertainties.

The "E" in EKF (**Extended** Kalman Filter) means the motion model is nonlinear — here, motion along circular arcs — and the filter linearizes it at each step.

### 2.2 The Motion Model — CTRV (Constant Turn Rate and Velocity)

The state is five numbers:

$$\mathbf{x} = [\,p_x,\ p_y,\ v,\ \psi,\ \omega\,]^T$$

| Symbol | Meaning | Unit |
|---|---|---|
| $p_x, p_y$ | position | pixels |
| $v$ | speed | pixels/frame |
| $\psi$ (psi) | heading angle | radians |
| $\omega$ (omega) | turn rate | radians/frame |

**Assumption:** between frames the object moves at constant speed along an arc of constant curvature. The prediction over one frame ($\Delta t = 1$):

For $|\omega| > 10^{-4}$ (turning):
$$p_x' = p_x + \frac{v}{\omega}\left[\sin(\psi+\omega) - \sin\psi\right] \qquad p_y' = p_y + \frac{v}{\omega}\left[\cos\psi - \cos(\psi+\omega)\right]$$

For $\omega \approx 0$ (straight-line limit, avoids dividing by zero):
$$p_x' = p_x + v\cos\psi \qquad p_y' = p_y + v\sin\psi$$

with $v' = v$, $\psi' = \psi + \omega$, $\omega' = \omega$. Real objects don't obey this exactly; the deviation is absorbed by **process noise** — two numbers describing how much acceleration you expect:

- $\sigma_a$ (`EKF_SIGMA_A` = 1.5 px/frame²): linear acceleration — speeding up / braking.
- $\sigma_\alpha$ (`EKF_SIGMA_ALPHA` = 0.15 rad/frame²): angular acceleration — how sharply turns change.

These feed the covariance growth in the predict step. **Bigger noise → filter trusts measurements more (follows erratic motion, smooths less). Smaller noise → filter trusts its physics more (smoother, but lags real maneuvers).**

### 2.3 The Measurement Update

The measurement is the template matcher's peak position $\mathbf{z} = (z_x, z_y)$. Its assumed noise is scaled by confidence:

$$R = \frac{R_{base}}{\max(\text{prob},\ 0.05)} \qquad (R_{base} = \texttt{EKF\_R\_BASE} = 4\ \text{px}^2)$$

A confident detection (prob 0.9) is treated as accurate to ~2 px; a weak one (prob 0.1) as accurate only to ~6 px — so weak detections barely move the filter. **This is the continuous hand-shake between Layer 1 and Layer 2: appearance confidence directly sets how much physics yields to appearance.**

### 2.4 The Mahalanobis Gate — physics as a veto

Before accepting a measurement, the filter computes how *surprising* it is. The **innovation** $\mathbf{y} = \mathbf{z} - \mathbf{prediction}$ is normalized by the total expected uncertainty $S$:

$$d^2 = \mathbf{y}^T S^{-1} \mathbf{y}$$

$d^2$ is the squared **Mahalanobis distance** — "how many standard deviations away is this measurement, accounting for direction?" Under correct operation, $d^2$ follows a chi-square distribution with 2 degrees of freedom, so:

$$d^2 > 9.21 \quad\Longleftrightarrow\quad \text{this jump has} < 1\%\ \text{probability of being real motion}$$

When the gate trips, the measurement noise is inflated by $d^2 / 9.21$ (capped at 50×), which crushes the correction — the filter coasts on physics instead of chasing the jump. **This is the straight-edge defense:** an edge-slide produces a confident but physically impossible lateral jump; appearance says "go", physics says "no".

### 2.5 Constants Reference

| Constant | Value | Where | Meaning |
|---|---|---|---|
| `EKF_SIGMA_A` | 1.5 px/frame² | src/CvTracker.cpp | linear accel. noise (↑ = follow erratic motion) |
| `EKF_SIGMA_ALPHA` | 0.15 rad/frame² | src/CvTracker.cpp | angular accel. noise (↑ = tolerate sharp turns) |
| `EKF_R_BASE` | 4.0 px² | src/CvTracker.cpp | base measurement variance (↓ = snap to matcher) |
| `MAHALANOBIS_GATE` | 9.21 | src/Ekf.cpp | χ²(2 dof) 99% gate (↓ = stricter physics) |
| `GATE_MAX_INFLATION` | 50 | src/Ekf.cpp | max damping of a gated measurement |
| Initial covariance | pos 4 px², speed 25, heading π², turn 0.25 | src/Ekf.cpp | uncertainty right after capture |

Suggested presets by platform:

| Scenario | σ_a | σ_α | R_base |
|---|---|---|---|
| Static camera, smooth target | 0.5 | 0.05 | 6.0 |
| Static camera, general (defaults) | 1.5 | 0.15 | 4.0 |
| Handheld / vehicle camera | 6.0 | 0.6 | 2.5 |
| Airborne / gimbal camera | 12.0 | 1.0 | 2.0 |

*Airborne note:* apparent on-screen motion is dominated by **aircraft** motion, not object motion. The CTRV model sees that as violent acceleration — hence the much larger noise values. The proper long-term fix is feeding gimbal/IMU ego-motion compensation upstream.

---

## Part 3 — Layer 3: The DNN Appearance Verifier

*Optional. Requires BOTH: an extractor (`tracker.setFeatureExtractor(...)`) AND the flag (`setParam(ENABLE_DNN_VERIFIER, 1.0f)`). Default OFF. This is the layer that was validated in the field for this build.*

### 3.1 Plain-English Working Principle

A neural network converts an image patch into an **embedding** — a list of ~1280 numbers that acts as a *fingerprint of appearance*. The network is trained so that two images of the **same object** produce nearby fingerprints even under rotation, scale and lighting changes, while **different objects** produce distant ones.

Similarity between two fingerprints $\mathbf{a}, \mathbf{b}$ is the **cosine similarity**:

$$\text{sim} = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} \in [-1, 1]$$

1.0 = identical direction (same appearance), ~0 = unrelated. In practice: same object ≈ 0.65–0.92, look-alike distractor ≈ 0.25–0.55, background ≈ 0.05–0.3 (with a decent embedder).

The verifier keeps a **template bank** — up to 12 fingerprints of *your* object collected over time — and periodically checks: "does what I'm tracking still match any remembered appearance of my object?"

### 3.2 The Template Bank — long-term appearance memory

| Rule | Value | Why |
|---|---|---|
| Seeded at capture | 1 embedding | ground truth of what you clicked; **never evicted** |
| Add condition | sim ≥ `DNN_ACCEPT_THRESHOLD` AND prob ≥ `CUSTOM_2` AND ≥ 30 frames since last add | only high-confidence, temporally spread samples — diversity over redundancy |
| Capacity | 12 | evicts oldest non-seed when full |
| Matching | **max** cosine over all stored templates | the object only has to match ONE remembered appearance (any past orientation/scale/lighting state) |

This is the antidote to Layer 1's 9-frame memory: the bank never decays and cannot be polluted by low-confidence frames.

### 3.3 What the Verifier Does — three interventions

Verification runs every `DNN_VERIFY_INTERVAL` frames (default 6) in TRACKING mode. The crop is 1.25 × the tracking rectangle at the current position, resampled to the network's input size.

1. **Learning freeze** — if sim < `DNN_VETO_THRESHOLD` (default 0.45): template learning rate is forced to 0. Costs nothing if it's a false alarm; prevents the template from absorbing a wrong object. A mismatch **streak counter** starts.
2. **Forced LOST** — 3 consecutive mismatches: the tracker stops pretending, freezes the template, switches to LOST and searches.
3. **Re-capture gate** — in LOST mode, a re-capture candidate must pass **both** the correlation threshold (`CUSTOM_2`) **and** sim ≥ `DNN_ACCEPT_THRESHOLD` (default 0.60). This is the look-alike killer.

### 3.4 Runtime Parameters

| Param | Default | Range | Meaning |
|---|---|---|---|
| `ENABLE_DNN_VERIFIER` | 0 | 0/1 | master switch (needs extractor too) |
| `DNN_VERIFY_INTERVAL` | 6 | ≥ 1 | frames between checks (↓ = faster drift detection, more GPU) |
| `DNN_VETO_THRESHOLD` | 0.45 | 0–1 | below = "not my object" |
| `DNN_ACCEPT_THRESHOLD` | 0.60 | 0–1 | required for re-capture & bank adds |
| `dnnSimilarity` (read-only) | — | 0–1 | last measured similarity — **log this** |

**Invariant that must always hold:** `DNN_ACCEPT_THRESHOLD > DNN_VETO_THRESHOLD` by at least 0.1. This gap is hysteresis — without it the verifier flip-flops.

### 3.5 The Extractor (Jetson / TensorRT)

```cpp
#include <cvtracker/TrtFeatureExtractor.h>
TrtExtractorConfig cfg;
cfg.onnxPath = "/home/user/models/embedder.onnx";  // static 1x3x128x128 input
auto fe = createTrtFeatureExtractor(cfg);           // builds + caches engine (first run: minutes)
if (fe) { tracker.setFeatureExtractor(fe);
          tracker.setParam(VTrackerParam::ENABLE_DNN_VERIFIER, 1.0f); }
```

Inference cost: ~1–3 ms FP16 on Xavier NX; at the default 6-frame cadence, negligible. Without any model, `TinyPatchExtractor` (built-in, dependency-free) exercises the same pipeline with weak (pixel-level) similarity — useful for smoke tests only.

---

## Part 4 — State Diagrams: How the Three Layers Cooperate

### 4.1 Operating-Mode State Machine

```
                        RESET (from any mode)
                              │
                              ▼
                        ┌───────────┐
        ┌──────────────▶│   FREE    │  idle; frames buffered; no detection
        │               └─────┬─────┘
        │                     │ CAPTURE / CAPTURE_PERCENTS
        │                     ▼
        │               ┌───────────┐◀──────────────────────────────┐
        │    ┌─────────▶│ TRACKING  │                               │
        │    │          └─────┬─────┘                               │
        │    │                │                                     │
        │    │   prob < CUSTOM_1 (matcher lost it)                  │
        │    │   OR 3× DNN mismatch (verifier: "wrong object")      │
        │    │   OR SET_LOST_MODE                                   │
        │    │                ▼                                     │
        │    │          ┌───────────┐    prob > CUSTOM_2            │
        │    │          │   LOST    │────AND sim ≥ DNN_ACCEPT───────┘
        │    │          └─────┬─────┘    (automatic re-capture)
        │    │                │
        │    │                │ lostModeFrameCounter ≥ maxFramesInLostMode
        │    └── SET_LOST_    ▼
        │        MODE   ┌───────────┐
        └───────────────│ auto RESET│──────▶ FREE
                        └───────────┘

   Side modes (entered/left only by explicit command):
   INERTIAL — coasts on velocity/EKF prediction, no detection, no learning
   STATIC   — position frozen, no detection, no learning
```

### 4.2 One TRACKING Frame — the three layers in sequence

```
 frame arrives
      │
      ▼
 ┌─────────────────────────────────────────────┐
 │ [EKF] PREDICT (if enabled)                  │  physics prior:
 │ state advances along CTRV arc,              │  "expect the object here"
 │ uncertainty grows by process noise          │
 └──────────────────┬──────────────────────────┘
                    ▼
 ┌─────────────────────────────────────────────┐
 │ [TEMPLATE MATCHER] correlate search window  │  appearance evidence:
 │ → peak position (zx, zy)                    │  "strongest match is here,
 │ → detectionProbability = peak × PSR terms   │   with this confidence"
 └──────────────────┬──────────────────────────┘
                    ▼
              ◇ prob < CUSTOM_1 ?
             yes ──────────────────────▶ enter LOST (template frozen)
              │ no
              ▼
 ┌─────────────────────────────────────────────┐
 │ [EKF GATE + UPDATE] (if enabled)            │  physics veto + fusion:
 │ d² = yᵀS⁻¹y                                 │
 │ ◇ d² > 9.21 ? → inflate R (up to 50×):      │  "that jump is impossible —
 │   measurement damped, filter coasts         │   mostly ignore it"
 │ position/velocity = fused estimate          │
 │ (EKF off → raw peak is used directly)       │
 └──────────────────┬──────────────────────────┘
                    ▼
 ┌─────────────────────────────────────────────┐
 │ [DNN VERIFIER] every Nth frame (if enabled) │  identity audit:
 │ embed 1.25×rect crop at current position    │
 │ sim = max cosine vs template bank           │
 │ ◇ sim < VETO ?  → streak++, LEARNING FROZEN │  "this doesn't look like
 │ ◇ streak ≥ 3 ?  → force LOST                │   my object anymore"
 │ ◇ sim ≥ ACCEPT AND prob ≥ CUSTOM_2 ?        │
 │        → add embedding to bank (≥30 f gap)  │
 └──────────────────┬──────────────────────────┘
                    ▼
 ┌─────────────────────────────────────────────┐
 │ ADAPT: scale probe (every 4th frame),       │
 │ template update at rate η·(confidence),     │
 │ rate = 0 if verifier mismatch pending       │
 └─────────────────────────────────────────────┘
```

### 4.3 One LOST Frame — the re-capture gauntlet

```
 frame arrives (mode = LOST)
      │
      ▼
 lostModeOption?  0: hold position   1/2: advance position by
                                     EKF prediction (or velocity)
      │
      ▼
 [TEMPLATE MATCHER] correlate at search center
      │            (sight limited to ~2.5×rect around center — Part 7.1)
      ▼
 ◇ prob > CUSTOM_2 ?  ──no──▶ ◇ lost too long? → auto-RESET, else wait next frame
      │ yes
      ▼
 ◇ DNN enabled AND bank non-empty ?
      │ yes:  embed candidate crop, sim = max cosine vs bank
      │       ◇ sim < DNN_ACCEPT ? ──yes──▶ REJECT (stay LOST)
      │ no / passed
      ▼
 RE-CAPTURE: mode → TRACKING
 (EKF enabled: measurement fused through the gate — see Part 7.2 caveat)
```

**Reading the diagrams as one sentence:** *the matcher proposes, the EKF gate disposes, the verifier audits — and in LOST mode the matcher and verifier must agree before a re-lock is allowed.*

---

## Part 5 — The Tuning Guide

### 5.0 The Three Fundamental Trade-offs

Every knob in this tracker sets one of exactly three trade-offs. Identify which one your problem lives in, and the right knob follows:

| Trade-off | Question | Knobs |
|---|---|---|
| **A. Plasticity vs Stability** | "Is what I see real change, or corruption?" | `CUSTOM_3` (template), σ_a/σ_α (motion), bank spacing (identity) |
| **B. Sensitivity vs False Alarm** | "When do I declare failure?" | `CUSTOM_1`, Mahalanobis gate, `DNN_VETO_THRESHOLD` + streak |
| **C. Strictness of Re-entry** | "How hard is it to (re)gain the lock?" | `CUSTOM_2`, `DNN_ACCEPT_THRESHOLD` |

Two invariants that must **always** hold (they create hysteresis; violating them causes mode flapping):
- `CUSTOM_2` > `CUSTOM_1` — gap of 3–4× (e.g. 0.1 / 0.4)
- `DNN_ACCEPT_THRESHOLD` > `DNN_VETO_THRESHOLD` — gap ≥ 0.1

### 5.1 Parameter Reference (complete)

| Parameter | Default | Valid range | Layer | One-line meaning |
|---|---|---|---|---|
| `rectWidth/Height` | 72 | ≥ 4 | 1 | template size — match the object tightly (80–100% of its bounding box) |
| `searchWindowWidth/Height` | 256 | ≥ 16 | 1 | search area per frame (but see sight bubble, Part 7.1) |
| `CUSTOM_1` loss threshold | 0.1 | (0, 1) | 1 | below → LOST; template freezes |
| `CUSTOM_2` re-capture threshold | 0.4 | (0, 1) | 1 | required to (re)lock from LOST; also full-η learning bar |
| `CUSTOM_3` learning rate | 0.075 | (0, 1) | 1 | template memory half-life ≈ 0.7/η frames |
| `lostModeOption` | 0 | 0/1/2 | — | when lost: 0 freeze, 1 predict, 2 predict+reset at edge |
| `maxFramesInLostMode` | 128 | ≥ 1 | — | patience before auto-reset (≈ seconds × FPS) |
| `frameBufferSize` | 128 | ≥ 1 | — | stop-frame capture depth (RAM: ~2 MB/frame @1080p) |
| `rectAutoSize/AutoPosition` | off | 0/1 | 1 | continuous rect adaptation from the object-size estimate |
| `multipleThreads` | off | 0/1 | 1 | multithreaded CPU FFT — enable on ≥4 cores |
| `ENABLE_EKF` | 0 | 0/1 | 2 | motion filter + gate (read Part 7.2 first) |
| `ENABLE_DNN_VERIFIER` | 0 | 0/1 | 3 | identity audit (needs extractor) |
| `DNN_VERIFY_INTERVAL` | 6 | ≥ 1 | 3 | frames between identity checks |
| `DNN_VETO_THRESHOLD` | 0.45 | 0–1 | 3 | below = mismatch (freeze learning; ×3 → LOST) |
| `DNN_ACCEPT_THRESHOLD` | 0.60 | 0–1 | 3 | re-capture / bank-add bar |

### 5.2 The Validated Baseline Configuration (start here)

This is the configuration field-validated on airborne footage with this build:

```cpp
p.rectWidth  = <object size>;  p.rectHeight = <object size>;   // tight!
p.searchWindowWidth = 256;     p.searchWindowHeight = 256;
p.lostModeOption = 0;          p.maxFramesInLostMode = 90;
p.multipleThreads = true;
// thresholds
tracker.setParam(VTrackerParam::CUSTOM_1, 0.20f);
tracker.setParam(VTrackerParam::CUSTOM_2, 0.55f);
tracker.setParam(VTrackerParam::CUSTOM_3, 0.075f);
// Layer 3 ON (the validated combination for this build)
tracker.setFeatureExtractor(fe);
tracker.setParam(VTrackerParam::ENABLE_DNN_VERIFIER, 1.0f);
// Layer 2 OFF in this build unless you accept the Part 7.2 caveat
tracker.setParam(VTrackerParam::ENABLE_EKF, 0.0f);
```

### 5.3 Systematic Tuning Procedure (do these in order, one change at a time)

1. **Geometry first.** Set rect to the object's tight bounding box; search window 2–4× rect. Capture; confirm lock. Fix this before touching anything else — a loose rect invalidates every other step.
2. **Confirm real-time budget.** Average `processingTimeMks` over 100 frames must be < 1,000,000/FPS (33,333 µs at 30 FPS). Too slow → smaller search window, `multipleThreads=1`, or CUDA build.
3. **Place `CUSTOM_1`.** Log `detectionProbability` on a normal tracking clip. Set `CUSTOM_1` ≈ 25% of the typical value (e.g. typical 0.7 → set ~0.15–0.20). It must never trip during clean tracking, and must trip within a few frames of true occlusion.
4. **Place `CUSTOM_2`.** Cover/uncover the object. Watch the probability the matcher reports when it re-sees the object; set `CUSTOM_2` just below that, and well above `CUSTOM_1`.
5. **Tune `CUSTOM_3`.** Drifts onto background → halve it. Loses the object on rotation/appearance change → raise by 50%. Note: **with the verifier enabled you can afford a higher η** (0.1 instead of 0.05) — drift is now caught by Layer 3, so the template can adapt faster.
6. **Place the DNN thresholds by measurement, never by guessing.** See 5.5.
7. **Adversarial pass.** Test: motion blur, partial/full occlusion, lighting jump, similar objects nearby. For each failure, use the scenario table (5.4). Re-test previous cases after every change — tuning is not monotonic.

### 5.4 Scenario Table (symptom → ordered fixes)

| # | Symptom | Root cause | Fixes, in order of impact |
|---|---|---|---|
| A | Loses object immediately after capture | rect too large (background in template) | 1) shrink rect 30–50% 2) lower `CUSTOM_1` to 0.05 3) check `getImage(0)` — object should fill the reference |
| B | Drifts onto background over time | η too high for scene | 1) `CUSTOM_3` → 0.03 2) tighten rect 3) **enable verifier** (it freezes learning on mismatch) |
| C | Loses on fast rotation/turn | template can't keep up | 1) `CUSTOM_3` → 0.10–0.15 2) `CUSTOM_1` → 0.05 3) verifier on + `DNN_VETO` → 0.35 (tolerate turned views) |
| D | Loses on fast motion | object exits search window between frames | 1) double search window 2) `lostModeOption=1` 3) higher camera FPS |
| E | Stays LOST though object is visible | object outside the **sight bubble** (Part 7.1) — not the threshold | 1) understand: in this build re-capture only works within ~1.25×rect of the search center 2) `lostModeOption=1` so the center follows the predicted path 3) re-issue `SET_SEARCH_WINDOW_POSITION` onto the object **each frame** (it is one-shot in this build) 4) manual re-CAPTURE |
| F | Flaps TRACKING↔LOST every few frames | hysteresis violated | enforce `CUSTOM_2 ≥ 2×CUSTOM_1`; check DNN gap ≥ 0.1 |
| G | Re-locks onto wrong object | correlation can't separate same-class objects | 1) **verifier on** — this is its purpose 2) `DNN_ACCEPT` → 0.65–0.70 3) `CUSTOM_2` → 0.5–0.6 4) `lostModeOption=2` |
| H | Fails after long occlusion | patience/memory | 1) `maxFramesInLostMode` = occlusion-seconds × FPS × 1.5 2) `lostModeOption=1` 3) verifier on (bank remembers pre-occlusion appearance) |
| I | Object scale changes (approach/recede) | scale probe too slow alone | 1) `rectAutoSize=1` 2) periodic `ADJUST_RECT_SIZE` 3) note Part 7.3 if verifier similarity decays with zoom |
| J | Erratic during aircraft maneuvers | ego-motion violates CTRV | 1) EKF off, or airborne noise preset (Part 2.5) 2) long-term: IMU ego-motion compensation upstream |
| K | Too slow on target hardware | FFT budget | 1) search window 128×128 (~4× faster) 2) `multipleThreads=1` 3) CUDA build 4) lower input resolution |
| L | Verifier false-vetoes on hard viewing angles | thresholds vs distributions | 1) `DNN_VETO` → 0.35 2) verify streak is 3 (compile-time) 3) shrink `DNN_BANK_MIN_SPACING` so turned views enter the bank sooner |
| M | Verifier lets distractors through | embedder can't separate the class | 1) `DNN_ACCEPT` → 0.70 2) smaller model input if objects are tiny 3) re-ID backbone (osnet) 4) fine-tune on your footage — the real fix |

### 5.5 Placing the DNN Thresholds — the measurement method

**You cannot pick these numbers a priori.** Measure, then place:

1. Log `dnnSimilarity` while locked on the true object through hard moments (turns, shadows). Note the **minimum** — call it $m_{obj}$.
2. Deliberately mis-lock on a distractor; log again. Note the **maximum** — call it $M_{dis}$.
3. If $m_{obj} > M_{dis}$ (clean gap):
   - `DNN_VETO_THRESHOLD` = just below $m_{obj}$ (e.g. $m_{obj}=0.55$ → 0.45–0.50)
   - `DNN_ACCEPT_THRESHOLD` = just above $M_{dis}$ (e.g. $M_{dis}=0.55$ → 0.62–0.65)
4. If the distributions **overlap**, no threshold works — the model is the problem: smaller input for tiny objects → re-ID backbone → fine-tune on your recordings (in that order of effort).

### 5.6 Presets

**Vehicle from airborne camera (validated config of this build)** — see 5.2.

**Person (surveillance, static camera):** rect 64×128, window 256×384, `lostModeOption=1`, `maxFrames=150`, `CUSTOM_1=0.05`, `CUSTOM_2=0.35`, `CUSTOM_3=0.12`, autoSize+autoPosition on, verifier on.

**Small distant target (telephoto):** rect 24×24, window 192×192, `CUSTOM_1=0.04`, `CUSTOM_2=0.25`, `CUSTOM_3=0.05`; verifier with 64-px model input.

**Thermal:** `type=1`, rect 64, window 256, `CUSTOM_1=0.08`, `CUSTOM_2=0.40`, `CUSTOM_3=0.10`.

---

## Part 6 — Diagnostics and Telemetry

### 6.1 The Three Debug Images (`getImage`)

| type | Image | Healthy looks like | Unhealthy looks like |
|---|---|---|---|
| 0 | Reference (running-average template) | your object, sharp | blurred blend of object + background = drift in progress |
| 1 | Object mask | clean blob outlining the object | scattered noise = size estimate unreliable |
| 2 | Correlation response surface | ONE sharp bright spot | flat (no lock) / a ridge (edge — aperture problem) / multiple spots (distractors) |

### 6.2 The Telemetry Triplet — log these every frame

```
mode, detectionProbability, dnnSimilarity
```

After any test flight, three checks:
1. **Probability histogram while TRACKING:** bulk above `CUSTOM_2`, dips landing between `CUSTOM_1` and `CUSTOM_2`, excursions below `CUSTOM_1` only at true occlusions. Anything else = thresholds misplaced (5.3 steps 3–4).
2. **Similarity histograms** (on-object / drift / rejected re-locks): clean gap = healthy; overlap = model problem (5.5).
3. **Mode timeline:** rapid TRACKING↔LOST alternation = hysteresis violation (scenario F) or the Part 7.2 EKF caveat.

### 6.3 Quick Health Checks

- `processingTimeMks` — per-frame cost; budget = 10⁶/FPS.
- `params.velX/velY` — with EKF on, these are the filtered velocity; wild swings while visually smooth = process noise too high.
- `frameId - processedFrameId` — should be 0 steady-state; growing = you are dropping frames upstream (never drop; the ring buffer and stop-frame depend on continuity).

---

## Part 7 — Known Limitations of This Build

These are honest, verified constraints of `verifier-stable`. Improved re-capture machinery exists on the `feature/dnn-verifier` branch but is **not** in this build (it traded these problems for distractor-grab problems and needs offline replay validation before another field attempt).

### 7.1 The re-capture sight bubble
The cosine window limits detection to **~2.5 × rect around the search center** (Part 1.3a). In LOST mode with `lostModeOption=0`, the center is frozen at the loss point: an object re-appearing farther than ~1.25×rect away is **mathematically invisible** — no threshold change can find it. The `searchWindowWidth/Height` you configure is honored during TRACKING recentering but NOT fully searchable during LOST.
**Practical rules:** use `lostModeOption=1` so the center follows the prediction; expect automatic re-capture only near the predicted path; otherwise re-capture manually. `SET_SEARCH_WINDOW_POSITION` helps but is **one-shot per frame** in this build — issue it repeatedly to hold the search somewhere.

### 7.2 EKF + re-capture interaction
On re-capture, this build fuses the re-detection through the ordinary gated EKF update. After a loss with `lostModeOption=0`, the filter's uncertainty is still small, so a legitimate re-detection some distance away trips the Mahalanobis gate and gets damped — the tracker can flicker TRACKING↔LOST at the *old* position.
**Practical rules:** if automatic re-capture matters to your mission, run `ENABLE_EKF=0` (the validated config). If you enable the EKF for its smoothing, use `lostModeOption=1` and treat re-capture as degraded.

### 7.3 Verifier crop does not track scale
The embedding crop is 1.25 × the tracking rectangle. With `rectAutoSize=0`, the rectangle stays at its capture size while the true object size follows the scale probe — after large zoom/altitude change the crop frames the object differently than the bank seed and `dnnSimilarity` decays without any real appearance change.
**Practical rules:** enable `rectAutoSize`, or issue `ADJUST_RECT_SIZE` after big scale changes, or re-capture; treat slowly decaying similarity during approach/climb as this artifact before blaming the model.

### 7.4 PSR bias for large objects
The PSR sidelobe exclusion is a fixed 11×11 box; objects larger than ~50 px have a wider response bump, so their PSR — and probability — read slightly low.
**Practical rule:** for large objects, place `CUSTOM_1/CUSTOM_2` using measured probabilities (5.3), not the defaults.

### 7.5 Object-size estimate at extreme scale
The mask-moment neighborhood grows with the scale factor, pulling background into the size estimate when scale is far from 1.
**Practical rule:** treat `objectWidth/Height` as approximate after large scale drift; `ADJUST_RECT_SIZE` + retrain resets the baseline.

---

## Part 8 — Vision-Only Robustness Aids (`feature/vision-robust-lockon` branch)

Three opt-in flags for unstabilized cameras (boresight fixed-wing mounts),
requiring **no IMU and no lens calibration** — all measurements are made
from the video itself, so changing focal length costs nothing.

| Flag | What it does | When to enable | Cost |
|---|---|---|---|
| `ENABLE_GMC` | Whole-frame phase correlation measures the camera-induced scene shift each frame and moves all stored positions with the scene — a violent jolt never appears as object motion. Telemetry: `gmcShiftX/Y`. | Always, on any unstabilized platform. This is the aid that addresses jolt-induced loss directly. | ~1–2 ms/frame |
| `ENABLE_DNN_ARBITRATION` | On ambiguous frames (multiple response peaks), embeds each candidate and locks onto the one that matches the template bank — not merely the brightest. Needs verifier. | Cluttered scenes, look-alike traffic | 2–3 inferences, ambiguous frames only |
| `ENABLE_DNN_REACQUISITION` | In LOST, sweeps embedding crops across the WHOLE search window (escapes the ~2.5×rect sight bubble); a bank match cues a confirming correlation detect. Needs verifier. | Whenever re-capture matters | ~4 inferences per LOST frame |

**Suggested rollout for A/B testing** (each flag independently toggleable
at runtime through your existing `setParam` path):
1. Baseline flight: all three off (identical to `verifier-stable` behavior).
2. Enable `ENABLE_GMC`; verify `gmcShiftX/Y` telemetry tracks known camera
   motion and jolt survival improves.
3. Add `ENABLE_DNN_REACQUISITION`; verify post-jolt/post-occlusion
   re-locks.
4. Add `ENABLE_DNN_ARBITRATION` last; watch for any lock stolen by a
   nearby object (raise `DNN_ACCEPT_THRESHOLD` if seen).

**Limitation closures shipped on this branch (always on):** Known
Limitation 7.2 (EKF re-capture deadlock — the filter is now re-seeded at
the re-detection) and 7.3 (verifier crop now follows `baseRect × scale`,
so `dnnSimilarity` no longer decays with zoom/altitude change). Limitation
7.1 (sight bubble) is bypassed by `ENABLE_DNN_REACQUISITION` when enabled;
7.4 and 7.5 remain.

**GMC limits to know:** translation only (aircraft roll is absorbed by the
matcher's rotation tolerance, not measured); rejects measurements on
featureless scenes (open water, uniform cloud) and falls back to
uncompensated behavior; requires contiguous `frameId`s (drop-free feed).

---

## Glossary

| Term | Meaning |
|---|---|
| **Luma** | The brightness channel of a video frame (the "Y" in YUV). The tracker uses only this. |
| **Template** | The learned image pattern of the object being tracked. |
| **Correlation** | Sliding a template over an image and scoring the match at every offset. |
| **FFT** | Fast Fourier Transform — converts correlation into fast multiplication. |
| **Response surface** | The heat map of match scores over the search window; its peak is the detected position. |
| **PSR** | Peak-to-Sidelobe Ratio — how much the peak stands out above the rest of the surface; a sharpness measure. |
| **Hann / cosine window** | A smooth weighting that fades to zero at the edges; prevents FFT artifacts and focuses the filter. |
| **MOSSE** | Minimum Output Sum of Squared Error — the classic formulation of the adaptive correlation filter used here. |
| **Learning rate (η)** | Fraction of the template replaced by the newest frame each update; sets memory length. |
| **Kalman filter** | An estimator that maintains a best guess *and its uncertainty*, blending predictions with measurements by their relative confidence. |
| **EKF** | Extended Kalman Filter — a Kalman filter for nonlinear motion models, linearized each step. |
| **CTRV** | Constant Turn Rate and Velocity — motion model: objects move along circular arcs. |
| **Covariance** | The filter's uncertainty, expressed as variances and correlations between state variables. |
| **Innovation** | Measurement minus prediction — the "surprise" in a measurement. |
| **Mahalanobis distance** | Distance in units of standard deviations, accounting for direction — "how surprising is this measurement?" |
| **Embedding** | A vector "fingerprint" of an image patch produced by a neural network; similar objects → similar vectors. |
| **Cosine similarity** | The angle-based similarity of two vectors (1 = same direction). |
| **Template bank** | The verifier's stored collection of embeddings of the tracked object across time. |
| **Drift** | The template gradually becoming an image of the wrong thing (background/occluder/distractor). |
| **Aperture problem** | On a straight edge, motion along the edge is unobservable — the response is a ridge, not a peak. |
| **Hysteresis** | Requiring a higher bar to enter a state than to stay in it; prevents rapid flip-flopping. |
