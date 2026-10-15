# Navigation options

Issue #45. The [PRD](prd.md#navigation) defers house-scale localisation until the core interaction loop has proven useful, and lists "Do not introduce ROS or LiDAR" until then. This page evaluates the options for this robot and records a recommendation. **Step 1 of the recommendation (vision room recognition) is built (#53, [behaviours](behaviours.md#room-recognition)); the rest is not.** Figures marked *estimate* are not measured on the robot; the "Open questions" section lists what to measure first.

## 1. What Laika can and cannot do today

Hardware relevant to navigation: a Raspberry Pi 4 with 1 GB RAM, a PiCar-X with two rear DC motors driven by PWM only, front steering, **one ultrasonic sensor pointing ahead**, a **3-channel grayscale sensor** under the front (used for cliff detection), and a camera (5 MP, as supplied for the PiCar-X; the vendor docs I could open do not state it) on a pan/tilt head. AI inference is off-board through the vision model.

- **No odometry.** The reviewed [`picarx.py`](../references/sunfounder/picar-x/picarx/picarx.py) has `forward(speed)`, steering, `get_distance()` and `get_grayscale_data()` and nothing else for motion feedback; no encoder or IMU appears anywhere in `references/`, and the [vendor documentation](https://docs.sunfounder.com/projects/picar-x-v20/en/latest/) does not mention encoders either. Speed is a control value, not a measured one ([hardware reference](hardware-reference.md)). Wheel slip depends on the floor, so "forward 800 ms" has no known distance or heading.
- **Exploration is a reactive wander** ([behaviours](behaviours.md#exploration)): short forward steps, a back-off arc on an obstacle or cliff, a head scan every few steps. It has no goal and no memory of where it has been.
- **Rooms come from the vision model.** (Written before #53; now each frame carries a room guess and a `RoomTracker` votes over recent frames, with hysteresis, landmarks and an "unknown" state, and publishes `RoomEntered`. The accuracy on the robot is still to be measured in the field trial.) It can still be wrong or missing, there are no room boundaries, and no position inside a room.
- **No map.** `find_object` can say "last seen in the kitchen" but cannot go there ([search](behaviours.md#searching-and-approaching)); `approach` only closes the last metre on something currently in view.
- **Autonomous driving is off on the robot by default** (`[robot] allow_autonomous_driving = false`), and all movement goes through the 20 Hz `SafetyController`. Any navigation feature must obey both.

## 2. The options

Fit with the PRD is judged against its priorities: prove the see/understand/ask/remember loop first, keep the Pi light, keep AI off-board, no premature complexity.

### (a) Room recognition from the vision model (already partly there)

- **Gives:** a semantic "which room am I in", good enough for memory ("the keys are usually in the hall") and for the conversation. It is what Laika already does.
- **Cost:** no new hardware, no new dependency; one vision request per observation, already rate-limited. Improving it means better prompts, asking for a room each scan, and keeping a few per-room landmarks (for example "the red sofa means living room") in the world model, then voting over recent scans so one wrong frame does not flip the room.
- **Risks:** it is only as good as the model; similar rooms (two bedrooms) and dark rooms fail; it never gives a direction to drive. It cannot make `find_object` go anywhere on its own.
- **Fit:** excellent. Do it first.

### (b) AprilTag / ArUco markers at doorways and on a home spot

- **Gives:** a cheap, deterministic "I see the kitchen-door tag" fix, an ID per room entry, a bearing to the tag, and a rough distance from its apparent size. A tag on the floor or wall is also a target to drive to, which is what `return_to_base` needs (section 3). It is not a map, but a graph of "doorway tags" is enough for "go to the kitchen" by chaining searches.
- **Cost:** a printed tag costs nothing; a few A4 sheets stuck on door frames. Software: [OpenCV's ArUco module](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) (`cv2.aruco.ArucoDetector`; it also ships AprilTag dictionaries, which I know from OpenCV but could not re-verify because the page returned 403) or [pupil-apriltags](https://github.com/pupil-labs/apriltags) (a wrapper over the C AprilTag 3 library). That would be Laika's first dependency beyond the standard library and `requests`, so it needs a decision (OpenCV is the heavier install; pupil-apriltags needs numpy; on a 1 GB Pi both must be checked against RAM). CPU: I found no published Pi 4 benchmark. Related figures: a [review cited in the Frappe paper](https://paperity.org/p/312781559/frappe-fast-fiducial-detection-on-low-cost-hardware) says ArUco ran at 640x480 and 20 Hz on a Pi 3 while AprilTag was judged too heavy there; one [forum user](https://www.chiefdelphi.com/t/using-apriltag-on-raspberry-pi/423250) saw about 11 Hz on a Pi, resolution unstated. *Estimate:* a few frames per second at 640x480 on a Pi 4 core, which is plenty for "glance at the door", because Laika is slow and does not need 30 Hz.
- **Range:** a tag's apparent size in pixels is roughly focal length in pixels times tag width divided by distance, and it shrinks further at an oblique angle ([Basler's AprilTag notes](https://docs.baslerweb.com/stereovisard/rc_cube/en/tagdetect) give family-dependent limits). Practical consequence: use large tags (15 to 20 cm), fewer-module families (16h5 or 25h9, or ArUco 4x4) for range, and mount them at the camera's height, because the head pans but the PiCar-X camera sits low.
- **Effort:** small to medium: a detector service, tag-to-room table in config, a world-model landmark, a tool or two.
- **Risks:** it works only where tags are visible; glare, motion blur while the rover moves, and camera height can all spoil detection; tags in the home need the owner's agreement (looks). The detector must run off the safety thread, rate-limited, and never starve the 20 Hz loop; the Pi has four cores but only 1 GB RAM.
- **Fit:** good. It is the smallest step that gives real navigation value, uses the existing search/approach tools, and keeps AI off-board.

### (c) Visual SLAM (ORB-SLAM3, RTAB-Map and similar)

- **Gives:** a metric map and a pose from the camera alone, which could in principle replace odometry.
- **Cost:** heavy. [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) is GPLv3 and C++ with several dependencies; I found [one Pi 4 write-up](https://habr.com/ru/post/689168) (on Raspbian Buster with ROS Noetic) but no published memory or frame-rate figures for a Pi 4, and none for 1 GB. [RTAB-Map](https://introlab.github.io/rtabmap/) has memory management for long runs; the Pi 4 result I found used an RGB-D camera, not a monocular one. *Estimate:* monocular SLAM would use a large share of the 1 GB and a full core or more, competing with the camera, audio, Python app and vision uploads.
- **Risks:** a monocular camera has no metric scale; a low, shaking, pan/tilt camera on a rover with no odometry is a hard case (fast rotation loses tracking); lighting changes in a home break it; it would have to be built and maintained for ARM.
- **Fit:** poor on this board. It could be run off-board (stream frames to a laptop), but that adds a second always-on machine and a network dependency for something the PRD wants local and simple.

### (d) 2D LiDAR + ROS 2 / Nav2

- **Gives:** a proper floor-plan map, reliable obstacle detection all around (today there is one ultrasonic cone ahead), and path planning with recovery behaviours. This is the standard answer for house-scale navigation.
- **Cost:** hardware: an RPLIDAR A1-class sensor is roughly US$80 to 110 from retailers ([Waveshare](https://www.waveshare.com/product/rplidar-a1.htm), [Adafruit](https://www.adafruit.com/product/4010)); spec: 12 m range, about 8000 samples/s, 5.5 Hz scan. It needs a mount on the PiCar-X chassis (the camera and pan/tilt head already occupy the top), USB power and a clear 360-degree view, which this low chassis may not give. SLAM and Nav2 want odometry, which this robot lacks; scan matching can partly substitute, but wheel slip makes it fragile. Software: [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Releases/Release-Jazzy-Jalisco.html) targets Ubuntu 24.04 (Tier 1) and lists Debian Bookworm only as Tier 3 amd64, so Raspberry Pi OS is unsupported; the PiCar-X vendor stack is tested on Raspberry Pi OS. Community reports show a [2 GB Pi 4 running SLAM and Nav2](https://forum.dexterindustries.com/t/ros-2-gopigo3-robot-humble-dave-doing-navigation/9538) on Ubuntu with all cores saturated; a [vendor guide](https://zbotic.in/slam-raspberry-pi-turtlebot3-navigation-mapping/) recommends 4 GB. There is no 1 GB report that I could find, and it would not leave room for the Laika app. An alternative is to run Nav2 on a laptop, which brings the off-board dependency back.
- **Effort:** large: new OS image, a ROS bridge between the SafetyController and the motors, a second safety story, mechanical work.
- **Risks:** it changes the whole architecture (ROS processes, DDS networking, a second way to command wheels); `docs/prd.md` forbids it until the core loop is useful.
- **Fit:** the right long-term answer if the owner wants real "go to the kitchen", but premature. It would also probably mean replacing the Pi 4 1 GB.

### (e) Hybrid

Combine the cheap pieces: room recognition (a) for meaning, tags (b) for anchors and a home spot, the ultrasonic and grayscale sensors for local safety, and a coarse dead-reckoning estimate that is **reset at every tag** (so drift never accumulates beyond one hop). If limits are then measured and LiDAR is added later, tags remain useful as global re-localisation landmarks and the dock marker. This is the recommended path, in steps (section 4).

| Option | Hardware | RAM/CPU on Pi 4 1 GB | Effort | Fit with PRD now |
| --- | --- | --- | --- | --- |
| (a) Vision room recognition | none | none extra | small | yes |
| (b) AprilTag / ArUco | printed tags | light (estimate) | small-medium | yes, later step |
| (c) Visual SLAM | none | heavy (estimate) | large | no |
| (d) LiDAR + ROS 2 / Nav2 | about US$80-110 plus mount, likely a bigger Pi | does not fit 1 GB | very large | no, not yet |
| (e) Hybrid | printed tags | light | medium | yes, staged |

## 3. `return_to_base`

The PRD excludes an automatic charging dock, so "base" here means a **home spot**: a marked place on the floor (a tag on the wall or a floor tag, plus perhaps a mat), not a charger. The owner plugs in the battery by hand.

How it could work with what exists:

1. A tag (a distinct ID, say `home`) is mounted at camera height at the home spot and registered in the world model as the landmark "home".
2. `return_to_base` first looks: a head scan checks whether the tag is in view. If so it uses the existing `approach` loop with the tag as the target, in short moves, a picture after each, stopping at `approach_stop_cm` or when the tag is large enough.
3. If it is not in view, it does a bounded search with `Explorer.run_goal` (the same exploration the search tools use), stopping on the first scan that shows the tag, limited by `max_seconds`. If a room tag is known to be near, it can drive towards that tag first. If it still fails it says so ("non trovo la base") rather than wandering.
4. It stops with the wheels off at the tag and reports.

Safety constraints, unchanged: it is an autonomous drive, so the `allow_autonomous_driving` switch must be on or it returns `autonomous_disabled` and only looks; every move goes through `SafetyController.execute` under a lease with heartbeat (obstacle, cliff, watchdog and emergency stop all apply); it is bounded in time, moves and steps; it never drives blind when tag detection fails or stalls; the detector runs off the safety thread. A tag can never authorise moving: the model and the detector only propose moves.

## 4. Recommendation

An ordered plan, each step gated on the previous one being useful:

1. **Now: stay with vision room recognition (a). Done in #53, awaiting the field-trial accuracy.** Improve it cheaply: ask for a room on every scan, keep per-room landmarks in the world model, vote over recent scans. No new dependency, no hardware.
2. **When the owner wants "go to the kitchen" or a home spot: AprilTags (b).** Print large tags for doorways and a `home` tag; detect them on the Pi with one library, on a worker thread with a frame-rate cap (a few frames per second) and its own CPU budget, publishing a `LandmarkSeen` event. Build `return_to_base` on `approach` and `run_goal`. Optionally reset a coarse dead-reckoning estimate at each tag. Pick OpenCV ArUco or pupil-apriltags after a benchmark on the Pi (the dependency decision is the owner's, since the project is standard-library-plus-`requests` today).
3. **Only after the core loop has run in the house for weeks and the limits are measured: reconsider SLAM, LiDAR and ROS 2 / Nav2 (c, d).** Do not start before then, in line with the PRD. If it is time, prefer a 2D LiDAR on a more capable board with Ubuntu (a Pi 5 or similar) over visual SLAM on this Pi, keep the SafetyController as the only path to the wheels, and keep tags as landmarks.

**Triggers for moving to the next step:**

- Step 1 to 2: the owner asks for a room destination or a home spot more than occasionally, or room recognition is wrong often enough to hurt answers (measure it from the observation log), and exploration alone cannot find the target in the time limit.
- Step 2 to 3: tags in every doorway still leave frequent "lost" or "stuck" outcomes, the owner wants a floor plan or automatic charging, or drift between tags makes routes unreliable.

## 5. Open questions to answer with measurements on the robot

- **CPU and RAM headroom:** with the app, camera, audio, and vision uploads running, how much of the Pi is left? Run the tag detector (OpenCV and pupil-apriltags) at 640x480 and measure time per frame, resident memory, and whether the 20 Hz safety loop's timing stays within limits.
- **Tag detection range and reliability:** at the camera's real resolution and the head's real height, from what distance and angle is a 15 and a 20 cm tag read, in daylight, at night and with the rover moving? Compare tag families.
- **Floor types:** how far does "forward 800 ms" go on each floor (tile, wood, carpet), and how much does it turn on a steering command? This decides whether dead reckoning between tags is worth anything.
- **Room-recognition accuracy:** how often does the vision model name the right room, from the observation log after a few weeks?
- **Does a LiDAR fit?** Only if step 3 arrives: can it be mounted with a clear view on the chassis, and is the 1 GB board still the right one?

## Sources and gaps

Verified from the repository: `picarx.py` has no encoders or IMU, and no mention of one exists under `references/`. Cited web pages are linked inline. Not found or not verified: any Pi 4 benchmark for AprilTag/ArUco detection, ORB-SLAM3 or RTAB-Map memory on a Pi 4 (let alone 1 GB), Nav2 memory on 1 GB, the OpenCV AprilTag dictionary list (page returned 403; stated from memory), and the PiCar-X camera resolution (the vendor docs reached did not give it).
