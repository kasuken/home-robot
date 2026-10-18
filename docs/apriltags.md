# AprilTag position fixes

`src/laika/tags.py`. Laika has no odometry, so nothing in her knows where she is except what the vision model *guesses*. A printed tag is the opposite of a guess: a black-and-white square (family **tag36h11**) that a detector reads from the pixels, with its id. This is step (b) of the [navigation options](navigation.md).

## What it does now

1. The camera takes a frame as it always did. The `TagReader` gets the same frame on its own thread (one frame at a time, an older waiting frame is dropped) and detects tags **locally**: nothing leaves the robot, the safety monitor is not involved, the wheels are not touched.
2. For each tag it estimates the size in pixels, the bearing (left or right of the camera axis) and the distance (pinhole model: `[tags] tag_size_cm` and `fov_deg`). Tags beyond `max_fix_cm` and ids the plan does not list are ignored.
3. The nearest known tag decides. The plan says where it is: `[[tag]] id, marks, seen_from` ([the plan of the house](floor-plan.md)). *marks* is the doorway that carries it, *seen_from* the room she is in when she sees it (a tag on a door frame, hall side, is seen from the hall).
4. `RoomTracker.tag_fix` makes that room certain, like a person's "questa è la cucina": the window agrees, `RoomEntered` is published with source `tag`, and the plan's plausibility check for jumps is skipped (a tag is not a guess).

It does **not** drive to tags, count steps or build a position. She still moves as before (reactive wander, no-go rooms, safety layer).

## Setup on the robot

```sh
sudo pip3 install --break-system-packages opencv-python-headless    # OpenCV 4.7 or newer, for cv2.aruco.ArucoDetector
```

The service runs `/usr/bin/python3` as root, so install it for that Python. Debian's `python3-opencv` may be older than 4.7, which lacks `ArucoDetector`. Without OpenCV she starts normally and prints `tags: OpenCV is not installed, so no tag fixes`. With OpenCV and a plan that has `[[tag]]` entries she prints `tags: reading N AprilTags from the camera frames`. RAM: OpenCV imports to roughly 100 MB on a Pi (an estimate, not measured here); the Pi has 1 GB.

Print the tags (a ready sheet, ids 0 to 6, was generated for the owner's flat; the tags come from [AprilRobotics/apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs), `tag36h11/tag36_11_00000.png` and the following). Print at 100% on matte paper, black square **16 cm**, white margin kept. Stick them flat, 15 to 25 cm above the floor, on the door frames facing the room she drives in. Avoid glossy surfaces and glare.

## What is verified

- The real detector (OpenCV 5.0, `DICT_APRILTAG_36h11`) finds tag id 0 in a JPEG of the image in `docs/images/apriltag-36h11-id0.png`, at 57 px and at about 28 px wide, and returns nothing for an empty or broken frame (`tests/floorplan/test_tags.py`, skipped where OpenCV is not installed, as in CI).
- The reader, the distance and bearing maths, the plan's tag table and the room fix are tested with a fake detector.

## What is not verified

- Anything on the robot: whether her low camera sees a 16 cm tag from the hall, motion blur while she moves, glare, how far away a tag is reliably read, and the CPU cost on the Pi 4.
- The distance is an estimate: `fov_deg = 62` is a guess for the PiCar-X camera. To check it, put a tag at a measured distance (for example 100 cm), and compare with `TagReader.last` (a `--check-tags` command for this is not written yet). A wrong field of view scales every distance, so `max_fix_cm` only roughly bounds how near she must be.
- Frames are read when the camera captures them (exploring, scans, looking), not continuously.

## Next steps

Driving to a doorway tag (turn until it is centred, approach to a stop distance, always through `SafetyController`) and a home spot tag she can dock to are the next step, and they need the field results above first.
