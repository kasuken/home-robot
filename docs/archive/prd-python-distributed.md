# Laika Home Robot

Requirements updated: 15 October 2026. The backend, simulator, and Raspberry Pi runtime will use Python so implementation can build on the working SunFounder libraries and examples. The [initial project brief](archive/prd-initial.md) is retained for historical reference; this document defines the current requirements.

## Project Overview

**Laika** is an autonomous home robot designed to move around the house, observe its surroundings, learn about the environment, remember information, ask questions, and proactively interact with people.

The interaction style is inspired by **TARS from Interstellar**:

- concise
- direct
- dry sense of humour
- occasionally sarcastic
- curious
- proactive without being annoying
- able to decide when not to interrupt
- aware of previous observations and conversations

The GitHub repository is named:

```text
home-robot
```

The robot itself is named:

```text
Laika
```

---

# Goals

Laika should eventually be able to:

- move autonomously around the house
- avoid obstacles and unsafe movement
- look around using its camera
- detect when a person is nearby
- proactively initiate conversations
- answer spoken questions
- ask questions about things it sees
- identify interesting or unusual changes
- remember objects, rooms, people, and previous observations
- build a persistent understanding of the house
- investigate things it does not understand
- recognize recurring patterns
- learn user preferences through conversation
- operate with either cloud-hosted or locally hosted AI
- continue operating safely if the AI backend is unavailable

The AI must never directly control motors or bypass the local safety system.

---

# Current Hardware

## Rover

The physical platform is a **SunFounder PiCar-X style rover**.

The rover provides capabilities such as:

- DC motor movement
- steering
- camera
- camera pan
- camera tilt
- ultrasonic distance sensor
- grayscale sensors
- speaker/audio support
- Robot HAT

The existing SunFounder Python libraries should be used instead of reimplementing low-level hardware access.

---

## Raspberry Pi

Current board:

```text
Raspberry Pi 4
RAM: 1 GB
```

The Raspberry Pi is intentionally **not responsible for running AI models**.

It should act as:

- hardware controller
- sensor gateway
- camera gateway
- audio gateway
- safety controller
- remote backend client

Use a lightweight headless operating system such as Raspberry Pi OS Lite.

Avoid installing unnecessary GUI software.

---

# Core Architecture

The system is split into two major components.

```text
                   ┌──────────────────────────────┐
                   │        Laika Backend         │
                   │                              │
                   │ Python                       │
                   │                              │
                   │ Agent                        │
                   │ Planning                     │
                   │ Memory                       │
                   │ World Model                  │
                   │ Vision                       │
                   │ Conversation                 │
                   │ Curiosity Engine             │
                   │ Interaction Policy           │
                   │                              │
                   └──────────────┬───────────────┘
                                  │
                         WebSocket / HTTPS
                                  │
                   ┌──────────────▼───────────────┐
                   │     Raspberry Pi Runtime     │
                   │                              │
                   │ Python                       │
                   │                              │
                   │ Hardware Controller          │
                   │ Camera                       │
                   │ Sensors                      │
                   │ Audio                        │
                   │ Safety                       │
                   │ Backend Connection           │
                   │                              │
                   └──────────────┬───────────────┘
                                  │
                           SunFounder API
                                  │
                   ┌──────────────▼───────────────┐
                   │          PiCar-X             │
                   │                              │
                   │ motors                       │
                   │ steering                     │
                   │ camera                       │
                   │ pan / tilt                   │
                   │ ultrasonic sensor            │
                   │ grayscale sensors            │
                   │ speaker                      │
                   └──────────────────────────────┘
```

---

# Architectural Principle

The most important rule is:

> The remote AI decides **what** Laika should do.  
> The robot runtime decides **how and whether it can be done safely**.

The AI must issue high-level commands.

Good:

```text
move_forward
stop
look_left
capture_image
scan_environment
investigate
```

Bad:

```text
set GPIO 17 high
set left motor PWM to 63%
```

Hardware implementation details must remain hidden from the AI layer.

---

# Robot Runtime

The Raspberry Pi runtime must be written in **Python** and reuse the installed SunFounder hardware libraries and working examples. The owner has confirmed that the software is installed and the examples work; see the [confirmed hardware baseline](current-status.md).

Suggested structure:

```text
robot/
└── laika-runtime/
    ├── main.py
    ├── hardware.py
    ├── movement.py
    ├── camera.py
    ├── sensors.py
    ├── audio.py
    ├── safety.py
    ├── protocol.py
    └── connection.py
```

The runtime should remain deliberately small.

It should not contain:

- LLM logic
- prompts
- embeddings
- AI memory
- world reasoning
- large computer vision models
- planning logic

---

# Backend

The backend and simulator must use **Python**, matching the Raspberry Pi runtime and the working official examples.

The backend must provide:

```text
HTTPS endpoints
persistent WebSocket sessions
asynchronous command and event handling
```

Keep the backend as one application with Python modules. The HTTP/WebSocket framework and package versions remain TBC until implementation. Select Python versions compatible with the dependencies used in each deployment, preserving the robot's working vendor stack.

Suggested backend structure:

```text
pyproject.toml

src/
└── laika/
    ├── __init__.py
    ├── api/
    ├── agent/
    ├── core/
    ├── infrastructure/
    ├── memory/
    ├── robot_protocol/
    ├── simulator/
    └── vision/

tests/
├── agent/
├── core/
├── robot_protocol/
└── simulator/
```

The [robot runtime](#robot-runtime) remains a separate process and deployment. Using Python throughout does not move AI models onto the Pi or allow the backend to access motors directly. Backend and simulator imports must work without Raspberry Pi hardware libraries installed.

Use the [local SunFounder reference](../references/sunfounder/README.md) to guide the hardware adapter. Keep that snapshot unchanged and implement Laika-specific behavior outside it.

---

# AI Provider

The backend must be **provider-independent**.

All AI access should go through an OpenAI-compatible abstraction.

Possible providers include:

- OpenAI
- Azure OpenAI
- local AI running on a laptop
- LM Studio
- Ollama through an OpenAI-compatible endpoint
- vLLM
- DeepSeek-compatible providers
- future OpenAI-compatible providers

Example configuration:

```json
{
  "AI": {
    "Endpoint": "http://localhost:1234/v1",
    "ApiKey": "local",
    "Model": "model-name"
  }
}
```

Cloud configuration should use the same application architecture.

Define a small Python AI client abstraction for the agent's model requests and tool results. Keep endpoint configuration, credentials, and provider-specific client code in infrastructure adapters so the agent does not depend on one provider or SDK.

Keep this abstraction in the backend. The Raspberry Pi runtime must not import AI clients, prompts, or planning code.

---

# Deployment

The same backend should work in multiple environments.

## Local

```text
Laika Backend
+
local OpenAI-compatible AI server
```

For example:

```text
Python backend
+
LM Studio
```

---

## Azure

Preferred deployment:

```text
Azure Container Apps
+
Laika Python backend
+
OpenAI-compatible AI endpoint
```

Azure Functions may later be useful for asynchronous workloads, but should **not be the main robot control plane**.

Potential Function workloads:

- scheduled processing
- memory consolidation
- summaries
- image processing jobs
- notifications
- background analysis

The robot requires a persistent session, so the main backend should support WebSockets.

---

# Robot Connection

The Raspberry Pi should initiate the connection to the backend.

```text
Laika
   │
   │ outbound WebSocket
   ▼
Backend
```

Do not require the backend to directly connect into the home network.

This avoids:

- inbound firewall rules
- port forwarding
- static home IP requirements
- unnecessary VPN configuration

Example:

```text
wss://backend.example.com/robot/connect
```

---

# Robot Identity

Example handshake:

```json
{
  "robotId": "laika-01",
  "name": "Laika",
  "version": "0.1.0",
  "capabilities": [
    "drive",
    "steer",
    "camera",
    "camera-pan",
    "camera-tilt",
    "ultrasonic",
    "grayscale",
    "speaker"
  ]
}
```

The backend should work from advertised capabilities rather than assuming every robot has identical hardware.

---

# Robot Protocol

Create a dedicated robot protocol independent of the PiCar-X implementation.

Example command:

```json
{
  "type": "command",
  "id": "cmd-123",
  "command": "move",
  "parameters": {
    "direction": "forward",
    "speed": 20,
    "durationMs": 500
  }
}
```

Response:

```json
{
  "type": "command_completed",
  "id": "cmd-123",
  "result": "success"
}
```

Telemetry example:

```json
{
  "type": "telemetry",
  "distanceCm": 83.4,
  "cliffDetected": false,
  "camera": {
    "pan": 15,
    "tilt": -5
  }
}
```

---

# Robot Abstraction

The backend must not know whether it is talking to:

- a physical PiCar-X
- the simulator
- a future robot
- another hardware platform

Conceptually:

```python
from __future__ import annotations

from typing import Protocol


class Robot(Protocol):
    async def get_state(self) -> RobotState: ...

    async def move(self, command: MovementCommand) -> None: ...
    async def stop(self) -> None: ...

    async def look(self, pan: float, tilt: float) -> None: ...
    async def capture_image(self) -> ImageObservation: ...

    async def speak(self, text: str) -> None: ...

    async def get_sensors(self) -> RobotSensorState: ...
```

The annotated state and command types are conceptual domain models to define during implementation.

Implementations:

```text
Robot
├── SimulatedRobot
└── RemoteRobot
```

---

# Simulator

Development should not depend on having the physical robot connected.

The Python `laika.simulator` module should simulate:

- rooms
- movement
- camera observations
- sensors
- person detection
- objects
- failures
- environmental changes

Example simulated house:

```text
house/
├── living-room/
├── kitchen/
├── office/
└── bedroom/
```

The same agent code must operate against both the simulator and real robot.

---

# Event-Driven Model

Robot activity should generate events.

Examples:

```text
RobotConnected
RobotDisconnected

MovementStarted
MovementCompleted
MovementFailed

PersonDetected
PersonLost

ImageCaptured

ObstacleDetected
CliffDetected

RoomEntered

ObservationCreated

InterestingChangeDetected

InteractionStarted
InteractionCompleted

BackendConnectionLost
```

The AI engine reacts to events rather than constantly polling everything.

---

# Safety

Safety must run locally on the Raspberry Pi.

The cloud AI cannot override it.

Conditions that should automatically stop movement include:

```text
obstacle too close
cliff detected
command timeout
backend heartbeat lost
invalid movement command
hardware error
emergency stop
```

Conceptually:

```python
if obstacle_detected():
    stop()

if cliff_detected():
    stop()

if backend_heartbeat_missing():
    stop()

if command_expired():
    stop()
```

If the backend disappears, Laika should become safe and boring.

It should never continue executing an uncontrolled queue of AI-generated movement commands.

---

# Vision

The Pi should capture images, but remote AI should interpret them.

Normal workflow:

```text
Camera
  ↓
Raspberry Pi
  ↓
Capture selected frame
  ↓
Backend
  ↓
Vision model
  ↓
Structured observation
```

Do not continuously upload full-resolution video to an AI endpoint.

Images should be captured:

- when requested by the agent
- periodically during exploration
- after movement
- when something interesting happens
- during active investigation

A separate live video stream may eventually exist for debugging or manual control.

---

# Camera Pan and Tilt

Camera pan/tilt is a key capability.

Laika should normally inspect an area by moving the camera before moving the whole rover.

Example:

```text
look left
capture

look center
capture

look right
capture
```

This enables an AI operation such as:

```text
scan_environment()
```

without unnecessary physical movement.

---

# World Model

Laika should maintain a semantic representation of the house.

Example:

```text
House
├── living_room
│   ├── sofa
│   ├── television
│   ├── keyboard
│   └── guitars
│
├── office
│   ├── desk
│   ├── monitor
│   └── chair
│
├── kitchen
│   ├── table
│   └── coffee_machine
│
└── bedroom
```

The world model should evolve from observations and conversations rather than being completely predefined.

---

# Observations

Example:

```json
{
  "timestamp": "2026-10-08T12:00:00",
  "location": "living_room",
  "description": "A backpack is next to the sofa.",
  "objects": [
    "backpack",
    "sofa"
  ],
  "confidence": 0.91
}
```

Observations should be persistent.

This enables reasoning such as:

```text
Have I seen this before?

Where did I last see this?

Is this normally here?

What changed since yesterday?

Is this object new?

Has this person been here recently?
```

---

# Memory

Laika needs multiple types of memory.

## Semantic memory

Facts such as:

```text
The black backpack normally stays near the entrance.

The office is where Emanuele usually works.

This object belongs to Leonardo.
```

## Episodic memory

Events such as:

```text
Yesterday the backpack was beside the sofa.

At 10:30 Laika asked about an unknown object.

The user said not to enter the bedroom.
```

## Conversation memory

Important information learned from conversations.

Not every message should become long-term memory.

---

# Curiosity Engine

One of Laika's defining features should be curiosity.

Every observation can receive an interestingness score.

Inputs might include:

```text
novelty
environmental change
uncertainty
person detected
unknown object
unusual object location
movement
sound
long-term pattern
user relevance
```

Conceptually:

```text
interestingness =
    novelty
  + environmentalChange
  + uncertainty
  + relevance
```

High interestingness can cause Laika to:

```text
inspect
take another picture
move closer
ask a question
remember something
find the user
```

Not every observation should generate an interaction.

---

# Proactive Interaction

Laika should sometimes initiate interaction when it sees the user nearby.

Example:

```text
Person detected
      ↓
Context evaluated
      ↓
Interaction policy
      ↓
Should interrupt?
      ↓
Agent generates interaction
```

The deterministic interaction policy decides **whether** to interact.

The LLM decides **what to say**.

The LLM must not independently decide when it is allowed to interrupt.

---

# Interaction Policy

Possible factors:

```text
person confidence
time since last interaction
current user activity
time of day
whether Laika has something useful to say
curiosity score
pending question
recent interactions
do-not-disturb state
```

Possible states:

```text
AVAILABLE
BUSY
DO_NOT_DISTURB
AWAY
SLEEPING
UNKNOWN
```

For the first version, keep this simple.

Example:

```text
person detected
AND
no interaction during previous 30 minutes
AND
not DND
AND
interestingness above threshold
```

Only then allow proactive interaction.

---

# Personality

Laika's personality is inspired by TARS but should be its own character.

Desired traits:

- concise
- confident
- curious
- dry humour
- occasional sarcasm
- highly direct
- not excessively friendly
- not constantly talking
- capable of challenging assumptions
- knows when to remain silent

Possible configurable parameters:

```json
{
  "personality": {
    "humor": 70,
    "sarcasm": 45,
    "verbosity": 20,
    "initiative": 60,
    "curiosity": 75,
    "interruptionTolerance": 25
  }
}
```

The most important design goal is not making Laika talk frequently.

It is making Laika occasionally say something surprisingly relevant.

---

# Example Interaction

```text
Laika sees Emanuele walking through the living room.

Laika has not interacted with him recently.

Earlier it discovered an unknown object beside the desk.

Laika turns the camera toward him.

Laika:
"Emanuele, quick question. What's that thing next to the desk?"

Emanuele:
"That's a camera stabilizer."

Laika:
"Useful information. I was going with suspicious plastic object."

Laika stores:
camera stabilizer → known object
```

---

# Agent Tools

The AI should operate through explicit tools.

Initial tools:

```text
get_robot_state()
move()
stop()

look()
look_left()
look_right()
look_up()
look_down()

capture_image()

get_distance()
get_sensors()

speak()

remember()
recall()
```

Later:

```text
scan_environment()
investigate()
explore()
find_person()
find_object()
approach()
return_to_base()
```

---

# Behaviour Priority

Laika should use a strict priority hierarchy.

```text
1. Safety
2. Direct user command
3. Active conversation
4. Pending follow-up
5. Important observation
6. Curiosity
7. Social interaction
8. Random exploration
```

Lower priority behavior must never interfere with higher priority behavior.

---

# Navigation

Robust autonomous indoor localization is intentionally deferred.

The initial rover sensors are useful for:

- obstacle avoidance
- collision prevention
- basic exploration
- cliff detection

They are not enough for reliable house-scale localization.

Possible future approaches:

- LiDAR
- ROS 2 / Nav2
- SLAM
- visual SLAM
- AprilTags
- room recognition
- hybrid localization

Do not introduce ROS or LiDAR until the core AI interaction model has proven useful.

---

# Development Roadmap

## V0.1 - Backend and Simulator

No physical robot required.

Build:

```text
AI abstraction
Agent
Robot interface
Simulator
Robot protocol
Events
Basic world model
```

Target:

```text
Agent
  ↓
Robot tools
  ↓
SimulatedRobot
```

---

## V0.2 - Physical Robot Control

Connect the Raspberry Pi and rover.

Verify:

```text
forward
backward
steering
stop
camera pan
camera tilt
camera capture
ultrasonic
grayscale sensors
speaker
```

The backend should control these through the same abstraction used by the simulator.

---

## V0.3 - AI Vision

Enable:

```text
"Look around."
```

Workflow:

```text
AI
↓
camera pan/tilt
↓
capture several images
↓
vision model
↓
structured environment description
```

---

## V0.4 - Exploration and Curiosity

Enable basic autonomous exploration.

Example:

```text
move slowly
↓
stop
↓
look around
↓
capture
↓
analyse
↓
calculate interestingness
↓
investigate or continue
```

---

## V0.5 - Persistent Memory

Add:

```text
rooms
objects
observations
semantic memory
episodic memory
conversation memory
environment changes
```

Laika should begin learning the house.

---

## V0.6 - Proactive Interaction

Add:

```text
person detection
interaction cooldown
user availability
curiosity-triggered conversations
questions
TARS-inspired personality
```

---

## V1 - Home Companion

Expected capabilities:

```text
explore
observe
remember
ask
answer
recognize environmental changes
find the user
proactively interact
investigate interesting observations
```

---

# Initial Technical Priorities

Implement in approximately this order:

1. Python `Robot` abstraction
2. `SimulatedRobot`
3. robot command protocol
4. event system
5. WebSocket connection
6. Raspberry Pi runtime
7. physical PiCar-X adapter
8. safety controller
9. AI abstraction
10. basic agent tools
11. vision integration
12. world model
13. observations
14. memory
15. curiosity engine
16. proactive interaction policy
17. personality
18. navigation improvements

---

# Things Not To Build Yet

Avoid premature complexity.

Do not initially build:

- ROS 2
- SLAM
- LiDAR integration
- Kubernetes
- microservices
- sophisticated facial recognition
- full continuous video analysis
- AI models on Raspberry Pi
- complicated knowledge graphs
- automatic charging dock
- multi-robot orchestration

First prove this loop:

```text
Laika sees
     ↓
Laika understands
     ↓
Laika notices something interesting
     ↓
Laika acts
     ↓
Laika asks
     ↓
Human answers
     ↓
Laika remembers
```

That is the core product.

---

# Core Design Principle

Laika is not intended to be a voice assistant mounted on wheels.

The intended experience is:

> A small autonomous machine living in the house, observing its environment, becoming familiar with it, occasionally investigating things, learning from people, and interacting when it has a reason to.

That principle should guide technical and product decisions throughout development.
