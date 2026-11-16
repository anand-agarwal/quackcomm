# Running in the Duckiematrix simulator

This runs the same stack as on the real robot (`dt-core` lane following plus this repository's container), on a virtual Duckiebot driving in the Duckiematrix. Nothing in the code changes; only the robot name does.

Tested with DTS 6.2.27 (`ente` profile), Duckiematrix renderer v0.14.0, on an x86_64 Ubuntu 22.04 workstation with an Intel iGPU (Vulkan through Mesa).

## What runs where

| Piece | Where | Started by |
|---|---|---|
| Engine (physics, world, sensor routing) | Docker container `dts-matrix-engine` on the workstation | `dts matrix engine run` |
| Renderer (3D window, renders the camera images) | Native window on the workstation | `dts matrix run` |
| Virtual Duckiebot (drivers, ROS master) | Docker container `dts-virtual-<name>` | `dts duckiebot virtual start` |
| `dt-core` lane following (FSM, car command switch) | Container `demo_lane_following` on the virtual bot | `./demo.sh up` |
| This stack (lane detector, Pure Pursuit, YOLO, ToF, safety) | Container `dts-run-quackcomm` on the virtual bot | `./demo.sh up` |

The renderer must stay open. It produces the robot's camera frames; if the window is closed, the camera goes silent and the safety supervisor stops the car.

## One-time setup

Create a virtual Duckiebot if you do not have one (`bot2` is used below):

```bash
dts duckiebot virtual create -t duckiebot -c DB21J bot2
dts duckiebot virtual list
```

The standard maps ship with `dt-core` assets. The ones used here are in:

```text
~/duckie-assets/dt-core/assets/duckiematrix/maps/
  loop                    plain lane loop (start here)
  intersections           loop with intersections
  loop_with_pedestrians   loop with duckies on the road (for YOLO / ToF stops)
```

Each map has one robot entity, `map_0/vehicle_0`, with a front camera and a front-center ToF.

## Run

Use three terminals. Keep the first two open for the whole session.

```bash
# Terminal 1: engine with the loop map
dts matrix engine run --no-pull -m ~/duckie-assets/dt-core/assets/duckiematrix/maps/loop

# Terminal 2: renderer window
dts matrix run
```

`--no-pull` uses the engine image you already have. Without it, dts tries to update the image first and fails if Docker Hub is unreachable.

```bash
# Terminal 3
dts duckiebot virtual start bot2              # skip if `virtual list` shows it Running
dts matrix attach bot2 map_0/vehicle_0

cd ~/quackcomm
ROBOT=bot2 ./demo.sh build                    # first time, and after code changes
ROBOT=bot2 ./demo.sh up                       # dt-core lane following + this stack
ROBOT=bot2 ./demo.sh status                   # check before driving (see below)
ROBOT=bot2 ./demo.sh autopilot on             # FSM -> LANE_FOLLOWING, the bot drives
```

The virtual bot is x86, so `dts devel build -H bot2` (which `./demo.sh build` calls) builds an `amd64` image. Do not use the `arm64v8` image from the main README.

Stop driving with `ROBOT=bot2 ./demo.sh autopilot off`. Stop both containers with `ROBOT=bot2 ./demo.sh down` only when you are finished.

## Check that it works

Before `autopilot on`, `ROBOT=bot2 ./demo.sh status` should look roughly like this:

```text
  fsm:            state=NORMAL_JOYSTICK_CONTROL
  allow_motion:   True
  safety_state:   state=STOP  reason=FSM is NORMAL_JOYSTICK_CONTROL  ...  yolo_fresh=True  tof_fresh=True
  lane_estimate:  valid=True  source=both  ...  confidence=0.99
  rates over 4 s:
    camera_node/image/compressed      15 Hz
    purepursuit/lane_estimate         13 Hz
    purepursuit/safety_state          20 Hz
```

After `autopilot on`, `safety_state` should read `CLEAR  reason=path clear` with `authorized_v` above zero, and the bot moves in the renderer window.

Other useful commands (all take `ROBOT=bot2`):

```bash
ROBOT=bot2 ./demo.sh view          # how to open the lane / YOLO debug images
ROBOT=bot2 ./demo.sh snap 6        # save debug frames to ~/quackcomm-snaps
ROBOT=bot2 ./demo.sh record 30     # bag + timeline to ~/quackcomm-bags
ROBOT=bot2 ./demo.sh logs          # follow this stack's logs
```

To test obstacle stops, restart the engine with `-m .../maps/loop_with_pedestrians` and attach again.

## Troubleshooting

**The bot does not appear or move, and `status` shows the camera at 0 Hz.**
Check, in order:

1. The renderer window is open (`dts matrix run`).
2. The bot is attached: run `dts matrix attach bot2 map_0/vehicle_0` again after every engine restart.
3. Measure the camera on the bot itself:
   ```bash
   docker -H bot2.local exec ros-interface bash -c \
     "source /environment.sh; rostopic hz /bot2/camera_node/image/compressed"
   ```
   It should be about 15 Hz, and `/bot2/front_center_tof_driver_node/range` about 8 Hz.
4. If both are silent with the renderer open and the bot attached, the engine is stuck. One symptom is that `docker logs dts-matrix-engine` shows nothing after its startup lines, not even the attach. Restart it:
   ```bash
   docker stop dts-matrix-engine; docker rm dts-matrix-engine
   dts matrix engine run --no-pull -m ~/duckie-assets/dt-core/assets/duckiematrix/maps/loop
   # restart the renderer, then attach again
   ```

**`rostopic hz` from `dts start_gui_tools bot2` says "no new messages".**
Inside the GUI-tools container `bot2.local` resolves to `127.0.0.2`, so it cannot reach the bot's publishers even though topics are listed. Measure from the bot (step 3 above) or use `./demo.sh status`.

**`demo.sh` says `bot2.local does not answer ping`.**
Use the plain dts commands instead:

```bash
dts devel build -H bot2 -f
dts duckiebot demo --demo_name lane_following --duckiebot_name bot2 --package_name duckietown_demos
dts devel run -H bot2
dts duckiebot keyboard_control bot2      # A = lane following, S = stop
```

**`dts matrix engine run` fails with `network is unreachable` while pulling.**
Add `--no-pull`.

**The bot is stopped and `safety_state` gives a reason.**
The supervisor only allows motion in `LANE_FOLLOWING`, holds still for 2 s after start, and stops on stale ToF/YOLO or a close obstacle. The `reason` field says which applies.

**The lane estimate is poor in simulation.**
The virtual bot uses default camera calibration, and the rendered colours differ from the real mat. Use `./demo.sh snap` to look at the masks and tune `packages/purepursuit/config/lane_vision.yaml`, then `ROBOT=bot2 ./demo.sh build && ROBOT=bot2 ./demo.sh restart`.
