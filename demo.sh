#!/usr/bin/env bash
# Run the QuackComm Pure Pursuit demo on a Duckiebot from this workstation.
#
#   ./demo.sh build        build our image on the robot (dts devel build)
#   ./demo.sh up           start dt-core lane following (FSM) + our stack, wait until ready
#   ./demo.sh status       containers, safety/controller/lane state, topic rates, CPU
#   ./demo.sh keyboard     keyboard control: A = lane following (go), S = joystick (stop)
#   ./demo.sh autopilot on|off|status   lane following on/off without the keyboard (FSM)
#   ./demo.sh go | stop    set allow_motion true | false on the running supervisor
#   ./demo.sh view         how to open the lane debug image
#   ./demo.sh snap [N]     save N lane debug frames (default 6) to ~/quackcomm-snaps and print the estimates
#   ./demo.sh record [S]   record S seconds (default 20) of a drive to ~/quackcomm-bags and print a timeline
#   ./demo.sh logs [ours|lane]   follow container logs (default: ours)
#   ./demo.sh ros <cmd...>       run a ROS command inside our container
#   ./demo.sh restart      stop and start our stack only (after a rebuild)
#   ./demo.sh down         stop both containers
#
# ROBOT=myrobot2 by default; override with  ROBOT=otherbot ./demo.sh up
# Background dts logs: ~/.local/state/quackcomm/

set -euo pipefail

ROBOT="${ROBOT:-myrobot2}"
HOST="${ROBOT}.local"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/quackcomm"
OURS="dts-run-quackcomm"
LANE="demo_lane_following"
D=(docker -H "$HOST")

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# No `grep -q` here: under `set -o pipefail` grep -q exits early, docker ps
# gets SIGPIPE and the pipeline reports failure even when the name matched.
running() { "${D[@]}" ps --filter "name=^${1}\$" --format '{{.Names}}' 2>/dev/null | grep -x "$1" >/dev/null; }

check_robot() {
  ping -c1 -W3 "$HOST" >/dev/null 2>&1 || die "$HOST does not answer ping. Is the robot on and on this network?"
  "${D[@]}" info >/dev/null 2>&1 || die "Docker on $HOST is not reachable."
}

check_repo_size() {
  local size
  size=$(du -sm --exclude=.git "$REPO" | cut -f1)
  if [ "$size" -gt 200 ]; then
    warn "The repo is ${size} MB. Keep datasets/venvs/runs in ~/quackcomm-training (dts --sync would copy them)."
  fi
}

# Run a ROS command inside a container on the robot.
rosx() {
  local container="$1"; shift
  "${D[@]}" exec -i -e PYTHONUNBUFFERED=1 \
    -e ROS_MASTER_URI="http://${HOST}:11311/" -e ROS_HOSTNAME="$HOST" \
    "$container" bash -c "source /environment.sh >/dev/null 2>&1; $*"
}

ros_container() {
  if running "$OURS"; then echo "$OURS"; elif running "$LANE"; then echo "$LANE"; else echo ""; fi
}

wait_for() {  # wait_for <container> <seconds> [logfile]: shows the log while waiting
  local name="$1" limit="$2" log="${3:-}" t=0 shown=0 total
  while ! running "$name"; do
    if [ -n "$log" ] && [ -f "$log" ]; then
      total=$(wc -l <"$log")
      if [ "$total" -gt "$shown" ]; then
        tail -n +"$((shown + 1))" "$log" | head -n "$((total - shown))" | sed 's/\r/\n   | /g; s/^/   | /'
        shown=$total
      fi
    fi
    sleep 2; t=$((t + 2))
    if [ "$t" -ge "$limit" ]; then return 1; fi
  done
  if [ -n "$log" ] && [ -f "$log" ]; then
    tail -n +"$((shown + 1))" "$log" | sed 's/\r/\n   | /g; s/^/   | /'
  fi
  return 0
}

start_bg() {  # start_bg <logname> <command...>; sets BG_LOG
  BG_LOG="$STATE_DIR/$1.log"; shift
  mkdir -p "$STATE_DIR"
  # exec + setsid: dts runs in its own session with no bash wrapper left behind,
  # so Ctrl-C here cannot reach it (dts devel run removes its container on exit).
  # All of the subshell's output goes to the log or /dev/null: nothing may keep
  # this script's stdout open, or anything waiting on that output never returns.
  (cd "$REPO" && exec setsid nohup "$@" </dev/null >"$BG_LOG" 2>&1) </dev/null >/dev/null 2>&1 &
  disown 2>/dev/null || true
}

cmd_build() {
  check_robot
  check_repo_size
  say "Building our image on $ROBOT (this can take a few minutes)"
  (cd "$REPO" && dts devel build -H "$ROBOT" -f)
}

start_lane() {
  if running "$LANE"; then
    say "$LANE already running"
    return
  fi
  say "Starting dt-core lane following ($LANE): FSM, command switch path"
  start_bg lane_following dts duckiebot demo --demo_name lane_following \
        --duckiebot_name "$ROBOT" --package_name duckietown_demos
  wait_for "$LANE" 120 "$BG_LOG" || die "$LANE did not start. Log: $BG_LOG"
  say "$LANE is up"
}

start_ours() {
  if running "$OURS"; then
    say "$OURS already running"
    return
  fi
  say "Starting our stack ($OURS)"
  start_bg quackcomm dts devel run -H "$ROBOT"
  wait_for "$OURS" 300 "$BG_LOG" || die "$OURS did not start. Log: $BG_LOG"
  say "$OURS is up"
}

cmd_up() {
  check_robot
  start_lane
  start_ours
  say "Our stack's startup log:"
  "${D[@]}" logs --tail 25 "$OURS" 2>&1 | sed 's/^/   | /' || true
  cmd_status
  cat <<EOF

Next:
  1. Put the robot in a straight lane.
  2. Check the debug image:   ./demo.sh view
  3. Start driving:           ./demo.sh autopilot on    (stop: ./demo.sh autopilot off)
                              or ./demo.sh keyboard and press A / S
EOF
}

cmd_status() {
  check_robot
  say "Containers"
  "${D[@]}" ps --format '  {{.Names}}\t{{.Status}}' | grep -E "$OURS|$LANE|car-interface" || warn "none of the demo containers are running"
  local c; c=$(ros_container)
  [ -z "$c" ] && { warn "No demo container is running; start with ./demo.sh up"; return; }
  running "$OURS" || warn "$OURS (our stack) is not running: safety/controller/lane fields will be empty. Start it with ./demo.sh up"
  say "Robot state (listening for 4 s)"
  # One ROS node for everything: each rostopic call costs ~9 s to start on the Nano.
  rosx "$c" "ROBOT=$ROBOT python3 -" <<'PY' || true
import os, time, rospy
robot = "/" + os.environ["ROBOT"]
rospy.init_node("demo_status", anonymous=True, disable_signals=True)
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import FSMState, Twist2DStamped
topics = {
    "fsm_node/mode": FSMState,
    "camera_node/image/compressed": CompressedImage,
    "purepursuit/desired_cmd": Twist2DStamped,
}
try:
    from purepursuit.msg import SafetyState, ControllerStatus, LaneEstimate, DetectionArray
    topics.update({
        "purepursuit/safety_state": SafetyState,
        "purepursuit/controller_status": ControllerStatus,
        "purepursuit/lane_estimate": LaneEstimate,
        "purepursuit/detection_array": DetectionArray,
    })
except ImportError:
    pass
last, count = {}, {}
def handler(name):
    def cb(msg):
        last[name] = msg
        count[name] = count.get(name, 0) + 1
    return cb
for name, kind in topics.items():
    rospy.Subscriber(robot + "/" + name, kind, handler(name), queue_size=50)
window = 4.0
time.sleep(window)
def fields(name, keys):
    msg = last.get(name)
    if msg is None:
        return "no message"
    out = []
    for key in keys:
        value = getattr(msg, key, None)
        out.append("%s=%s" % (key, ("%.3f" % value) if isinstance(value, float) else value))
    return "  ".join(out)
param = rospy.get_param(robot + "/safety_supervisor_node/allow_motion", "unset")
print("  fsm:            %s" % fields("fsm_node/mode", ["state"]))
print("  allow_motion:   %s" % param)
print("  safety_state:   %s" % fields("purepursuit/safety_state", ["state", "reason", "authorized_v", "yolo_fresh", "tof_fresh"]))
print("  controller:     %s" % fields("purepursuit/controller_status", ["source", "status", "desired_v", "desired_omega"]))
print("  lane_estimate:  %s" % fields("purepursuit/lane_estimate", ["valid", "source", "d", "phi", "curvature", "confidence", "process_ms"]))
print("  rates over %.0f s:" % window)
for name in topics:
    if name != "fsm_node/mode":
        print("    %-32s %5.1f Hz" % (name, count.get(name, 0) / window))
PY
  say "CPU (top containers)"
  "${D[@]}" stats --no-stream --format '  {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' | sort -t$'\t' -k2 -rn | head -6
}

cmd_keyboard() {
  check_robot
  say "Keyboard control: A = lane following (go), S = joystick (stop). Close the window to exit."
  dts duckiebot keyboard_control "$ROBOT"
}

cmd_autopilot() {  # cmd_autopilot on|off|status
  check_robot
  running "$OURS" || die "$OURS is not running; start it with ./demo.sh up"
  local action
  case "${1:-status}" in
    on|start) action=start ;;
    off|stop) action=stop ;;
    status)   action=status ;;
    *) die "usage: ./demo.sh autopilot on|off|status" ;;
  esac
  if [ "$action" = start ]; then
    say "Autopilot ON: FSM -> LANE_FOLLOWING. The robot drives once it sees a lane and the path is clear."
  elif [ "$action" = stop ]; then
    say "Autopilot OFF: FSM -> NORMAL_JOYSTICK_CONTROL (the supervisor stops the car)"
  fi
  rosx "$OURS" "rosrun purepursuit autopilot.py $action $ROBOT"
}

cmd_motion() {  # cmd_motion true|false
  check_robot
  running "$OURS" || die "$OURS is not running"
  rosx "$OURS" "rosparam set /$ROBOT/safety_supervisor_node/allow_motion $1"
  say "allow_motion = $1"
}

cmd_view() {
  cat <<EOF
Open rqt_image_view connected to the ROBOT's ROS master (the robot name matters:
without it the GUI talks to this laptop's master and lists nothing):

  dts start_gui_tools $ROBOT
  rqt_image_view

In rqt_image_view pick one of these, then press the refresh button (circular arrow)
if it was open before the demo started or after ./demo.sh restart:

  /$ROBOT/purepursuit/lane_debug_image/compressed   lane detector (5 fps, drawn only while viewed)
  /$ROBOT/purepursuit/yolo_debug_image/compressed   YOLO boxes
  /$ROBOT/camera_node/image/compressed              raw camera

Overlay: orange tint = yellow mask, blue tint = white mask, yellow/white lines = fitted arcs,
green = lane centre sent to the controller, RED = centre REJECTED (the controller ignores it;
the reason is printed under the numbers, with any line dropped as not parallel).
Text = source d phi curvature confidence.
No GUI handy? ./demo.sh snap saves frames to ~/quackcomm-snaps instead.
Tune HSV thresholds in packages/purepursuit/config/lane_vision.yaml, then ./demo.sh build && ./demo.sh restart.
EOF
}

cmd_snap() {  # cmd_snap [N]
  check_robot
  running "$OURS" || die "$OURS is not running; start it with ./demo.sh up"
  local n="${1:-6}" dir
  dir="$HOME/quackcomm-snaps/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$dir"
  say "Capturing $n lane debug frames (the node only draws them while someone listens)"
  rosx "$OURS" "ROBOT=$ROBOT N=$n python3 -" <<'PY' | python3 -c '
import base64, json, os, sys
out = sys.argv[1]
for line in sys.stdin:
    if not line.startswith("{"):
        continue
    rec = json.loads(line)
    name = "frame_%02d.jpg" % rec["i"]
    with open(os.path.join(out, name), "wb") as handle:
        handle.write(base64.b64decode(rec["jpg"]))
    e = rec["est"]
    print("  %s  %s" % (name, e if e else "no lane_estimate yet"))
' "$dir"
import base64, json, os, time, rospy
from sensor_msgs.msg import CompressedImage
from purepursuit.msg import LaneEstimate
robot, n = "/" + os.environ["ROBOT"], int(os.environ["N"])
rospy.init_node("demo_snap", anonymous=True, disable_signals=True)
state = {"est": None, "i": 0}
def on_est(m):
    state["est"] = "valid=%s source=%s d=%+.3f phi=%+.3f k=%+.2f conf=%.2f yellow_pts=%d white_pts=%d %.0fms" % (
        m.valid, m.source, m.d, m.phi, m.curvature, m.confidence, m.yellow_points, m.white_points, m.process_ms)
def on_img(m):
    if state["i"] >= n:
        return
    print(json.dumps({"i": state["i"], "jpg": base64.b64encode(bytes(m.data)).decode(), "est": state["est"]}), flush=True)
    state["i"] += 1
rospy.Subscriber(robot + "/purepursuit/lane_estimate", LaneEstimate, on_est, queue_size=5)
rospy.Subscriber(robot + "/purepursuit/lane_debug_image/compressed", CompressedImage, on_img, queue_size=2)
start = time.time()
while state["i"] < n and time.time() - start < 30:
    time.sleep(0.1)
PY
  say "Saved to $dir"
  echo "  Overlay: tinted = colour masks (orange = yellow, blue = white), yellow/white lines = fits, green = lane centre (red = rejected)"
  command -v xdg-open >/dev/null && xdg-open "$dir" >/dev/null 2>&1 &
}

cmd_record() {  # cmd_record [seconds]
  check_robot
  running "$OURS" || die "$OURS is not running; start it with ./demo.sh up"
  local secs="${1:-20}" dir name r="/$ROBOT"
  [[ "$secs" =~ ^[0-9]+$ ]] || die "usage: ./demo.sh record [seconds]"
  dir="$HOME/quackcomm-bags"
  name="drive-$(date +%Y%m%d-%H%M%S).bag"
  mkdir -p "$dir"
  # The camera stream lets the lane pipeline be replayed offline; recording the
  # debug image makes the lane node draw it, which costs some CPU on the Nano.
  local topics="$r/camera_node/image/compressed $r/camera_node/camera_info \
    $r/purepursuit/lane_estimate $r/purepursuit/lane_debug_image/compressed \
    $r/purepursuit/controller_status $r/purepursuit/desired_cmd $r/purepursuit/safety_state \
    $r/lane_controller_node/car_cmd $r/kinematics_node/velocity $r/fsm_node/mode"
  say "Recording ${secs} s: drive the robot through the turn now (./demo.sh autopilot on in another terminal)"
  rosx "$OURS" "rosbag record -q --duration=${secs} -O /tmp/$name $topics" || warn "rosbag exited with an error"
  say "Timeline"
  rosx "$OURS" "rosrun purepursuit bag_summary.py /tmp/$name" || warn "summary failed"
  "${D[@]}" cp "$OURS:/tmp/$name" "$dir/$name" && rosx "$OURS" "rm -f /tmp/$name"
  say "Saved $dir/$name"
}

cmd_logs() {
  check_robot
  case "${1:-ours}" in
    ours) "${D[@]}" logs -f --tail 50 "$OURS" ;;
    lane) "${D[@]}" logs -f --tail 50 "$LANE" ;;
    *) die "logs takes 'ours' or 'lane'" ;;
  esac
}

cmd_ros() {
  check_robot
  local c; c=$(ros_container)
  [ -z "$c" ] && die "No demo container is running"
  rosx "$c" "$*"
}

cmd_restart() {
  check_robot
  if running "$OURS"; then
    say "Stopping $OURS"
    "${D[@]}" stop "$OURS" >/dev/null
    local t=0; while running "$OURS" && [ "$t" -lt 30 ]; do sleep 2; t=$((t + 2)); done
  fi
  start_ours
}

cmd_down() {
  check_robot
  for c in "$OURS" "$LANE"; do
    if running "$c"; then say "Stopping $c"; "${D[@]}" stop "$c" >/dev/null; fi
  done
  say "Demo stopped (drivers and car-interface keep running)"
}

case "${1:-help}" in
  build)    cmd_build ;;
  up)       cmd_up ;;
  status)   cmd_status ;;
  keyboard) cmd_keyboard ;;
  autopilot) shift; cmd_autopilot "${1:-status}" ;;
  go)       cmd_motion true ;;
  stop)     cmd_motion false ;;
  view)     cmd_view ;;
  snap)     shift; cmd_snap "${1:-6}" ;;
  record)   shift; cmd_record "${1:-20}" ;;
  logs)     shift; cmd_logs "${1:-ours}" ;;
  ros)      shift; [ $# -gt 0 ] || die "usage: ./demo.sh ros <command>"; cmd_ros "$@" ;;
  restart)  cmd_restart ;;
  down)     cmd_down ;;
  help|-h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command '$1' (try ./demo.sh help)" ;;
esac
