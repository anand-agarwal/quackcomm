# Lane pose measured by lane_detector_node from one camera frame.
# header.stamp is the camera frame time (not the processing time), so the
# controller can predict the pose forward over the pipeline delay.
# Conventions match dt-core LanePose: d > 0 robot left of the lane centre,
# phi > 0 robot yawed left; curvature > 0 road turning left (1/m).
std_msgs/Header header
bool valid
float32 d
float32 phi
float32 curvature
float32 confidence
float32 lane_width
string source            # both | yellow | white | none
int32 yellow_points
int32 white_points
float32 process_ms
