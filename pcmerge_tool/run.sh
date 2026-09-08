#!/usr/bin/env bash
# Launcher for the PointCloud Merge & Edit Tool.
# Sources ROS2 if available, makes sure DISPLAY is set, then runs the GUI.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fall back to :0 if running from a non-graphical context
export DISPLAY="${DISPLAY:-:0}"

# Pick up ROS2 (humble by default) so rosbag2_py / rclpy are importable.
# Skip silently if not present.
if [ -z "${ROS_DISTRO:-}" ] && [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

cd "$HERE"
exec python3 pcmerge_tool.py "$@"
