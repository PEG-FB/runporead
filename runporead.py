#!/usr/bin/env python3
"""
runporead.py

Reads the large 7-segment numeric readout and battery icon from a Runpotec
cable-measuring display via the Pi Camera, and publishes over ROS 2.

Tunable parameters live in the CONFIGURATION block below. Debug frames are
written to /tmp/debug_full.png, /tmp/debug_crop_gray.png, and
/tmp/debug_crop_thresh.png on every cycle to aid tuning.

Topics published:
    /cable_length_display  (sensor_msgs/Image)        - cropped, thresholded digit image
    /cable_length_full     (sensor_msgs/Image)        - full frame with crop region overlaid
    /cable_length          (std_msgs/Float32)         - extracted numeric reading (2 Hz)
    /diagnostics           (diagnostic_msgs/DiagnosticArray) - battery level (0.1 Hz)

Run directly with: python3 runporead.py
Requires: rclpy, cv_bridge, opencv-python, picamera2, ssocr (on PATH)
          ros-jazzy-diagnostic-msgs
"""

import subprocess
import tempfile
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from cv_bridge import CvBridge

from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Configuration - tune these for your physical camera mount / lighting
# ---------------------------------------------------------------------------

CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480
RATE_HZ        = 2.0    # cable length reading rate
BATTERY_HZ     = 0.1    # battery diagnostics rate (every 10 s)

# Crop region (x1, y1, x2, y2) in pixels, against the 640x480 capture,
# applied AFTER the 180-degree rotation below.
# Updated for the new ~60° FOV lens (was ~120° previously).
CROP_REGION = (95, 185, 530, 335)

# Threshold for digit binarization (0-255).
# Below this → dark (segment filled), above → white (background).
# 120 gives a wide, clean margin in the new lens's brightness histogram.
THRESHOLD = 120

# Pixels of white border added around the thresholded crop before ssocr.
# Prevents digits touching the image edge, which confuses ssocr segmentation.
BORDER_PX = 10

# Morphological closing kernel size — bridges tiny gaps in segments from
# soft focus / sensor noise that otherwise cause ssocr to misread digits.
MORPH_KERNEL_SIZE = 3

# ---------------------------------------------------------------------------
# Battery icon sample points (x, y) in the full 640x480 frame, AFTER rotation.
# One point per segment, placed in the centre of each filled rectangle.
# Same THRESHOLD used: gray < THRESHOLD → segment filled, else empty.
# Updated for the new ~60° FOV lens — determined from fresh.jpg inspection.
# ---------------------------------------------------------------------------
BATTERY_SAMPLE_POINTS = [
    (428, 152),   # left segment
    (452, 152),   # middle segment
    (478, 152),   # right segment
]

# ---------------------------------------------------------------------------
# Manual exposure / gain / white-balance controls.
# Lock these so THRESHOLD stays valid across reboots and sessions.
# ---------------------------------------------------------------------------
MANUAL_EXPOSURE_ENABLED = True
EXPOSURE_TIME_US = 16000    # µs — increase if too dark, decrease if blown out
ANALOGUE_GAIN    = 1.0      # sensor gain (~ISO). Try 1.0–4.0.
AWB_ENABLED      = False    # lock white balance for consistent grayscale
COLOUR_GAINS     = (1.5, 1.5)  # (red, blue) — only used when AWB_ENABLED=False

# ssocr binary — on PATH after `sudo make install`
SSOCR_BIN = "ssocr"


class RunporeadNode(Node):
    def __init__(self):
        super().__init__('runporead')

        self.bridge = CvBridge()
        self.display_pub  = self.create_publisher(Image,           '/cable_length_display', 10)
        self.full_pub     = self.create_publisher(Image,           '/cable_length_full',    10)
        self.value_pub    = self.create_publisher(Float32,         '/cable_length',         10)
        self.diag_pub     = self.create_publisher(DiagnosticArray, '/diagnostics',          10)

        self.get_logger().info('Initializing camera...')
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        if MANUAL_EXPOSURE_ENABLED:
            controls = {
                "AeEnable":    False,
                "ExposureTime": EXPOSURE_TIME_US,
                "AnalogueGain": ANALOGUE_GAIN,
                "AwbEnable":   AWB_ENABLED,
            }
            if not AWB_ENABLED:
                controls["ColourGains"] = COLOUR_GAINS
            self.picam2.set_controls(controls)
            self.get_logger().info(
                f'Manual exposure: ExposureTime={EXPOSURE_TIME_US}µs  '
                f'Gain={ANALOGUE_GAIN}  AWB={AWB_ENABLED}'
            )

        # Two timers — cable length at RATE_HZ, battery diagnostics at BATTERY_HZ
        self.create_timer(1.0 / RATE_HZ,   self.length_callback)
        self.create_timer(1.0 / BATTERY_HZ, self.battery_callback)

        # Cache the latest frame so battery_callback can reuse it without
        # an extra capture (saves time and avoids two captures per second)
        self._latest_frame = None

        # Track display on/off state to avoid spamming the log every cycle
        self._display_on = True

        self.get_logger().info(
            f'runporead started — {RATE_HZ} Hz length, '
            f'{BATTERY_HZ} Hz battery | crop={CROP_REGION} | threshold={THRESHOLD}'
        )

    # ------------------------------------------------------------------
    # Cable length: capture → rotate → crop → threshold → ssocr → publish
    # ------------------------------------------------------------------
    def length_callback(self):
        frame = self.picam2.capture_array()          # RGB888, H×W×3
        frame = cv2.rotate(frame, cv2.ROTATE_180)   # camera mounted upside-down
        self._latest_frame = frame                   # share with battery_callback

        x1, y1, x2, y2 = CROP_REGION

        # Full frame with green crop rectangle for monitoring via rqt_image_view
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Also mark battery sample points in blue
        for (bx, by) in BATTERY_SAMPLE_POINTS:
            cv2.circle(annotated, (bx, by), 4, (255, 0, 0), -1)

        full_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='rgb8')
        full_msg.header.stamp     = self.get_clock().now().to_msg()
        full_msg.header.frame_id  = 'cable_length_full'
        self.full_pub.publish(full_msg)

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.copyMakeBorder(
            thresh, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
            cv2.BORDER_CONSTANT, value=255
        )

        # Debug dumps — pull with: scp fb@pi5:/tmp/debug_*.png .
        cv2.imwrite('/tmp/debug_full.png',       cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        cv2.imwrite('/tmp/debug_crop_gray.png',  gray)
        cv2.imwrite('/tmp/debug_crop_thresh.png', thresh)

        disp_msg = self.bridge.cv2_to_imgmsg(thresh, encoding='mono8')
        disp_msg.header.stamp    = self.get_clock().now().to_msg()
        disp_msg.header.frame_id = 'cable_length_display'
        self.display_pub.publish(disp_msg)

        value = self.run_ssocr(thresh)
        if value is None:
            # Only log once on transition to avoid spamming at 2 Hz
            if self._display_on:
                self._display_on = False
                self.get_logger().warn(
                    'Display off or unreadable — ssocr extraction failed. '
                    'Will resume silently when display comes back on.'
                )
        else:
            if not self._display_on:
                self._display_on = True
                self.get_logger().info('Display back on — resuming cable length readings.')
            msg = Float32()
            msg.data = value
            self.value_pub.publish(msg)

    # ------------------------------------------------------------------
    # Battery: sample three pixels, publish a single battery_level float
    # in diagnostics (0.0 / 0.33 / 0.66 / 1.0 for 0/1/2/3 bars filled)
    # ------------------------------------------------------------------
    def battery_callback(self):
        frame = self._latest_frame
        if frame is None:
            return   # no frame yet — wait for the first length_callback

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        bars_filled = sum(
            1 for (bx, by) in BATTERY_SAMPLE_POINTS
            if int(gray[by, bx]) < THRESHOLD
        )

        battery_level = round(bars_filled / 3.0, 2)  # 0.0, 0.33, 0.66, or 1.0

        status = DiagnosticStatus()
        status.name        = 'runpotec/battery'
        status.hardware_id = 'runpotec_display'
        status.values      = [KeyValue(key='battery_level', value=str(battery_level))]

        if bars_filled == 0:
            status.level   = DiagnosticStatus.ERROR
            status.message = f'Battery ERROR — unreadable (display off or camera misaligned), battery_level={battery_level}'
            self.get_logger().error('Runpotec battery unreadable (0/3 bars)')
        elif bars_filled == 1:
            status.level   = DiagnosticStatus.WARN
            status.message = f'Battery LOW — 1/3 bar remaining, battery_level={battery_level}'
            self.get_logger().warn('Runpotec battery LOW (1/3 bar)')
        else:
            status.level   = DiagnosticStatus.OK
            status.message = f'Battery OK — {bars_filled}/3 bars, battery_level={battery_level}'

        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [status, self._display_status()]
        self.diag_pub.publish(arr)

    def _display_status(self):
        """Separate DiagnosticStatus for display on/off state."""
        s = DiagnosticStatus()
        s.name        = 'runpotec/display'
        s.hardware_id = 'runpotec_display'
        s.values      = []
        if self._display_on:
            s.level   = DiagnosticStatus.OK
            s.message = 'Display ON — readings active'
        else:
            s.level   = DiagnosticStatus.WARN
            s.message = 'Display OFF — no cable length readings'
        return s

    # ------------------------------------------------------------------
    # ssocr helper
    # ------------------------------------------------------------------
    def run_ssocr(self, thresh_img):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, thresh_img)
            result = subprocess.run(
                [SSOCR_BIN, '-d', '-1', tmp_path],
                capture_output=True, text=True, timeout=2.0,
            )
            if result.returncode != 0:
                self.get_logger().debug(f'ssocr exit {result.returncode}: {result.stderr.strip()}')
                return None
            text = result.stdout.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                self.get_logger().debug(f'ssocr output not parseable as float: "{text}"')
                return None
        except subprocess.TimeoutExpired:
            self.get_logger().debug('ssocr timed out')
            return None
        finally:
            os.unlink(tmp_path)

    def destroy_node(self):
        self.picam2.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RunporeadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()