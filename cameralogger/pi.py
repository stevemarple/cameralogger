# Cameralogger - record and decorate camera images for timelapses etc.
# Copyright (C) 2017  Steve Marple.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from fractions import Fraction
import logging
from picamera import PiCamera, PiCameraNotRecording
from picamera.array import PiRGBArray
from PIL import Image
import threading
import time
import traceback

from cameralogger import get_config_option


__author__ = 'Steve Marple'
__version__ = '0.1.1'
__license__ = 'MIT'


class Camera(object):
    """Raspberry Pi camera.

    Uses the :class:`picamera.PiCamera` class."""

    def __init__(self, config):
        self.config = config
        self.capture_image_lock = threading.Lock()
        self.camera = PiCamera()
        self.use_video_port = False
        self.splitter_port = None

    def __del__(self):
        self.stop_recording()
        self.camera.close()

    def apply_settings(self, section):
        # Stop any previous recordings
        self.stop_recording()

        self.use_video_port = get_config_option(self.config, section, 'use_video_port', get='getboolean')
        if self.use_video_port:
            self.splitter_port = get_config_option(self.config, section, 'splitter_port', 1, get='getint')

        # Fractions
        framerate = get_config_option(self.config, section, 'framerate')
        if framerate is not None:
            logger.debug('setting framerate=%s', str(framerate))
            self.camera.framerate = fraction_or_float(framerate)

        # Booleans
        for k in ('hflip', 'image_denoise', 'vflip', 'video_denoise'):
            val = get_config_option(self.config, section, k, get='getboolean')
            if val is not None:
                logger.debug('setting %s=%s', k, 'true' if val else 'false')
                setattr(self.camera, k, val)

        # Ints
        for k in ('brightness', 'contrast', 'exposure_compensation', 'iso', 'rotation', 'saturation', 'sensor_mode',
                  'sharpness', 'shutter_speed'):
            val = get_config_option(self.config, section, k, get='getint')
            if val is not None:
                logger.debug('setting %s=%d', k, val)
                setattr(self.camera, k, val)

        # Strings
        for k in ('awb_mode', 'drc_strength', 'exposure_mode', 'meter_mode', 'resolution', 'still_stats'):
            val = get_config_option(self.config, section, k)
            if val is not None:
                logger.debug('setting %s=%s', k, val)
                setattr(self.camera, k, val)

        awb_gains = get_config_option(self.config, section, 'awb_gains')
        if awb_gains:
            awb_gains = awb_gains.split()
            if len(awb_gains) == 1:
                self.camera.awb_gains = fraction_or_float(awb_gains[0])
            else:
                # awb_gains = (Fraction(*map(int, awb_gains[0])), Fraction(*map(int, awb_gains[0])))
                self.camera.awb_gains = (fraction_or_float(awb_gains[0]), fraction_or_float(awb_gains[1]))

        if self.use_video_port:
            # Enable recording
            try:
                self.camera.start_recording(NullStream(), format='rgb', splitter_port=self.splitter_port)
            finally:
                self.stop_recording()

    def capture_image(self, section):
        t = time.time()
        logger.debug('capture_image: acquiring lock')
        if self.capture_image_lock.acquire(False):
            try:
                logger.debug('capture_image: acquired lock')
                if self.use_video_port and not self.camera.frame.complete:
                    # With long exposures the camera has not started producing valid frames. Attempting to
                    # record now mean the thread hangs and prevents other captures later.
                    raise Exception('no video frame to capture')

                shape = list(self.camera.resolution)
                shape.append(3)
                data = PiRGBArray(self.camera)
                self.camera.capture(data,
                                    format='rgb',
                                    use_video_port=self.use_video_port,
                                    splitter_port=self.splitter_port)
                img = Image.fromarray(data.array)
                img_info = self.get_image_settings(t)

                return img, img_info, t

            finally:
                logging.debug('capture_image: released lock')
                self.capture_image_lock.release()
        else:
            logger.warning('capture_image: could not acquire lock')
            raise Exception('could not acquire lock')

    def get_image_settings(self, t):
        r = {'Exposure_s': self.camera.exposure_speed / 1000000.0,
             'ExposureMode': self.camera.exposure_mode,
             'SystemTemperature': float('NaN'),
             'ISO': int(self.camera.iso),
             'DigitalGain': float(self.camera.digital_gain),
             'AnalogGain': float(self.camera.analog_gain),
             'SensorMode': int(self.camera.sensor_mode),
             }
        # Take CPU temperature as system temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                r['SystemTemperature'] = float(f.read().strip()) / 1000
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.warning('could not read system temperature')
            logger.debug(traceback.format_exc())
        return r

    def stop_recording(self):
        # Stop any previous video recording. How to tell if this has to be done? It is not clear to
        # which splitter port camera.recording refers.
        if self.use_video_port:
            try:
                self.camera.stop_recording(self.splitter_port)
            except PiCameraNotRecording:
                pass
            finally:
                self.splitter_port = None
                self.use_video_port = False



class NullStream(object):
    def __init__(self):
        pass

    def write(self, _):
        pass


def fraction_or_float(s):
    if '/' in s:
        return Fraction(*map(int, s.split('/')))
    else:
        return float(s)


logger = logging.getLogger(__name__)
