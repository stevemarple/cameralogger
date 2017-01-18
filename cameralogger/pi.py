from fractions import Fraction
import logging
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from PIL import Image
import threading
import time
import traceback

from cameralogger import get_config_option


class Camera(object):
    def __init__(self, config):
        self.config = config
        self.capture_image_lock = threading.Lock()
        self.camera = PiCamera()
        self.use_video_port = None

        self.initialise()
        time.sleep(2)

    def __del__(self):
        self.camera.close()

    def initialise(self):
        if self.camera.recording:
            self.camera.stop_recording()

        # self.camera.resolution = self.camera.MAX_RESOLUTION

        # Booleans
        for k in ('hflip', 'image_denoise', 'vflip', 'video_denoise'):
            val = get_config_option(self.config, 'camera', k, get='getboolean')
            if val is not None:
                setattr(self.camera, k, val)

        self.use_video_port = get_config_option(self.config, 'camera', 'use_video_port', get='getboolean')

        # Ints
        for k in ('brightness', 'contrast', 'exposure_compensation', 'iso', 'rotation', 'saturation', 'sensor_mode',
                  'sharpness', 'shutter_speed'):
            val = get_config_option(self.config, 'camera', k, get='getint')
            if val is not None:
                setattr(self.camera, k, val)

        # Ints (uppercase in PiCamera)
        for k in ('capture_timeout', ):
            val = get_config_option(self.config, 'camera', k, get='getint')
            if val is not None:
                setattr(self.camera, k.upper(), val)

        # Strings
        for k in ('awb_mode', 'drc_strength', 'exposure_mode', 'meter_mode', 'resolution', ):
            val = get_config_option(self.config, 'camera', k)
            if val is not None:
                setattr(self.camera, k, val)

        # Fractions
        framerate = get_config_option(self.config, 'camera', 'framerate', '1/6')
        if framerate:
            self.camera.framerate = fraction_or_float(framerate)

        awb_gains = get_config_option(self.config, 'camera', 'awb_gains')
        if awb_gains:
            awb_gains = awb_gains.split()
            if len(awb_gains) == 1:
                self.camera.awb_gains = fraction_or_float(awb_gains[0])
            else:
                # awb_gains = (Fraction(*map(int, awb_gains[0])), Fraction(*map(int, awb_gains[0])))
                self.camera.awb_gains = (fraction_or_float(awb_gains[0]), fraction_or_float(awb_gains[1]))

        # Enable recording
        # self.camera.start_recording()

    def capture_image(self):
        logger.debug('capture_image: acquiring lock')
        if self.capture_image_lock.acquire(False):
            try:
                logger.debug('capture_image: acquired lock')
                t = time.time()
                shape = list(self.camera.resolution)
                shape.append(3)
                data = PiRGBArray(self.camera)
                self.camera.capture(data, format='rgb', use_video_port=self.use_video_port)
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
        r = {'DateTime': time.strftime('%Y-%m-%d %H:%M:%S+00:00', time.gmtime(t)),
             'Exposure_s': self.camera.exposure_speed / 1000000.0,
             'SystemTemperature': float('NaN'),
             'DigitalGain': float(self.camera.digital_gain),
             'AnalogGain': float(self.camera.analog_gain),
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


def fraction_or_float(s):
    if '/' in s:
        return Fraction(*map(int, s.split('/')))
    else:
        return float(s)


logger = logging.getLogger(__name__)
