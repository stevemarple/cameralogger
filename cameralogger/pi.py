from fractions import Fraction
import logging
import numpy as np
import threading
import time
from picamera import PiCamera
from cameralogger import get_config_option

class Camera(object):
    def __init__(self, config):
        self.config = config
        self.capture_image_lock = threading.Lock()
        self.camera = PiCamera()

        self.initialise()
        time.sleep(2)

    def initialise(self):
        self.camera.stop_recording()
        self.camera.resolution = self.camera.MAX_RESOLUTION

        # Booleans
        for k in ('hflip', 'image_denoise', 'vflip', 'video_denoise'):
            val = get_config_option(self.config, 'camera', k, get='getboolean')
            if val is not None:
                setattr(self.camera, k, val)

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
        framerate = get_config_option(self.config, 'camera', 'framerate', '1/6', get='getfraction')
        if framerate:
            self.camera.framerate = framerate

        awb_gains = get_config_option(self.config, 'camera', 'awb_gains')
        if awb_gains:
            awb_gains = awb_gains.split()
            awb_gains = (Fraction(*map(int, awb_gains[0])), Fraction(*map(int, awb_gains[0])))
            self.camera.awb_gains = awb_gains

    def capture_image(self):
        logger.debug('capture_image: acquiring lock')
        if self.capture_image_lock.acquire(False):
            try:
                logger.debug('capture_image: acquired lock')
                t = time.time()
                img_info = {} # self.get_control_values()
                shape = list(self.camera.resolution)
                shape.append(3)
                img = np.empty(shape, dtype=np.uint8)
                self.camera.capture(img)
                img_info['DateTime'] = time.strftime('%Y-%m-%d %H:%M:%S+00:00', time.gmtime(t))

                # Take CPU temperature as system temperature
                img_info['SystemTemperature'] = float('NaN')
                with open('/sys/class/thermal/thermal_zone0/temp') as f:
                    img_info['SystemTemperature'] = float(f.read().strip()) / 1000
                return img, img_info, t

            finally:
                logging.debug('capture_image: released lock')
                self.capture_image_lock.release()
        else:
            logger.warning('capture_image: could not acquire lock')
            raise Exception('could not acquire lock')

    capture_image.lock = threading.Lock()


logger = logging.getLogger(__name__)
