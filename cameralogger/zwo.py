import logging
from PIL import Image
import threading
import time
import zwoasi
from cameralogger import get_config_option


__author__ = 'Steve Marple'
__version__ = '0.0.10'
__license__ = 'MIT'


class Camera(object):
    def __init__(self, config):
        num_cameras = zwoasi.get_num_cameras()
        if num_cameras == 0:
            raise Exception('no camera present')
        self.camera = zwoasi.Camera(0)
        self.config = config
        self.capture_image_lock = threading.Lock()

    def __del__(self):
        self.camera.stop_video_capture()
        self.camera.close()

    def apply_settings(self, section):
        # Initialise
        controls = self.camera.get_controls()
        self.camera.start_video_capture()

        # Read all camera controls defined in the config file
        for c in controls:
            cl = c.lower()
            value = get_config_option(self.config, section, cl)
            if value is not None:
                default_value = controls[c]['DefaultValue']
                control_type = getattr(zwoasi, 'ASI_' + c.upper())
                logger.debug('set control value %s to %s', cl, value)
                if value == 'auto':
                    self.camera.set_control_value(control_type, default_value, auto=True)
                else:
                    # Cast value to same type as default_value
                    self.camera.set_control_value(control_type, type(default_value)(value), auto=False)

        image_type = get_config_option(self.config, section, 'image_type')
        if image_type is not None:
            logger.debug('set image type to %s', image_type)
            self.camera.set_image_type(getattr(zwoasi, 'ASI_IMG_' + image_type.upper()))

    def capture_image(self, _):
        logger.debug('capture_image: acquiring lock')
        if self.capture_image_lock.acquire(False):
            try:
                logger.debug('capture_image: acquired lock')
                t = time.time()
                img_info = self.get_control_values()
                img = Image.fromarray(self.camera.capture_video_frame()[:, :, ::-1])  # Swap from BGR order
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

    def get_control_values(self):
        controls = self.camera.get_controls()
        r = {}
        for k in controls:
            r[k] = self.camera.get_control_value(controls[k]['ControlType'])[0]

        # Fix up certain keys
        r['Exposure_s'] = r['Exposure'] / 1000000.0
        if 'Temperature' in r:
            r['SensorTemperature'] = r['Temperature'] / 10.0
        if 'Flip' in r:
            r['Flip'] = {0: 'None', 1: 'Horizontal', 2: 'Vertical', 3: 'Both'}[r['Flip']]

        return r


logger = logging.getLogger(__name__)
