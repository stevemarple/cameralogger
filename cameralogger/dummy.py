import logging
import glob
from PIL import Image
import threading
import time


__author__ = 'Steve Marple'
__version__ = '0.0.8'
__license__ = 'MIT'


class Camera(object):
    def __init__(self, config):
        self.config = config
        self.file_gen = self.get_next_file()
        self.capture_image_lock = threading.Lock()

    def capture_image(self):
        logger.debug('capture_image: acquiring lock')
        if self.capture_image_lock.acquire(False):
            try:
                logger.debug('capture_image: acquired lock')
                t = time.time()
                filename = next(self.file_gen)
                logger.debug('loading image from %s', filename)
                img = Image.open(filename)
                # Fake some image details
                img_info = dict(Exposure_s=1.0,
                                Gain=1,
                                Aperture=2.0,
                                DateTime=time.strftime('%Y-%m-%d %H:%M:%S+00:00', time.gmtime(t)),
                                SensorTemperature=25.0,
                                SystemTemperature=float('NaN'))

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

    def get_next_file(self):
        while True:
            files = glob.glob(self.config.get('camera', 'images'))
            for f in sorted(files):
                yield f


logger = logging.getLogger(__name__)
