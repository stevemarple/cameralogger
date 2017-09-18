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


import logging
import glob
from PIL import Image
import threading
import time
from cameralogger import get_config_option


__author__ = 'Steve Marple'
__version__ = '0.2.1'
__license__ = 'MIT'


class Camera(object):
    """Dummy camera.

    Dummy camera for testing. Does not use real hardware but 'captures' images by reading images files
    from a directory. Files are selected by :func:`glob.glob()`."""

    def __init__(self, config):
        self.config = config
        self.file_gen = None
        self.capture_image_lock = threading.Lock()

    def apply_settings(self, section):
        self.file_gen = self.get_next_file(section)

    def capture_image(self, _):
        logger.debug('capture_image: acquiring lock')
        if self.capture_image_lock.acquire(False):
            try:
                logger.debug('capture_image: acquired lock')
                t = time.time()
                filename = next(self.file_gen)
                if filename is None:
                    raise Exception('no file to read')
                logger.debug('loading image from %s', filename)
                img = Image.open(filename)
                # Fake some image details
                img_info = dict(Exposure_s=1.0,
                                Gain=1,
                                Aperture=2.0,
                                SensorTemperature=25.0,
                                SystemTemperature=float('NaN'))

                # Take CPU temperature as system temperature
                img_info['SystemTemperature'] = float('NaN')
                with open('/sys/class/thermal/thermal_zone0/temp') as f:
                    img_info['SystemTemperature'] = float(f.read().strip()) / 1000
                return img, img_info, t

            finally:
                self.capture_image_lock.release()
                logging.debug('capture_image: released lock')
        else:
            logger.warning('capture_image: could not acquire lock')
            raise Exception('could not acquire lock')

    def get_next_file(self, section):
        while True:
            pattern = get_config_option(self.config, section, 'images')
            files = glob.glob(pattern)
            if not files:
                # raise Exception('no files matched %s', pattern)
                yield None
            for f in sorted(files):
                yield f


logger = logging.getLogger(__name__)
