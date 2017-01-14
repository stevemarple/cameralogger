
import logging
import six
import threading
import time
import zwoasi


class Camera(zwoasi.Camera):
    def __init__(self, config):
        num_cameras = zwoasi.get_num_cameras()
        if num_cameras == 0:
            raise Exception('no camera present')
        super(Camera, self).__init__(0)
        self.config = config

        # Initialise
        controls = self.get_controls()
        self.start_video_capture()

        # Read all camera controls defined in the config file
        for c in controls:
            cl = c.lower()
            if self.config.has_option('camera', cl):
                value = self.config.get('camera', cl)
                default_value = controls[c]['DefaultValue']
                control_type = getattr(zwoasi, 'ASI_' + c.upper())
                logger.debug('set control value %s to %s', cl, value)
                if value == 'auto':
                    self.set_control_value(control_type, default_value, auto=True)
                else:
                    # Cast value to same type as default_value
                    self.set_control_value(control_type, type(default_value)(value), auto=False)

        if self.config.has_option('camera', 'image_type'):
            image_type = self.config.get('camera', 'image_type')
            logger.debug('set image type to %s', image_type)
            self.set_image_type(getattr(zwoasi, 'ASI_IMG_' + image_type.upper()))
        time.sleep(2)

    def capture_image(self):
        logger.debug('capture_image: acquiring lock')
        if self.capture_image.lock.acquire(False):
            try:
                logger.debug('capture_image: acquired lock')
                t = time.time()
                img_info = self.get_control_values()
                img = self.capture_video_frame()
                img_info['DateTime'] = time.strftime('%Y-%m-%d %H:%M:%S+00:00', time.gmtime(t))

                # Take CPU temperature as system temperature
                img_info['SystemTemperature'] = 'unknown'
                with open('/sys/class/thermal/thermal_zone0/temp') as f:
                    img_info['SystemTemperature'] = '%.2f' % (float(f.read().strip()) / 1000)
                return img, img_info, t

            finally:
                logging.debug('capture_image: released lock')
                self.capture_image.lock.release()
        else:
            logger.warning('capture_image: could not acquire lock')
            raise Exception('could not acquire lock')

    capture_image.lock = threading.Lock()

    def get_control_values(self):
        controls = self.get_controls()
        r = {}
        for k in controls:
            r[k] = self.get_control_value(controls[k]['ControlType'])[0]

        # Fix up certain keys
        if 'Exposure' in r:
            r['Exposure'] = '%.6f' % (r['Exposure'] / 1000000.0)
        if 'Temperature' in r:
            r['Temperature'] = '%.1f' % (r['Temperature'] / 10.0)
        if 'Flip' in r:
            r['Flip'] = {0: 'None', 1: 'Horizontal', 2: 'Vertical', 3: 'Both'}[r['Flip']]

        # Convert any remaining non-string types to string
        for k, v in six.iteritems(r):
            if isinstance(v, six.string_types):
                pass
            elif isinstance(v, float):
                r[k] = '.1f' % v
            else:
                r[k] = str(v)
        return r


logger = logging.getLogger(__name__)
