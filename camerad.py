#!/usr/bin/env python

import argparse
import astral
import cv2
import datetime
import glob
import logging
import lxml.etree as ET
import numpy as np
import operator
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import requests
from scipy.stats import trim_mean
import signal
import six
import struct
import sys
import threading
import time
import traceback
import zwoasi as asi


if sys.version_info[0] >= 3:
    import configparser
    from configparser import SafeConfigParser
else:
    import ConfigParser
    from ConfigParser import SafeConfigParser


class Task(object):
    def __init__(self, config=None, schedule=None):
        self.config = config
        self.schedule = schedule
        self.time = None
        self.buffers = {}
        self.capture_info = {}

    def run_tasks(self, tasks):
        for section in tasks:
            if not config.has_section(section):
                raise ValueError('no section called %s' % section)
            act = self.config.get(section, 'action')
            if not hasattr(self, act):
                raise ValueError('unknown action (%s)' % act)
            logger.debug('calling action %s(%s)' % (act, repr(section)))
            getattr(self, act)(section)

    def get_option(self, section, option, default=None, get=None):
        return get_config_option(self.config, section, option, default=default, get=get)

    def capture(self, section):
        dst = self.get_option(section, 'dst')
        img, self.capture_info[dst], t = capture_image()
        if self.time is None:
            self.time = t # Record time of first capture
        self.buffers[dst] = Image.fromarray(img[:, :, ::-1]) # Swap from BGR order

    def save(self, section):
        src = self.get_option(section, 'src')
        tm = time.gmtime(self.time)
        filename = self.get_option(section, 'filename')
        filename = time.strftime(filename, tm)
        self.buffers[src].save(time.strftime(filename, tm))
        logger.info('saved %s', filename)

    def copy(self, section):
        src = self.get_option(section, 'src')
        dst = self.get_option(section, 'dst')
        self.buffers[dst] = self.buffers[src].copy()

    def convert(self, section):
        src = self.get_option(section, 'src')
        dst = self.get_option(section, 'dst', src)
        mode = self.get_option(section, 'mode')
        self.buffers[dst] = self.buffers[src].convert(mode)

    def crop(self, section):
        src = self.get_option(section, 'src')
        dst = self.get_option(section, 'dst', src)
        position = map(int, self.get_option(section, 'position').split())
        self.buffers[dst] = self.buffers[src].crop(position)
        self.buffers[dst].load() # crop() is lazy operation; break connection

    def flip(self, section):
        src = self.get_option(section, 'src')
        dst = self.get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.flip(self.buffers[src])

    def mirror(self, section):
        src = self.get_option(section, 'src')
        dst = self.get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.mirror(self.buffers[src])

    def delete(self, section):
        src = self.get_option(section, 'src')
        del self.buffers[src]

    def list_buffers(self, section):
        a = []
        for buf in sorted(self.buffers.keys()):
            a.append('%s(mode=%s size=%dx%d)' %
                     (buf, self.buffers[buf].mode, self.buffers[buf].size[0], self.buffers[buf].size[1]))
        logger.info('buffers: %s' % (', '.join(a)))

    def add_text(self, section):
        src = self.get_option(section, 'src')
        # dst = self.get_option(section, 'dst', src)
        font_name = self.get_option(section, 'font')
        font_size = self.get_option(section, 'size', get='getint')
        text = self.get_option(section, 'text')
        color = hex_to_rgb(self.get_option(section, 'color'))
        position = map(int, self.get_option(section, 'position').split())
        font = ImageFont.truetype(font_name, font_size)
        draw = ImageDraw.Draw(self.buffers[src])
        draw.text(position, text, color, font=font)


def read_config_file(filename):
    """Read config file."""
    logger.info('Reading config file ' + filename)
    config = SafeConfigParser()


    config.add_section('daemon')
    config.set('daemon', 'user', 'pi')
    config.set('daemon', 'group', 'pi')

    config.add_section('aurorawatchuk')
    config.set('aurorawatchuk', 'status_url', 'http://aurorawatch-api.lancs.ac.uk/0.2/status/current-status.xml')
    config.set('aurorawatchuk', 'status_cache', '/home/pi/tmpfs/aurorawatchuk_status.ini')

    config.add_section('upload')
    # User must add appropriate values


    if filename:
        config_files_read = config.read(filename)
        if filename not in config_files_read:
            raise UserWarning('Could not read ' + filename)
        logger.debug('Successfully read ' + ', '.join(config_files_read))

    return config


def get_config_option(config, section, option,
                      default=None,
                      fallback_section=None,
                      get=None):
    sec = None
    read_file = False
    if config.has_option(section, option):
        sec = section
    elif fallback_section and config.has_option(fallback_section, option):
        sec = fallback_section
    else:
        return default

    if get is None:
        return config.get(sec, option)
    else:
        # For 'getboolean' etc
        return getattr(config, get)(sec, option)


def cmp_value_with_option(value, config, section, option, fallback_section='common'):
    def in_operator(a, b):
        return a in b

    def not_in_operator(a, b):
        return a not in b

    ops = {'<': operator.lt,
           '<=': operator.le,
           '==': operator.eq,
           '>': operator.gt,
           '>=': operator.ge,
           '!=': operator.ne,
           'is': operator.is_,
           'is not': operator.is_not,
           #'in': lambda(a, b): operator.contains(b,a), # Fix reversed operands
           #'not in': lambda(a, b): not operator.contains(b,a), # Fix reversed operands
           'in': in_operator,
           'not in': not_in_operator,
           }
    op_name = get_config_option(config, section, option + '_operator',
                                default='==',
                                fallback_section=fallback_section)
    if op_name not in ops:
        raise Exception('Unknown operator (%s)' % op_name)
    if isinstance(value, bool):
        cast = bool
    elif isinstance(value, (int, np.int64)):
        cast = int
    elif isinstance(value,(float, np.float64)):
        cast = float
    else:
        # Keep as str
        cast = lambda(x): x

    option_str = get_config_option(config, section, option,
                                   fallback_section=fallback_section)
    if op_name in ['in', 'not_in']:
        conf_value = []
        for s in option_str.split():
            conf_value.append(cast(s))
    else:
            conf_value = cast(option_str)
    return ops[op_name](value, conf_value)


def get_schedule(config):
    if args.schedule is not None:
        return args.schedule

    t = time.time()
    for sec in config.sections():
        if sec in ['DEFAULT', 'common', 'daemon']:
            continue
        if not config.has_option(sec, 'tasks'):
            continue

        if config.has_option(sec, 'solar_elevation'):
            latitude = get_config_option(config, sec, 'latitude',
                                         fallback_section='common', default=0, get='getfloat')
            longitude = get_config_option(config, sec, 'longitude',
                                          fallback_section='common', default=0, get='getfloat')
            val = get_solar_elevation(latitude, longitude, t)
            if not cmp_value_with_option(val, config, sec, 'solar_elevation', fallback_section='common'):
                continue

        if config.has_option(sec, 'aurorawatchuk_status'):
            val = get_aurorawatchuk_status(config)
            if not cmp_value_with_option(val, config, sec, 'aurorawatchuk_status', fallback_section='common'):
                continue

        # All tests passed
        return sec

    return 'common'


def get_solar_elevation(latitude, longitude, t):
    loc = astral.Location(('', '', latitude, longitude, 'UTC', 0))
    return loc.solar_elevation(datetime.datetime.utcfromtimestamp(t))


def get_aurorawatchuk_status(config, use_cache=True):
    filename = config.get('aurorawatchuk', 'status_cache')
    # Try to get status from cached value if possible and current
    if use_cache and os.path.exists(filename):
        try:
            cache = SafeConfigParser()
            cache.read(filename)
            status = cache.get('status', 'value')
            expires_str = cache.get('status', 'expires')
            expires = time.mktime(datetime.datetime.strptime(expires_str, '%a, %d %b %Y %H:%M:%S %Z').utctimetuple())
            time_left = expires - time.time()
            if time_left > 0:
                # Present and current
                try:
                    if time_left < 30:
                        print('Starting new thread to update status')
                        thread = threading.Thread(target=get_aurorawatchuk_status,
                                                  args=(config,),
                                                  kwargs=dict(use_cache=False))
                        thread.start()
                except:
                    pass
                return status

        except:
            logger.error('could not read cached AuroraWatch UK status')
            logger.debug(traceback.format_exc())

    # Fetch current status
    url = config.get('aurorawatchuk', 'status_url')
    status = 'green'
    try:
        logger.info('fetching status from %s', url)
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception('could not access %s' % url)
        xml = r.text
        xml_tree = ET.fromstring(xml.encode('UTF-8'))
        if xml_tree.tag != 'current_status':
            raise Exception('wrong root element')
        site_status = xml_tree.find('site_status')
        status = site_status.attrib['status_id']
        expires = r.headers['Expires']

        # Cache results for next time
        try:
            new_cache = SafeConfigParser()
            new_cache.add_section('status')
            new_cache.set('status', 'value', status)
            new_cache.set('status', 'expires', expires)
            with open(filename, 'w') as fh:
                new_cache.write(fh)
        except:
            logger.error('could not save AuroraWatch UK status to cache file %s', filename)
            logger.debug(traceback.format_exc())


    except:
        logger.error('could not get AuroraWatch UK status')
        logger.debug(traceback.format_exc())
    return status


def init_camera(camera, config):
    controls = camera.get_controls()
    camera.start_video_capture()

    # Read all camera controls defined in the config file
    for c in controls:
        cl = c.lower()
        if config.has_option('camera', cl):
            value = config.get('camera', cl)
            default_value = controls[c]['DefaultValue']
            control_type = getattr(asi, 'ASI_' + c.upper())
            logger.debug('set control value %s to %s', cl, value)
            if value == 'auto':
                camera.set_control_value(control_type, default_value, auto=True)
            else:
                # Cast value to same type as default_value
                camera.set_control_value(control_type, type(default_value)(value), auto=False)


    if config.has_option('camera', 'image_type'):
        image_type = config.get('camera', 'image_type')
        logger.debug('set image type to %s', image_type)
        camera.set_image_type(getattr(asi, 'ASI_IMG_' + image_type.upper()))
    time.sleep(2)


def get_control_values(camera):
    controls = camera.get_controls()
    r = {}
    for k in controls:
        r[k] = camera.get_control_value(controls[k]['ControlType'])[0]

    # Fix up certain keys
    if 'Exposure' in r:
        r['Exposure'] = '%.6f' % (r['Exposure'] / 1000000.0)
    if 'Temperature' in r:
        r['Temperature'] = '%.1f' % (r['Temperature']/10.0)
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


def run_camera():
    global config
    global camera
    global sampling_interval

    # This should be called after dropping root privileges because
    # it uses safe_eval to convert strings to numbers or
    # lists (not guaranteed safe!)
    camera = get_camera(config)
    init_camera(camera, config)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    try:
        get_log_file_for_time(time.time(), log_filename)
        logger.info('Starting sampling thread')

        do_every(config, process_tasks)
        while take_images:
            time.sleep(2)

        # Wait until all other threads have (or should have)
        # completed
        try:
            sampling_interval_lock.acquire(True)
            si = sampling_interval
        finally:
            sampling_interval_lock.release()

        for n in range(int(round(si)) + 1):
            if threading.activeCount() == 1:
                break
            time.sleep(1)


    except Exception as e:
        get_log_file_for_time(time.time(), log_filename)
        logger.error(traceback.format_exc())
        time.sleep(5)

       
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result


def cancel_sampling_threads():
    threads = threading.enumerate()
    for t in threads[1:]:
        t.cancel()        
    
    # If all the other threads have completed then exit; exit anyway
    # after a short time has passed
    t = time.time()
    while time.time() < t + 1 and len(threading.enumerate()) > 2:
        time.sleep(0.1)
    #sys.exit()


take_images = True
def stop_handler(signal, frame):
    global take_images
    global camera
    get_log_file_for_time(time.time(), log_filename)
    logger.info('Stopping sampling threads')
    take_images = False
    cancel_sampling_threads()
    camera.stop_video_capture()
    camera.close()


def do_every (config, worker_func, iterations = 0):
    global sampling_interval
    if iterations != 1:
        # Identify current operating condition to find the actions to do and sampling interval
        schedule = get_schedule(config)
        logger.info('using schedule %s', schedule)

        # Schedule the next worker thread. Aim to start at the next
        # multiple of sampling interval. Take current time, add 1.25
        # of the interval and then find the nearest
        # multiple. Calculate delay required.
        now = time.time()
        interval = get_config_option(config, schedule, 'sampling_interval',
                                     fallback_section='common',
                                     default=default_sampling_interval,
                                     get='getfloat')
        delay = round_to(now + (1.25 * interval), interval) - now
        # Avoid lockups by many threads piling up. Impose a minimum delay
        if delay < 0.1:
            delay = 0.1
        t = threading.Timer(delay,
                            do_every, 
                            [config, worker_func,
                             0 if iterations == 0 else iterations-1])
        t.daemon = True
        t.start()

        # Update the global sampling_interval so that cancel_sampling_threads knows how long to wait.
        # This could be attempted by multiple capture threads so must use a lock. Don't wait for the lock to be
        # available.
        if sampling_interval_lock.acquire(False):
            try:
                logger.debug('sampling_interval_lock: acquired lock')
                sampling_interval = interval
            finally:
                logging.debug('sampling_interval_lock: released lock')
                sampling_interval_lock.release()
        else:
            logger.error('sampling_interval_lock: could not acquire lock')


    try:
        worker_func(schedule)
    except Exception as e:
        get_log_file_for_time(time.time(), log_filename)
        logger.error(traceback.format_exc())


def round_to(n, nearest):
    return round(n / float(nearest)) * nearest


def get_camera(config):
    '''Return camera object based on config settings'''
    return asi.Camera(0)


def capture_image():
    global camera
    with capture_image.lock:
        t = time.time()
        img_info = get_control_values(camera)
        img = camera.capture_video_frame()
        img_info['DateTime'] = time.strftime('%Y-%m-%d %H:%M:%S+00:00', time.gmtime(t))

        # Take CPU temperature as system temperature
        img_info['SystemTemperature'] = 'unknown'
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            img_info['SystemTemperature'] = '%.2f' % (float(f.read().strip())/1000)

        return img, img_info, t

capture_image.lock = threading.Lock()

# Each sampling action is made by a new thread. This function uses a
# lock to avoid contention for the camera. If the lock cannot be
# acquired the attempt is abandoned. The lock is released after the
# sample has been taken. This means two instances of process_schedule()
# can occur at the same time, whilst the earlier one writes data and
# possibly sends a real-time data packet over the network. A second
# lock (write_to_csv_file.lock) is used to avoid contention on writing
# results to a file.
def process_tasks(schedule):
    tasks = get_config_option(config, schedule, 'tasks', fallback_section='common', default='')
    print('tasks: ' + repr(tasks))

    act = Task(config, schedule)
    act.run_tasks(tasks.split())

    return

process_tasks.lock = threading.Lock()


def data_to_str(data, separator=',', comments='#', want_header=False):
    separator = ','
    header = comments + 'sample_time'
    fstr = '{sample_time:.3f}'
    d = dict(sample_time=data['sample_time'], separator=separator)
    for c in ('x', 'y', 'z', 'sensor_temperature'):
        if c in data:
            d[c] = data[c]
            header += separator + c
            fstr += '{separator}{' + c + ':.3f}'
    header += '\n'
    fstr += '\n'
    s = fstr.format(**d)
    if want_header:
        return s, header
    else:
        return s


def get_log_file_for_time(t, fstr, 
                          mode='a', 
                          delay=True, 
                          name=__name__):
    if fstr is None:
        return
    fh = get_log_file_for_time.fh
    tmp_name = time.strftime(fstr, time.gmtime(t))

    if fh is not None:
        if fh.stream and fh.stream.closed:
            fh = None
        elif fh.stream.name != tmp_name:
            # Filename has changed
            fh.close()
            fh = None
        
    if fh is None:
        # File wasn't open or filename changed
        p = os.path.dirname(tmp_name)
        if not os.path.isdir(p):
            os.makedirs(p)

        fh = logging.FileHandler(tmp_name, mode=mode, delay=delay)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%dT%H:%M:%SZ')
        fh.setFormatter(formatter)
        logger = logging.getLogger(name)
        for h in logger.handlers:
            # Only remove file handlers 
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
        logger.addHandler(fh)


# http://stackoverflow.com/questions/4296249/how-do-i-convert-a-hex-triplet-to-an-rgb-tuple-and-back
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


# http://stackoverflow.com/questions/4296249/how-do-i-convert-a-hex-triplet-to-an-rgb-tuple-and-back
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


get_log_file_for_time.fh = None


logger = logging.getLogger(__name__)
camera = None
default_sampling_interval = 120
sampling_interval = default_sampling_interval
sampling_interval_lock = threading.Lock()


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    # Parse command line arguments
    progname = os.path.basename(sys.argv[0]).partition('.')[0]
    default_config_file = \
        os.path.join(os.path.sep, 'etc', 'camera.ini')

    parser = \
        argparse.ArgumentParser(description='AuroraWatch camera daemon')

    parser.add_argument('-c', '--config-file', 
                        default=default_config_file,
                        help='Configuration file')
    parser.add_argument('--log-level', 
                        choices=['debug', 'info', 'warning', 
                                 'error', 'critical'],
                        default='info',
                        help='Control how much detail is printed',
                        metavar='LEVEL')
    parser.add_argument('--log-format',
                        default='%(levelname)s:%(message)s',
                        help='Set format of log messages',
                        metavar='FORMAT')
    parser.add_argument('--schedule',
                        help='Override automatic scheduling')

    args = parser.parse_args()

    config = read_config_file(args.config_file)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format=args.log_format, datefmt='%Y-%m-%dT%H:%M:%SZ')

    if (args.schedule is not None and
            (not config.has_section(args.schedule) or not config.has_option(args.schedule, 'tasks'))):
        raise Exception('%s is not a valid schedule', args.schedule)

    log_filename = None
    if config.has_option('logfile', 'filename'):
        log_filename = config.get('logfile', 'filename')
        
    get_log_file_for_time(time.time(), log_filename)
    logger.info(progname + ' started')

    run_camera()

