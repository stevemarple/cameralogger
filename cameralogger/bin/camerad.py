#!/usr/bin/env python

import argparse
import astral
import datetime
import logging
import lxml.etree as ET
import numpy as np
import operator
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import re
import requests
import signal
import six
import sys
import threading
import time
import traceback
import zwoasi as asi

__author__ = 'Steve Marple'
__version__ = '0.0.1'
__license__ = 'PSF'

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

    def _get_color(self, section, default=None, fallback_section=None, get=None, raise_=True, option='color'):
        color = self._get_option(section, option, default=default, fallback_section=fallback_section, get=get,
                                 raise_=raise_)
        if re.match('^[0-9a-f]{3,6}$', color, re.IGNORECASE):
            return '#' + color
        else:
            return color

    def _get_mode(self, section, default=None, fallback_section=None, get=None, raise_=True, option='mode'):
        mode = self._get_option(section, option, default=default, fallback_section=fallback_section, get=get,
                                raise_=raise_)
        if len(mode) and mode[0] == '@':
            mode = mode.lstrip('@')
            if mode not in self.buffers:
                raise ValueError('no buffer named "%s"' % mode)
            return self.buffers[mode].mode
        else:
            return mode

    def _get_option(self, section, option, default=None, fallback_section=None, get=None, raise_=True):
        r = get_config_option(self.config, section, option, default=default, fallback_section=fallback_section, get=get)
        if raise_ and r is None:
            if fallback_section is None:
                raise Exception('could not find value for "%s" in section "%s"' % (option, section))
            else:
                raise Exception(
                    'could not find value for "%s" in sections "%s" or "%s"' % (option, section, fallback_section))
        return r

    def _get_size(self, section, default=None, fallback_section=None, get=None, raise_=True, option='size'):
        size = self._get_option(section, option, default=default, fallback_section=fallback_section, get=get,
                                raise_=raise_)
        if len(size) and size[0] == '@':
            size = size.lstrip('@')
            if size not in self.buffers:
                raise ValueError('no buffer named "%s"' % size)
            return self.buffers[size].size
        else:
            return map(int, size.split())

    def run_tasks(self, tasks):
        for section in tasks:
            if not config.has_section(section):
                raise ValueError('no section called %s' % section)
            act = self.config.get(section, 'action')
            if not hasattr(self, act):
                raise ValueError('unknown action (%s)' % act)
            logger.debug('calling action %s(%s)' % (act, repr(section)))
            getattr(self, act)(section)

    # Tasks are below
    def add_text(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        font_name = self._get_option(section, 'font', fallback_section='common')
        font_size = self._get_option(section, 'fontsize', fallback_section='common', get='getint')
        text = self._get_option(section, 'text')
        color = self._get_color(section, fallback_section='common')
        position = map(int, self._get_option(section, 'position').split())
        font = ImageFont.truetype(font_name, font_size)
        if dst != src:
            self.buffers[dst] = self.buffers[src].copy()
        draw = ImageDraw.Draw(self.buffers[dst])
        draw.text(position, text, color, font=font)

    def alpha_composite(self, section):
        src1 = self._get_option(section, 'src1')
        src2 = self._get_option(section, 'src2')
        dst = self._get_option(section, 'dst', src1)
        self.buffers[dst] = Image.alpha_composite(self.buffers[src1], self.buffers[src2])

    def autocontrast(self, section):
        src = self._get_option(section, 'src')
        cutoff = self._get_option(section, 'cutoff', 0, get='getfloat')
        ignore = self._get_option(section, 'ignore', raise_=False)
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.autocontrast(self.buffers[src], cutoff, ignore)

    def blend(self, section):
        src1 = self._get_option(section, 'src1')
        src2 = self._get_option(section, 'src2')
        dst = self._get_option(section, 'dst', src1)
        alpha = self._get_option(section, 'alpha', get='getfloat')
        self.buffers[dst] = Image.blend(self.buffers[src1], self.buffers[src2], alpha)

    def capture(self, section):
        dst = self._get_option(section, 'dst')
        img, self.capture_info[dst], t = capture_image()
        if self.time is None:
            self.time = t  # Record time of first capture
        self.buffers[dst] = Image.fromarray(img[:, :, ::-1])  # Swap from BGR order

    def colorize(self, section):
        src = self._get_option(section, 'src')
        black = self._get_option(section, 'black')
        white = self._get_option(section, 'white')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.colorize(self.buffers[src], black, white)

    def composite(self, section):
        src1 = self._get_option(section, 'src1')
        src2 = self._get_option(section, 'src2')
        mask = self._get_option(section, 'mask')
        dst = self._get_option(section, 'dst', src1)
        self.buffers[dst] = Image.composite(self.buffers[src1], self.buffers[src2], self.buffers[mask])

    def convert(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        mode = self._get_mode(section)
        self.buffers[dst] = self.buffers[src].convert(mode)

    def copy(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst')
        self.buffers[dst] = self.buffers[src].copy()

    def crop(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        position = map(int, self._get_option(section, 'position').split())
        self.buffers[dst] = self.buffers[src].crop(position)
        self.buffers[dst].load()  # crop() is lazy operation; break connection

    def delete(self, section):
        src = self._get_option(section, 'src')
        del self.buffers[src]

    def equalize(self, section):
        src = self._get_option(section, 'src')
        mask = self._get_option(section, 'mask', raise_=False)
        white = self._get_option(section, 'white')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.equalize(self.buffers[src], mask)

    def expand(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        # border is left, top, right, bottom
        border = self._get_option(section, 'border', raise_=False)
        if border is not None:
            border = tuple(map(int, self._get_option(section, 'border').split()))
        else:
            # size: width, height
            # position: left, top
            size = self._get_size(section)
            if len(size) == 1:
                size.append(size[0])
            position = map(int, self._get_option(section, 'position', 0).split())
            if len(position) == 1:
                position.append(position[0])
            src_size = self.buffers[src].size
            border = (
            position[0], position[1], size[0] - src_size[0] - position[0], size[1] - src_size[1] - position[1])

        fill = self._get_option(section, 'fill', 0)
        self.buffers[dst] = ImageOps.expand(self.buffers[src], border, fill)

    def fit(self, section):
        src = self._get_option(section, 'src')
        size = self._get_size(section)
        method = self._get_option(section, 'method', Image.NEAREST)
        if method is not 0:
            method = getattr(Image, method.upper())
        bleed = self._get_option(section, 'bleed', 0, get='getfloat')
        centering = tuple(map(float, self._get_option(section, 'centering', '0.5 0.5').split()))
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.fit(self.buffers[src], size, method, bleed, centering)

    def flip(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.flip(self.buffers[src])

    def grayscale(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.grayscale(self.buffers[src])

    def invert(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.invert(self.buffers[src])

    def list_buffers(self, section):
        a = []
        for buf in sorted(self.buffers.keys()):
            a.append('%s(mode=%s size=%dx%d)' %
                     (buf, self.buffers[buf].mode, self.buffers[buf].size[0], self.buffers[buf].size[1]))
        logger.info('buffers: %s' % (', '.join(a)))

    def load(self, section):
        dst = self._get_option(section, 'dst')
        filename = self._get_option(section, 'filename')
        self.buffers[dst] = Image.open(filename)

    def mirror(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.mirror(self.buffers[src])

    def merge(self, section):
        mode = self._get_mode(section)
        src1 = self._get_option(section, 'src1')
        bands = [self.buffers[src1]]
        for n in range(1, len(Image.new('mode', (1, 1)).getbands())):
            src = self._get_option(section, 'src%d' % n)
            bands.append(self.buffers[src])
        dst = self._get_option(section, 'dst', src1)
        self.buffers[dst] = Image.merge(mode, bands)

    def new(self, section):
        dst = self._get_option(section, 'dst')
        mode = self._get_option(section, 'mode')
        size = self._get_size(section)
        color = self._get_color(section, default=0)
        self.buffers[dst] = Image.new(mode, size, color)

    def paste(self, section):
        src1 = self._get_option(section, 'src1')
        src2 = self._get_option(section, 'src2')
        position = map(int, self._get_option(section, 'position', '0 0').split())
        mask = self._get_option(section, 'mask', raise_=False)
        # In keeping with most other functions allow paste() to be nondestructive.
        dst = self._get_option(section, 'dst', src1)
        if dst == src1:
            self.buffers[src1].paste(src2, position, mask)
        else:
            self.buffers[dst] = self.buffers[src1].copy()
            self.buffers[dst].paste(src2, position, mask)

    def posterize(self, section):
        src = self._get_option(section, 'src')
        bits = self._get_option(section, 'bits', get='getint')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.posterize(self.buffers[src], bits)

    def resize(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        size = self._get_size(section)
        resample = self._get_option(section, 'resample', Image.NEAREST)
        if resample is not 0:
            resample = getattr(Image, resample.upper())
        self.buffers[dst] = self.buffers[src].resize(size, resample)

    def rotate(self, section):
        src = self._get_option(section, 'src')
        angle = self._get_option(section, 'angle', get='getfloat')
        resample = self._get_option(section, 'resample', Image.NEAREST)
        if resample is not 0:
            resample = getattr(Image, resample.upper())
        expand = self._get_option(section, 'expand', get='getbool', raise_=False)
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = self.buffers[src].rotate(angle, resample, expand)

    def save(self, section):
        src = self._get_option(section, 'src')
        tm = time.gmtime(self.time)
        filename = self._get_option(section, 'filename')
        filename = time.strftime(filename, tm)
        self.buffers[src].save(time.strftime(filename, tm))
        logger.info('saved %s', filename)

    def split(self, section):
        src = self._get_option(section, 'src')
        images = self.buffers[src]
        for n in range(len(images)):
            dst = self._get_option(section, 'dst%d' % n)
            self.buffers[dst] = images[n]

    def solarize(self, section):
        src = self._get_option(section, 'src')
        threshold = self._get_option(section, 'bits', 128, get='getint')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.solarize(self.buffers[src], threshold)

    def transpose(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        method = getattr(Image, self._get_option(section, 'method'))
        self.buffers[dst] = self.buffers[src].tranpose(method)


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
           # 'in': lambda(a, b): operator.contains(b,a), # Fix reversed operands
           # 'not in': lambda(a, b): not operator.contains(b,a), # Fix reversed operands
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
    elif isinstance(value, (float, np.float64)):
        cast = float
    else:
        # Keep as str
        cast = lambda (x): x

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
        if get_config_option(config, sec, 'tasks', fallback_section='common') is None:
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
        # sys.exit()


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


def do_every(config, worker_func, iterations=0):
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
                             0 if iterations == 0 else iterations - 1])
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
    """Return camera object based on config settings"""
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
            img_info['SystemTemperature'] = '%.2f' % (float(f.read().strip()) / 1000)

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
