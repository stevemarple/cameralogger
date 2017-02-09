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

import astral
from atomiccreate import atomic_symlink
from atomiccreate import smart_open
import datetime
from fractions import Fraction
import logging
import numpy as np
import operator
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import re
import six
import subprocess
import sys
import time
import traceback

from cameralogger.formatters import MyFormatter
from cameralogger.formatters import LatLon
from aurorawatchuk.snapshot import AuroraWatchUK_SS

if sys.version_info[0] >= 3:
    # noinspection PyCompatibility
    from configparser import RawConfigParser
else:
    # noinspection PyCompatibility
    from ConfigParser import RawConfigParser

__author__ = 'Steve Marple'
__version__ = '0.1.1'
__license__ = 'MIT'


class Tasks(object):
    """A collection of tasks than can be performed to create, act upon and save image buffers.

    Camera: a camera object such as :class:`.dummy.Camera`, :class:`.pi.Camera`, :class:`.zwo.Camera`"""

    def __init__(self, camera=None, config=None, schedule=None, schedule_info={}):
        self.camera = camera
        self.config = config
        self.schedule = schedule
        self.schedule_info = schedule_info
        self.buffers = {}
        self.time = None
        self.capture_info = None
        self.format_dict = None

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

    def _make_dict(self, section):
        if self.format_dict is not None:
            return self.format_dict

        d = {}
        if self.capture_info is not None:
            # Copy all exposure settings
            d = self.capture_info.copy()
            # Remove subsecond part from time (datetime %S does not work as expected)
            d['DateTime'] = datetime.datetime.utcfromtimestamp(int(self.time))
        d['Schedule'] = self.schedule
        d['Section'] = section
        lat = self._get_option(section, 'latitude', fallback_section='common', get='getfloat')
        lon = self._get_option(section, 'longitude', fallback_section='common', get='getfloat')
        lat_lon = LatLon(lat, lon)
        d['LatLon'] = lat_lon

        for k, v in six.iteritems(self.schedule_info):
            d[k] = v
        self.format_dict = d
        return self.format_dict

    def format_str(self, section, s):
        return MyFormatter().format(s, **self._make_dict(section))

    def run_tasks(self, tasks):
        for section in tasks:
            if not self.config.has_section(section):
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
        text = self._get_option(section, 'text')
        font_name = self._get_option(section, 'font', fallback_section='common')
        font_size = self._get_option(section, 'fontsize', fallback_section='common', get='getint')
        color = self._get_color(section, fallback_section='common')
        unicode = self._get_option(section, 'unicode', False, fallback_section='common', get='getboolean')
        position = map(int, self._get_option(section, 'position').split())
        font = ImageFont.truetype(font_name, font_size)
        if unicode:
            text = unescape_unicode(text)
        text = self.format_str(section, text)
        color = self.format_str(section, color)
        if dst != src:
            self.buffers[dst] = self.buffers[src].copy()
        draw = ImageDraw.Draw(self.buffers[dst])
        draw.text(position, text, color, font=font)

    def add_multiline_text(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        text = self._get_option(section, 'text')
        font_name = self._get_option(section, 'font', fallback_section='common')
        font_size = self._get_option(section, 'fontsize', fallback_section='common', get='getint')
        color = self._get_color(section, fallback_section='common')
        unicode = self._get_option(section, 'unicode', False, fallback_section='common', get='getboolean')
        spacing = self._get_option(section, 'spacing', fallback_section='common', get='getint')
        align = self._get_option(section, 'spacing', 'left', fallback_section='common')
        position = list(map(int, self._get_option(section, 'position').split()))
        font = ImageFont.truetype(font_name, font_size)
        if unicode:
            text = unescape_unicode(text)
        text = self.format_str(section, text)
        color = self.format_str(section, color)
        if dst != src:
            self.buffers[dst] = self.buffers[src].copy()
        draw = ImageDraw.Draw(self.buffers[dst])
        if hasattr(draw, 'multiline_text'):
            draw.multiline_text(position, text, color, font=font, spacing=spacing, align=align)
        else:
            # Compatibility, but without align
            for s in text.splitlines():
                draw.text(position, s, color, font=font)
                position[1] += spacing

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
        img, info, t = self.camera.capture_image(section)
        if self.time is None:
            self.time = t  # Record time of first capture
            self.capture_info = info
        self.buffers[dst] = img

    def colorize(self, section):
        src = self._get_option(section, 'src')
        black = self._get_option(section, 'black')
        white = self._get_option(section, 'white')
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = ImageOps.colorize(self.buffers[src], black, white)

    def command(self, section):
        """Run shell command."""
        cmd = self._get_option(section, 'cmd')
        check_call = self._get_option(section, 'check_call', True, get='getboolean')
        background = self._get_option(section, 'background', False, get='getboolean')
        cmd = self.format_str(section, cmd)
        if check_call:
            logger.debug('running command "%s"', cmd)
            subprocess.check_call(cmd, shell=True)
        elif background:
            logger.debug('running command "%s" in background', cmd)
            raise Exception('not implemented')
        else:
            logger.debug('running command "%s" (no checks)', cmd)
            subprocess.call(cmd, shell=True)

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
            position = list(map(int, self._get_option(section, 'position', 0).split()))
            if len(position) == 1:
                position.append(position[0])
            src_size = self.buffers[src].size
            border = (position[0], position[1],
                      size[0] - src_size[0] - position[0], size[1] - src_size[1] - position[1])

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

    def list_buffers(self, _):
        a = []
        for buf in sorted(self.buffers.keys()):
            a.append('%s(mode=%s size=%dx%d)' %
                     (buf, self.buffers[buf].mode, self.buffers[buf].size[0], self.buffers[buf].size[1]))
        logger.info('buffers: %s' % (', '.join(a)))

    def load(self, section):
        dst = self._get_option(section, 'dst')
        filename = self.format_str(section, self._get_option(section, 'filename'))
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
        expand = self._get_option(section, 'expand', get='getboolean', raise_=False)
        dst = self._get_option(section, 'dst', src)
        self.buffers[dst] = self.buffers[src].rotate(angle, resample, expand)

    def save(self, section):
        src = self._get_option(section, 'src')
        filename = self.format_str(section, self._get_option(section, 'filename'))
        tempfile = self._get_option(section, 'tempfile', fallback_section='common', get='getboolean', raise_=False)
        chmod = self._get_option(section, 'chmod', fallback_section='common', raise_=False)
        if chmod:
            chmod = int(chmod, 8)
        with smart_open(filename, 'wb', use_temp=tempfile, chmod=chmod) as f:
            self.buffers[src].save(f)
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

    def symlink(self, section):
        src_filename = self.format_str(section, self._get_option(section, 'src_filename'))
        dst_filename = self.format_str(section, self._get_option(section, 'dst_filename'))
        raise_ = self._get_option(section, 'raise', True, fallback_section='common', get='getboolean')
        tempfile = self._get_option(section, 'tempfile', False, fallback_section='common', get='getboolean', raise_=False)
        try:
            if tempfile:
                atomic_symlink(src_filename, dst_filename)
            else:
                os.symlink(src_filename, dst_filename)
            logger.info('created symlink %s -> %s', src_filename, dst_filename)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            if raise_:
                raise
            else:
                logger.error('Could not create symlink %s -> %s', src_filename, dst_filename)
                logger.debug(traceback.format_exc())

    def transpose(self, section):
        src = self._get_option(section, 'src')
        dst = self._get_option(section, 'dst', src)
        method = getattr(Image, self._get_option(section, 'method'))
        self.buffers[dst] = self.buffers[src].tranpose(method)


def unescape_unicode(s):
    if sys.version_info[0] >= 3:
        return bytes(s, 'UTF-8').decode('unicode-escape')
    else:
        return s.decode('unicode-escape')

def read_config_file(filename):
    """Read config file."""
    logger.info('Reading config file ' + filename)
    config = RawConfigParser()

    config.add_section('daemon')
    config.set('daemon', 'user', 'pi')
    config.set('daemon', 'group', 'pi')

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
    if config.has_option(section, option):
        sec = section
    elif fallback_section and config.has_option(fallback_section, option):
        sec = fallback_section
    else:
        return default

    if get is None:
        return config.get(sec, option)
    elif get == 'getfraction':
        s = config.get(sec, option)
        return Fraction(*map(int, s.split('/')))
    else:
        # For 'getboolean' etc
        return getattr(config, get)(sec, option)


def cmp_value_with_option(value, config, section, option, fallback_section='common'):
    def in_operator(a, b):
        return a in b

    def not_in_operator(a, b):
        return a not in b

    def no_op(a):
        return a

    ops = {'<': operator.lt,
           '<=': operator.le,
           '==': operator.eq,
           '>': operator.gt,
           '>=': operator.ge,
           '!=': operator.ne,
           'is': operator.is_,
           'is not': operator.is_not,
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
        cast = no_op

    option_str = get_config_option(config, section, option,
                                   fallback_section=fallback_section)
    if op_name in ['in', 'not_in']:
        conf_value = map(cast, option_str.split())
    else:
        conf_value = cast(option_str)
    return ops[op_name](value, conf_value)


def get_schedule(config, forced_schedule=None):
    """Get schedule to use.

    Allow for schedule to be overridden for testing.
    """

    t = time.time()
    info = {
        # TODO: Allow cache directory and baseurl to be overridden from config settings
        'AuroraWatchUK': AuroraWatchUK_SS(),
    }

    if forced_schedule:
        sections = [forced_schedule]
    else:
        sections = [x for x in config.sections() if (x not in ['DEFAULT', 'common', 'daemon'] and
                                                     get_config_option(config, x, 'sampling_interval') is not None)]

    for sec in sections:
        # Compute solar elevation and make it available for output. Allow for location to be customised in each section
        lat = get_config_option(config, sec, 'latitude',
                                             fallback_section='common', default=0, get='getfloat')
        lon = get_config_option(config, sec, 'longitude',
                                              fallback_section='common', default=0, get='getfloat')
        info['SolarElevation'] = get_solar_elevation(lat, lon, t)

        # Compare with solar elevation
        if config.has_option(sec, 'solar_elevation') \
            and not cmp_value_with_option(info['SolarElevation'], config, sec, 'solar_elevation',
                                          fallback_section='common'):
            continue

        # Compare with AuroraWatch
        if config.has_option(sec, 'aurorawatchuk_status') \
            and not cmp_value_with_option(info['AuroraWatchUK'].status.level, config, sec, 'aurorawatchuk_status',
                                          fallback_section='common'):
            continue

        # Test file existence
        if config.has_option(sec, 'file_exists') and not os.path.exists(config.get(sec, 'file_exists')):
            continue

        # Test file non-existence
        if config.has_option(sec, 'file_not_exist') and os.path.exists(config.get(sec, 'file_not_exist')):
            continue

        # All tests passed
        return sec, info

    if forced_schedule:
        return forced_schedule, info

    return 'common', info


def get_solar_elevation(latitude, longitude, t):
    loc = astral.Location(('', '', latitude, longitude, 'UTC', 0))
    return loc.solar_elevation(datetime.datetime.utcfromtimestamp(t))


# Each sampling action is made by a new thread. This function uses a
# lock to avoid contention for the camera. If the lock cannot be
# acquired the attempt is abandoned. The lock is released after the
# sample has been taken. This means two instances of process_schedule()
# can occur at the same time, whilst the earlier one writes data and
# possibly sends a real-time data packet over the network.
def process_tasks(camera, config, schedule, schedule_info):
    task_list = get_config_option(config, schedule, 'tasks', fallback_section='common', default='')
    logger.debug('tasks: ' + repr(task_list))
    tasks = Tasks(camera, config, schedule, schedule_info)
    tasks.run_tasks(task_list.split())
    return


logger = logging.getLogger(__name__)
