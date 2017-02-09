#!/usr/bin/env python

# timelapse - create timelapses from camera images by calling FFmpeg.
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

import argparse
import datetime
import logging
import numpy as np
import os
from PIL import Image
import six
import subprocess
import time


__author__ = 'Steve Marple'
__version__ = '0.1.1'
__license__ = 'MIT'


def find_start_end_frames(st, et, step, filename_fstr):
    r_st = r_et = None
    t = st
    while t < et:
        dt = datetime.datetime.fromtimestamp(t)
        filename = filename_fstr.format(DateTime=dt)
        if os.path.exists(filename):
            r_st = t
            break
        t += step
    if r_st is None:
        return None, None

    t = et
    while t >= r_st:
        dt = datetime.datetime.fromtimestamp(t)
        filename = filename_fstr.format(DateTime=dt)
        if os.path.exists(filename):
            r_et = t
            break
        t -= step

    return r_st, r_et


def ffmpeg(start_time, end_time, step, filename_fstr, output_filename,
           size=None, duration=None, speed_up=None, ifr=None, ofr=None, jitter=None):
    def get_filename(t):
        return filename_fstr.format(DateTime=datetime.datetime.fromtimestamp(t))

    processing_start_time = time.time()
    img_mode = 'RGB'  # Mode for all images
    # Remember et is inclusive
    start_time_s = start_time.astype('datetime64[s]').astype(int)
    end_time_s = end_time.astype('datetime64[s]').astype(int)
    st, et = find_start_end_frames(start_time_s,
                                   end_time_s,
                                   step,
                                   filename_fstr)
    if st is None:
        raise ValueError('could not find any images in time range')

    if jitter:
        if step <= jitter:
            raise ValueError('step size must be more than 2 * jitter')
        # Remove jitter effects from computed start/end times
        if abs(start_time_s - st) <= jitter:
            st = start_time_s
        if abs(end_time_s - et) <= jitter:
            et = end_time_s


    input_duration = et - st
    frames = input_duration / step  # Includes repeated frames
    if ifr is not None:
        pass
    elif duration is not None:
        ifr = round(float(frames) / duration)
    elif speed_up is not None:
        ifr = speed_up * float(frames) / input_duration
    elif ifr is None:
        ifr = float(frames) / input_duration

    if ofr is None:
        ofr = 60
        # Can a lower sensible rate be used, but one which is still larger than ifr?
        for fr in (5, 10, 15, 20, 30, 60):
            if fr >= ifr:
                ofr = fr
                break

    first_frame = Image.open(filename_fstr.format(DateTime=datetime.datetime.fromtimestamp(st))).convert('RGB')

    # TODO: Add code to run tasks at start/end of time lapse, eg add text, dissolving to/from first/last frames

    if not size:
        size = tuple(first_frame.size)  # Force to be tuple so that img.size comparison works later
    else:
        if isinstance(size, six.string_types):
            if size[-1] == '%':
                # Percentage, like 50%
                ratio = float(size[:-1]) / 100
                size = (int(round(first_frame.size[0] * ratio)), int(round(first_frame.size[1] * ratio)))
            else:
                # width x height, possibly with one missing
                width, _, height = size.partition('x')
                if width == '':
                    if height == '':
                        # Will not occur as long an empty strings caught previously
                        raise ValueError('unknown size format (%s)' % size)
                    else:
                        height = int(height)
                        width = int(round(first_frame.size[0] * height / float(first_frame.size[1])))
                else:
                    width = int(width)
                    if height == '':
                        height = int(round(first_frame.size[1] * width / float(first_frame.size[0])))
                    else:
                        height = int(height)
                size = (width, height)
        elif isinstance(size, float):
            size = (int(round(first_frame.size[0] * size)), int(round(first_frame.size[1] * size)))
        elif len(size) == 2:
            # Assume iterable of numbers
            size = tuple(map(int, size))
        else:
            raise ValueError('unknown size format (%s)' % size)

    # Set up a subprocess to run ffmpeg
    cmd = ('ffmpeg',
           '-loglevel', 'error',
           '-y',  # overwrite
           '-framerate', str(ifr),  # input frame rate
           '-s', '%dx%d' % (size[0], size[1]),  # size of image
           '-pix_fmt', 'rgb24',  # format
           '-f', 'rawvideo',
           '-i', '-',  # read from stdin
           '-vcodec', 'libx264',  # set output encoding
           '-r', str(ofr),  # output frame rate
           output_filename)
    logger.info('Running command ' + ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    img = None
    t = st
    stats = {'frames': 0,
             'images': 0,
             'time_taken': None}

    while t <= et:
        found = False
        tries = [t]
        if jitter:
            tries += list(range(t+1, t+jitter+1))  # Additional tries for late samples
            # tries += list(range(t-1, t-jitter-1, -1))  # Additional tries for early samples
        for t2 in tries:
            filename = get_filename(t2)
            if os.path.exists(filename):
                found = True
                logger.info('reading %s', filename)
                img = Image.open(filename)
                stats['images'] += 1
                break

        if found:
            pass
        elif img is None:
            logger.debug('missing %s', filename)
            img = Image.new('RGB', size)
        else:
            if jitter and logger.level <= logging.DEBUG:
                # Log the target filename
                filename = get_filename(t)
            logger.debug('missing intermediate image %s', filename)

        if img.mode != img_mode:
            logger.debug('converting image to mode %s', img_mode)
            img = img.convert(img_mode)
        if tuple(img.size) != size:
            logger.debug('resizing image')
            img = img.resize(size, Image.BILINEAR)

        proc.stdin.write(img.tobytes())
        stats['frames'] += 1
        t += step
    proc.communicate()
    stats['time_taken'] = int(time.time() - processing_start_time)

    logger.info('saved to %s', output_filename)
    stats['output_filename'] = output_filename
    logger.info('used {stats[images]} images to construct {stats[frames]} frames in {stats[time_taken]} s'
                .format(stats=stats))
    return stats


logger = logging.getLogger(__name__)
if __name__ == '__main__':
    default_config_file = os.path.join(os.path.sep, 'etc', 'camera.ini')
    parser = \
        argparse.ArgumentParser(description='')

    parser.add_argument('-c', '--config-file',
                        default=default_config_file,
                        help='Configuration file')

    parser.add_argument('-e', '--end-time',
                        required=True,
                        help='End time for time lapse')
    parser.add_argument('-f', '--fstr',
                        required=True,
                        help='Filenames format string')
    parser.add_argument('-j', '--jitter',
                        type=int,
                        default=0,
                        help='Accept jitter',
                        metavar='SECONDS')
    parser.add_argument('--log-level',
                        choices=['debug', 'info', 'warning',
                                 'error', 'critical'],
                        default='info',
                        help='Control how much detail is printed',
                        metavar='LEVEL')
    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output filename')
    parser.add_argument('-r', '--output-frame-rate',
                        type=float,
                        help='Input frame rate',
                        metavar='FPS')
    parser.add_argument('--size',
                        help='Video size (WxH or percentage')
    parser.add_argument('-s', '--start-time',
                        required=True,
                        help='Start time for time lapse')
    parser.add_argument('--step',
                        default=1,
                        type=int,
                        help='Time step between images',
                        metavar='SECONDS')

    # Offer multiple methods to set the input frame rate
    ifr_group = parser.add_mutually_exclusive_group()
    ifr_group.add_argument('-d', '--duration',
                           type=float,
                           help='Set output duration for timelapse',
                           metavar='SECONDS')
    ifr_group.add_argument('-i', '--input-frame-rate',
                           type=float,
                           help='Input frame rate',
                           metavar='FPS')
    ifr_group.add_argument('--speed-up',
                           type=float,
                           help='Playback speed as ratio to real time',
                           metavar='FPS')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    start_time = np.datetime64(args.start_time, 's')
    end_time = np.datetime64(args.end_time, 's')

    kwargs = {}
    if args.input_frame_rate is not None:
        kwargs['ifr'] = args.input_frame_rate
    elif args.speed_up is not None:
        kwargs['speed_up'] = args.speed_up
    else:
        kwargs['duration'] = args.duration

    ffmpeg(start_time, end_time, args.step, args.fstr, args.output,
           ofr=args.output_frame_rate,
           size=args.size,
           jitter=args.jitter,
           **kwargs)
