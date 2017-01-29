#!/usr/bin/env python

import argparse
import datetime
import logging
import numpy as np
import os
from PIL import Image
import subprocess


__author__ = 'Steve Marple'
__version__ = '0.0.7'
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
           resolution=None, duration=None, speed_up=None, ifr=None, ofr=None):
    img_mode = 'RGB'  # Mode for all images
    # Remember et is inclusive
    st, et = find_start_end_frames(start_time.astype('datetime64[s]').astype(int),
                                   end_time.astype('datetime64[s]').astype(int),
                                   step,
                                   filename_fstr)
    if st is None:
        raise ValueError('could not find any images in time range')

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
    if resolution is None:
        resolution = tuple(first_frame.size)
    else:
        resolution = tuple(resolution)  # Force to be tuple so that img.size comparison works later

    # Set up a subprocess to run ffmpeg
    cmd = ('ffmpeg',
           '-loglevel', 'error',
           '-y',  # overwrite
           '-framerate', str(ifr),  # input frame rate
           '-s', '%dx%d' % (resolution[0], resolution[1]),  # size of image
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
    while t <= et:
        dt = datetime.datetime.fromtimestamp(t)
        filename = filename_fstr.format(DateTime=dt)
        if os.path.exists(filename):
            logger.debug('reading %s', filename)
            img = Image.open(filename)
        elif img is None:
            logger.debug('missing %s', filename)
            img = Image.new('RGB', resolution)
        else:
            logger.debug('missing intermediate image %s', filename)

        if img.mode != img_mode:
            logger.debug('converting image to mode %s', img_mode)
            img = img.convert(img_mode)
        if tuple(img.size) != resolution:
            logger.debug('resizing image')
            img = img.resize(resolution, Image.BILINEAR)

        proc.stdin.write(img.tobytes())
        t += step
    proc.communicate()
    logger.info('saved to %s', output_filename)


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
    parser.add_argument('--resolution',
                        nargs=2,
                        type=int,
                        help='Video resolution',
                        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('-s', '--start-time',
                        required=True,
                        help='Start time for time lapse')
    parser.add_argument('--step',
                        default=1,
                        type=float,
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
           resolution=args.resolution,
           **kwargs)
