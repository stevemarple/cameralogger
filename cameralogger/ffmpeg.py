# ffmpeg.py - wrapper to FFmpeg program.
#
# This file is part of Cameralogger - software to record and decorate
# camera images for timelapses etc.
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
#
#
#  - create timelapses from camera images by calling FFmpeg.
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

from cameralogger.utils import apply_alpha
import datetime
import logging
import numpy as np
import os
from PIL import Image
import subprocess
import time


__author__ = 'Steve Marple'
__version__ = '0.1.1'
__license__ = 'MIT'


class FFmpeg(object):
    def __init__(self, filename, size, ifr, ofr,
                 vcodec='libx264',
                 loglevel='error',
                 background='black'):
        self.filename = filename
        self.ifr = ifr
        self.ofr = ofr
        self.vcodec = vcodec
        self.size = tuple(size)
        self.loglevel = loglevel

        # Set up a subprocess to run ffmpeg
        cmd = ('ffmpeg',
               '-loglevel', 'error',
               '-y',  # overwrite
               '-framerate', str(ifr),  # input frame rate
               '-s', '%dx%d' % (self.size[0], self.size[1]),  # size of image
               '-pix_fmt', 'rgb24',  # format
               '-f', 'rawvideo',
               '-i', '-',  # read from stdin
               '-vcodec', 'libx264',  # set output encoding
               '-r', str(ofr),  # output frame rate
               self.filename)
        logger.debug('Running command ' + ' '.join(cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.background = None
        self.background_rgba = None  # For efficiency keep an RGBA version too
        self.set_background(background)
        self._frames = 0

    def __del__(self):
        self.close()

    def close(self):
        if self.proc:
            self.proc.communicate()
            self.proc = None

    def set_background(self, background):
        if background is None:
            self.background = Image.new('RGB', self.size, 'black')
        elif isinstance(background, Image.Image):
            if background.size != self.size:
                raise ValueError('background image is incorrect size')
            if 'A' in background.getbands():
                # Remove transparency by compositing over black
                self.background = Image.alpha_composite(Image.new('RGBA', background.size, 'black'),
                                                        background).convert('RGB')
            else:
                self.background = background.convert('RGB')
        else:
            self.background = Image.new('RGB', self.size, background)
        self.background_rgba = self.background.convert('RGBA')

    def add_frame(self, image):
        if image.size != self.size:
            raise ValueError('Image is incorrect size (%s)' % repr(image.size))
        if image.mode != 'RGB':
            if 'A' in image.getbands():
                # Must alpha composite over background color
                image = Image.alpha_composite(self.background_rgba, image)

            image = image.convert('RGB')
        self.proc.stdin.write(image.tobytes())
        self._frames += 1

    def dissolve(self, image1, image2, num_frames):
        im2 = image2
        if 'A' in im2.getbands():
            im2 = Image.alpha_composite(image1, im2)
        for alpha in my_linspace(0, 1, num_frames):
            self.add_frame(Image.blend(image1, im2, alpha))

    def fade_in(self, image, num_frames):
        im2 = image.copy()
        for alpha in my_linspace(0, 1, num_frames):
            self.add_frame(apply_alpha(im2, alpha))

    def fade_out(self, image, num_frames):
        im = image.copy()
        for alpha in my_linspace(1, 0, num_frames):
            self.add_frame(apply_alpha(im, alpha))

    def freeze(self, image, num_frames):
        """Show image as freeze-frame for given duration."""
        for n in range(num_frames):
            self.add_frame(image)

    @property
    def closed(self):
        return self.proc is None

    @property
    def frames(self):
        return self._frames


def my_linspace(a, b, n):
    if n == 1:
        return [(a + b) / 2.0]  # numpy.linspace() doesn't interpolate
    else:
        return list(np.linspace(a, b, n))


def timelapse(ffmpeg, start_time, end_time, step, filename_fstr, jitter=None, fade_in=0, fade_out=0):
    def get_filename(t):
        return filename_fstr.format(DateTime=datetime.datetime.fromtimestamp(t))

    img_mode = 'RGB'  # Mode for all images
    st = time.mktime(start_time.timetuple())
    et = time.mktime(end_time.timetuple())

    if jitter and not step > 2 * jitter:
        raise ValueError('step size must be more than 2 * jitter')

    first_frame = None
    img = None
    t = st
    frames = 0

    fade_in_alpha = my_linspace(0, 1, fade_in)
    fade_out_alpha = my_linspace(1, 0, fade_out)
    total_frames = int((et-st) / step)
    fade_out_start = total_frames - fade_out

    while t < et:
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
                break

        if found:
            if first_frame is None:
                first_frame = img
        elif img is None:
            logger.debug('missing %s', filename)
            img = Image.new('RGB', ffmpeg.size)
            # img.putalpha(0)
        else:
            if jitter and logger.level <= logging.DEBUG:
                # Log the target filename
                filename = get_filename(t)
            logger.debug('missing intermediate image %s', filename)

        if img.mode != img_mode:
            logger.debug('converting image to mode %s', img_mode)
            img = img.convert(img_mode)
        if tuple(img.size) != ffmpeg.size:
            logger.debug('resizing image')
            img = img.resize(ffmpeg.size, Image.BILINEAR)

        if len(fade_in_alpha):
            img = apply_alpha(img, fade_in_alpha.pop(0))
        if frames >= fade_out_start:
            img = apply_alpha(img, fade_out_alpha.pop(0))

        ffmpeg.add_frame(img)
        t += step
        frames += 1

    return first_frame, img

logger = logging.getLogger(__name__)
