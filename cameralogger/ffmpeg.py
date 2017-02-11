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


import logging
import numpy as np
from PIL import Image
import subprocess


__author__ = 'Steve Marple'
__version__ = '0.1.1'
__license__ = 'MIT'


class FFmpeg(object):
    def __init__(self, filename, size, ifr, ofr,
                 vcodec='libx264',
                 loglevel='error',
                 bg_color='black'):
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
        self.set_bg_color(bg_color)

    def __del__(self):
        self.close()

    def close(self):
        if self.proc:
            self.proc.communicate()
            self.proc = None

    def set_bg_color(self, color):
        if color is None:
            self.background = None
        else:
            self.background = Image.new('RGBA', self.size, color)

    def add_frame(self, image):
        if image.size != self.size:
            raise ValueError('Image is incorrect size (%s)' % repr(image.size))
        if image.mode != 'RGB':
            if 'A' in image.getbands() and self.background:
                # Must alpha composite over background color
                image = Image.alpha_composite(self.background, image)

            image = image.convert('RGB')
        self.proc.stdin.write(image.tobytes())

    def dissolve(self, image1, image2, num_frames):
        for alpha in my_linspace(0, 1, num_frames):
            self.add_frame(Image.blend(image1, image2, alpha))

    def fade_in(self, image, num_frames):
        im2 = image.copy()
        for alpha in my_linspace(0, 1, num_frames):
            im2.putalpha(int(round(alpha * 255)))
            self.add_frame(im2)

    def fade_out(self, image, num_frames):
        im2 = image.copy()
        for alpha in my_linspace(1, 0, num_frames):
            im2.putalpha(int(round(alpha * 255)))
            self.add_frame(im2)

    def freeze(self, image, num_frames):
        """Show image as freeze-frame for given duration."""
        for n in range(num_frames):
            self.add_frame(image)

    @property
    def closed(self):
        return self.proc is None


def my_linspace(a, b, n):
    if n == 1:
        return (a + b) / 2.0  # numpy.linspace() doesn't interpolate
    else:
        return list(np.linspace(a, b, n))


logger = logging.getLogger(__name__)
