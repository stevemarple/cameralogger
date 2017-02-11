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
import subprocess


__author__ = 'Steve Marple'
__version__ = '0.1.1'
__license__ = 'MIT'


class FFmpeg(object):
    def __init__(self, filename, size, ifr, ofr,
                 vcodec='libx264',
                 loglevel='error'):
        self.filename = filename
        self.ifr = ifr
        self.ofr = ofr
        self.vcodec = vcodec
        self.size = size
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

    def __del__(self):
        self.close()

    def close(self):
        if self.proc:
            self.proc.communicate()
            self.proc = None

    def add_frame(self, image):
        self.proc.stdin.write(image.tobytes())

    @property
    def closed(self):
        return self.proc is None


logger = logging.getLogger(__name__)
