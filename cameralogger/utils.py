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


import numpy as np
from PIL import Image


def apply_alpha(image, alpha):
    """Apply alpha to entire image, taking into account any existing alpha mask."""
    im = image.copy()
    if 'A' not in image.getbands():
        im.putalpha(int(round(alpha * 255)))
        return im
    elif image.mode == 'RGBA':
        a = bytes(bytearray((np.array(bytearray(image.split()[3].tobytes())) * alpha + 0.5).astype('int8')))
        im.putalpha(Image.frombytes('L', image.size, a))
        return im
    else:
        raise ValueError('incorrect image mode (was %s)' % image.mode)
