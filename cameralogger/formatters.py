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

from string import Formatter


__author__ = 'Steve Marple'
__version__ = '0.1.1'
__license__ = 'MIT'


class MyFormatter(Formatter):
    def __init__(self):
        Formatter.__init__(self)

    def convert_field(self, value, conversion):
        # do any conversion on the resulting object
        if conversion is None:
            return value
        elif conversion == 'a':
            return value.encode('ascii')
        elif conversion == 's':
            return str(value)
        elif conversion == 'r':
            return repr(value)
        elif conversion == 'l':
            return str(value).lower()
        elif conversion == 'u':
            return str(value).upper()
        elif conversion == 'c':
            return str(value).capitalize()
        elif conversion == 't':
            return str(value).title()
        elif conversion == '.':
            return str(value).rstrip('.')
        raise ValueError("Unknown conversion specifier {0!s}".format(conversion))


class LatLon(object):
    def __init__(self, latitude, longitude):
        self._latitude = DegMS(latitude, ('N', 'S'))
        self._longitude = DegMS(longitude, ('E', 'W'))

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude


class DegMS(object):
    def __init__(self, val, hemi):
        self._val = val
        self._hemi = tuple(hemi)

    def __format__(self, *args, **kwargs):
        """Format the class as if it was a floating point value."""
        return self._val.__format__(*args, **kwargs)

    def _hemisphere(self):
        if self._val > 0:
            return self._hemi[0]
        elif self._val < 0:
            return self._hemi[1]
        else:
            return ''

    def _dmsh(self):
        m, s = divmod(abs(self._val) * 3600, 60)
        d, m = divmod(m, 60)
        # d = d if self._val >= 0 else -d
        return int(d), int(m), s, self._hemisphere()

    @property
    def unsigned_degrees(self):
        return abs(self._val)

    @property
    def signed_degrees(self):
        return self._val

    @property
    def minutes(self):
        return self._dmsh()[1]

    @property
    def seconds(self):
        return self._dmsh()[2]

    @property
    def hemisphere(self):
        return self._hemisphere()

    @property
    def dhms(self):
        return self._dmsh()

