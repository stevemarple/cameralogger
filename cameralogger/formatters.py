from string import Formatter

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
    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    def _dmsh(self, val, hemi):
        is_positive = val >= 0

        val = abs(val)
        m, s = divmod(val * 3600, 60)
        d, m = divmod(m, 60)
        d = d if is_positive else -d
        return int(d), int(m), s, self._hemisphere(val, hemi)

    def _hemisphere(self, val, hemi):
        if val > 0:
           return hemi[0]
        elif val < 0:
            return hemi[1]
        else:
            return ''

    @property
    def latitude(self):
        return abs(self._lat)

    @property
    def latitude_signed(self):
        return self._lat

    @property
    def latitude_degrees(self):
        return int(abs(self._lat))

    @property
    def latitude_minutes(self):
        return self._dmsh(self._lat, ('N', 'S'))[1]

    @property
    def latitude_seconds(self):
        return self._dmsh(self._lat, ('N', 'S'))[2]

    @property
    def latitude_hemisphere(self):
        return self._hemisphere(self._lat, ('N', 'S'))

    @property
    def longitude(self):
        return abs(self._lon)

    @property
    def longitude_signed(self):
        return self._lon

    @property
    def longitude_degrees(self):
        return int(abs(self._lon))

    @property
    def longitude_minutes(self):
        return self._dmsh(self._lon, ('E', 'W'))[1]

    @property
    def longitude_seconds(self):
        return self._dmsh(self._lon, ('E', 'W'))[2]

    @property
    def longitude_hemisphere(self):
        return self._hemisphere(self._lon, ('E', 'W'))




