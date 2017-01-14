from string import Formatter

class MyFormatter(Formatter):
    def __init__(self):
        Formatter.__init__(self)

    def convert_field(self, value, conversion):
        # do any conversion on the resulting object
        if conversion is None:
            return value
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
    def __init__(self, lat=None, lon=None):
        self.lat = lat
        self.lon = lon

    def __format__(self, fmt):
        print(fmt)
        if fmt == 'd':
            # decimal degrees
            pass
        return '???'



