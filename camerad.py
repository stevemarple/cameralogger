#!/usr/bin/env python

import argparse
import copy
import cv2
import glob
import logging
import numpy as np
import os
from scipy.stats import trim_mean
import signal
import six
import struct
import sys
import threading
import time
import traceback
import zwoasi as asi

import aurorawatchnet as awn

if sys.version_info[0] >= 3:
    import configparser
    from configparser import SafeConfigParser
else:
    import ConfigParser
    from ConfigParser import SafeConfigParser



logger = logging.getLogger(__name__)

camera = None


def read_config_file(filename):
    """Read config file."""
    logger.info('Reading config file ' + filename)

    config = SafeConfigParser()

    config.add_section('daemon')
    # The configuration file is the same for the original
    # AuroraWatchNet magnetometer system (Calunium microcontroller,
    # Raspberry Pi or other data logger) or the Raspberry Pi
    # magnetometer system (sensors connected directly to Raspberry
    # Pi). These systems are supported by two different daemons,
    # awnetd and raspmagd.
    config.set('daemon', 'name', 'awnetd')

    config.set('daemon', 'user', 'pi')
    config.set('daemon', 'group', 'pi')
    config.set('daemon', 'sampling_interval', '30')


    config.add_section('upload')
    # User must add appropriate values

    # Monitor for the existence of a file to indicate possible adverse
    # data quality
    config.add_section('dataqualitymonitor')
    config.set('dataqualitymonitor', 'extension', '.bad')
    config.set('dataqualitymonitor', 'username', 'pi')
    config.set('dataqualitymonitor', 'group', 'dialout')

    if filename:
        config_files_read = config.read(filename)
        if filename not in config_files_read:
            raise UserWarning('Could not read ' + filename)
        logger.debug('Successfully read ' + ', '.join(config_files_read))

    return config


def init_camera(camera):
    controls = camera.get_controls()
    camera.start_video_capture()
    camera.set_control_value(asi.ASI_GAIN,
                             controls['Gain']['MinValue'],
                             auto=True)
    camera.set_control_value(asi.ASI_EXPOSURE,
                             controls['Exposure']['MinValue'],
                             auto=True)
    camera.set_control_value(asi.ASI_WB_R, 75)
    camera.set_control_value(asi.ASI_WB_B, 99)
    camera.set_control_value(asi.ASI_GAMMA, 50)
    camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
    camera.set_image_type(asi.ASI_IMG_RGB24)
    time.sleep(2)


def get_control_values(camera):
    controls = camera.get_controls()
    print(controls)
    r = {}
    for k in controls:
        r[k] = camera.get_control_value(controls[k]['ControlType'])[0]

    # Fix up certain keys
    if 'Exposure' in r:
        r['Exposure'] = '%.6f' % (r['Exposure'] / 1000000.0)
    if 'Temperature' in r:
        r['Temperature'] = '%.1f' % (r['Temperature']/10.0)
    if 'Flip' in r:
        r['Flip'] = {0: 'None', 1: 'Horizontal', 2: 'Vertical', 3: 'Both'}[r['Flip']]

    # Convert any remaining non-string types to string
    for k, v in r.iteritems():
        if isinstance(v, six.string_types):
            pass
        elif isinstance(v, float):
            r[k] = '.1f' % v
        else:
            r[k] = str(v)

    print('#######')
    print(r)
    return r


def run_camera():
    global camera

    # This should be called after dropping root privileges because
    # it uses safe_eval to convert strings to numbers or
    # lists (not guaranteed safe!)
    camera = get_camera(config)
    init_camera(camera)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    try:
        get_log_file_for_time(time.time(), log_filename)
        logger.info('Starting sampling thread')

        do_every(config.getfloat('daemon', 'sampling_interval'),
                 record_image)
        while take_samples:
            time.sleep(2)

        # Wait until all other threads have (or should have)
        # completed
        for n in range(int(round(config.getfloat('daemon', 
                                                 'sampling_interval')))
                       + 1):
            if threading.activeCount() == 1:
                break
            time.sleep(1)


    except Exception as e:
        print(e)
        get_log_file_for_time(time.time(), log_filename)
        logger.error(traceback.format_exc())
        time.sleep(5)

       
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result


def cancel_sampling_threads():
    threads = threading.enumerate()
    for t in threads[1:]:
        t.cancel()        
    
    # If all the other threads have completed then exit; exit anyway
    # after a short time has passed
    t = time.time()
    while time.time() < t + 1 and len(threading.enumerate()) > 2:
        time.sleep(0.1)
    #sys.exit()


take_samples = True
def stop_handler(signal, frame):
    global take_samples
    global camera
    get_log_file_for_time(time.time(), log_filename)
    logger.info('Stopping sampling threads')
    take_samples = False
    cancel_sampling_threads()
    camera.stop_video_capture()
    camera.close()


def do_every (interval, worker_func, iterations = 0):
    if iterations != 1:
        # Schedule the next worker thread. Aim to start at the next
        # multiple of sampling interval. Take current time, add 1.25
        # of the interval and then find the nearest
        # multiple. Calculate delay required.
        now = time.time()
        delay = round_to(now + (1.25 * interval), interval) - now
        # Avoid lockups by many threads piling up. Impose a minimum delay
        if delay < 0.1:
            delay = 0.1
        t = threading.Timer(delay,
                            do_every, 
                            [interval, worker_func, 
                             0 if iterations == 0 else iterations-1])
        t.daemon = True
        t.start()
    try:
        worker_func()
    except Exception as e:
        get_log_file_for_time(time.time(), log_filename)
        logger.error(traceback.format_exc())


def round_to(n, nearest):
    return round(n / float(nearest)) * nearest


def get_camera(config):
    '''Return camera object based on config settings'''
    return asi.Camera(0)


def capture():
    global camera
    t = time.time()
    img_info = get_control_values(camera)
    img = camera.capture_video_frame()
    img_info['DateTime'] = time.strftime('%Y-%m-%d %H:%M:%S+00:00', time.gmtime(t))
    print(repr(img_info))

    # Take CPU temperature as system temperature
    img_info['SystemTemperature'] = 'unknown'
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        img_info['SystemTemperature'] = '%.2f' % (float(f.read().strip())/1000)

    return img, img_info, t


# Each sampling action is made by a new thread. This function uses a
# lock to avoid contention for the camera. If the lock cannot be
# acquired the attempt is abandoned. The lock is released after the
# sample has been taken. This means two instances of record_image()
# can occur at the same time, whilst the earlier one writes data and
# possibly sends a real-time data packet over the network. A second
# lock (write_to_csv_file.lock) is used to avoid contention on writing
# results to a file.
def record_image():
    global data_quality_ok
    global ntp_ok
    img = None
    img_info = None
    t = None
    get_log_file_for_time(time.time(), log_filename, delay=False)
    logger.debug('record_image(): acquiring lock')
    if record_image.lock.acquire(False):
        try:
            logger.debug('record_image(): acquired lock')
            img, img_info, now = capture()
        finally:
            logging.debug('record_image(): released lock')
            record_image.lock.release()
    else:
        logger.error('record_image(): could not acquire lock')

    if img is None:
        return

    # A warning about data quality from any source
    data_quality_warning = False
    if config.has_option('dataqualitymonitor', 'directory'):
        # Any file/directory counts as a warning
        try:
            data_quality_warning = \
                bool(os.listdir(config.get('dataqualitymonitor',
                                           'directory')))
        except:
            pass
    elif config.has_option('dataqualitymonitor', 'filename'):
        data_quality_warning = \
            os.path.isfile(config.get('dataqualitymonitor', 'filename'))

    if data_quality_warning:
        # Problem with data quality
        if data_quality_ok:
            # Not known previously, log
            get_log_file_for_time(time.time(), log_filename)
            logger.warning('Data quality warning detected')
            data_quality_ok = False
    elif not data_quality_ok:
            get_log_file_for_time(time.time(), log_filename)
            logger.info('Data quality warning removed')
            data_quality_ok = True
    
    if (config.has_option('ntp_status', 'filename') and 
        config.has_option('ntp_status', 'max_age') and 
        (not os.path.exists(config.get('ntp_status', 'filename')) or 
         time.time() - os.stat(config.get('ntp_status', 'filename')).st_mtime
         > config.get('ntp_status', 'max_age'))):
        # NTP status file is missing/old, assume NTP not running
        if ntp_ok:
            get_log_file_for_time(time.time(), log_filename)
            logger.warning('NTP not running/synchronized')
            ntp_ok = False
    elif not ntp_ok:
        get_log_file_for_time(time.time(), log_filename)
        logger.info('NTP problem resolved')
        ntp_ok = True

    ext = None
    if not data_quality_ok or not ntp_ok:
        ext = config.get('dataqualitymonitor', 'extension')

    ###########################
    # Save image
    dir = '/home/pi/tmpfs'
    filename = '%Y%m%dT%H%M%S.png'
    info_filename = '%Y%m%dT%H%M%S.txt'
    if os.path.isdir(dir):
        filename = os.path.join(dir, filename)
        info_filename = os.path.join(dir, info_filename)

    tm = time.gmtime(t)
    filename = time.strftime(filename, tm)
    info_filename = time.strftime(info_filename, tm)
    cv2.imwrite(filename, img)
    logger.debug('wrote %s', filename)
    with open(info_filename, 'w') as fh:
        for k in sorted(img_info.keys()):
            fh.write('%s: %s\n' % (k, str(img_info[k])))
    # if config.has_option('awnettextdata', 'filename'):
    #     write_to_txt_file(img, ext)
    #
    # if config.has_option('raspitextdata', 'filename'):
    #     write_to_csv_file(img, ext)

    return

record_image.lock = threading.Lock()


def data_to_str(data, separator=',', comments='#', want_header=False):
    separator = ','
    header = comments + 'sample_time'
    fstr = '{sample_time:.3f}'
    d = dict(sample_time=data['sample_time'], separator=separator)
    for c in ('x', 'y', 'z', 'sensor_temperature'):
        if c in data:
            d[c] = data[c]
            header += separator + c
            fstr += '{separator}{' + c + ':.3f}'
    header += '\n'
    fstr += '\n'
    s = fstr.format(**d)
    if want_header:
        return s, header
    else:
        return s


def get_log_file_for_time(t, fstr, 
                          mode='a', 
                          delay=True, 
                          name=__name__):
    if fstr is None:
        return
    fh = get_log_file_for_time.fh
    tmp_name = time.strftime(fstr, time.gmtime(t))

    if fh is not None:
        if fh.stream and fh.stream.closed:
            fh = None
        elif fh.stream.name != tmp_name:
            # Filename has changed
            fh.close()
            fh = None
        
    if fh is None:
        # File wasn't open or filename changed
        p = os.path.dirname(tmp_name)
        if not os.path.isdir(p):
            os.makedirs(p)

        fh = logging.FileHandler(tmp_name, mode=mode, delay=delay)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%dT%H:%M:%SZ')
        fh.setFormatter(formatter)
        logger = logging.getLogger(name)
        for h in logger.handlers:
            # Only remove file handlers 
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
        logger.addHandler(fh)


get_log_file_for_time.fh = None


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    # Parse command line arguments
    progname = os.path.basename(sys.argv[0]).partition('.')[0]
    default_config_file = \
        os.path.join(os.path.sep, 'etc', 'camera.ini')

    parser = \
        argparse.ArgumentParser(description='AuroraWatch camera daemon')

    parser.add_argument('-c', '--config-file', 
                        default=default_config_file,
                        help='Configuration file')
    parser.add_argument('--log-level', 
                        choices=['debug', 'info', 'warning', 
                                 'error', 'critical'],
                        default='info',
                        help='Control how much detail is printed',
                        metavar='LEVEL')
    parser.add_argument('--log-format',
                        default='%(levelname)s:%(message)s',
                        help='Set format of log messages',
                        metavar='FORMAT')

    args = parser.parse_args()

    config = read_config_file(args.config_file)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format=args.log_format, datefmt='%Y-%m-%dT%H:%M:%SZ')

    log_filename = None
    if config.has_option('logfile', 'filename'):
        log_filename = config.get('logfile', 'filename')
        
    get_log_file_for_time(time.time(), log_filename)
    logger.info(progname + ' started')

    data_quality_ok = True
    ntp_ok = True
    run_camera()

