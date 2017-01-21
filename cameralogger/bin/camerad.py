#!/usr/bin/env python

import argparse
import importlib
import logging
import os
import signal
import sys
import threading
import time
import traceback

import cameralogger

__author__ = 'Steve Marple'
__version__ = '0.0.3'
__license__ = 'PSF'


def run_camera(forced_schedule):
    global config
    global camera
    global sampling_interval

    # This should be called after dropping root privileges because
    # it uses safe_eval to convert strings to numbers or
    # lists (not guaranteed safe!)
    #camera = cameralogger.get_camera(config)
    #cameralogger.init_camera(camera, config)

    camera_type = config.get('camera', 'type')
    logger.debug('camera type: %s', camera_type)

    Camera = getattr(importlib.import_module('cameralogger.' + camera_type), 'Camera')
    camera = Camera(config)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    try:
        get_log_file_for_time(time.time(), log_filename)
        logger.info('Starting sampling thread')

        do_every(camera, config, forced_schedule, cameralogger.process_tasks)
        while take_images:
            time.sleep(2)

        # Wait until all other threads have (or should have)
        # completed
        try:
            sampling_interval_lock.acquire(True)
            si = sampling_interval
        finally:
            sampling_interval_lock.release()

        for n in range(int(round(si)) + 1):
            if threading.activeCount() == 1:
                break
            time.sleep(1)

    except Exception:
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
        # sys.exit()


take_images = True


def stop_handler(signal, frame):
    global take_images
    global camera
    get_log_file_for_time(time.time(), log_filename)
    logger.info('Stopping sampling threads')
    take_images = False
    cancel_sampling_threads()
    if camera:
        camera = None


def do_every(camera, config, forced_schedule, worker_func, iterations=0):
    global sampling_interval
    if iterations != 1:
        # Identify current operating condition to find the actions to do and sampling interval
        schedule, schedule_info = cameralogger.get_schedule(config, forced_schedule)
        logger.info('using schedule %s', schedule)
        logger.info('schedule info: %s', repr(schedule_info))
        # Schedule the next worker thread. Aim to start at the next
        # multiple of sampling interval. Take current time, add 1.25
        # of the interval and then find the nearest
        # multiple. Calculate delay required.
        now = time.time()
        interval = cameralogger.get_config_option(config, schedule, 'sampling_interval',
                                                  fallback_section='common',
                                                  default=default_sampling_interval,
                                                  get='getfloat')
        delay = round_to(now + (1.25 * interval), interval) - now
        # Avoid lockups by many threads piling up. Impose a minimum delay
        if delay < 0.1:
            delay = 0.1
        t = threading.Timer(delay,
                            do_every,
                            [camera, config, forced_schedule, worker_func,
                             0 if iterations == 0 else iterations - 1])
        t.daemon = True
        t.start()

        # Update the global sampling_interval so that cancel_sampling_threads knows how long to wait.
        # This could be attempted by multiple capture threads so must use a lock. Don't wait for the lock to be
        # available.
        if sampling_interval_lock.acquire(False):
            try:
                logger.debug('sampling_interval_lock: acquired lock')
                sampling_interval = interval
            finally:
                logging.debug('sampling_interval_lock: released lock')
                sampling_interval_lock.release()
        else:
            logger.error('sampling_interval_lock: could not acquire lock')

    try:
        worker_func(camera, config, schedule, schedule_info)
    except Exception:
        get_log_file_for_time(time.time(), log_filename)
        logger.error(traceback.format_exc())


def round_to(n, nearest):
    return round(n / float(nearest)) * nearest


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
        log = logging.getLogger(name)
        for h in log.handlers:
            # Only remove file handlers 
            if isinstance(h, logging.FileHandler):
                log.removeHandler(h)
        log.addHandler(fh)


get_log_file_for_time.fh = None

camera = None
default_sampling_interval = 120
sampling_interval = default_sampling_interval
sampling_interval_lock = threading.Lock()

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
    parser.add_argument('--schedule',
                        help='Override automatic scheduling')

    args = parser.parse_args()

    config = cameralogger.read_config_file(args.config_file)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format=args.log_format, datefmt='%Y-%m-%dT%H:%M:%SZ')

    if (args.schedule is not None and
            (not config.has_section(args.schedule) or not config.has_option(args.schedule, 'sampling_interval'))):
        raise Exception('%s is not a valid schedule' % args.schedule)

    log_filename = None
    if config.has_option('logfile', 'filename'):
        log_filename = config.get('logfile', 'filename')

    get_log_file_for_time(time.time(), log_filename)
    logger.info(progname + ' started')

    run_camera(args.schedule)


logger = logging.getLogger(__name__)
