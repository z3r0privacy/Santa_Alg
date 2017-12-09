#!/usr/bin/env python

import logging
from functools import wraps

import numpy as np

import coloredlogs
import pandas as pd
from haversine import haversine

NORTH_POLE = (90, 0)
WEIGHT_LIMIT = 1000.0
SLEIGH_WEIGHT = 10.0

CACHE_HIT = 0
CACHE_MISS = 0

GIFT = 0
TRIP = 1
LAT = 2
LON = 3
WEIGHT = 4

def memoize(func):
  cache = {}
  @wraps(func)
  def wrap(*args):
    global CACHE_HIT, CACHE_MISS
    if args not in cache:
      cache[args] = func(*args)
      CACHE_MISS += 1
    else:
      CACHE_HIT += 1
    return cache[args]
  return wrap


def get_logger(name):
  debug_levelv_num = 21
  # add "success" log level
  logging.addLevelName(debug_levelv_num, "SUCCESS")
  def success(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(debug_levelv_num):
      self._log(debug_levelv_num, message, args, **kws)
  logging.Logger.success = success

  # set up logger
  coloredlogs.install(level="DEBUG")
  coloredlogs.DEFAULT_LEVEL_STYLES = {
      "debug": {"color": "white", "bold": False},
      "info": {"color": "white", "bold": True},
      "success": {"color": "green", "bold": True},
      "warning": {"color": "yellow", "bold": True},
      "error": {"color": "red", "bold": True},
      "fatal": {"color": "magenta", "bold": True},
      }
  logger = logging.getLogger(name)
  handler = logging.StreamHandler()
  log_format = "%(asctime)s %(module)s.%(funcName)s:%(lineno)d %(levelname)-8s %(message)s"
  formatter = coloredlogs.ColoredFormatter(log_format)
  handler.setFormatter(formatter)
  logger.propagate = False
  logger.handlers = []
  logger.addHandler(handler)
  logger.setLevel(logging.DEBUG)
  return logger

def log_success_or_error(log, success, message):
  log_method = log.success if success else log.error
  log_method(message)

@memoize
def distance(a, b):
  return haversine(a, b)

def weighted_trip_length(stops, weights):
  tuples = [tuple(x) for x in stops.values]
  # adding the last trip back to north pole, with just the sleigh weight
  tuples.append(NORTH_POLE)
  weights.append(SLEIGH_WEIGHT)

  dist = 0.0
  prev_stop = NORTH_POLE
  prev_weight = sum(weights)
  for location, weight in zip(tuples, weights):
    dist = dist + distance(location, prev_stop) * prev_weight
    prev_stop = location
    prev_weight = prev_weight - weight

  return dist

def verify_weights(all_trips):
  uniq_trips = all_trips.TripId.unique()

  if all(all_trips.groupby("TripId").Weight.sum() < WEIGHT_LIMIT):
      return True
  else :
      for i,trip in enumerate(all_trips.groupby("TripId")):
          if trip.Weight.sum() > WEIGHT_LIMIT:
              self.log.warn("Weight too high: {}".format(trip.Weight.sum()))

def weighted_reindeer_weariness(all_trips):
  uniq_trips = all_trips.TripId.unique()

  dist = 0.0
  for t in uniq_trips:
    this_trip = all_trips[all_trips.TripId==t]
    dist = dist + weighted_trip_length(this_trip[["Latitude","Longitude"]], this_trip.Weight.tolist())

  return dist
