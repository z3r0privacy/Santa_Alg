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
LOCATION = [LAT, LON]

def memoize(func):
  """Uses a dict to cache the results of the decorated function.

  :func: Function to decorate

  :returns: Decorated function
  """
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

def get_location(gift):
  """Extracts the location of a gift as a tuple.

  :gift: Numpy array row of the gift

  :returns: Tuple with location
  """
  return tuple(gift[LOCATION])

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

def distance(a, b):
  """Use cache to avoid computing unnecessary haversine distances.
  Also avoid computing symmetric distances ((a,b) and (b,a)).

  :a: First location
  :b: Second location

  :returns: Haversine distance between the locations
  """
  aa = tuple(a)
  bb = tuple(b)
  return _actually_get_distance(aa, bb) if aa < bb else _actually_get_distance(bb, aa)

@memoize
def _actually_get_distance(a, b):
  return haversine(a, b)

def weighted_trip_length(stops, weights):
  """Calculates the cost of the trip.

  :stops: Pandas DataFrame or Numpy array with the locations of the trip
  :weights: Pandas Series or Numpy array with the weights of the gifts

  :returns: The cost of the trip
  """
  tuples = [tuple(x) for x in (stops.values if isinstance(stops, pd.DataFrame) else stops)]
  weights = (weights.values if isinstance(weights, pd.Series) else weights).tolist()
  if len(tuples) != len(weights):
      raise ValueError("Stops/weights dimension mismatch!")
  # print("length of trip", len(tuples))

  # adding the last trip back to north pole, with just the sleigh weight
  tuples.append(NORTH_POLE)
  weights.append(SLEIGH_WEIGHT)

  cost = 0.0
  prev_stop = NORTH_POLE
  prev_weight = sum(weights)
  dist = 0.0
  for location, weight in zip(tuples, weights):
    cost += distance(location, prev_stop) * prev_weight
    dist += distance(location, prev_stop)
    prev_stop = location
    prev_weight = prev_weight - weight
  # print("DISTANCE\t", dist)
  return cost

def verify_weights(all_trips, log):
  """Verifies that none of the trips exceeds the weight limit.

  :all_trips: Pandas DataFrame with the trips to verify
  :log: Log for informing about invalid trips

  :returns: True if all trips are valid
  """
  has_invalid_trip = False
  for trip in all_trips.groupby("TripId"):
    this_trip = trip[1]
    if this_trip.Weight.sum() > WEIGHT_LIMIT:
      log.warning("Weight too high: {}".format(this_trip.Weight.sum()))
      has_invalid_trip = True
  return not has_invalid_trip

def weighted_reindeer_weariness(all_trips):
  """Calculates the total cost of all the trips.

  :all_trips: Pandas DataFrame with all trips

  :returns: Total cost of the trips
  """
  cost = 0.0
  for trip in all_trips.groupby("TripId"):
    this_trip = trip[1]
    cost += weighted_trip_length(this_trip[["Latitude","Longitude"]], this_trip.Weight)
  return cost

