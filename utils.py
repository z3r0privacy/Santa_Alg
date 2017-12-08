#!/usr/bin/env python

import logging

import numpy as np
import pandas as pd

import coloredlogs

from haversine import haversine

NORTH_POLE = (90, 0)
WEIGHT_LIMIT = 1000.0
SLEIGH_WEIGHT = 10.0


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

def weighted_trip_length(stops, weights):
  tuples = [tuple(x) for x in stops.values]
  # adding the last trip back to north pole, with just the sleigh weight
  tuples.append(NORTH_POLE)
  weights.append(SLEIGH_WEIGHT)

  dist = 0.0
  prev_stop = NORTH_POLE
  prev_weight = sum(weights)
  for location, weight in zip(tuples, weights):
    dist = dist + haversine(location, prev_stop) * prev_weight
    prev_stop = location
    prev_weight = prev_weight - weight

  return dist

def verify_weights(all_trips, gifts):
  all_trips = all_trips.merge(gifts, on="GiftId")
  uniq_trips = all_trips.TripId.unique()

  return all(all_trips.groupby("TripId").Weight.sum() < WEIGHT_LIMIT)

def weighted_reindeer_weariness(all_trips, gifts):
  all_trips = all_trips.merge(gifts, on="GiftId")
  uniq_trips = all_trips.TripId.unique()

  dist = 0.0
  for t in uniq_trips:
    this_trip = all_trips[all_trips.TripId==t]
    dist = dist + weighted_trip_length(this_trip[["Latitude","Longitude"]], this_trip.Weight.tolist())

  return dist
