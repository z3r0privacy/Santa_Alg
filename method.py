#!/usr/bin/env python

import abc
import pickle

import numpy as np
import pandas as pd

import utils


class Method(abc.ABC):
  def __init__(self, gifts, log):
    self.current_best = 435032574114.5575
    self.gifts = gifts
    self.log = log
    self.trips = pd.DataFrame(columns=["GiftId", "TripId"])

  @property
  @abc.abstractmethod
  def name(self):
    pass

  @abc.abstractmethod
  def run(self, args):
    """Applies the method to try and find optimal solutions.
    The solution should then be stored in `trips`.

    :args: Additional arguments depending on the method.
    """
    pass

  def evaluate_trips(self):
    """Verifies that the constraints aren't violated and checks solution quality.

    :returns: True if the trips are valid
    """
    self.log.info("Evaluating trips...")

    if not set(self.gifts.GiftId) == set(self.trips.GiftId):
      self.log.error("Mismatch in the delivered gifts: {} to deliver, {} delivered".format(
        self.gifts.GiftId.shape[0], self.trips.GiftId.shape[0]))
      return False

    unique_trips = self.trips.TripId.unique()
    merged = self.trips.merge(self.gifts, on="GiftId")
    trips = [merged[merged.TripId == t] for t in unique_trips]

    if not utils.verify_weights(merged):
      self.log.error("One or more trip is invalid!")
      return False

    score = utils.weighted_reindeer_weariness(merged)
    utils.log_success_or_error(self.log, score < self.current_best, "Cost of the {} trips: {:.5f}B ({:.5f}B)".format(
      unique_trips.shape[0], score / 1e9, (score - self.current_best) / 1e9))

    weights = np.asarray([trip.Weight.sum() for trip in trips])
    self.log.info("Sleigh utilization: min {:.2f}, max {:.2f}, avg {:.2f}, std {:.2f}".format(
      weights.min(), weights.max(), weights.mean(), weights.std()))

    costs = np.asarray([utils.weighted_trip_length(trip[["Latitude","Longitude"]], trip.Weight.tolist()) for trip in trips])
    self.log.info("Trip costs: min {:.2f}M, max {:.2f}M, avg {:.2f}M, std {:.2f}k".format(
      costs.min() / 1e6, costs.max() / 1e6, costs.mean() / 1e6, costs.std() / 1e3))

    stops = np.asarray([trip.shape[0] for trip in trips])
    self.log.info("Stops per trip: min {}, max {}, avg {:.2f}, std {:.2f}".format(
      stops.min(), stops.max(), stops.mean(), stops.std()))

    return True

  def write_trips(self, file_name):
    """Creates a submission file from the calculated trips

    :file_name: Name of the file to write

    """
    self.log.info("Writing trips to {}".format(file_name))
    self.trips[["GiftId", "TripId"]].astype(int).to_csv(file_name, index=False)

