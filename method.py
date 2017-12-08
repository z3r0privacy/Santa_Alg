#!/usr/bin/env python

import abc
import pickle

import numpy as np
import pandas as pd

import utils


class Method(abc.ABC):
  def __init__(self, gifts, log):
    self.current_best = 460338437682.3982
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
    utils.log_success_or_error(self.log, score < self.current_best, "Score of the {} trips: {}".format(
      unique_trips.shape[0], score))

    weights = np.asarray([trip.Weight.sum() for trip in trips])
    self.log.info("Sleigh utilization: min {:.2f}, max {:.2f}, avg {:.2f}, std {:.2f}".format(
      weights.min(), weights.max(), weights.mean(), weights.std()))

    costs = np.asarray([utils.weighted_trip_length(trip[["Latitude","Longitude"]], trip.Weight.tolist()) for trip in trips])
    self.log.info("Trip costs: min {:.2f}, max {:.2f}, avg {:.2f}, std {:.2f}".format(
      costs.min(), costs.max(), costs.mean(), costs.std()))

    return True

  def write_trips(self, file_name):
    """Creates a submission file from the calculated trips

    :file_name: Name of the file to write

    """
    self.log.info("Writing trips to {}".format(file_name))
    self.trips.to_csv(file_name, index=False, columns=["GiftId", "TripId"])

