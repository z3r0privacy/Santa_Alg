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

    if not utils.verify_weights(self.trips, self.gifts):
      self.log.error("One or more trip is invalid!")
      return False

    score = utils.weighted_reindeer_weariness(self.trips, self.gifts)
    utils.log_success_or_error(self.log, score < self.current_best, "Score of the trip: {}".format(score))

    # TODO extend with more metrics such as trip count, weight utilization, etc.
    return True

  def write_trips(self, file_name):
    """Creates a submission file from the calculated trips

    :file_name: Name of the file to write

    """
    self.log.info("Writing trips to {}".format(file_name))
    self.trips.to_csv(file_name, index=False, columns=["GiftId", "TripId"])

