#!/usr/bin/env python

import abc
import glob

import numpy as np

import pandas as pd
import utils


class Method(abc.ABC):
  def __init__(self, gifts, log):
    self.current_best = 29.34056 * 1e9 # heavy antarctica with north->south ordering
    self.current_best = 22.26235 * 1e9 # balanced antarctica with north->south ordering
    self.current_best = 19.63026 * 1e9 # single-trip optimized heavy antarctica with north->south ordering
    self.current_best = 16.87566 * 1e9 # single-trip optimized balanced antarctica with north->south ordering
    self.current_best = 16.57840 * 1e9 # slightly sa-optimized, single-trip optimized balanced antarctica with north->south ordering
    self.current_score = None
    self.current_trip_count = None
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

  def _load_trips_from_file(self, args):
    matches = glob.glob("data/*{}*".format(args.from_file))
    if not matches:
      self.log.warning("No matching file found, aborting!")
      return
    if len(matches) > 1:
      self.log.warning("More than one matching file found, aborting! ({})".format(matches))
      return
    self.log.info("Using file {} from {} matching files ({})".format(matches[0], len(matches), matches))
    data = pd.read_csv(matches[0]).merge(self.gifts, on="GiftId")
    self.current_score = utils.weighted_reindeer_weariness(data)
    self.current_trip_count = len(data.TripId.unique())
    return data

  def verify_trips(self):
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

    if not utils.verify_weights(merged, self.log):
      self.log.error("One or more trip is invalid!")
      return False

    self.log.info("Trips are valid")
    return True

  def evaluate_trips(self):
    unique_trips = self.trips.TripId.unique()
    merged = self.trips.merge(self.gifts, on="GiftId")
    trips = [merged[merged.TripId == t] for t in unique_trips]

    score = utils.weighted_reindeer_weariness(merged)
    utils.log_success_or_error(self.log, score < self.current_score, "Cost of the {} trips: {:.5f}B ({:.5f}M with {} trips)".format(
      unique_trips.shape[0], score / 1e9, (score - self.current_score) / 1e6, self.current_trip_count))
    utils.log_success_or_error(self.log, score < self.current_best, "Compared to best: {:.5f}M".format(
      (score - self.current_best) / 1e6))

    weights = np.asarray([trip.Weight.sum() for trip in trips])
    self.log.info("Sleigh utilization: min {:.2f}, max {:.2f}, avg {:.2f}, std {:.2f}".format(
      weights.min(), weights.max(), weights.mean(), weights.std()))

    costs = np.asarray([utils.weighted_trip_length(trip[["Latitude","Longitude"]], trip.Weight) for trip in trips])
    self.log.info("Trip costs: min {:.2f}M, max {:.2f}M, avg {:.2f}M, std {:.2f}k".format(
      costs.min() / 1e6, costs.max() / 1e6, costs.mean() / 1e6, costs.std() / 1e3))

    stops = np.asarray([trip.shape[0] for trip in trips])
    self.log.info("Stops per trip: min {}, max {}, avg {:.2f}, std {:.2f}".format(
      stops.min(), stops.max(), stops.mean(), stops.std()))

    cache_info = utils.get_cache_info()
    self.log.info("Distance cache info: {} ({:.2f}% hits))".format(
      cache_info, 100.0 * cache_info.hits / (cache_info.hits + cache_info.misses)))

  def write_trips(self, file_name):
    """Creates a submission file from the calculated trips

    :file_name: Name of the file to write

    """
    self.log.debug("Writing trips to {}".format(file_name))
    self.trips[["GiftId", "TripId"]].astype(int).to_csv(file_name, index=False)

