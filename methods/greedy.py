#!/usr/bin/env python

import pickle
from os import path

import numpy as np

import pandas as pd
import utils
from method import Method
from neighbor import Neighbor


class StupidGreedyMethod(Method):
  @property
  def name(self):
    return "stupidgreedy"

  def run(self, args):
    """
    Idea: Create valid trips by picking gifts in order and start new trip when needed.
    """
    current_trip = 1
    current_capacity = utils.WEIGHT_LIMIT
    trips = []
    for _, gift in self.gifts.iterrows():
      if current_capacity >= gift.Weight:
        current_capacity -= gift.Weight
      else:
        current_trip += 1
        current_capacity = utils.WEIGHT_LIMIT - gift.Weight
      trips.append({"GiftId": int(gift.GiftId), "TripId": current_trip})

    self.trips = self.trips.append(trips)


class BiggestGiftsFirstGreedyMethod(Method):
  @property
  def name(self):
    return "biggestfirstgreedy"

  def run(self, args):
    """
    Idea: Improve on StupidGreedyMethod by treating the ~2000 gifts with largest size 50 differently:
    Always deliver them at the beginning of a trip.
    """
    biggest_gifts = self.gifts[self.gifts.Weight == 50]
    biggest_gifts_count = biggest_gifts.shape[0]
    other_gifts = self.gifts[self.gifts.Weight < 50]

    current_trip = 1
    trips = []
    trips.append({"GiftId": int(biggest_gifts.iloc[0].GiftId), "TripId": current_trip})
    trips.append({"GiftId": int(biggest_gifts.iloc[1].GiftId), "TripId": current_trip})
    biggest_gift_index = 2
    current_capacity = utils.WEIGHT_LIMIT - biggest_gifts.iloc[0].Weight - biggest_gifts.iloc[1].Weight

    for _, gift in other_gifts.iterrows():
      if current_capacity >= gift.Weight:
        current_capacity -= gift.Weight
      else:
        current_trip += 1
        current_capacity = utils.WEIGHT_LIMIT - gift.Weight

        if biggest_gift_index < biggest_gifts_count:
          big_gift_1 = biggest_gifts.iloc[biggest_gift_index]
          trips.append({"GiftId": int(big_gift_1.GiftId), "TripId": current_trip})
          current_capacity -= big_gift_1.Weight
        if biggest_gift_index+1 < biggest_gifts_count:
          big_gift_2 = biggest_gifts.iloc[biggest_gift_index+1]
          trips.append({"GiftId": int(big_gift_2.GiftId), "TripId": current_trip})
          current_capacity -= big_gift_2.Weight
        biggest_gift_index += 2

      trips.append({"GiftId": int(gift.GiftId), "TripId": current_trip})

    self.trips = self.trips.append(trips)


class MinimumTripsGreedMethod(Method):
  @property
  def name(self):
    return "minimumtripsgreedy"

  def run(self, args):
    """
    Idea: Minimize the number of trips needed to distribute all gifts. Doesn't work.
    """
    sorted_gifts = self.gifts.sort_values("Weight", ascending=False)
    current_trip = 1
    current_capacity = utils.WEIGHT_LIMIT
    trips = []
    while sorted_gifts.shape[0] > 0:
      gift_added = False
      # find gift that fits into sleigh
      for _, gift in sorted_gifts.iterrows():
        if current_capacity < gift.Weight:
          continue

        # found gift, add to trip and remove from remaining gifts
        current_capacity -= gift.Weight
        sorted_gifts = sorted_gifts.drop(gift.name)
        trips.append({"GiftId": int(gift.GiftId), "TripId": current_trip})
        gift_added = True
        break

      if not gift_added:
        # we couldn't find a gift that fits into the current sleigh, start new trip
        current_trip += 1
        current_capacity = utils.WEIGHT_LIMIT
        self.log.debug("Started trip {}, remaining gifts: {}".format(current_trip, sorted_gifts.shape[0]))

    self.trips = self.trips.append(trips)


class HeavyAntarcticaGreedyMethod(Method):
  @property
  def name(self):
    return "antarctica"

  def calculate_heavy_trip_assignments(self):
    antarctica_gifts = self.gifts[self.gifts.Latitude < -60]
    heavy_antarctica_gifts = antarctica_gifts[antarctica_gifts.Weight > 30]
    self.log.info("Creating {} trips for the heaviest gifts that need to be delievered furthest".format(heavy_antarctica_gifts.shape[0]))
    trips = {gift.Longitude: (gift.Weight, [gift]) for _, gift in heavy_antarctica_gifts.iterrows()}

    remaining_gifts = self.gifts[~self.gifts.isin(heavy_antarctica_gifts)].dropna().sort_values("Weight", ascending=False)
    self.log.info("Spreading the remaining {} gifts across the existing trips".format(remaining_gifts.shape[0]))

    for _, gift in remaining_gifts.iterrows():
      # find trip where the heavy antarctica gift has the most similar longitude
      trip_longitude = None
      minimum_distance = None

      for longitude, gift_info in trips.items():
        if gift.Weight + gift_info[0] > utils.WEIGHT_LIMIT:
          # avoid invalid assignments
          continue
        current_distance = abs(gift.Longitude - longitude)
        if trip_longitude is None or current_distance < minimum_distance:
          minimum_distance = current_distance
          trip_longitude = longitude

      # assign current gift to the closest trip
      trips[trip_longitude] = (trips[trip_longitude][0] + gift.Weight, trips[trip_longitude][1] + [gift])

    return trips

  def get_optimally_sorted_trips(self, trips):
    self.log.info("Putting trips into optimal order")
    total_trip_count = len(trips)
    all_trips = []
    for i, trip in enumerate(trips.values()):
      self.log.debug("Processing trip {}/{} with length {}".format(i+1, total_trip_count, len(trip[1])))
      sorted_trip = sorted(trip[1], key=lambda tup: tup.Weight, reverse=True)
      this_trip = []
      for gift in sorted_trip:
        if len(this_trip) == 0:
          this_trip.append(gift)
          continue
        gift[utils.TRIP] = len(all_trips)
        best_index, _ = Neighbor.find_best_insertion_index(np.asarray(this_trip), gift.values,
            lat_index=1, lon_index=2, weight_index=3)
        this_trip.insert(best_index, gift)
      if this_trip:
        all_trips.append(this_trip)

    # extract gift/trip mapping
    combined_trips = np.concatenate(all_trips)[:, [utils.GIFT, utils.TRIP]]
    return pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

  def get_geographically_sorted_trips(self, trips):
    # drop off gifts from north to south
    self.log.info("Putting trips into geographical order")
    all_trips = []
    for trip in trips.values():
      sorted_trip = sorted(trip[1], key=lambda tup: tup.Latitude, reverse=True)
      this_trip = []
      for t in sorted_trip:
        t[utils.TRIP] = len(all_trips)
        this_trip.append(t)
      if this_trip:
        all_trips.append(this_trip)

    # extract gift/trip mapping
    combined_trips = np.concatenate(all_trips)[:, [utils.GIFT, utils.TRIP]]
    return pd.DataFrame(combined_trips, columns=["GiftId", "TripId"])

  def run(self, args):
    """
    Idea: Create valid trips by creating one for each of the heaviest gifts in antarctica
    and then assign the other gifts to these based on their Longitude.
    """
    trips_file = "unsorted-heavy-trips.pkl"
    if path.exists(trips_file):
      self.log.info("Loading unsorted trips from '{}'".format(trips_file))
      with open(trips_file, "rb") as fh:
        trips = pickle.load(fh)
    else:
      trips = self.calculate_heavy_trip_assignments()
      self.log.info("Writing unsorted trips to '{}'".format(trips_file))
      with open(trips_file, "wb") as fh:
        pickle.dump(trips, fh)

    self.trips = self.get_geographically_sorted_trips(trips)


class BalancedAntarcticaGreedyMethod(Method):
  @property
  def name(self):
    return "balancedantarctica"

  def calculate_balanced_trip_assignments(self):
    antarctica_gifts = self.gifts[self.gifts.Latitude < -60]
    heavy_antarctica_gifts = antarctica_gifts[antarctica_gifts.Weight > 35]
    self.log.info("Creating {} trips for the heaviest gifts that need to be delievered furthest".format(heavy_antarctica_gifts.shape[0]))
    trips = {gift.Longitude: (gift.Weight, [gift]) for _, gift in heavy_antarctica_gifts.iterrows()}

    remaining_gifts = self.gifts[~self.gifts.isin(heavy_antarctica_gifts)].dropna().sort_values("Weight", ascending=False)

    self.log.info("Inserting additional boundaries to prevent long trips")
    # create histogram of all gifts, distributed across the current trip longitudes
    bounds = sorted(heavy_antarctica_gifts.Longitude)
    histogram = np.histogram(self.gifts.Longitude, bounds)

    # keep inserting bounds until no very long trips remain
    max_trip_length = 210
    added_bounds = []
    while histogram[0].max() > max_trip_length:
      bounds = histogram[1][:-1]
      new_bounds = []
      for bound in bounds[histogram[0] > max_trip_length]:
        i = np.where(histogram[1] == bound)[0][0]
        if i > 0:
          new_bound_before = bounds[i-1] + (bound-bounds[i-1])/2
          new_bounds.append(new_bound_before)
          added_bounds.append(new_bound_before)
        if i < len(bounds)-1:
          new_bound_after = bound + (bounds[i+1]-bound)/2
          new_bounds.append(new_bound_after)
          added_bounds.append(new_bound_after)
      bounds = sorted(np.append(bounds, new_bounds))
      histogram = np.histogram(self.gifts.Longitude, bounds)
    for added_bound in added_bounds:
      trips[added_bound] = (0, [])
    self.log.info("Added bounderies for a total of {} trips (expected gifts/trip average: {}, std: {})"
        .format(len(trips.keys()), histogram[0].mean(), histogram[0].std()))

    self.log.info("Spreading the remaining {} gifts across the existing trips".format(remaining_gifts.shape[0]))
    for _, gift in remaining_gifts.iterrows():
      # find trip where the heavy antarctica gift has the most similar longitude
      trip_longitude = None
      minimum_distance = None

      for longitude, gift_info in trips.items():
        if gift.Weight + gift_info[0] > utils.WEIGHT_LIMIT:
          # avoid invalid assignments
          continue
        current_distance = abs(gift.Longitude - longitude)
        if trip_longitude is None or current_distance < minimum_distance:
          minimum_distance = current_distance
          trip_longitude = longitude

      # assign current gift to the closest trip
      trips[trip_longitude] = (trips[trip_longitude][0] + gift.Weight, trips[trip_longitude][1] + [gift])

    gifts_per_trip_counts = [len(t[1]) for t in trips.values()]
    self.log.info("Spread all gifts across {} trips (gifts/trip average: {}, std: {})".format(
      len(trips.values()), np.mean(gifts_per_trip_counts), np.std(gifts_per_trip_counts)))

    return trips

  def run(self, args):
    """
    Idea: Create valid trips by creating
    and then assign the other gifts to these based on their Longitude.
    """
    trips_file = "unsorted-balanced-trips.pkl"
    if path.exists(trips_file):
      self.log.info("Loading unsorted trips from '{}'".format(trips_file))
      with open(trips_file, "rb") as fh:
        trips = pickle.load(fh)
    else:
      trips = self.calculate_balanced_trip_assignments()
      self.log.info("Writing unsorted trips to '{}'".format(trips_file))
      with open(trips_file, "wb") as fh:
        pickle.dump(trips, fh)

    self.trips = HeavyAntarcticaGreedyMethod.get_optimally_sorted_trips(self, trips) # naughty!

