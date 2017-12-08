#!/usr/bin/env python

import utils
from method import Method


class GreedyMethod(Method):
  @property
  def name(self):
    return "greedy"

  def run(self, args):
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

