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


class BiggestGiftsFirstGreedyMethod(Method):
  @property
  def name(self):
    return "biggestfirstgreedy"

  def run(self, args):
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

