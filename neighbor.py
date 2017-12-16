#!/usr/bin/env python

import abc

import numpy as np

import utils


class Neighbor(abc.ABC):
  VERIFY_COST_DELTA = False
  log = None

  def __init__(self):
    self.cost = None

  @abc.abstractmethod
  def cost_delta(self):
    """
    Calculates the difference in cost that applying the neighbor would bring.
    Manual memoization is strongly encouraged.
    """
    pass

  @abc.abstractmethod
  def apply(self):
    """
    Modifies the trip to apply the neighbor.
    """
    pass

  @staticmethod
  def get_cost_of_tour_of_three(a, b, c, cumulative_weight_at_a, weight_at_b):
    return utils.distance(a, b) * cumulative_weight_at_a + \
        utils.distance(b, c) * (cumulative_weight_at_a - weight_at_b)

  """
  @staticmethod
  def _get_cost_of_inserting_here(trip, i, row, gift, index_to_be_removed, lat_index, lon_index, weight_index):
    if index_to_be_removed is not None and (i == index_to_be_removed or i+1 == index_to_be_removed):
      # don't compute insertion where we're about to remove
      return None

    # note: we evaluate inserting before the current node - that means we won't try to insert
    # in the very end of the tour

    # we need to consider the distance from NP to the first gift - unless we're evaluating the first gift
    distance = 0 if i == 0 else utils.distance(utils.NORTH_POLE, tuple(trip[0][[lat_index, lon_index]]))
    # add distances up until the one before the current
    for k in range(i-1):
      distance += utils.distance(
          tuple(trip[k][[lat_index, lon_index]]),
          tuple(trip[k+1][[lat_index, lon_index]])
          )
    cost_to_carry_gift = distance * gift[weight_index]

    previous_location = tuple(trip[i-1][[lat_index, lon_index]]) if i > -0 else utils.NORTH_POLE
    location_of_current = tuple(row[[lat_index, lon_index]])
    cum_weight = np.sum(trip[i:][:, weight_index]) + utils.SLEIGH_WEIGHT + gift[weight_index]
    cost_to_move_here = Neighbor.get_cost_of_tour_of_three(
        previous_location, tuple(gift[[lat_index, lon_index]]), location_of_current,
        cum_weight, gift[weight_index])

    cost_for_old_path = utils.distance(location_of_current, previous_location) * (cum_weight - gift[weight_index])

    cost_to_insert_here = cost_to_carry_gift + cost_to_move_here - cost_for_old_path

    return cost_to_insert_here

  @staticmethod
  def _try_insertion_places(results, trip, args, gift, index_to_be_removed, lat_index, lon_index, weight_index):
    print(12)
    minimum_cost = np.finfo(np.float64).max
    best_index = None
    for i, row in args:
      # job = Process(target=Neighbor._try_insertion_places, args=(trip, i, row, gift,
        # index_to_be_removed, lat_index, lon_index, weight_index))

      cost_to_insert_here = Neighbor._get_cost_of_inserting_here(trip, i, row, gift,
          index_to_be_removed, lat_index, lon_index, weight_index)
      if cost_to_insert_here is None:
        continue
      if cost_to_insert_here < minimum_cost:
        minimum_cost = cost_to_insert_here
        best_index = i

    print("FOO")
    # results.append((minimum_cost, best_index))
    return minimum_cost, best_index

  @staticmethod
  def find_best_insertion_index(trip, gift, index_to_be_removed=None,
      lat_index=utils.LAT, lon_index=utils.LON, weight_index=utils.WEIGHT):
    minimum_cost = np.finfo(np.float64).max
    best_index = None

    if len(trip) > 1e0:
      n_jobs = 2
      all_args = [(i, row) for i, row in enumerate(trip)]
      job_size = int(len(all_args)/n_jobs)
      jobs = []
      # print("all", len(all_args))
      # print("size", job_size)

      per_job_args = []

      results = []
      for j in range(n_jobs):
        if j < n_jobs-1:
          # print(j*job_size, (j+1)*job_size)
          args = all_args[j*job_size:(j+1)*job_size]
        else:
          # print(j*job_size, "END")
          args = all_args[j*job_size:]
        per_job_args.append(args)
        # print(len(args))
        arg_tuple = (results, trip, args, gift, index_to_be_removed, lat_index, lon_index, weight_index)
        job = Neighbor.pool.apply_async(foobar, zip([Neighbor]*len(arg_tuple), arg_tuple))
        jobs.append(job)
      for j in jobs:
        print("FOO", j)
        results.append(j.get())

      # Neighbor.pool.map(Neighbor._try_insertion_places, per_job_args)
      raise ValueError()

      for result in results:
        if result[0] < minimum_cost:
          minimum_cost = result[0]
          best_index = result[1]
      return best_index, minimum_cost
    else:
      for i, row in enumerate(trip):
        cost_to_insert_here = Neighbor._get_cost_of_inserting_here(trip, i, row, gift,
            index_to_be_removed, lat_index, lon_index, weight_index)
        if cost_to_insert_here is None:
          continue
        if cost_to_insert_here < minimum_cost:
          minimum_cost = cost_to_insert_here
          best_index = i
      return best_index, minimum_cost
  """

  @staticmethod
  def find_best_insertion_index(trip, gift, index_to_be_removed=None,
      lat_index=utils.LAT, lon_index=utils.LON, weight_index=utils.WEIGHT):
    minimum_cost = np.finfo(np.float64).max
    best_index = None
    for i, row in enumerate(trip):
      if index_to_be_removed is not None and (i == index_to_be_removed or i+1 == index_to_be_removed):
        # don't compute insertion where we're about to remove
        continue

      # note: we evaluate inserting before the current node - that means we won't try to insert
      # in the very end of the tour

      # we need to consider the distance from NP to the first gift - unless we're evaluating the first gift
      distance = 0 if i == 0 else utils.distance(utils.NORTH_POLE, tuple(trip[0][[lat_index, lon_index]]))
      # add distances up until the one before the current
      for k in range(i-1):
        distance += utils.distance(
            tuple(trip[k][[lat_index, lon_index]]),
            tuple(trip[k+1][[lat_index, lon_index]])
            )
      cost_to_carry_gift = distance * gift[weight_index]

      previous_location = tuple(trip[i-1][[lat_index, lon_index]]) if i > -0 else utils.NORTH_POLE
      location_of_current = tuple(row[[lat_index, lon_index]])
      cum_weight = np.sum(trip[i:][:, weight_index]) + utils.SLEIGH_WEIGHT + gift[weight_index]
      cost_to_move_here = Neighbor.get_cost_of_tour_of_three(
          previous_location, tuple(gift[[lat_index, lon_index]]), location_of_current,
          cum_weight, gift[weight_index])

      cost_for_old_path = utils.distance(location_of_current, previous_location) * (cum_weight - gift[weight_index])

      cost_to_insert_here = cost_to_carry_gift + cost_to_move_here - cost_for_old_path

      if cost_to_insert_here < minimum_cost:
        minimum_cost = cost_to_insert_here
        best_index = i

    return best_index, minimum_cost

  def _cost_to_remove_gift(self, trip, index_to_be_removed):
    gift_to_remove = trip[index_to_be_removed]
    i = index_to_be_removed

    # we need to consider the distance from NP to the first gift - unless we're evaluating the first gift
    distance = 0 if i == 0 else utils.distance(utils.NORTH_POLE, tuple(trip[0][[utils.LAT, utils.LON]]))
    # add distances up until the one before the current
    for k in range(i-1):
      distance += utils.distance(
          tuple(trip[k][[utils.LAT, utils.LON]]),
          tuple(trip[k+1][[utils.LAT, utils.LON]])
          )
    cost_to_not_carry_gift = distance * -gift_to_remove[utils.WEIGHT]

    previous_location = tuple(trip[i-1][[utils.LAT, utils.LON]]) if i > -0 else utils.NORTH_POLE
    next_location = tuple(trip[i+1][[utils.LAT, utils.LON]]) if i < len(trip)-1 else utils.NORTH_POLE
    location_of_current = tuple(gift_to_remove[[utils.LAT, utils.LON]])
    cum_weight = np.sum(trip[i:][:, utils.WEIGHT]) + utils.SLEIGH_WEIGHT # + gift_to_remove[utils.WEIGHT]
    cost_of_old_tour = Neighbor.get_cost_of_tour_of_three(
        previous_location, location_of_current, next_location,
        cum_weight, gift_to_remove[utils.WEIGHT])

    cost_for_new_path = utils.distance(previous_location, next_location) * (cum_weight - gift_to_remove[utils.WEIGHT])

    total_cost = cost_to_not_carry_gift - cost_of_old_tour + cost_for_new_path

    return total_cost

