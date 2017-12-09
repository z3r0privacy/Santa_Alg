#!/usr/bin/env python

import abc

import utils


class Neighbor(abc.ABC):
  def __init__(self, log):
    self.log = log

  @property
  @abc.abstractmethod
  def cost_delta(self):
    """
    Calculates the difference in cost that applying the neighbor would bring.
    Memoization is strongly encouraged (after @property).
    """
    pass

  @abc.abstractmethod
  def apply(self):
    """
    Modifies the trip to apply the neighbor.
    """
    pass

  def get_cost_of_tour_of_three(self, a, b, c, cumulative_weight_at_a, weight_at_b):
    return utils.distance(a, b) * cumulative_weight_at_a + \
        utils.distance(b, c) * (cumulative_weight_at_a - weight_at_b)

