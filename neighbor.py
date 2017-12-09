#!/usr/bin/env python

import abc


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

