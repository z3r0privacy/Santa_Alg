#!/usr/bin/env python

import utils
from method import Method


class SimulatedAnnealingMethod(Method):
  @property
  def name(self):
    return "sim"

  def run(self, args):
    self.log.fatal("IMPLEMENT ME")
