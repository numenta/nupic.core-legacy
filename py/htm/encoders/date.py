# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2013, Numenta, Inc.
#               2019, David McDougall
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

import datetime

import numpy
import math

from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.sdr import SDR


class DateEncoder:
  """
  A date encoder encodes a time and date.  The input to a date encoder is a
  datetime.datetime object. The output is the concatenation of several sub-
  encodings, each of which encodes a different aspect of the date. Which sub-
  encodings are present, and details of those sub-encodings, are specified in
  the DateEncoder constructor.
  """
  def __init__(self,
          season=0,
          dayOfWeek=0,
          weekend=0,
          holiday=0,
          timeOfDay=0,
          customDays=0,
          holidays=((12, 25),)):
    """
    Each parameter describes one attribute to encode. By default, the attribute
    is not encoded.

    Argument season: (int | tuple) Season of the year, where units = day.

        - (int) width of attribute; default radius = 91.5 days (1 season)
        - (tuple)  season[0] = width; season[1] = radius

    Argument dayOfWeek: (int | tuple) Day of week, where monday = 0, units = 1 day.
        The timestamp is compared against day:noon, so encodings of a day switch representation on midnight.

        - (int) width of attribute; default radius = 1 day
        - (tuple) dayOfWeek[0] = width; dayOfWeek[1] = radius

    Argument weekend: (int) Is a weekend or not. A block of bits either 0s or 1s.
    Note: the implementation treats "weekend" as starting Fri 6pm, till Sun midnight. 

        - (int) width of attribute
        - TODO remove and replace by customDays=(width, ["Saturday", "Sunday"]) ?

    Argument holiday: (int) Is a holiday or not, boolean: 0, 1

        - (int) width of attribute

    Argument timeOfday: (int | tuple) Time of day, where midnight = 0, units = hour.

        - (int) width of attribute: default radius = 4 hours
        - (tuple) timeOfDay[0] = width; timeOfDay[1] = radius

    Argument customDays: (tuple) A way to custom encode specific days of the week.

        - [0] (int) Width of attribute
        - [1] (str | list) Either a string representing a day of the week like
          "Monday" or "mon", or a list of these strings.

    Argument holidays: (list) a list of tuples for holidays.

        - Each holiday is either (month, day) or (year, month, day).
          The former will use the same month day every year eg: (12, 25) for Christmas.
          The latter will be a one off holiday eg: (2018, 4, 1) for Easter Sunday 2018
        - By default the only holiday is December 25.
    """
    self.size = 0

    self.seasonEncoder = None
    if season != 0:
      p = ScalarEncoderParameters()
      # Ignore leapyear differences -- assume 366 days in a year
      # Radius = 91.5 days = length of season
      # Value is number of days since beginning of year (0 - 355)
      p.minimum  = 0
      p.maximum  = 366
      p.periodic = True
      try:
        activeBits, radius = season
      except TypeError:
        p.activeBits = season
        p.radius     = 91.5
      else:
        p.activeBits = season[0]
        p.radius     = season[1]
      self.seasonEncoder = ScalarEncoder(p)
      self.size += self.seasonEncoder.size

    self.dayOfWeekEncoder = None
    if dayOfWeek != 0:
      p = ScalarEncoderParameters()
      # Value is day of week (floating point)
      # Radius is 1 day
      p.minimum  = 0
      p.maximum  = 7
      p.periodic = True
      try:
        activeBits, radius = dayOfWeek
      except TypeError:
        p.activeBits = dayOfWeek
        p.radius     = 1
      else:
        p.activeBits = dayOfWeek[0]
        p.radius     = dayOfWeek[1]
      self.dayOfWeekEncoder = ScalarEncoder(p)
      self.size += self.dayOfWeekEncoder.size

    self.weekendEncoder = None
    if weekend != 0:
      p = ScalarEncoderParameters()
      # Binary value.
      p.minimum    = 0
      p.maximum    = 1
      p.category   = True
      p.activeBits = weekend
      self.weekendEncoder = ScalarEncoder(p)
      self.size += self.weekendEncoder.size

    # Set up custom days encoder, first argument in tuple is width
    # second is either a single day of the week or a list of the days
    # you want encoded as ones.
    self.customDaysEncoder = None
    if customDays !=0:
      daysToParse = []
      assert len(customDays)==2, "Please provide a w and the desired days"
      if isinstance(customDays[1], list):
        daysToParse=customDays[1]
      elif isinstance(customDays[1], str):
        daysToParse = [customDays[1]]
      else:
        raise ValueError("You must provide either a list of days or a single day")
      # Parse days
      self.customDays = []
      for day in daysToParse:
        if(day.lower() in ["mon","monday"]):
          self.customDays += [0]
        elif day.lower() in ["tue","tuesday"]:
          self.customDays += [1]
        elif day.lower() in ["wed","wednesday"]:
          self.customDays += [2]
        elif day.lower() in ["thu","thursday"]:
          self.customDays += [3]
        elif day.lower() in ["fri","friday"]:
          self.customDays += [4]
        elif day.lower() in ["sat","saturday"]:
          self.customDays += [5]
        elif day.lower() in ["sun","sunday"]:
          self.customDays += [6]
        else:
          raise ValueError("Unable to understand %s as a day of week" % str(day))

      p = ScalarEncoderParameters()
      p.activeBits = customDays[0]
      p.minimum    = 0
      p.maximum    = 1
      p.category   = True
      self.customDaysEncoder = ScalarEncoder(p)
      self.size += self.customDaysEncoder.size

    self.holidayEncoder = None
    if holiday != 0:
      p = ScalarEncoderParameters()
      # A "continuous" binary value. = 1 on the holiday itself and smooth ramp
      # 0->1 on the day before the holiday and 1->0 on the day after the
      # holiday.
      p.minimum    = 0
      p.maximum    = 2
      p.radius     = 1
      p.periodic   = True
      p.activeBits = holiday
      self.holidayEncoder = ScalarEncoder(p)
      self.size += self.holidayEncoder.size

      for h in holidays:
        if not (hasattr(h, "__getitem__") or len(h) not in [2,3]):
          raise ValueError("Holidays must be an iterable of length 2 or 3")
      self.holidays = holidays

    self.timeOfDayEncoder = None
    if timeOfDay != 0:
      p = ScalarEncoderParameters()
      p.minimum  = 0
      p.maximum  = 24
      p.periodic = True
      # Value is time of day in hours
      # Radius = 4 hours, e.g. morning, afternoon, evening, early night,  late
      # night, etc.
      try:
        activeBits, radius = timeOfDay
      except TypeError:
        p.activeBits = timeOfDay
        p.radius     = 4
      else:
        p.activeBits = timeOfDay[0]
        p.radius     = timeOfDay[1]

      self.timeOfDayEncoder = ScalarEncoder(p)
      self.size += self.timeOfDayEncoder.size

    self.dimensions = (self.size,)
    assert(self.size > 0)

  def reset(self):
    """ Does nothing, DateEncoder holds no state. """
    pass

  def encode(self, inp, output=None):
    """
    Argument inp: (datetime) representing the time being encoded
    """
    if output is None:
      output = SDR(self.dimensions)
    else:
      assert( isinstance(output, SDR) )
      assert( all(x == y for x, y in zip( output.dimensions, self.dimensions )))

    if inp is None or (isinstance(inp, float) and math.isnan(inp)):
      output.zero()
      return output

    elif not isinstance(inp, datetime.datetime):
      raise ValueError("Input is type %s, expected datetime. Value: %s" % (
          type(inp), str(inp)))

    # -------------------------------------------------------------------------
    # Encode each sub-field
    sdrs      = []
    timetuple = inp.timetuple()
    timeOfDay = timetuple.tm_hour + float(timetuple.tm_min)/60.0

    if self.seasonEncoder is not None:
      # Number the days starting at zero, intead of 1 like the datetime does.
      dayOfYear = timetuple.tm_yday - 1
      assert(dayOfYear >= 0)
      # dayOfYear -= self.seasonEncoder.parameters.radius / 2. # Round towards the middle of the season.
      sdrs.append( self.seasonEncoder.encode(dayOfYear) )

    if self.dayOfWeekEncoder is not None:
      hrs_ = float(timeOfDay) / 24.0 # add hours as decimal value in extension to day  
      dayOfWeek = timetuple.tm_wday + hrs_
      dayOfWeek -= .5 # Round towards noon, not midnight, this means similarity of representations changes at midnights, not noon.
      # handle underflow: on Mon before noon -> move to Sun
      if dayOfWeek < 0:
        dayOfWeek += 7
      assert(dayOfWeek >= 0 and dayOfWeek < 7)
      sdrs.append( self.dayOfWeekEncoder.encode(dayOfWeek) )

    if self.weekendEncoder is not None:
      # saturday, sunday or friday evening
      if (timetuple.tm_wday == 6 or timetuple.tm_wday == 5 or
         (timetuple.tm_wday == 4 and timeOfDay > 18)):
        weekend = 1
      else:
        weekend = 0
      sdrs.append( self.weekendEncoder.encode(weekend) )

    if self.customDaysEncoder is not None:
      if timetuple.tm_wday in self.customDays:
        customDay = 1
      else:
        customDay = 0
      sdrs.append( self.customDaysEncoder.encode(customDay) )

    if self.holidayEncoder is not None:
      # A "continuous" binary value. = 1 on the holiday itself and smooth ramp
      #  0->1 on the day before the holiday and 1->0 on the day after the holiday.
      # holidays is a list of holidays that occur on a fixed date every year
      val = 0
      for h in self.holidays:
        # hdate is midnight on the holiday
        if len(h) == 3:
          hdate = datetime.datetime(h[0], h[1], h[2], 0, 0, 0)
        else:
          hdate = datetime.datetime(timetuple.tm_year, h[0], h[1], 0, 0, 0)
        if inp > hdate:
          diff = inp - hdate
          if diff.days == 0:
            # return 1 on the holiday itself
            val = 1
            break
          elif diff.days == 1:
            # ramp smoothly from 1 -> 0 on the next day
            val = 1.0 + (float(diff.seconds) / 86400)
            break
        else:
          diff = hdate - inp
          if diff.days == 0:
            # ramp smoothly from 0 -> 1 on the previous day
            val = 1.0 - (float(diff.seconds) / 86400)

      sdrs.append( self.holidayEncoder.encode(val) )

    if self.timeOfDayEncoder is not None:
      sdrs.append( self.timeOfDayEncoder.encode(timeOfDay) )

    if len(sdrs) > 1:
      output.concatenate( sdrs )
    else:
      output.setSDR( sdrs[0] )
    return output
