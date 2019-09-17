# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc. 
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
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import pickle



class LoggingDecorator(object):
    """
    Decorator class for logging calls to be used to debug and reconstruct
    experiments.

    Usage:

    class Foo(object):
        @LoggingDecorator()
        def __init__(self, logCalls=False):
            # Calls to this method are logged
            self.logCalls = logCalls

        @LoggingDecorator()
        def bar(self, *args, **kwargs):
            # Calls to this method are logged
            pass

        def baz(self):
            # Calls to this method are NOT logged
            pass


    foo = Foo(logCalls=True)
    foo.bar(1, two=2)
    foo.baz()

    print "\n========== CALL LOG ==============="

    for call in foo.callLog:
        print call
        print

    print "=====================================\n"

    LoggingDecorator.save(foo.callLog, "callLog.pkl")

    for call in LoggingDecorator.load("callLog.pkl"):
        print call
        print

    print "=====================================\n"

    """

    @staticmethod
    def __call__(fn):
        """ 
        Returns decorated function that logs calls
        """
        def _fn(instance, *args, **kwargs):
            if not hasattr(instance, "callLog"):
                instance.callLog = []

            # Log if, and only if logCalls is set to True, either as an attr on instance
            # or as a kwarg to the called function (e.g. constructor/__init__) AND
            # call originated internally
            if getattr(instance, "logCalls", kwargs.get("logCalls", False)):
                instance.callLog.append([fn.__name__, {"args": args, "kwargs": kwargs}])

            return fn(instance, *args, **kwargs)


        return _fn


    @staticmethod
    def save(callLog, logFilename):
        """
        Save the call log history into this file.

        @param logFilename (path) Filename in which to save a pickled version of the call logs.

        """
        with open(logFilename, "wb") as outp:
            pickle.dump(callLog, outp)


    @staticmethod
    def load(logFilename):
        """
        Load a previously saved call log history from file, returns new
        LoggingDecorator instance separate from singleton.

        @param logFilename (path) Filename from which to load a pickled version of the call logs.
        """
        with open(logFilename, "rb") as inp:
            return pickle.load(inp)
