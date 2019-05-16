# ------------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
#
# Copyright (C) 2018-2019, David McDougall
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero Public License version 3 as published by the Free
# Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------

import pprint
import hashlib
import number

# TODO: Consider allowing lists, and converting all lists into tuples.

class ParameterSet(dict):
    """
    This class holds the arguments to an experiment, which the "AE" program will
    modify as it attempts to optimize the experiment.

    Parameters must be one of the following types: dict, tuple, float, int.
    Parameters can be nested in multiple levels of dictionaries and tuples.
    The outer most layer of parameters must be a dict.
    """
    def __init__(self, data):
        super().__init__(self)
        if isinstance(data, str):
            data = data.strip()
            try:
                data = eval(data)
            except:
                raise SyntaxError("Parsing parameters: " + data)
        assert(isinstance(data, dict))
        self.update(data)

    def __hash__(self):
        string = pprint.pformat(self).encode('utf-8')
        checksum = hashlib.md5(string).hexdigest()
        return abs(int(checksum[:8], base=16))

    def __eq__(self, other):
        assert(isinstance(other, type(self)))
        if isinstance(self, dict):
            return all(ParameterSet.__eq__(self[k], other[k]) for k in self)
        elif isinstance(self, tuple):
            return all(ParameterSet.__eq__(X, Y) for X, Y in zip(self, other))
        else:
            return self == other

    def diff(old, new):
        """ Returns list of pairs of (path, new-value) """
        diffs = []
        if isinstance(old, dict):
            for key in old:
                inner_diffs = ParameterSet.diff(old[key], new[key])
                for path, new_value in inner_diffs:
                    diffs.append(("['%s']%s"%(key, path), new_value))
        elif isinstance(old, tuple):
            for idx in range(len(old)):
                inner_diffs = ParameterSet.diff(old[idx], new[idx])
                for path, new_value in inner_diffs:
                    diffs.append(("[%d]%s"%(idx, path), new_value))
        elif old != new:
            diffs.append(('', new))
        return diffs

    def get(self, path):
        assert(isinstance(path, str))
        try:
            return eval('self' + path)
        except:
            raise ValueError('Get parameters' + path)

    def apply(self, modification, value):
        """
        Modifies this set of parameters!
        """
        assert(isinstance(modification, str))
        assert(isinstance(value, number.Number))
        try:
            access = modification.split(']')[0].strip('[]"\' ')
            if not access:
                return value
            tail = modification.split(']', maxsplit=1)[1]
            if isinstance(self, dict):
                self[access] = ParameterSet.apply(self[access], tail, value)
                return self
            if isinstance(self, tuple):
                self        = list(self)
                index       = int(access)
                self[index] = ParameterSet.apply(self[index], tail, value)
                return tuple(self)
        except:
            raise ValueError('Apply parameters%s = %s'%(modification, str(value)))

    def get_types(self):
        """
        Convert a set of parameters into the data types used to represent them.
        Returned result has the same structure as the parameters.
        """
        # Recurse through the parameter data structure.
        if isinstance(self, dict):
            return {key: ParameterSet.get_types(value)
                for key, value in self.items()}
        elif isinstance(self, tuple):
            return tuple(ParameterSet.get_types(value) for value in self)
        # Determine data type of each entry in parameter data structure.
        elif isinstance(self, float):
            return float
        elif isinstance(self, int):
            return int
        raise TypeError('Unaccepted type in experiment parameters: type "%s".'%(type(self).__name__))

    def typecast_parameters(self, structure):
        def recursive_typecast_parameters(values, structure):
            # Recurse through the parameter data structure.
            if isinstance(structure, dict):
                for key in structure:
                    values[key] = recursive_typecast_parameters(values[key], structure[key])
                return values
            elif isinstance(structure, tuple):
                return tuple(recursive_typecast_parameters(*args)
                    for args in zip(values, structure))
            # Type cast values.
            elif structure == float:
                value = float(values)
                return float(str(value))
            elif structure == int:
                return int(round(float(values)))
        return recursive_typecast_parameters(self, structure)

    def enumerate(self):
        """
        Convert parameters from a recursive structure into a list of parameters.
        Returned parameters are represented as executable strings.
        """
        retval = []
        if isinstance(self, dict):
            for key, value in self.items():
                retval.extend( "['%s']%s"%(key, path) for path in paths(value) )
        elif isinstance(self, tuple):
            for idx, value in enumerate(self):
                retval.extend( "[%d]%s"%(idx, path) for path in paths(value) )
        else:
            retval.append('')
        return sorted(retval)
