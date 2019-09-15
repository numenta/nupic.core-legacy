
from collections.abc import Iterable

def executeCommand(command_name, region_instance, *args):
    """
    Send the command to the region instance together with the arguments.
    """
    string_args = [command_name]
    for arg in args:
            
        if type(arg) == str:
            string_args.append(arg)

        elif isinstance(arg, Iterable):
            s = ','.join([str(a) for a in arg])
            string_args.append(s)
            
        elif type(arg) == bool:
            string_args.append(str(int(arg)))
            
        else:
            string_args.append(str(arg))
            
    return region_instance.executeCommand(string_args)

def extractList(list_string, dataType):
    """
    Extract a list if dataType from list string string.
    """
    if list_string:
        data_list = list_string.split(',')
        if data_list:
            return [dataType(s) for s in data_list]

    return []
    