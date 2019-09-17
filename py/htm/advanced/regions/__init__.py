
import json

def extractList(list_string, dataType=None):
    """
    Extract a list of dataType from list_string string.
    The same separator must be used consistently in the string. Either ',' or single spaces.
    If dataType is None, just return a parsed list.
    """
    data_list = []
    
    if list_string:
        try:
            data_list = json.loads(list_string)
            
        except json.decoder.JSONDecodeError:
            try:
                list_string = (',').join(list_string.split(' '))
                data_list = json.loads(list_string)
                
            except json.decoder.JSONDecodeError:
                list_string = list_string.replace('.,', '.0,')
                list_string = list_string.replace('.]', '.0]')
                data_list = json.loads(list_string)
                
        if data_list:
            if dataType is None:
                return data_list
            else:
                return [dataType(s) for s in data_list]

    return []
    
def asBool(arg):
    """
    Convert arg to a bool. 
    Accepts int or ints as strings or bool or bool as string
    """
    if type(arg) == bool:
        return arg
    elif type(arg) == str and arg.upper() == 'TRUE':
        return True
    elif type(arg) == str and arg.upper() == 'FALSE':
        return False
    
    return bool(int(arg))