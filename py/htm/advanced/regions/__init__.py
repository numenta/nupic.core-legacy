
import json

def extractList(listString, dataType=None):
    """
    Extract a list of dataType from list_string string.
    The same separator must be used consistently in the string. Either ',' or single spaces.
    If dataType is None, just return a parsed list.
    """
    data_list = []
    list_string = listString
    
    if list_string:
        try:
            data_list = json.loads(list_string)
            
        except json.decoder.JSONDecodeError:
            try:
                list_string = list_string.replace('   ', ' ')
                list_string = list_string.replace('  ', ' ')
                if list_string.startswith('[ '):
                    list_string = list_string.replace('[ ', '[')
                list_string = (',').join(list_string.split(' '))
                data_list = json.loads(list_string)
                
            except json.decoder.JSONDecodeError:
                list_string = list_string.replace('.,', '.0,')
                list_string = list_string.replace('.]', '.0]')
                try:
                    data_list = json.loads(list_string)
                except json.decoder.JSONDecodeError as ex:
                    # Got something really out of bounds
                    # Catch and then ass on as a debugging hook
                    raise ex
                
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