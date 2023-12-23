import json

def load_json(file_path, args=None):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if args is not None:
        args.update(data)
    else:
        args = data
    
    return args
