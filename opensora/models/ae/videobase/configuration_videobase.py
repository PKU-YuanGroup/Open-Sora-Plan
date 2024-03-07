import json

"""
We are currently working on the abstraction of the code structure, 
so there is no content here for the moment. 
It will be gradually optimized over time. 
Contributions from the open-source community are welcome.
"""
class VideoBaseConfiguration:
    def __init__(self, **kwargs):
        pass
    
    def to_json_string(self):
        json_string = json.dumps(vars(self))
        return json_string
    
    def to_dict(self):
        return vars(self)
    
    @classmethod
    def load_from_file(cls, config_path):
        with open(config_path, 'r') as json_file:
            config_dict = json.load(json_file)
        return cls(**config_dict)