import json
import yaml
from typing import TypeVar, Dict, Any
from diffusers import ConfigMixin

T = TypeVar('T', bound='VideoBaseConfiguration')
class VideoBaseConfiguration(ConfigMixin):
    config_name = "VideoBaseConfiguration"
    _nested_config_fields: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for key, value in vars(self).items():
            if isinstance(value, VideoBaseConfiguration):
                d[key] = value.to_dict()  # Serialize nested VideoBaseConfiguration instances
            elif isinstance(value, tuple):
                d[key] = list(value)
            else:
                d[key] = value
        return d
    
    def to_yaml_file(self, yaml_path: str):
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(self.to_dict(), yaml_file, default_flow_style=False)
    
    @classmethod
    def load_from_yaml(cls: T, yaml_path: str) -> T:
        with open(yaml_path, 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
        for field, field_type in cls._nested_config_fields.items():
            if field in config_dict:
                config_dict[field] = field_type.load_from_dict(config_dict[field])
        return cls(**config_dict)

    @classmethod
    def load_from_dict(cls: T, config_dict: Dict[str, Any]) -> T:
        # Process nested configuration objects
        for field, field_type in cls._nested_config_fields.items():
            if field in config_dict:
                config_dict[field] = field_type.load_from_dict(config_dict[field])
        return cls(**config_dict)