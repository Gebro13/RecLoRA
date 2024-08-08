import copy
import json
import logging
import os
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from transformers.utils import cached_property

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, save_directory):
        assert os.path.isdir(save_directory), f"not a directory: {save_directory}"
        output_config_file = os.path.join(save_directory, "config.json")
        self.to_json_file(output_config_file)

    @classmethod
    def load(cls, load_directory):
        output_config_file = os.path.join(load_directory, "config.json")
        config_dict = cls.from_json_file(output_config_file)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        config = cls(**config_dict)
        return config

    @classmethod
    def from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
