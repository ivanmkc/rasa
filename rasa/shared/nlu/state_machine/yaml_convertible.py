import abc
from typing import Dict


class YAMLConvertable(abc.ABC):
    @abc.abstractmethod
    def as_nlu_yaml(self):
        "Save to YAML"
        pass


class StoryYAMLConvertable(abc.ABC):
    @abc.abstractmethod
    def as_story_yaml(self) -> Dict:
        "Save to YAML"
        pass
