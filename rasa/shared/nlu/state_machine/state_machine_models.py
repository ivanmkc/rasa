import abc
from typing import Any, Dict, Optional, List, Union
from rasa.shared.nlu.state_machine.yaml_convertible import YAMLConvertable
from rasa.shared.nlu.state_machine.condition import Condition
import rasa.shared.core.slots as rasa_slots


class Intent(YAMLConvertable):
    def __init__(
        self, examples: Union[str, List[str]] = [], name: Optional[str] = None
    ):
        if isinstance(examples, str):
            examples = [examples]

        if len(examples) == 0:
            raise ValueError("No examples provided.")

        self.examples = examples

        if name:
            self.name = name
        else:
            text_stripped = "".join(
                e.lower() for e in examples[0] if e.isalnum() or e.isspace()
            )
            self.name = "_".join(text_stripped.split(" "))

    def as_yaml(self) -> Dict[str, Any]:
        return {
            "intent": self.name,
            "examples": "\n".join(
                [f"- {example}" for example in self.examples]
            ),
        }


class Action(abc.ABC):
    @abc.abstractproperty
    def name(self) -> str:
        pass


# class ResetSlotsAction(Action):

#     name: str = "ResetSlotsAction"
#     slots: List[Slot]

#     def __init__(self, slots: List[Slot], name: str):
#         self.slots = slots
#         self.name = name


class Utterance(Action):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, text: str, name: Optional[str] = None):
        self.text = text

        if name:
            self._name = name
        else:
            text_stripped = "".join(
                e.lower() for e in text if e.isalnum() or e.isspace()
            )
            self._name = "utter_" + "_".join(text_stripped.split(" "))


# class Entity(abc.ABC):
#     @property
#     def name(self) -> str:
#         pass


# enum SpacyEntity: Entity {
#     case GPE
#     case PERSON
#     case LOC

#     var name: str {
#         switch (self) {
#         case .GPE:
#             return "GPE"
#         case .PERSON:
#             return "PERSON"
#         case .LOC:
#             return "LOC"
#         }
#     }
# }


class Slot(abc.ABC):
    def __init__(
        self,
        name: str,
        condition: Optional[Condition] = None,
        only_fill_when_prompted: bool = False,
        entities: List[str] = [],
        intents: Dict[Intent, Any] = {},
        prompt_actions: List[Action] = [],
    ):
        self.name = name
        self.condition = condition
        self.entities = entities
        self.intents = intents
        self.only_fill_when_prompted = only_fill_when_prompted
        self.prompt_actions = prompt_actions

    @abc.abstractmethod
    def as_rasa_slot(self) -> rasa_slots.Slot:
        pass


class TextSlot(Slot):
    def as_rasa_slot(self) -> rasa_slots.Slot:
        return rasa_slots.TextSlot(name=self.name)


class BooleanSlot(Slot):
    def as_rasa_slot(self) -> rasa_slots.Slot:
        return rasa_slots.BooleanSlot(name=self.name)
