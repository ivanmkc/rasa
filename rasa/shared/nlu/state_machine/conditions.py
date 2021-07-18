from typing import Any, List, Optional

# from rasa.shared.core.trackers import "DialogueStateTracker"
import abc
from typing import Union
from rasa.shared.nlu.state_machine.state_machine_models import (
    Action,
    Intent,
    Slot,
)
from rasa.shared.nlu.state_machine.condition import Condition
from rasa.shared.core.events import ActionExecuted, StateMachineLifecycle


class OnEntryCondition(Condition):
    def is_valid(self, tracker: "DialogueStateTracker"):
        return tracker.state_machine_lifecycle == StateMachineLifecycle.ENTRY


class IntentCondition(Condition):
    intent: Intent

    ignores_previous_tracker = True

    def __init__(self, intent: Intent):
        self.intent = intent

    def is_valid(self, tracker: "DialogueStateTracker"):
        last_intent_name = tracker.latest_message.intent.get("name")
        return self.intent.name == last_intent_name


class ActionCondition(Condition):
    action: Action

    ignores_previous_tracker = True

    def __init__(self, action: Action):
        self.action = action

    def is_valid(self, tracker: "DialogueStateTracker"):
        last_action_name: Optional[str] = None
        for event in reversed(tracker.events):
            if isinstance(event, ActionExecuted):
                last_action_name = event.action_name
                break

        return self.action.name == last_action_name


class SlotsFilledCondition(Condition):
    slots: List[Slot]

    def __init__(self, slots: [Slot]):
        self.slots = slots

    def is_valid(self, tracker: "DialogueStateTracker"):
        return all([tracker.slots.get(slot.name).value for slot in self.slots])


class SlotEqualsCondition(Condition):
    slot: Union[Slot, str]
    value: Any

    def __init__(self, slot: Union[Slot, str], value: Any):
        self.slot = slot
        self.value = value

    def is_valid(self, tracker: "DialogueStateTracker"):
        slot_name = self.slot.name if isinstance(self.slot, Slot) else self.slot

        tracker_slot = tracker.slots.get(slot_name)

        if tracker_slot:
            return tracker_slot.value == self.value
        else:
            raise RuntimeError("Required slot not found in tracker")


class ConditionWithConditions(abc.ABC):
    @abc.abstractproperty
    def conditions(self) -> List[Condition]:
        pass

    @property
    def intents(self) -> List[Intent]:
        all_intents: List[Intent] = []
        for condition in self.conditions:
            if isinstance(condition, ConditionWithConditions):
                all_intents += condition.intents
            elif isinstance(condition, IntentCondition):
                all_intents.append(condition.intent)

        return all_intents


class AndCondition(Condition, ConditionWithConditions):
    @property
    def conditions(self) -> List[Condition]:
        return self._conditions

    def __init__(self, conditions: List[Condition]):
        self._conditions = conditions

    def is_valid(self, tracker: "DialogueStateTracker"):
        return all([condition.is_valid(tracker) for condition in self.conditions])


class OrCondition(Condition, ConditionWithConditions):
    @property
    def conditions(self) -> List[Condition]:
        return self._conditions

    def __init__(self, conditions: List[Condition]):
        self._conditions = conditions

    def is_valid(self, tracker: "DialogueStateTracker"):
        return any([condition.is_valid(tracker) for condition in self.conditions])
