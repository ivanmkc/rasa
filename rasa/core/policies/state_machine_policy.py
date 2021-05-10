import logging
from typing import Any, List, Dict, Text, Optional, Set, Tuple, TYPE_CHECKING

from rasa.shared.constants import DOCS_URL_RULES
from rasa.shared.core.constants import STATE_MACHINE_ACTION_NAME
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.shared.core.events import (
    LoopInterrupted,
    SlotSet,
    StateMachineQueueActions,
    StateMachineSetLifecycle,
    StateMachineLifecycle,
)
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData, PolicyPrediction
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    get_active_loop_name,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import (
    DEFAULT_CORE_FALLBACK_THRESHOLD,
    RULE_POLICY_PRIORITY,
)
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.nlu.constants import ACTION_NAME, INTENT_NAME_KEY
import rasa.core.test
import rasa.core.training.training

from rasa.shared.nlu.state_machine.state_machine_state import (
    Response,
    StateMachineState,
)
from rasa.shared.nlu.state_machine.state_machine_models import (
    Intent,
    Slot,
    Utterance,
)
from rasa.core.actions.state_machine_action import StateMachineAction
from rasa.core.actions.action import Action

if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble

logger = logging.getLogger(__name__)

# These are Rasa Open Source default actions and overrule everything at any time.
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,
    USER_INTENT_BACK: ACTION_BACK_NAME,
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,
}

RULES = "rules"
RULES_FOR_LOOP_UNHAPPY_PATH = "rules_for_loop_unhappy_path"
RULES_NOT_IN_STORIES = "rules_not_in_stories"

LOOP_WAS_INTERRUPTED = "loop_was_interrupted"
DO_NOT_PREDICT_LOOP_ACTION = "do_not_predict_loop_action"

DEFAULT_RULES = "predicting default action with intent "
LOOP_RULES = "handling active loops and forms - "
LOOP_RULES_SEPARATOR = " - "


class InvalidRule(RasaException):
    """Exception that can be raised when rules are not valid."""

    def __init__(self, message: Text) -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> Text:
        return self.message + (
            f"\nYou can find more information about the usage of "
            f"rules at {DOCS_URL_RULES}. "
        )


class StateMachinePolicy(MemoizationPolicy):
    """Policy which handles all the rules"""

    # rules use explicit json strings
    ENABLE_FEATURE_STRING_COMPRESSION = False

    # number of user inputs that is allowed in case rules are restricted
    ALLOWED_NUMBER_OF_USER_INPUTS = 1

    def _metadata(self) -> Dict[Text, Any]:
        return {
            "priority": self.priority,
            "lookup": self.lookup,
            "core_fallback_threshold": self._core_fallback_threshold,
            "core_fallback_action_name": self._fallback_action_name,
            "enable_fallback_prediction": self._enable_fallback_prediction,
        }

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "rule_policy.json"

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        Returns:
            The data type supported by this policy (ML and rule data).
        """
        return SupportedData.ML_AND_RULE_DATA

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = RULE_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
        core_fallback_threshold: float = DEFAULT_CORE_FALLBACK_THRESHOLD,
        core_fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
        enable_fallback_prediction: bool = True,
        restrict_rules: bool = True,
        check_for_contradictions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a `StateMachinePolicy` object.

        Args:
            featurizer: `Featurizer` which is used to convert conversation states to
                features.
            priority: Priority of the policy which is used if multiple policies predict
                actions with the same confidence.
            lookup: Lookup table which is used to pick matching rules for a conversation
                state.
            core_fallback_threshold: Confidence of the prediction if no rule matched
                and de-facto threshold for a core fallback.
            core_fallback_action_name: Name of the action which should be predicted
                if no rule matched.
            enable_fallback_prediction: If `True` `core_fallback_action_name` is
                predicted in case no rule matched.
            restrict_rules: If `True` rules are restricted to contain a maximum of 1
                user message. This is used to avoid that users build a state machine
                using the rules.
            check_for_contradictions: Check for contradictions.
        """
        self._core_fallback_threshold = core_fallback_threshold
        self._fallback_action_name = core_fallback_action_name
        self._enable_fallback_prediction = enable_fallback_prediction
        self._restrict_rules = restrict_rules
        self._check_for_contradictions = check_for_contradictions

        self._rules_sources = None

        # max history is set to `None` in order to capture any lengths of rule stories
        super().__init__(
            featurizer=featurizer,
            priority=priority,
            max_history=None,
            lookup=lookup,
            **kwargs,
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers.

        Args:
            training_trackers: The list of the trackers.
            domain: The domain.
            interpreter: Interpreter which can be used by the polices for featurization.
        """
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> "PolicyPrediction":
        """Predicts the next action (see parent class for more information)."""

        # If the lifecycle is None, set to ENTRY
        if tracker.state_machine_lifecycle is None:
            tracker.update_with_events(
                new_events=[
                    StateMachineSetLifecycle(StateMachineLifecycle.ENTRY)
                ],
                domain=domain,
            )

        # Queue actions if not already queued
        if tracker.queued_state_actions is None:
            action_names = StateMachinePolicy._get_actions_names(
                tracker=tracker, domain=domain
            )

            tracker.update_with_events(
                [StateMachineQueueActions(action_names)], domain
            )

            tracker.update_with_events(
                new_events=[
                    StateMachineSetLifecycle(StateMachineLifecycle.IN_STATE)
                ],
                domain=domain,
            )

        # Check if there are any queued actions
        queued_action_name = None
        if len(tracker.queued_state_actions) > 0:
            queued_action_name = tracker.queued_state_actions.pop(0)
        else:
            # All queued actions were run, set queue to None
            tracker.update_with_events(
                [StateMachineQueueActions(None)], domain
            )

        return self._rule_prediction(
            self._prediction_result(queued_action_name, tracker, domain),
            None,
            returning_from_unhappy_path=False,
            is_end_to_end_prediction=False,
        )

    @staticmethod
    def _get_valid_slots(
        slots: List[Slot],
        tracker: DialogueStateTracker,
        only_empty_slots: bool = False,
    ) -> List[Slot]:
        valid_slots = slots

        # Get only unfilled slots
        if only_empty_slots:
            valid_slots = [
                slot
                for slot in valid_slots
                if tracker.get_slot(slot.name) == None
            ]

        # Get slots that pass conditions
        valid_slots = [
            slot
            for slot in valid_slots
            if not slot.condition
            or (slot.condition and slot.condition.is_valid(tracker))
        ]

        return valid_slots

    @staticmethod
    def _get_slot_values(
        slots: List[Slot], tracker: "DialogueStateTracker"
    ) -> Dict[str, Any]:
        slot_values = {}

        # Get valid slots from latest utterance
        valid_slots = StateMachinePolicy._get_valid_slots(
            slots=slots, tracker=tracker, only_empty_slots=False
        )

        for slot in valid_slots:
            last_bot_uttered_action_name = (
                tracker.latest_bot_utterance.metadata.get("utter_action")
            )

            if slot.only_fill_when_prompted:
                if not last_bot_uttered_action_name or (
                    last_bot_uttered_action_name
                    not in [action.name for action in slot.prompt_actions]
                ):
                    continue

            # Extract values using entities
            values_for_slots: List[Any] = [
                StateMachinePolicy.get_entity_value(
                    entity, tracker, None, None
                )
                for entity in slot.entities
            ]

            # Extract values using intents
            last_intent_name = tracker.latest_message.intent.get("name")
            if last_intent_name:
                for slot_intent, value in slot.intents.items():
                    if isinstance(slot_intent, Intent):
                        if slot_intent.name == last_intent_name:
                            values_for_slots.append(value)
                    elif isinstance(slot_intent, str):
                        if slot_intent == last_intent_name:
                            values_for_slots.append(value)

            # Filter out None's
            values_for_slots = [
                value for value in values_for_slots if value is not None
            ]

            if len(values_for_slots) > 0:
                # Take first entity extracted
                slot_values.update({slot.name: values_for_slots[0]})

        return slot_values

    @staticmethod
    def _get_response_actions(
        responses: List[Response], tracker: "DialogueStateTracker"
    ) -> List[Action]:
        valid_responses = [
            response
            for response in responses
            if response.condition.is_valid(tracker=tracker)
        ]

        ## Choose a random action
        # valid_responses_action_names = [
        #     valid_response.actions[
        #         random.randint(0, len(valid_response.actions) - 1)
        #     ].name
        #     for valid_response in valid_responses
        #     if len(valid_response.actions) > 0
        # ]

        valid_responses_actions = [
            action
            for valid_response in valid_responses
            for action in valid_response.actions
        ]

        return valid_responses_actions

    @staticmethod
    def _get_next_slot_actions(
        slots: List[Slot],
        tracker: "DialogueStateTracker",
    ) -> List[Action]:
        # Get non-filled slots
        empty_slots = StateMachinePolicy._get_valid_slots(
            slots=slots, tracker=tracker, only_empty_slots=True
        )

        if len(empty_slots) > 0:
            empty_slot = empty_slots[0]
            if len(empty_slot.prompt_actions) == 0:
                raise []

            return empty_slot.prompt_actions

        return []

    @staticmethod
    def _temporary_tracker(
        current_tracker: DialogueStateTracker,
        additional_events: List[Event],
        domain: Domain,
    ) -> DialogueStateTracker:
        tracker = current_tracker.copy()
        tracker.update_with_events(new_events=additional_events, domain=domain)
        return tracker

    @staticmethod
    def _get_actions_names(
        tracker: DialogueStateTracker, domain: Domain
    ) -> List[Action]:
        # Get current state info
        state_machine_state: StateMachineState = (
            domain.active_state_machine_state
        )

        if not state_machine_state:
            # Return no prediction
            return []

        # If there are slots to fill, predict slot fill
        # Check if there are slots to fill
        slot_values = StateMachinePolicy._get_slot_values(
            slots=state_machine_state.slots, tracker=tracker
        )

        # Set slots
        slot_set_events = [
            SlotSet(key=slot_name, value=slot_value)
            for slot_name, slot_value in slot_values.items()
        ]

        # Perhaps move to action
        tracker.update_with_events(new_events=slot_set_events, domain=domain)

        # Create temporary tracker with the validation events applied
        # Otherwise, the slots will not be set
        temp_tracker = StateMachinePolicy._temporary_tracker(
            tracker, slot_set_events, domain
        )

        # Find valid utterances
        # TODO: Handle slots that are filled but not uttered.
        slot_filled_utterance: Optional[Utterance] = None

        if len(slot_values) > 0:
            number_of_slots_in_utterance = 0
            for utterance in state_machine_state.slot_fill_utterances:
                uttered_slot_names = [
                    slot_name
                    for slot_name in slot_values.keys()
                    if f"{{{slot_name}}}" in utterance.text
                ]

                if len(uttered_slot_names) > number_of_slots_in_utterance:
                    slot_filled_utterance = utterance
                    number_of_slots_in_utterance = len(uttered_slot_names)

        # Add slot_filled_action
        slot_filled_actions: List[str] = (
            [slot_filled_utterance] if slot_filled_utterance else []
        )

        # Check if any response conditions are met
        response_actions = StateMachinePolicy._get_response_actions(
            responses=state_machine_state.responses, tracker=temp_tracker
        )

        # Ask for next slot
        next_slot_actions = StateMachinePolicy._get_next_slot_actions(
            state_machine_state.slots, tracker=temp_tracker
        )

        # TODO: Handle transitions

        action_names: List[str] = [
            action.name
            for action in slot_filled_actions
            + response_actions
            + next_slot_actions
        ]

        return action_names

    def _rule_prediction(
        self,
        probabilities: List[float],
        prediction_source: Text,
        returning_from_unhappy_path: bool = False,
        is_end_to_end_prediction: bool = False,
        is_no_user_prediction: bool = False,
    ) -> PolicyPrediction:
        return PolicyPrediction(
            probabilities,
            self.__class__.__name__,
            self.priority,
            events=[LoopInterrupted(True)]
            if returning_from_unhappy_path
            else [],
            is_end_to_end_prediction=is_end_to_end_prediction,
            is_no_user_prediction=is_no_user_prediction,
            hide_rule_turn=False,
        )

    def _default_predictions(self, domain: Domain) -> List[float]:
        result = super()._default_predictions(domain)

        if self._enable_fallback_prediction:
            result[
                domain.index_for_action(self._fallback_action_name)
            ] = self._core_fallback_threshold
        return result
