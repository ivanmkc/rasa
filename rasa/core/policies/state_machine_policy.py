import logging
from typing import Any, List, Dict, Text, Optional, Set, Tuple, TYPE_CHECKING

from rasa.shared.constants import DOCS_URL_RULES
from rasa.shared.exceptions import RasaException
from rasa.shared.core.events import (
    LoopInterrupted,
    SlotSet,
    StateMachineQueueActions,
    StateMachineSetLifecycle,
    StateMachineLifecycle,
    StateMachineTransition,
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
from rasa.core.constants import RULE_POLICY_PRIORITY
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, ActionExecuted

import rasa.core.test
import rasa.core.training.training

from rasa.shared.nlu.state_machine.state_machine_state import (
    Response,
    StateMachineState,
    Transition,
)
from rasa.shared.nlu.state_machine.state_machine_models import (
    Intent,
    Slot,
    Utterance,
)
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
            "intent_threshold_for_slot_fill": self._intent_threshold_for_slot_fill,
            "next_slot_action_confidence": self._next_slot_action_confidence,
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
        intent_threshold_for_slot_fill: float = 0,
        next_slot_action_confidence: float = 0.8,
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
            intent_threshold_for_slot_fill: Min confidence needed for slot fill. Defaults to 0, meaning any matched intent is accepted.
            next_slot_action_confidence: Confidence of the prediction for the asking for the next slot.
                This means that if a next slot is requested, another policy has to have a higher confidence to override the request.
            enable_fallback_prediction: If `True` `core_fallback_action_name` is
                predicted in case no rule matched.
            restrict_rules: If `True` rules are restricted to contain a maximum of 1
                user message. This is used to avoid that users build a state machine
                using the rules.
            check_for_contradictions: Check for contradictions.
        """
        self._intent_threshold_for_slot_fill = intent_threshold_for_slot_fill
        self._next_slot_action_confidence = next_slot_action_confidence
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

    @staticmethod
    def was_interrupted_by_other_policy(
        tracker: DialogueStateTracker,
    ) -> bool:
        for event in reversed(tracker.events):
            # Only look for ActionExecuted events
            if isinstance(event, ActionExecuted):
                if event.policy:
                    if isinstance(event.policy, str) and event.policy.endswith(
                        "StateMachinePolicy"
                    ):
                        return False
                    else:
                        return True
                else:
                    return False

        # If no ActionExecuted was detected, then return False
        return False

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> "PolicyPrediction":
        """Predicts the next action (see parent class for more information)."""

        initial_tracker = tracker.copy()

        # Set initial state name if it exists in the domain
        target_state_machine_state_name = domain.initial_state_machine_state_name

        if (
            not tracker.active_state_machine_state_name
            and target_state_machine_state_name
        ):
            tracker.update_with_events(
                [
                    StateMachineTransition(target_state_machine_state_name),
                ],
                domain,
            )

            logger.debug(
                f"StateMachinePolicy found transition to: {target_state_machine_state_name}"
            )

        state_machine_state = tracker.get_active_state_machine_state(domain)

        # Exit if no state found
        if not state_machine_state:
            logger.debug(
                f"StateMachinePolicy found no state, deferring to other policies."
            )

            return self._rule_prediction(
                self._prediction_result(None, tracker, domain),
                None,
                returning_from_unhappy_path=False,
                is_end_to_end_prediction=False,
            )

        # Queue actions if not already queued
        was_interrupted_by_other_policy = (
            StateMachinePolicy.was_interrupted_by_other_policy(tracker)
        )
        if (
            tracker.queued_state_action_probabilities is None
            or was_interrupted_by_other_policy
        ):

            if was_interrupted_by_other_policy:
                logger.debug(
                    f"StateMachinePolicy was interrupted by other policy, looking for applicable actions."
                )
            else:
                logger.debug(
                    f"StateMachinePolicy found no queue, looking for applicable actions."
                )

            actions = StateMachinePolicy.get_actions(
                state_machine_state=state_machine_state,
                prev_tracker=initial_tracker,
                tracker=tracker,
                domain=domain,
                intent_threshold_for_slot_fill=self._intent_threshold_for_slot_fill,
            )

            # Update state if changed
            state_machine_state = tracker.get_active_state_machine_state(domain)
            # Get next slot actions
            next_slot_actions: List[Action] = StateMachinePolicy._get_next_slot_actions(
                state_machine_state.slots,
                tracker=tracker,
            )

            actions_to_queue = [(action.name, 1.0) for action in actions] + [
                (
                    next_slot_action.name,
                    self._next_slot_action_confidence,
                )
                for next_slot_action in next_slot_actions
            ]

            if len(actions_to_queue) > 0:
                actions_to_queue += [(ACTION_LISTEN_NAME, 1.0)]

            tracker.update_with_events(
                [
                    StateMachineQueueActions(actions_to_queue),
                ],
                domain,
            )

            logger.debug(
                f"StateMachinePolicy set queue to: {tracker.queued_state_action_probabilities}."
            )

        # Check if there are any queued actions
        queued_action_probability = None
        if len(tracker.queued_state_action_probabilities) > 0:
            queued_state_action_probabilities = (
                tracker.queued_state_action_probabilities.copy()
            )
            queued_action_probability = queued_state_action_probabilities.pop(0)

            tracker.update_with_events(
                [
                    StateMachineQueueActions(queued_state_action_probabilities),
                ],
                domain,
            )

            logger.debug(
                f"StateMachinePolicy found queued action: {queued_action_probability}"
            )

        # If there are no more actions, queue is finished. Set to None
        if len(tracker.queued_state_action_probabilities) == 0:
            tracker.update_with_events(
                [
                    StateMachineQueueActions(None),
                ],
                domain,
            )
            logger.debug(
                f"StateMachinePolicy set queue to None due to no more queued actions."
            )

        if queued_action_probability:
            # Predict action
            logger.debug(
                f"StateMachinePolicy predicted '{queued_action_probability}' by queued_action_probability."
            )

            queued_action_name = queued_action_probability[0]
            queued_action_probability_only = queued_action_probability[1]

            predictions = self._prediction_result(None, tracker, domain)

            # if self._enable_fallback_prediction:
            predictions[
                domain.index_for_action(queued_action_name)
            ] = queued_action_probability_only

            logger.debug(
                f"StateMachinePolicy says remaining tracker.queued_state_action_probabilities is {tracker.queued_state_action_probabilities}."
            )

            return self._rule_prediction(
                predictions,
                None,
                returning_from_unhappy_path=False,
                is_end_to_end_prediction=False,
            )
        else:
            logger.debug(f"StateMachinePolicy deferring to next policy.")

            logger.debug(
                f"StateMachinePolicy says tracker.queued_state_action_probabilities is {tracker.queued_state_action_probabilities}."
            )

            # Defer to next policy
            return self._rule_prediction(
                self._prediction_result(None, tracker, domain),
                None,
                returning_from_unhappy_path=False,
                is_end_to_end_prediction=False,
            )

    @staticmethod
    def _get_valid_transition(
        transitions: List[Transition],
        prev_tracker: DialogueStateTracker,
        tracker: DialogueStateTracker,
    ) -> Optional[Transition]:
        # Check for transitions
        valid_transitions = [
            transition
            for transition in transitions
            if transition.condition.is_valid_and_changed(
                prev_tracker=prev_tracker, tracker=tracker
            )
        ]

        if len(valid_transitions) > 0:
            return valid_transitions[0]
        else:
            return None

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
                slot for slot in valid_slots if tracker.get_slot(slot.name) == None
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
    def get_entity_value(
        name: Text,
        tracker: DialogueStateTracker,
        role: Optional[Text] = None,
        group: Optional[Text] = None,
    ) -> Any:
        """Extract entities for given name and optional role and group.

        Args:
            name: entity type (name) of interest
            tracker: the tracker
            role: optional entity role of interest
            group: optional entity group of interest

        Returns:
            Value of entity.
        """
        # list is used to cover the case of list slot type
        value = list(
            tracker.get_latest_entity_values(name, entity_group=group, entity_role=role)
        )
        if len(value) == 0:
            value = None
        elif len(value) == 1:
            value = value[0]
        return value

    @staticmethod
    def _get_slot_values(
        slots: List[Slot],
        tracker: DialogueStateTracker,
        intent_threshold_for_slot_fill: float,
    ) -> Dict[str, Any]:
        slot_values = {}

        # Get valid slots from latest utterance
        valid_slots = StateMachinePolicy._get_valid_slots(
            slots=slots,
            tracker=tracker,
            only_empty_slots=False,
        )

        last_bot_uttered_action_name = tracker.latest_bot_utterance.metadata.get(
            "utter_action"
        )

        for slot in valid_slots:
            if slot.only_fill_when_prompted:
                if not last_bot_uttered_action_name or (
                    last_bot_uttered_action_name
                    not in [action.name for action in slot.prompt_actions]
                ):
                    continue

            # Extract values using entities
            values_for_slots: List[Any] = [
                StateMachinePolicy.get_entity_value(entity, tracker, None, None)
                for entity in slot.entities
            ]

            # Extract values using intents
            last_intent_name = tracker.latest_message.intent.get("name")
            last_intent_confidence = tracker.latest_message.intent.get("confidence")
            if (
                last_intent_name
                and last_intent_confidence > intent_threshold_for_slot_fill
            ):
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
        responses: List[Response],
        prev_tracker: DialogueStateTracker,
        tracker: DialogueStateTracker,
    ) -> List[Action]:
        valid_responses = [
            response
            for response in responses
            if response.condition.is_valid_and_changed(
                prev_tracker=prev_tracker, tracker=tracker
            )
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
        tracker: DialogueStateTracker,
    ) -> List[Action]:
        # Get non-filled slots
        empty_slots = StateMachinePolicy._get_valid_slots(
            slots=slots, tracker=tracker, only_empty_slots=True
        )

        if len(empty_slots) > 0:
            empty_slot = empty_slots[0]
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
    def get_actions(
        state_machine_state: StateMachineState,
        prev_tracker: DialogueStateTracker,
        tracker: DialogueStateTracker,
        domain: Domain,
        intent_threshold_for_slot_fill: float,
        triggered_transition_state_names: Set[str] = set(),
    ) -> List[Action]:
        # If there are slots to fill, predict slot fill
        # Check if there are slots to fill
        slot_values = StateMachinePolicy._get_slot_values(
            slots=state_machine_state.slots,
            tracker=tracker,
            intent_threshold_for_slot_fill=intent_threshold_for_slot_fill,
        )

        if len(slot_values) > 0:
            logger.debug(f"StateMachinePolicy found slot_values = '{slot_values}'")

        # Set slots
        slot_set_events = [
            SlotSet(key=slot_name, value=slot_value)
            for slot_name, slot_value in slot_values.items()
        ]

        # # Create temporary tracker with the validation events applied
        # # Otherwise, the slots will not be set
        # temp_tracker = StateMachinePolicy._temporary_tracker(
        #     tracker, slot_set_events, domain
        # )
        tracker.update_with_events(new_events=slot_set_events, domain=domain)

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
        slot_filled_actions: List[Action] = (
            [slot_filled_utterance] if slot_filled_utterance else []
        )

        if len(slot_filled_actions) > 0:
            logger.debug(
                f"StateMachinePolicy found slot_filled_actions = '{slot_filled_actions}'"
            )

        # Check if any response conditions are met
        response_actions = StateMachinePolicy._get_response_actions(
            responses=state_machine_state.responses,
            prev_tracker=prev_tracker,
            tracker=tracker,
        )

        if len(response_actions) > 0:
            logger.debug(
                f"StateMachinePolicy found response_actions = '{[action.name for action in response_actions]}'"
            )

        tracker.update_with_events(
            new_events=[StateMachineSetLifecycle(StateMachineLifecycle.IN_STATE)],
            domain=domain,
        )

        # Check transitions
        valid_transition = StateMachinePolicy._get_valid_transition(
            transitions=[
                transition
                for transition in state_machine_state.transitions
                if transition.destination_state_name
                not in triggered_transition_state_names  # Only use transitions that weren't used before
            ],
            prev_tracker=prev_tracker,
            tracker=tracker,
        )

        transition_actions: List[str] = []
        if valid_transition:
            # Add exit utterances
            transition_actions += valid_transition.transition_utterances

            # Default to the last transitioned state name
            destination_state = domain.get_state_machine_state(
                valid_transition.destination_state_name
                or tracker.last_transitioned_state_name
            )

            # Prevent transition to same state
            if destination_state.name != state_machine_state.name:
                # Transition to state and get results
                tracker.update_with_events(
                    new_events=[StateMachineTransition(destination_state.name)],
                    domain=domain,
                )
                transition_actions += StateMachinePolicy.get_actions(
                    state_machine_state=destination_state,
                    prev_tracker=prev_tracker,
                    tracker=tracker,
                    domain=domain,
                    intent_threshold_for_slot_fill=intent_threshold_for_slot_fill,
                    triggered_transition_state_names=triggered_transition_state_names.union(
                        set([valid_transition.destination_state_name])
                    ),
                )

        return slot_filled_actions + response_actions + transition_actions

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
            events=[LoopInterrupted(True)] if returning_from_unhappy_path else [],
            is_end_to_end_prediction=is_end_to_end_prediction,
            is_no_user_prediction=is_no_user_prediction,
            hide_rule_turn=False,
        )
