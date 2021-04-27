import rasa

from rasa.shared.nlu.state_machine.state_machine_state import (
    Response,
    StateMachineState,
    Transition,
)

from rasa.shared.nlu.state_machine.state_machine_models import TextSlot
from rasa.shared.nlu.state_machine.conditions import (
    IntentCondition,
    OnEntryCondition,
    SlotEqualsCondition,
    SlotsFilledCondition,
)

from rasa.shared.nlu.state_machine.state_machine_models import (
    Intent,
    ActionName,
    Utterance,
    TextSlot,
    FloatSlot,
    BooleanSlot,
)


from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain, StateMachineProvider
from rasa.shared.core.events import BotUttered, StateMachineTransition, UserUttered
from rasa.shared.core.slots import Slot as RasaSlot
from rasa.core.policies.state_machine_policy import StateMachinePolicy

from typing import Dict, List, Optional
import yaml


# class MockStateMachineProvider(StateMachineProvider):
#     def __init__(self, initial_state_name: str, states: List[StateMachineState]):
#         self._initial_state_name = initial_state_name
#         self._state_dict = {state.name: state for state in states}

#     def initial_state_machine_state_name(self) -> Optional[str]:
#         return self._initial_state_name

#     def get_state_machine_state(self, state_name: str) -> Optional[StateMachineState]:
#         return self._state_dict.get(state_name)


SLOT_NAME = "slot_confirm"

intent_hello = Intent(examples=["hello"], name="intent_hello")
intent_goodbye = Intent(examples=["goodbye"], name="intent_goodbye")


def get_start_state() -> StateMachineState:
    slot_test = TextSlot(
        name=SLOT_NAME,
        intents={"intent_affirm": "yes", "deny": "no"},
        prompt_actions=[Utterance("Confirm this", name="utter_confirm")],
    )

    state = StateMachineState(
        "start_state",
        slots=[slot_test],
        slot_fill_utterances=[
            Utterance(f"The slot is {{{SLOT_NAME}}}", name="utter_slot_value"),
            Utterance(f"The slot is filled"),
        ],
        transitions=[
            Transition(
                IntentCondition(intent_goodbye),
                transition_utterances=[Utterance("Going", name="utter_going")],
                destination_state_name="destination_state",
            )
        ],
        responses=[
            Response(
                condition=OnEntryCondition(),
                actions=[Utterance("how are you?", name="utter_how_are_you")],
                repeatable=False,
            ),
            Response(
                condition=IntentCondition(intent=intent_hello),
                actions=[Utterance("Hello!", name="utter_hello")],
            ),
        ],
    )

    return state


def get_destination_state() -> StateMachineState:
    intent_back = Intent(examples=["back"], name="intent_back")
    state = StateMachineState(
        "destination_state",
        slots=[],
        slot_fill_utterances=[],
        transitions=[
            Transition(
                IntentCondition(intent_back),
                transition_utterances=[
                    Utterance("Going back", name="utter_going_back")
                ],
                destination_state_name=None,
            )
        ],
        responses=[
            Response(
                condition=OnEntryCondition(),
                actions=[Utterance("welcome", name="utter_welcome")],
                repeatable=False,
            )
        ],
    )

    return state


intent_book_tour = Intent(
    examples=[
        "I’d like to book a tour",
        "I want to book a tour.",
        "Can I ask you about tours?",
        "Can I book a tour?",
    ],
    name="intent_book_tour",
)


def get_visitor_center_state() -> StateMachineState:
    return StateMachineState(
        name="start_state",
        slots=[],
        slot_fill_utterances=[],
        transitions=[
            Transition(
                condition=IntentCondition(intent_book_tour),
                transition_utterances=[
                    Utterance(
                        text="Sure, I can help you with that.",
                        name="utter_i_can_help_you",
                    )
                ],
                destination_state_name="book_tour",
            ),
        ],
        responses=[
            Response(
                condition=OnEntryCondition(),
                actions=[
                    Utterance(
                        "Welcome to the Visitor Center! How can I help you?",
                        name="utter_welcome_to_the_visitor_center",
                    ),
                ],
            ),
        ],
    )


intent_select_boat_tour = Intent(
    examples=[
        "The boat tour",
        "boat",
        "I would prefer the boat one",
        "The 3pm",
        "The one at 3 o clock",
        "The tour at 3",
        "3 sounds good",
    ],
    name="intent_select_boat_tour",
)

intent_select_bus_tour = Intent(
    examples=[
        "The bus tour",
        "bus",
        "I would prefer the bus one",
        "The 4pm",
        "The one at 4 o clock",
        "The tour at 4",
        "4 sounds good",
        "Four",
    ],
    name="intent_select_bus_tour",
)

intent_select_three = Intent(
    examples=["3", "three", "a few"], name="intent_select_three"
)

utterance_tour_booked = Utterance(
    "Great. Booked you for the {tour_type} tour.", name="utter_tour_booked"
)
utterance_tour_num_tickets = Utterance(
    "Okay, {tour_num_tickets} tickets", name="utter_tour_num_tickets"
)


def get_book_tour_state() -> StateMachineState:
    slot_tour = TextSlot(
        name="tour_type",
        intents={intent_select_bus_tour: "bus", intent_select_boat_tour: "boat",},
        prompt_actions=[
            Utterance(
                "We have a boat tour of the city at 3:00pm. We also have a bus tour of the city at 4:00pm. Which one would you prefer?",
                name="utter_tour_info",
            )
        ],
    )

    slot_number_tickets = FloatSlot(
        name="tour_num_tickets",
        entities=[],  # TODO: Test this
        intents={intent_select_three: 3},
        prompt_actions=[
            Utterance(
                "How many tickets do you need for the {tour_type} tour?",
                name="utter_ask_tour_num_tickets",
            )
        ],
        only_fill_when_prompted=True,
    )

    slot_tour_confirmed = BooleanSlot(
        name="tour_confirmed",
        intents={"intent_affirm": True, "intent_deny": False,},
        prompt_actions=[
            Utterance(
                "Okay, just to confirm. I've booked you for the {tour_type} tour for {tour_num_tickets} persons. Is that correct?",
                name="utter_ask_tour_confirmation",
            ),
        ],
        only_fill_when_prompted=True,
    )

    slots = [
        slot_tour,
        slot_number_tickets,
        slot_tour_confirmed,
    ]

    return StateMachineState(
        name="book_tour",
        slots=slots,
        slot_fill_utterances=[utterance_tour_booked, utterance_tour_num_tickets,],
        transitions=[
            Transition(
                condition=SlotsFilledCondition(slots),
                transition_utterances=[],
                destination_state_name=None,
            ),
        ],
        responses=[
            Response(
                condition=OnEntryCondition(),
                actions=[
                    Utterance("Let's book your tour.", name="utter_lets_book_tour"),
                ],
            ),
            Response(
                condition=SlotEqualsCondition(slot=slot_tour_confirmed, value=False),
                actions=[
                    Utterance("No? Okay, what would you like then?"),
                    ActionName("action_reset_tour_slots"),
                ],
            ),
            Response(
                condition=SlotEqualsCondition(slot=slot_tour_confirmed, value=True),
                actions=[
                    Utterance(
                        "Okay great, please make sure you're here 15 min before departure.",
                        "utter_tour_confirmed",
                    ),
                ],
            ),
        ],
    )


def get_domain(states: List[StateMachineState]) -> Domain:
    intent_names = [
        intent.name for state in states for intent in list(state.all_intents())
    ]

    slots = [
        slot.as_rasa_slot() for state in states for slot in list(state.all_slots())
    ]

    action_names = [
        action.name for state in states for action in list(state.all_actions())
    ]

    state_machine_states = {
        state.name: {
            "is_initial_state": state == states[0],
            "state_yaml": yaml.dump(state),
        }
        for state in states
    }

    return Domain(
        intents=intent_names,
        slots=slots,
        action_names=action_names,
        entities={},
        responses={},
        forms={},
        state_machine_states=state_machine_states,
    )


# domain = MockStateMachineProvider(
#     initial_state_name="start_state", states=[get_start_state(), get_destination_state()]
# )


def test_get_actions_response_actions():
    domain = get_domain([get_start_state(), get_destination_state()])
    tracker = DialogueStateTracker(sender_id="test_id", slots=domain.slots)
    tracker.latest_message = UserUttered(
        "Hi", intent={"name": intent_hello.name, "confidence": 1}
    )
    tracker.update_with_events(
        [StateMachineTransition(domain.initial_state_machine_state_name)], domain=None
    )

    prev_tracker = tracker.copy()

    state = get_start_state()

    actions = StateMachinePolicy.get_actions(
        state_machine_state=state,
        prev_tracker=prev_tracker,
        tracker=tracker,
        domain=domain,
        intent_threshold_for_slot_fill=0,
    )

    action_names = [action.name for action in actions]

    # Test initial state
    assert (
        tracker.active_state_machine_state_name
        == domain.initial_state_machine_state_name
    )

    # Test response actions
    assert action_names == ["utter_how_are_you", "utter_hello"]


def test_get_actions_slot_fill_and_actions():
    domain = get_domain([get_start_state(), get_destination_state()])
    tracker = DialogueStateTracker(sender_id="test_id", slots=domain.slots)
    tracker.latest_message = UserUttered(
        "yes", intent={"name": "intent_affirm", "confidence": 1}
    )
    tracker.update_with_events(
        [StateMachineTransition(domain.initial_state_machine_state_name)], domain=None
    )

    prev_tracker = tracker.copy()

    state = get_start_state()

    actions = StateMachinePolicy.get_actions(
        state_machine_state=state,
        prev_tracker=prev_tracker,
        tracker=tracker,
        domain=domain,
        intent_threshold_for_slot_fill=0,
    )

    action_names = [action.name for action in actions]

    # Test initial state
    assert (
        tracker.active_state_machine_state_name
        == domain.initial_state_machine_state_name
    )

    # Test slot-fill
    assert tracker.current_slot_values().get(SLOT_NAME) == "yes"

    # Test slot-fill actions
    assert action_names == [
        "utter_slot_value",
        "utter_how_are_you",
    ]


def test_get_actions_transitions():
    domain = get_domain([get_start_state(), get_destination_state()])
    tracker = DialogueStateTracker(sender_id="test_id", slots=domain.slots)
    tracker.latest_message = UserUttered(
        "bye", intent={"name": intent_goodbye.name, "confidence": 1}
    )
    tracker.update_with_events(
        [StateMachineTransition(domain.initial_state_machine_state_name)], domain=None
    )

    prev_tracker = tracker.copy()

    state = get_start_state()

    # Test initial state
    assert (
        tracker.active_state_machine_state_name
        == domain.initial_state_machine_state_name
    )

    actions = StateMachinePolicy.get_actions(
        state_machine_state=state,
        prev_tracker=prev_tracker,
        tracker=tracker,
        domain=domain,
        intent_threshold_for_slot_fill=0,
    )

    action_names = [action.name for action in actions]

    # Test transition
    assert tracker.active_state_machine_state_name == "destination_state"

    # Test actions
    assert action_names == [
        "utter_how_are_you",
        "utter_going",
        "utter_welcome",
    ]

    # Next message from user
    tracker.latest_message = UserUttered(
        "back", intent={"name": "intent_back", "confidence": 1}
    )
    prev_tracker = tracker.copy()

    next_actions = StateMachinePolicy.get_actions(
        state_machine_state=tracker.get_active_state_machine_state(domain),
        prev_tracker=prev_tracker,
        tracker=tracker,
        domain=domain,
        intent_threshold_for_slot_fill=0,
    )

    next_action_names = [action.name for action in next_actions]

    # Test initial state
    assert (
        tracker.active_state_machine_state_name
        == domain.initial_state_machine_state_name
    )

    # Test actions
    assert next_action_names == [
        "utter_going_back",
        "utter_how_are_you",
    ]


def run_and_assert_events(
    expected_events: List[str],
    tracker: DialogueStateTracker,
    policy: StateMachinePolicy,
    domain: Domain,
):
    last_predicted_action_name = None
    for event in expected_events:
        if event.startswith("intent_"):
            tracker.latest_message = UserUttered(
                "random text", intent={"name": event, "confidence": 1}
            )
        else:
            assert last_predicted_action_name == event

        if event != "action_listen":
            prediction = policy.predict_action_probabilities(
                tracker=tracker, domain=domain, interpreter=None
            )

            if prediction.max_confidence > 0:
                action = rasa.core.actions.action.action_for_index(
                    prediction.max_confidence_index, domain, None
                )
                last_predicted_action_name = action.name()

                if last_predicted_action_name != "action_listen":
                    # Update tracker
                    tracker.update_with_events(
                        new_events=[
                            BotUttered(
                                "",
                                metadata={"utter_action": last_predicted_action_name},
                            )
                        ],
                        domain=domain,
                    )


def test_predict_action_probabilities():
    domain = get_domain([get_start_state(), get_destination_state()])
    tracker = DialogueStateTracker(sender_id="test_id", slots=domain.slots)
    policy = StateMachinePolicy()

    # Test initial state
    assert tracker.active_state_machine_state_name == None

    expected_events = [
        intent_hello.name,
        "utter_how_are_you",
        "utter_hello",
        "utter_confirm",
        "action_listen",
        intent_goodbye.name,
        "utter_going",
        "utter_welcome",
        "action_listen",
    ]

    run_and_assert_events(
        expected_events=expected_events, tracker=tracker, policy=policy, domain=domain
    )


def test_predict_action_probabilities_book_tour():
    domain = get_domain([get_visitor_center_state(), get_book_tour_state()])
    tracker = DialogueStateTracker(sender_id="test_id", slots=domain.slots)
    policy = StateMachinePolicy()

    # Test initial state
    assert tracker.active_state_machine_state_name == None

    expected_events = [
        intent_hello.name,
        "utter_welcome_to_the_visitor_center",
        "action_listen",
        intent_book_tour.name,  # Transition
        "utter_i_can_help_you",  # Transition utterance
        "utter_lets_book_tour",  # Entry utterance
        "utter_tour_info",  # Slot ask
        "action_listen",
        intent_select_three.name,  # Slot fill
        "utter_tour_info",  # Slot ask
        "action_listen",
        intent_select_bus_tour.name,  # Slot fill
        utterance_tour_booked.name,
        "utter_ask_tour_num_tickets",  # Slot ask
        "action_listen",
        intent_select_three.name,  # Slot fill
        utterance_tour_num_tickets.name,
        "utter_ask_tour_confirmation",  # Slot ask
        "action_listen",
    ]

    run_and_assert_events(
        expected_events=expected_events, tracker=tracker, policy=policy, domain=domain
    )

    # Test transition
    assert tracker.active_state_machine_state_name == "book_tour"


def test_predict_action_probabilities_book_tour_return_to_start():
    domain = get_domain([get_visitor_center_state(), get_book_tour_state()])
    tracker = DialogueStateTracker(sender_id="test_id", slots=domain.slots)
    policy = StateMachinePolicy()

    # Test initial state
    assert tracker.active_state_machine_state_name == None

    expected_events = [
        intent_hello.name,
        "utter_welcome_to_the_visitor_center",
        "action_listen",
        intent_book_tour.name,  # Transition
        "utter_i_can_help_you",  # Transition utterance
        "utter_lets_book_tour",  # Entry utterance
        "utter_tour_info",  # Slot ask
        "action_listen",
        intent_select_bus_tour.name,  # Slot fill
        utterance_tour_booked.name,
        "utter_ask_tour_num_tickets",  # Slot ask
        "action_listen",
        intent_select_three.name,  # Slot fill
        utterance_tour_num_tickets.name,
        "utter_ask_tour_confirmation",  # Slot ask
        "action_listen",
        "intent_affirm",  # Slot fill
        "utter_tour_confirmed",
        "utter_welcome_to_the_visitor_center",  # Entry utterance
        "action_listen",
    ]

    run_and_assert_events(
        expected_events=expected_events, tracker=tracker, policy=policy, domain=domain
    )

    # Test transition
    assert tracker.active_state_machine_state_name == "start_state"
