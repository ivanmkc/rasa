import abc


class Condition(abc.ABC):
    @abc.abstractmethod
    def is_valid(self, tracker: "DialogueStateTracker"):
        raise NotImplementedError()

    ignores_previous_tracker: bool = False

    def is_valid_and_changed(
        self, prev_tracker: "DialogueStateTracker", tracker: "DialogueStateTracker",
    ):
        # if self.ignores_previous_tracker:
        return self.is_valid(tracker)
        # else:
        #     return not self.is_valid(prev_tracker) and self.is_valid(tracker)
