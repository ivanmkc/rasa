import abc


class Condition(abc.ABC):
    @abc.abstractmethod
    def is_valid(self, tracker: "DialogueStateTracker"):
        raise NotImplementedError()
