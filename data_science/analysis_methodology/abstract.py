from abc import ABC, abstractmethod


class Methodology(ABC):
    NAME = "method_name"

    def __init__(self, compare_limit):
        self.compare_limit = compare_limit

    def set_dt(self, dt):
        self.dt = dt

    @abstractmethod
    def send_sample(self, sample):
        pass

    @abstractmethod
    def are_samples_distinct(self):
        pass

    @abstractmethod
    def compare_value(self):
        pass
