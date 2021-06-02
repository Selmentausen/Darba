class Pigeon:

    def __init__(self, name, flight, load) -> None:
        self.name = name
        self.flight = flight
        self.load = load
        self.count = -1
        self.food = 0

    def __str__(self) -> str:
        return f'Pigeon {self.name}, {self.load}'

    def eat(self, food):
        self.food += food
        self.load += self.food // 10
        self.food %= 10

    def carry(self, load, range):
        if load <= self.load and self.flight >= range:
            return True
        return False

    def flap__wings(self):
        self.count += 1
        if self.count % 2 == 0:
            return 'up'
        return 'down'

    def lt(self, other):  # <
        return (self.load, self.flight, len(self.name), self.name) \
            < (other.load, other.flight, len(other.name), other.name)

    def __le__(self, other):  # <=
        return (self.load, self.flight, len(self.name), self.name) \
            <= (other.load, other.flight, len(other.name), other.name)

    def __eq__(self, other):  # ==
        return (self.load, self.flight, len(self.name), self.name) \
            == (other.load, other.flight, len(other.name), other.name)

    def __gt__(self, other):  # >
        return (self.load, self.flight, len(self.name), self.name) \
            > (other.load, other.flight, len(other.name), other.name)

    def __ge__(self, other):  # >=
        return (self.load, self.flight, len(self.name), self.name) \
            >= (other.load, other.flight, len(other.name), other.name)

    def __ne__(self, other):
        return (self.load, self.flight, len(self.name), self.name) \
            != (other.load, other.flight, len(other.name), other.name)