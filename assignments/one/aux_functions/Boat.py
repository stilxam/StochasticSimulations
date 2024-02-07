import numpy


class Boat:
    def __init__(self, n: int = 8):
        self.n_seats: int = n
        self.filled_seats: int = 0

    def fill_boat(self, group:int):
        self.filled_seats += group

    def is_filling_possible(self, group:int):
        if self.n_seats < self.filled_seats+group:
            return False
        elif self.n_seats >= self.filled_seats+group:
            return True

    def is_boat_full(self):
        if self.filled_seats < 8:
            return False
        elif self.filled_seats == 8:
            return True
