class Event:
    ARRIVAL = 0 # constant for arrival type
    DEPARTURE = -1 # constant for departure type

    def __init__(self, typ, time):
        self.type = typ
        self.time = time
    
    def __lt__(self, other):
        return self.time < other.time
    
    def __repr__(self):
        s = ("Arrival", "Departure")
        return f"{s[self.type]} at {self.time}"