class Event:
    """
    Purpose: Represents an event in the simulation, either an arrival or a departure.
    Attributes: Type of event (ARRIVAL or DEPARTURE), time of event, and server ID.
    Methods:
    __init__: Initializes the event with given parameters.
    __lt__: Defines the less-than operation for sorting events by time.
    __repr__: Returns a string representation of the event
    """
    ARRIVAL = 0 # constant for arrival type
    DEPARTURE = -1 # constant for departure type

    def __init__(self, typ, time, server_id):
        self.type = typ
        self.time = time
        self.server_id = server_id
    
    def __lt__(self, other):
        return self.time < other.time
    
    def __repr__(self):
        s = ("Arrival", "Departure")
        return f"{s[self.type]} at {self.time} at server {self.server_id}"