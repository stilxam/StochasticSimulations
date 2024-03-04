class Event:
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