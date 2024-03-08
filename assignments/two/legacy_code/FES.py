import heapq

class FES:
    """
    Purpose: Manages the events in the simulation.
    Attributes: A list of events.
    Methods:
    add: Adds an event to the list.
    next: Returns and removes the next event from the list.
    __iter__ and __next__: Implements iterator protocol for the class.
    isEmpty: Checks if there are no events left.
    __repr__: Returns a string representation of the events
    """
    def __init__(self):
        self.events = []
        
    def add(self, event):
        heapq.heappush(self.events, event)
        
    def next(self):
        if self.events:
            return heapq.heappop(self.events)
        else:
            raise StopIteration
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.events:
            return self.next()
        else:
            raise StopIteration
    
    def isEmpty(self):
        return len(self.events) == 0
        
    def __repr__(self):
        # Note that if you print self.events, it would not appear to be sorted
        # (although they are sorted internally).
        # For this reason we use the function 'sorted'
        s = ''
        sortedEvents = sorted(self.events)
        for e in sortedEvents :
            s += f'{e}\n'
        return s



