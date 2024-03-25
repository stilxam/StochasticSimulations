import numpy as np
import scipy
from scipy import stats
import heapq

class Customer:

    NO_PREFERENCE = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self, cust_id, arrival_time = 0.0, 
                 fuel_time = 0.0, 
                 shop_time = 0.0, 
                 payment_time = 0.0,
                 parking_preference = NO_PREFERENCE,
                 shop_yes_no = False):
        
        self.system_entry_time = 0 # time the customer entered the system
        # self.arrival_time = arrival_time
        # self.fuel_time = fuel_time
        # self.shop_time = shop_time
        # self.payment_time = payment_time
        self.cust_id = cust_id # customer id
        self.payment_queue_time = 0 # time spent in payment queue
        self.entrance_queue_time = 0 # time spent in entrance queue
        self.parking_preference = parking_preference
        self.wants_to_shop = shop_yes_no
        self.fuel_pump = None
        self.time_spent_in_system = 0 # time spent in the system


class FES:

    def __init__(self):
        self.events = []

    def add(self, event):
        heapq.heappush(self.events, event)

    def next(self):
        return heapq.heappop(self.events)

    def isEmpty(self):
        return len(self.events) == 0

    def __repr__(self):
        # Note that if you print self.events, it would not appear to be sorted
        # (although they are sorted internally).
        # For this reason we use the function 'sorted'
        s = ''
        sortedEvents = sorted(self.events)
        for e in sortedEvents:
            s += f'{e}\n'
        return s

class Event:

    ARRIVAL = 0  # constant for arrival type
    FUEL_DEPARTURE = 1  # constant for fuel departure type
    SHOP_DEPARTURE = 2  # constant for shop departure type
    PAYMENT_DEPARTURE = 3  # constant for payment departure type

    def __init__(self, type, customer: Customer, time_of_event):
        self.type = type
        self.customer = customer
        self.time = time_of_event 

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        s = ("Arrival", "Fuel Departure", "Shop Departure", "Payment Departure")
        return f"customer {self.customer.cust_id} has event type {s[self.type]} at time {self.time}"

class Queue:

    # type of queue status
    EMPTY = 4  # constant for queue is empty
    NOT_EMPTY = 5  # constant for queue is not empty

    def __init__(self):
        # self.mu = departure_rate
        # self.servDist = scipy.stats.expon(scale=1 / self.mu)
        self.number_of_customers = 0
        self.customers_in_queue = []
        self.S = 0

    def join_queue(self, customer: Customer):
        self.number_of_customers += 1
        self.customers_in_queue.append(customer)


    def leave_queue(self):
        self.number_of_customers -= 1
        self.customers_in_queue.pop(0)
    
    def get_queue_status(self):
        return Queue.EMPTY if len(self.customers_in_queue) == 0 else Queue.NOT_EMPTY


class Server:
    IDLE = 0
    BUSY = 1
    def __init__(self, server_id):

        # this is for the cashier and the fuel pumps
        # casheir id will look like C1, C2, C3
        # fuel pump id will look like F1, F2, F3
        self.server_id = server_id
        self.status = Server.IDLE
        self.current_customer = None
    
    def customer_arrive(self, customer: Customer):
        self.current_customer = customer
        self.status = Server.BUSY
    
    def customer_leave(self):
        self.status = Server.IDLE
    


class Simulation:

    def __init__(self, interarrival_dist: stats.gamma, 
                 fuel_time_dist:  stats.gamma,
                 shop_time_dist: stats.gamma,
                 payment_time_dist: stats.gamma,
                #  parking_preference_dist: stats.gamma,
                #  shop_yes_no_dist: stats.bernoulli,
                 ):
        
        self.interarrival_dist = interarrival_dist
        self.fuel_time_dist = fuel_time_dist
        self.shop_time_dist = shop_time_dist
        self.payment_time_dist = payment_time_dist
        # self.parking_preference_dist = parking_preference_dist
        # self.shop_yes_no_dist = shop_yes_no_dist

        # total simulation duration is 960 minutes (i.e. 16 hours from 6 am to 10 pm)
        # check whether the input data is in seconds or minutes
        # Note, it is in seconds
        self.max_time = 960 * 60 # in seconds
        self.fes = FES()
        
        # queues
        self.entry_queue = Queue()
        self.shop_queue = Queue()
        self.payment_queue = Queue()
        self.waiting_to_leave = Queue()

        # customer id
        self.customer_id = 0

        # system time
        self.current_time = 0
        self.old_time = 0

        # the cashier server
        self.cashier = Server("C1")
        
        # the fuel pump servers
        self.pump_stations = [Server(f"F{i}") for i in range(4)]

        # measurements
        self.waiting_time_entrance_queue = [] # waiting times at the fuel station
        self.queue_length_fuel_station = [] # queue length of the entrance queue
        self.queue_length_shop = [] # queue length of the shop queue
        self.waiting_time_payment_queue = [] # waiting times at the payment queue
        self.queue_length_payment = [] # queue length of the payment queue
        self.total_time_spent_in_system = [] # total time spent in the system


    # given a customer, it will set up the customer's data
    def set_customer_data(self, customer: Customer):
        temp_customer = customer
        # temp_customer.fuel_time = self.fuel_time_dist.rvs()
        # temp_customer.shop_time = self.shop_time_dist.rvs()
        # temp_customer.payment_time = self.payment_time_dist.rvs()
        temp_customer.parking_preference = np.random.choice([Customer.NO_PREFERENCE, Customer.LEFT, Customer.RIGHT], 1,
                                               p=[0.828978622327791, 0.09501187648456057, 0.07600950118764846])

        temp_customer.wants_to_shop = np.random.choice([True, False], 1, p=[0.22327790973871733, 0.7767220902612827])


        return temp_customer

    def handle_no_preference(self, customer: Customer):

        # returns -1 if the customer is not assigned to a pump, otherwise returns the fuel pump id

        # check if there is a fuel pump is available
        if self.pump_stations[1].status == Server.IDLE:
            if self.pump_stations[0].status == Server.IDLE:
                self.pump_stations[0].customer_arrive(customer.cust_id)
                return "F0"
            else:
                self.pump_stations[1].customer_arrive(customer.cust_id)
                return "F1"
        
        elif self.pump_stations[3].status == Server.IDLE:
            if self.pump_stations[2].status == Server.IDLE:
                self.pump_stations[2].customer_arrive(customer.cust_id)
                return "F2"
            else: 
                self.pump_stations[3].customer_arrive(customer.cust_id)
                return "F3"
        
        else:
            return -1 
    
    def handle_left_preference(self, customer: Customer):

        # check pumps 3 and 4
        if  self.pump_stations[3].status == Server.IDLE:
            if self.pump_stations[2].status == Server.IDLE:
                self.pump_stations[2].customer_arrive(customer.cust_id)
                return "F2"
            else: 
                self.pump_stations[3].customer_arrive(customer.cust_id)
                return "F3"
        
        else: 
            return -1 # cannot assign the customer to a pump
    
    def handle_right_preference(self, customer: Customer):

        # check pumps 1 and 2
        if  self.pump_stations[1].status == Server.IDLE:
            if self.pump_stations[0].status == Server.IDLE:
                self.pump_stations[0].customer_arrive(customer.cust_id)
                return "F0"
            else: 
                self.pump_stations[1].customer_arrive(customer.cust_id)
                return "F1"
        
        else: 
            return -1 # cannot assign the customer to a pump
    

    def find_pump_and_customer(self, event: Event):
        for pump in self.pump_stations:
            if pump.current_customer.cust_id == event.cust_id:
                return pump, pump.current_customer
    

    def base_simulation (self):

        #logic goes here 
        self.customer_id += 1
        arrival_time = self.interarrival_dist.rvs()
        current_customer = Customer(self.customer_id)
        current_customer = self.set_customer_data(current_customer)
        current_customer.system_entry_time = arrival_time

        self.fes.add(Event(Event.ARRIVAL, current_customer, arrival_time))

        while self.current_time < self.max_time:

            event = self.fes.next()
            self.old_time = self.current_time
            self.current_time = event.time
            current_customer = event.customer
            current_customer.system_entry_time = self.current_time

            # update the queue length of the entrance queue, payment queue and shop queue
            self.queue_length_fuel_station.append((self.entry_queue.number_of_customers) * (self.current_time - self.old_time))
            self.queue_length_shop.append((self.shop_queue.number_of_customers) * (self.current_time - self.old_time))
            self.queue_length_payment.append((self.payment_queue.number_of_customers) * (self.current_time - self.old_time))

            if event.type == Event.ARRIVAL: 
                self.entry_queue.join_queue(current_customer)

                # if (self.entry_queue.get_queue_status() == Queue.NOT_EMPTY):

                # check the preference of the first customer
                temp_preference = self.entry_queue.customers_in_queue[0].parking_preference
                
                if temp_preference == Customer.NO_PREFERENCE:
                    status = self.handle_no_preference(self.entry_queue.customers_in_queue[0])
                
                elif temp_preference == Customer.LEFT:
                    status = self.handle_left_preference(self.entry_queue.customers_in_queue[0])
                
                elif temp_preference == Customer.RIGHT:
                    status = self.handle_right_preference(self.entry_queue.customers_in_queue[0])
                
                if status != -1:
                    # update wating time of customer in entry queue
                    self.entry_queue.customers_in_queue[0].entrance_queue_time = self.current_time - self.entry_queue.customers_in_queue[0].system_entry_time
                    self.entry_queue.customers_in_queue[0].fuel_pump = status
                    self.fes.add(Event(Event.FUEL_DEPARTURE, self.entry_queue.customers_in_queue[0], self.current_time + self.fuel_time_dist.rvs()))
                    self.entry_queue.leave_queue()
            
                # generate the next arrival 
                self.customer_id += 1
                next_arrival_time = self.interarrival_dist.rvs()
                next_customer = Customer(self.customer_id)
                next_customer = self.set_customer_data(next_customer)
                self.fes.add(Event(Event.ARRIVAL, next_customer, self.current_time + next_arrival_time))

            
            if event.type == Event.FUEL_DEPARTURE:

                # find the fuel pump that the customer was assigned to
                # pump, current_customer = self.find_pump_and_customer(event)
                
                if current_customer.wants_to_shop:
                    # add customer to the shop queue and create a shop departure event
                    self.shop_queue.join_queue(current_customer)
                    self.fes.add(Event(Event.SHOP_DEPARTURE, current_customer, self.current_time + self.shop_time_dist.rvs()))
                    # NOTE :customer only leaves when they pay
                    # self.pump_stations[pump].customer_leave()
                
                else: 
                    # if cashier is idle, customer can go to pay straight away
                    if self.cashier.status == Server.IDLE:
                        current_customer.payment_queue_time = 0
                        self.cashier.customer_arrive(current_customer)
                        self.fes.add(Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + self.payment_time_dist.rvs()))
                    
                    else:
                        # otherwise, they join the payment queue
                        current_customer.payment_queue_time = self.current_time # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                        self.payment_queue.join_queue(current_customer)
                    # self.pump_stations[pump].customer_leave()
            
            if event.type == Event.SHOP_DEPARTURE:                
                # if cashier is idle, customer can go to pay straight away
                if self.cashier.status == Server.IDLE:
                    current_customer.payment_queue_time = 0
                    self.cashier.customer_arrive(current_customer)
                    self.fes.add(Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + self.payment_time_dist.rvs()))
                    self.shop_queue.leave_queue()
                    
                else:
                    # otherwise, they join the payment queue
                    current_customer.payment_queue_time = self.current_time # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                    self.payment_queue.join_queue(current_customer)
                    self.shop_queue.leave_queue()
                    # self.pump_stations[pump].customer_leave()

            
            if event.type == Event.PAYMENT_DEPARTURE:
                # implement logic
                # cashier becomes idle, checks if there is someone waiting in the payment queue, if there is, they join the cashier, otherwise cashier remains idle

                current_customer.payment_queue_time = self.current_time - current_customer.payment_queue_time

                #check if customer can leave
                pump = current_customer.fuel_pump

                if pump == "F0":
                    current_customer.time_spent_in_system = self.current_time - current_customer.system_entry_time
                    self.pump_stations[0].customer_leave()

                    self.total_time_spent_in_system.append(current_customer.time_spent_in_system)
                    self.waiting_time_entrance_queue.append(current_customer.entrance_queue_time)
                    self.waiting_time_payment_queue.append(current_customer.payment_queue_time)


                    for cust in self.waiting_to_leave.customers_in_queue:
                        if cust.fuel_pump == "F1":
                            self.waiting_to_leave.leave_queue()
                            cust.time_spent_in_system = self.current_time - cust.system_entry_time

                            self.total_time_spent_in_system.append(cust.time_spent_in_system)
                            self.waiting_time_entrance_queue.append(cust.entrance_queue_time)
                            self.waiting_time_payment_queue.append(cust.payment_queue_time)

                            self.pump_stations[1].customer_leave()
                
                elif pump == "F1":
                    if self.pump_stations[0].status == Server.IDLE:
                        current_customer.time_spent_in_system = self.current_time - current_customer.system_entry_time

                        self.total_time_spent_in_system.append(current_customer.time_spent_in_system)
                        self.waiting_time_entrance_queue.append(current_customer.entrance_queue_time)
                        self.waiting_time_payment_queue.append(current_customer.payment_queue_time)

                        self.pump_stations[1].customer_leave()
                    
                    else:
                        self.waiting_to_leave.join_queue(current_customer)
                
                elif pump == "F2":
                    current_customer.time_spent_in_system = self.current_time - current_customer.system_entry_time

                    self.total_time_spent_in_system.append(current_customer.time_spent_in_system)
                    self.waiting_time_entrance_queue.append(current_customer.entrance_queue_time)
                    self.waiting_time_payment_queue.append(current_customer.payment_queue_time)

                    self.pump_stations[2].customer_leave()

                    for cust in self.waiting_to_leave.customers_in_queue:
                        if cust.fuel_pump == "F3":
                            self.waiting_to_leave.leave_queue()
                            cust.time_spent_in_system = self.current_time - cust.system_entry_time

                            self.total_time_spent_in_system.append(cust.time_spent_in_system)
                            self.waiting_time_entrance_queue.append(cust.entrance_queue_time)
                            self.waiting_time_payment_queue.append(cust.payment_queue_time)

                            self.pump_stations[3].customer_leave()
                
                elif pump == "F3":
                    if self.pump_stations[2].status == Server.IDLE:
                        current_customer.time_spent_in_system = self.current_time - current_customer.system_entry_time

                        self.total_time_spent_in_system.append(current_customer.time_spent_in_system)
                        self.waiting_time_entrance_queue.append(current_customer.entrance_queue_time)
                        self.waiting_time_payment_queue.append(current_customer.payment_queue_time)

                        self.pump_stations[3].customer_leave()
                    
                    else:
                        self.waiting_to_leave.join_queue(current_customer)
                
                # if there are customers waiting to pay, the first customer in the queue goes to the cashier
                if self.payment_queue.get_queue_status() == Queue.NOT_EMPTY:
                    next_customer = self.payment_queue.customers_in_queue[0]
                    next_customer.payment_queue_time = self.current_time - next_customer.payment_queue_time
                    self.cashier.customer_arrive(next_customer)
                    self.fes.add(Event(Event.PAYMENT_DEPARTURE, next_customer, self.current_time + self.payment_time_dist.rvs()))
                    self.payment_queue.leave_queue()

                # else, cashier becomes idle
                elif self.payment_queue.get_queue_status() == Queue.EMPTY:
                    self.cashier.customer_leave()
                
                # check if there is a customer waiting in the entrance queue
                if self.entry_queue.get_queue_status() == Queue.NOT_EMPTY:
                    temp_preference = self.entry_queue.customers_in_queue[0].parking_preference
                
                    if temp_preference == Customer.NO_PREFERENCE:
                        status = self.handle_no_preference(self.entry_queue.customers_in_queue[0])
                    
                    elif temp_preference == Customer.LEFT:
                        status = self.handle_left_preference(self.entry_queue.customers_in_queue[0])
                    
                    elif temp_preference == Customer.RIGHT:
                        status = self.handle_right_preference(self.entry_queue.customers_in_queue[0])
                    
                    if status != -1:
                        # update wating time of customer in entry queue
                        self.entry_queue.customers_in_queue[0].entrance_queue_time = self.current_time - self.entry_queue.customers_in_queue[0].system_entry_time
                        self.entry_queue.customers_in_queue[0].fuel_pump = status
                        self.fes.add(Event(Event.FUEL_DEPARTURE, self.entry_queue.customers_in_queue[0], self.current_time + self.fuel_time_dist.rvs()))
                        self.entry_queue.leave_queue()
    
        results = []
        results.append(np.mean(self.waiting_time_entrance_queue))
        results.append(np.mean(self.waiting_time_payment_queue))
        results.append(np.mean(self.total_time_spent_in_system))
        results.append(np.sum(self.queue_length_fuel_station) / self.current_time)
        results.append(np.sum(self.queue_length_shop) / self.current_time)
        results.append(np.sum(self.queue_length_payment) / self.current_time)

        return results

def main():
    
    fuel_time_dist = scipy.stats.gamma(a = 3.7407418789843607, scale = 1 / 0.7739719530752498)
    shop_time_dist = scipy.stats.gamma(a = 0.9896321424751771, scale = 1 / 0.8437679944913072)
    service_time_payment_dist = scipy.stats.gamma(a = 64.16085452169962, scale = 1 / 85.58827329763147)
    interarrival_dist = scipy.stats.gamma(a = 1.044611732553164, scale = 1 / 0.43845438113895135)

    # parking_preference_dist = np.random.choice([Customer.NO_PREFERENCE, Customer.LEFT, Customer.RIGHT], 1,
    #                                            p=[0.828978622327791, 0.09501187648456057, 0.07600950118764846])

    # shop_yes_no_dist = np.random.choice([True, False], 1, p=[0.22327790973871733, 0.7767220902612827])

    sim = Simulation(interarrival_dist, fuel_time_dist, shop_time_dist, service_time_payment_dist)

    results = sim.base_simulation()

    print("--------------------Results-------------------")
    print("Average waiting time in the entrance queue: ", results[0])
    print("Average waiting time in the payment queue: ", results[1])
    print("Average time spent in the system: ", results[2])
    print("Average queue length of the fuel station: ", results[3])
    print("Average queue length of the shop: ", results[4])
    print("Average queue length of the payment queue: ", results[5])

    # print(results)


if __name__ == "__main__":
    main()

