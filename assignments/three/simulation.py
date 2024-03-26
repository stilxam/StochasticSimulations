import numpy as np
import pandas as pd
import scipy
from scipy import stats
from tabulate import tabulate
import heapq


class Customer:
    NO_PREFERENCE = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self, cust_id,
                 parking_preference=NO_PREFERENCE,
                 shop_yes_no=False):
        self.system_entry_time = 0  # time the customer entered the system
  
        self.cust_id = cust_id  # customer id
        self.payment_queue_time = 0  # time spent in payment queue
        self.entrance_queue_time = 0  # time spent in entrance queue
        self.parking_preference = parking_preference
        self.wants_to_shop = shop_yes_no
        self.fuel_pump = None



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
        self.customers_in_queue = []
        self.S = 0

    def join_queue(self, customer: Customer):
        self.customers_in_queue.append(customer)

    def leave_queue(self, customer: Customer):
        self.customers_in_queue.remove(customer)

    def get_queue_status(self):
        return Queue.EMPTY if len(self.customers_in_queue) == 0 else Queue.NOT_EMPTY

    def __len__(self):
        return len(self.customers_in_queue)


class Server:
    IDLE = 0
    BUSY = 1

    def __init__(self, server_id):
        # cahsier and fuel pumps ar instances of the server class
        # casheir id will look like C1, C2, C3
        # fuel pump id will look like 0,1,2,3
        self.server_id = server_id
        self.status = Server.IDLE
        self.current_customer = None

    def customer_arrive(self, customer: Customer):
        self.current_customer = customer
        self.status = Server.BUSY

    def customer_leave(self):
        self.status = Server.IDLE
        self.current_customer = None


class Simulation:

    def __init__(self,
                 alphas,
                 betas,
                 ):


        self.fuel_time_dist = stats.gamma(a=alphas[0], scale=1 / betas[0])
        self.shop_time_dist = stats.gamma(a=alphas[1], scale=1 / betas[1])
        self.payment_time_dist = stats.gamma(a=alphas[2], scale=1 / betas[2])
        self.interarrival_dist = stats.gamma(a=alphas[3], scale=1 / betas[3])

        # total simulation duration is 960 minutes (i.e. 16 hours from 6 am to 10 pm)
        # check whether the input data is in seconds or minutes
        # Note, it is in seconds
        self.max_time = 960 * 60  # in seconds
        self.fes = FES()

        # queues
        self.station_entry_queue = Queue()
        self.shop_queue = Queue()
        self.payment_queue = Queue()

        # queue for customers waiting to leave because they are blocked by another customer
        self.waiting_to_leave = Queue()

        # customer id
        self.customer_id = 0

        # system time
        self.old_time = 0
        self.current_time = 0

        # the cashier server
        self.cashier = Server("C1")

        # the fuel pump servers
        self.pump_stations = [Server(f"F{i}") for i in range(4)]

        # measurements
        self.waiting_time_entrance_queue = []  # waiting times at the fuel station
        self.area_queue_length_fuel_station = []  # queue length of the entrance queue
        self.area_queue_length_shop = []  # queue length of the shop queue
        self.waiting_time_payment_queue = []  # waiting times at the payment queue
        self.area_queue_length_payment = []  # queue length of the payment queue
        self.total_time_spent_in_system = []  # total time spent in the system

        self.testing = [[],[],[],[]]

    # given a customer, it will set up the customer's data
    def set_customer_data(self, customer: Customer):
        temp_customer = customer # simply another pointer to the same instance of the customer class instance 

        temp_customer.parking_preference = np.random.choice([Customer.NO_PREFERENCE, Customer.LEFT, Customer.RIGHT],
                                                            p=[0.828978622327791, 0.09501187648456057, 0.07600950118764846])
        
        temp_customer.wants_to_shop = np.random.choice([True, False], p=[0.22327790973871733, 0.7767220902612827])
        return temp_customer

    def non_arrival_events_left(self):
        for event in self.fes.events:
            if event.type != Event.ARRIVAL:
                return True
        return False

    def handle_no_preference(self):

        # returns -1 if the customer is not assigned to a pump, otherwise returns the fuel pump id

        all_pumps_available = False

        # if all pumps are available, then all pumps available is set to True
        for pump in self.pump_stations:
            if pump.status == Server.IDLE:
                all_pumps_available = True
            else:
                all_pumps_available = False
                break
        
        if all_pumps_available:
            random_pump = np.random.choice([0, 2])
            
            return random_pump

        # check if there is a fuel pump is available
        if self.pump_stations[1].status == Server.IDLE:
            if self.pump_stations[0].status == Server.IDLE:
                return 0
            else:
                return 1

        elif self.pump_stations[3].status == Server.IDLE:
            if self.pump_stations[2].status == Server.IDLE:
                return 2
            else:
                return 3
        else:
            return -1

    def handle_left_preference(self):

        # check pumps 3 and 4
        if self.pump_stations[3].status == Server.IDLE:
            if self.pump_stations[2].status == Server.IDLE:
                return 2
            else:
                return 3
        else:
            return -1  # cannot assign the customer to a pump

    def handle_right_preference(self):

        # check pumps 1 and 2
        if self.pump_stations[1].status == Server.IDLE:
            if self.pump_stations[0].status == Server.IDLE:
                return 0
            else:
                return 1
        else:
            return -1  # cannot assign the customer to a pump

    def base_simulation(self):

        self.customer_id += 1
        arrival_time = self.interarrival_dist.rvs()
        current_customer = Customer(self.customer_id)
        current_customer = self.set_customer_data(current_customer)
        current_customer.system_entry_time = arrival_time

        self.fes.add(Event(Event.ARRIVAL, current_customer, arrival_time))

        while self.current_time < self.max_time or self.non_arrival_events_left():
            event = self.fes.next()

            # customers that arrive after the closing time are not served (our policy)
            if self.current_time >= self.max_time and event.type == Event.ARRIVAL:
                continue

            print(repr(event))
            self.old_time = self.current_time
            self.current_time = event.time

            # retrieve the current customer
            current_customer = event.customer

            # update the queue length of the entrance queue, payment queue and shop queue
            num_customers_at_fuel_pumps = 0
            for pump in self.pump_stations:
                if pump.status == Server.BUSY:
                    num_customers_at_fuel_pumps += 1

            if self.cashier.status == Server.BUSY:
                customer_at_cashier = 1
            else:
                customer_at_cashier = 0

            self.station_entry_queue.S += (len(self.station_entry_queue) + num_customers_at_fuel_pumps) * (self.current_time - self.old_time)
            self.shop_queue.S += len(self.shop_queue) * (self.current_time - self.old_time)
            self.payment_queue.S += (len(self.payment_queue) + customer_at_cashier) * (self.current_time - self.old_time)

            if event.type == Event.ARRIVAL:
                current_customer.system_entry_time = self.current_time
                self.station_entry_queue.join_queue(current_customer)

                # retrieve the preference of the first customer
                temp_preference = self.station_entry_queue.customers_in_queue[0].parking_preference

                # Try to assign the customer to a fuel pump
                if temp_preference == Customer.NO_PREFERENCE:
                    status = self.handle_no_preference()

                elif temp_preference == Customer.LEFT:
                    status = self.handle_left_preference()

                elif temp_preference == Customer.RIGHT:
                    status = self.handle_right_preference()

                if status != -1:
                    
                    # add the time the customer spent in the entrance queue
                    self.waiting_time_entrance_queue.append(self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                    # assign the customer to the fuel pump
                    self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                    # store the pump id the customer is assigned to
                    self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                    # generate the fuel departure event time
                    event_time = self.fuel_time_dist.rvs()
                    self.testing[0].append(event_time)

                    # customer was assigned to a fuel pump, hence we schedule a fuel departure event
                    self.fes.add(
                        Event(
                            Event.FUEL_DEPARTURE,
                            self.station_entry_queue.customers_in_queue[0],
                            self.current_time + event_time
                        )
                    )

                    # customer leaves the entrance queue
                    self.station_entry_queue.leave_queue(self.station_entry_queue.customers_in_queue[0])

                # generate the next arrival 
                self.customer_id += 1
                next_customer = Customer(self.customer_id)
                next_customer = self.set_customer_data(next_customer)
                next_arrival_time = self.interarrival_dist.rvs()
                self.testing[3].append(next_arrival_time)
                self.fes.add(Event(Event.ARRIVAL, next_customer, self.current_time + next_arrival_time))
                
            #----------------------------------------------------------Handling the fuel departure event----------------------------------------------# 
            if event.type == Event.FUEL_DEPARTURE:
                if current_customer.wants_to_shop:
                    # add customer to the shop queue and create a shop departure event
                    self.shop_queue.join_queue(current_customer)
                    event_time = self.shop_time_dist.rvs()
                    self.testing[1].append(event_time) # for testing purposes
                    self.fes.add(
                        Event(Event.SHOP_DEPARTURE, current_customer, self.current_time + event_time)
                    )

                elif not(current_customer.wants_to_shop):
                    # if cashier is idle, customer can go to pay straight away
                    if self.cashier.status == Server.IDLE:
                        current_customer.payment_queue_time = 0 # customer did not have to wait in the payment queue
                        self.cashier.customer_arrive(current_customer)
                        event_time = self.payment_time_dist.rvs()
                        self.testing[2].append(event_time)
                        # customer is with the cashier, hence we schedule a payment departure event
                        self.fes.add(
                            Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + event_time)
                        )

                    else:
                        # if cahsier is busy, the customer joins the payment queue
                        current_customer.payment_queue_time = self.current_time  # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                        self.payment_queue.join_queue(current_customer)

            # ----------------------------------------------------------Handling the shop departure event----------------------------------------------#
            if event.type == Event.SHOP_DEPARTURE:
                # if cashier is idle, customer can go to pay straight away
                if self.cashier.status == Server.IDLE:

                    # record the time the customer spent in the payment queue
                    self.waiting_time_payment_queue.append(0)

                    self.cashier.customer_arrive(current_customer)
                    event_time = self.payment_time_dist.rvs()
                    self.testing[2].append(event_time)

                    # schedule a payment departure event for the customer
                    self.fes.add(Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + event_time))

                    # remove the customer from the shop queue
                    self.shop_queue.leave_queue(current_customer)

                else:
                    # if cahsier is busy, the customer joins the payment queue
                    current_customer.payment_queue_time = self.current_time  # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                    self.payment_queue.join_queue(current_customer)
                    self.shop_queue.leave_queue(current_customer)

            # ----------------------------------------------------------Handling the payment departure event----------------------------------------------#
            if event.type == Event.PAYMENT_DEPARTURE:
                
                # retieve the fuel pump the customer was being served at
                pump = current_customer.fuel_pump

                # check if customer can leave the system or if they are blocked by another customer
                if pump == 0:
                    # customer can leave the system, hence we record the time they spent in the system
                    self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                    
                    # fuel pump becomes idle
                    self.pump_stations[0].customer_leave()

                    # check if any customers were blocked behind the current customer and allow them to leave if yes.
                    for cust in self.waiting_to_leave.customers_in_queue:
                        if cust.fuel_pump == 1:

                            # record the time the customer spent in the system
                            self.total_time_spent_in_system.append(self.current_time - cust.system_entry_time)
                            # customer leaves the system
                            self.waiting_to_leave.leave_queue(cust)
                            # fuel pump becomes idle
                            self.pump_stations[1].customer_leave()

                elif pump == 1:
                    if self.pump_stations[0].status == Server.IDLE:
                        self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                        self.pump_stations[1].customer_leave()

                    else:
                        # customer is blocked by the customer at pump 0, hence they join the waiting to leave queue
                        self.waiting_to_leave.join_queue(current_customer)

                elif pump == 2:
                    self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                    self.pump_stations[2].customer_leave()

                    # check if any customers were blocked behind the current customer and allow them to leave if yes.
                    for cust in self.waiting_to_leave.customers_in_queue:
                        if cust.fuel_pump == 3:

                            # record the time the customer spent in the system
                            self.total_time_spent_in_system.append(self.current_time - cust.system_entry_time)
                            # customer leaves the system
                            self.waiting_to_leave.leave_queue(cust)
                            # fuel pump becomes idle
                            self.pump_stations[3].customer_leave()

                elif pump == 3:
                    if self.pump_stations[2].status == Server.IDLE:
                        self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                        self.pump_stations[3].customer_leave()

                    else:
                        # customer is blocked by the customer at pump 2, hence they join the waiting to leave queue
                        self.waiting_to_leave.join_queue(current_customer)

                # if there are still customers in the payment queue, the cashier serves the next customer
                if self.payment_queue.get_queue_status() == Queue.NOT_EMPTY:
                    next_customer = self.payment_queue.customers_in_queue[0]
                    self.waiting_time_payment_queue.append(self.current_time - next_customer.payment_queue_time)
                    self.cashier.customer_arrive(next_customer)
                    event_time = self.payment_time_dist.rvs()
                    self.testing[2].append(event_time)
                    self.fes.add(
                        Event(Event.PAYMENT_DEPARTURE, next_customer, self.current_time + event_time)
                    )

                    self.payment_queue.leave_queue(next_customer)

                # if the payment queue is empty, cashier becomes idle
                elif self.payment_queue.get_queue_status() == Queue.EMPTY:
                    self.cashier.customer_leave()

                # check if there is a customer waiting in the entrance queue that can be assigned to a fuel pump
                if self.station_entry_queue.get_queue_status() == Queue.NOT_EMPTY:
                    # retrieve the preference of the first customer
                    temp_preference = self.station_entry_queue.customers_in_queue[0].parking_preference

                    # Try to assign the customer to a fuel pump
                    if temp_preference == Customer.NO_PREFERENCE:
                        status = self.handle_no_preference()

                    elif temp_preference == Customer.LEFT:
                        status = self.handle_left_preference()

                    elif temp_preference == Customer.RIGHT:
                        status = self.handle_right_preference()

                    if status != -1:
                        
                        # add the time the customer spent in the entrance queue
                        self.waiting_time_entrance_queue.append(self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                        # assign the customer to the fuel pump
                        self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                        # store the pump id the customer is assigned to
                        self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                        # generate the fuel departure event time
                        event_time = self.fuel_time_dist.rvs()
                        self.testing[0].append(event_time)

                        # customer was assigned to a fuel pump, hence we schedule a fuel departure event
                        self.fes.add(
                            Event(
                                Event.FUEL_DEPARTURE,
                                self.station_entry_queue.customers_in_queue[0],
                                self.current_time + event_time
                            )
                        )

                        # customer leaves the entrance queue
                        self.station_entry_queue.leave_queue(self.station_entry_queue.customers_in_queue[0])

        results = {}
        results["average_waiting_time_entrance_queue"] = np.mean(self.waiting_time_entrance_queue)
        results["average_waiting_time_payment_queue"] = np.mean(self.waiting_time_payment_queue)
        results["average_time_spent_in_system"] = np.mean(self.total_time_spent_in_system)
        results["average_queue_length_fuel_station"] = self.station_entry_queue.S / self.current_time
        results["average_queue_length_shop"] = self.shop_queue.S / self.current_time
        results["average_queue_length_payment"] = self.payment_queue.S / self.current_time

        results["avg fuel time"] = np.mean(self.testing[0])
        results["avg shop time"] = np.mean(self.testing[1])
        results["avg payment time"] = np.mean(self.testing[2])
        results["avg arrival time"] = np.mean(self.testing[3])

        results = pd.DataFrame(results, index=[f"Results for Runtime: {self.max_time}"]).T

        # results = []
        # results.append(np.mean(self.waiting_time_entrance_queue))
        # results.append(np.mean(self.waiting_time_payment_queue))
        # results.append(np.mean(self.total_time_spent_in_system))
        # results.append(np.sum(self.area_queue_length_fuel_station) / self.current_time)
        # results.append(np.sum(self.area_queue_length_shop) / self.current_time)
        # results.append(np.sum(self.area_queue_length_payment) / self.current_time)
        return results

    def simulation_no_shop(self):
        return 0

    def simulation_four_lines_of_pumps(self):
        return 0


def main():
    # fuel_time_dist = scipy.stats.gamma(a = 3.7407418789843607, scale = 1 / 0.7739719530752498)
    # shop_time_dist = scipy.stats.gamma(a = 0.9896321424751771, scale = 1 / 0.8437679944913072)
    # service_time_payment_dist = scipy.stats.gamma(a = 64.16085452169962, scale = 1 / 85.58827329763147)
    # interarrival_dist = scipy.stats.gamma(a = 1.044611732553164, scale = 1 / 26.307262868337094)

    # fuel_time_dist = scipy.stats.gamma(a = 3.740741878984356, scale = 1 / 0.014062799908188449)
    # shop_time_dist = scipy.stats.gamma(a = 0.9896321424751765, scale = 1 / 0.014062799908188449)
    # service_time_payment_dist = scipy.stats.gamma(a = 64.16085452170083, scale = 1 / 1.426471221627218)
    # interarrival_dist = scipy.stats.gamma(a = 1.044611732553164, scale = 1 / 0.43845438113895135)

    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.43845438113895135]

    # parking_preference_dist = np.random.choice([Customer.NO_PREFERENCE, Customer.LEFT, Customer.RIGHT], 1,
    #                                            p=[0.828978622327791, 0.09501187648456057, 0.07600950118764846])

    # shop_yes_no_dist = np.random.choice([True, False], 1, p=[0.22327790973871733, 0.7767220902612827])

    # sim = Simulation(interarrival_dist, fuel_time_dist, shop_time_dist, service_time_payment_dist)
    sim = Simulation(alphas, betas)

    results = sim.base_simulation()

    # print("--------------------Results-------------------")
    # print("Average waiting time in the entrance queue: ", results[0])
    # print("Average waiting time in the payment queue: ", results[1])
    # print("Average time spent in the system: ", results[2])
    # print("Average queue length of the fuel station: ", results[3])
    # print("Average queue length of the shop: ", results[4])
    # print("Average queue length of the payment queue: ", results[5])

    # print(results)
    print(tabulate(results, headers='keys', tablefmt='pretty'))


if __name__ == "__main__":
    main()
