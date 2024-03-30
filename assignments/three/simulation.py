import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy import stats
from tabulate import tabulate
import heapq
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")

# print(Path.cwd())
os.chdir(Path.cwd().parent.parent)
np.random.seed(420420)


class Customer:
    NO_PREFERENCE = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self, cust_id, arrival_time=0, fuel_time=0, shop_time=0, payment_time=0,
                 parking_preference=NO_PREFERENCE,
                 shop_yes_no=False):
        self.system_entry_time = 0  # time the customer entered the system

        self.cust_id = cust_id  # customer id
        self.payment_queue_time = 0  # entry time to the payment queue
        self.entrance_queue_time = 0  # entry time to the entrance queue
        self.parking_preference = parking_preference
        self.wants_to_shop = shop_yes_no
        self.fuel_pump = None

        # for simulation with the actual dataset
        self.arrival_time = arrival_time
        self.fuel_time = fuel_time
        self.shop_time = shop_time
        self.payment_time = payment_time


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
                 betas, mu
                 ):

        self.alphas = alphas
        self.betas = betas
        self.poisson_mean = mu
        self.setup_simulation()

    def setup_simulation(self):
        # Initialize distributions
        self.fuel_time_dist = stats.gamma(a=self.alphas[0], scale=1 / self.betas[0])
        self.shop_time_dist = stats.gamma(a=self.alphas[1], scale=1 / self.betas[1])
        # self.payment_time_dist = stats.gamma(a=self.alphas[2], scale=1 / self.betas[2])
        self.interarrival_dist = stats.gamma(a=self.alphas[3], scale=1 / self.betas[3])

        self.payment_time_dist = stats.poisson(mu=self.poisson_mean)

        # Reset simulation time and queues
        self.max_time = 960 * 60  # 16 hours in seconds
        self.fes = FES()  # Assuming FES is some form of future event scheduler
        self.station_entry_queue = Queue()
        self.shop_queue = Queue()
        self.payment_queue = Queue()
        self.waiting_to_leave = Queue()

        # Reset counters and measurements
        self.customer_id = 0
        self.old_time = 0
        self.current_time = 0
        self.number_of_customers_servered = 0

        # Initialize servers
        self.cashier = Server("C1")
        self.pump_stations = [Server(i) for i in range(4)]

        # Reset statistics
        self.waiting_time_entrance_queue = []
        self.area_queue_length_fuel_station = []
        self.area_queue_length_shop = []
        self.waiting_time_payment_queue = []
        self.area_queue_length_payment = []
        self.total_time_spent_in_system = []

    # given a customer, it will set up the customer's data
    def set_customer_data(self, customer: Customer):
        temp_customer = customer  # simply another pointer to the same instance of the customer class instance

        temp_customer.parking_preference = np.random.choice([Customer.NO_PREFERENCE, Customer.LEFT, Customer.RIGHT],
                                                            p=[0.828978622327791, 0.09501187648456057,
                                                               0.07600950118764846])

        temp_customer.wants_to_shop = np.random.choice([True, False], p=[0.22327790973871733, 0.7767220902612827])

        return temp_customer

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

    def handle_preferences_scenario_three(self, cust_preference):

        if cust_preference == Customer.NO_PREFERENCE:
            available_pumps = []
            for pump in self.pump_stations:
                if pump.status == Server.IDLE:
                    available_pumps.append(pump.server_id)

            if len(available_pumps) == 0:
                return -1
            else:
                choice = np.random.choice(available_pumps)
                # print(choice)
                return choice

        elif cust_preference == Customer.RIGHT:
            if self.pump_stations[0].status == Server.IDLE:
                return 0
            elif self.pump_stations[1].status == Server.IDLE:
                return 1
            else:
                return -1

        elif cust_preference == Customer.LEFT:
            if self.pump_stations[2].status == Server.IDLE:
                return 2
            elif self.pump_stations[3].status == Server.IDLE:
                return 3
            else:
                return -1

    # -----------------------------------------Base simulation (with fitted distributions)------------------------------------------#
    def base_simulation_fitted(self):

        self.customer_id += 1
        arrival_time = self.interarrival_dist.rvs()
        current_customer = Customer(self.customer_id)
        current_customer = self.set_customer_data(current_customer)
        current_customer.system_entry_time = arrival_time

        self.fes.add(Event(Event.ARRIVAL, current_customer, arrival_time))

        while len(self.fes.events) > 0:
            event = self.fes.next()

            # customers that arrive after the closing time are not served (our policy)
            if self.current_time >= self.max_time and event.type == Event.ARRIVAL:
                # use the following print line only for testing purposes
                # print(f"Customer {event.customer.cust_id} arrived after closing time -> customer not served")                        
                continue

            # use the following print line only for testing purposes
            # print(repr(event))
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

            self.station_entry_queue.S += (len(self.station_entry_queue) + num_customers_at_fuel_pumps) * (
                    self.current_time - self.old_time)
            self.shop_queue.S += len(self.shop_queue) * (self.current_time - self.old_time)
            self.payment_queue.S += (len(self.payment_queue) + customer_at_cashier) * (
                    self.current_time - self.old_time)

            if event.type == Event.ARRIVAL:
                self.number_of_customers_servered += 1
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
                    self.waiting_time_entrance_queue.append(
                        self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                    # store the pump id the customer is assigned to
                    self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                    # assign the customer to the fuel pump
                    self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                    # generate the fuel departure event time
                    event_time = self.fuel_time_dist.rvs()

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

                self.fes.add(Event(Event.ARRIVAL, next_customer, self.current_time + next_arrival_time))

            # ----------------------------------------------------------Handling the fuel departure event----------------------------------------------#
            if event.type == Event.FUEL_DEPARTURE:
                if current_customer.wants_to_shop:
                    # add customer to the shop queue and create a shop departure event
                    self.shop_queue.join_queue(current_customer)
                    event_time = self.shop_time_dist.rvs()

                    self.fes.add(
                        Event(Event.SHOP_DEPARTURE, current_customer, self.current_time + event_time)
                    )

                elif not (current_customer.wants_to_shop):
                    # if cashier is idle, customer can go to pay straight away
                    if self.cashier.status == Server.IDLE:

                        # record the time the customer spent in the payment queue
                        self.waiting_time_payment_queue.append(0)

                        self.cashier.customer_arrive(current_customer)
                        event_time = self.payment_time_dist.rvs()

                        # customer is with the cashier, hence we schedule a payment departure event
                        self.fes.add(
                            Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + event_time)
                        )

                    else:
                        # if cahsier is busy, the customer joins the payment queue
                        # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                        current_customer.payment_queue_time = self.current_time
                        self.payment_queue.join_queue(current_customer)

            # ----------------------------------------------------------Handling the shop departure event----------------------------------------------#
            if event.type == Event.SHOP_DEPARTURE:
                # if cashier is idle, customer can go to pay straight away
                if self.cashier.status == Server.IDLE:

                    # record the time the customer spent in the payment queue
                    self.waiting_time_payment_queue.append(0)

                    self.cashier.customer_arrive(current_customer)
                    event_time = self.payment_time_dist.rvs()

                    # schedule a payment departure event for the customer
                    self.fes.add(Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + event_time))

                    # remove the customer from the shop queue
                    self.shop_queue.leave_queue(current_customer)

                else:
                    # if cahsier is busy, the customer joins the payment queue
                    # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                    current_customer.payment_queue_time = self.current_time
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
                        self.waiting_time_entrance_queue.append(
                            self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                        # store the pump id the customer is assigned to
                        self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                        # assign the customer to the fuel pump
                        self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                        # generate the fuel departure event time
                        event_time = self.fuel_time_dist.rvs()

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
        results["Waiting time\nFuel station (s)"] = np.mean(self.waiting_time_entrance_queue)
        results["Queue length\nFuel station (Customers)"] = self.station_entry_queue.S / self.current_time

        results["Queue length\nshop (Customers)"] = self.shop_queue.S / self.current_time

        results["Waiting time\nPayment queue (s)"] = np.mean(self.waiting_time_payment_queue)
        results["Queue length\nPayment queue (Customers)"] = self.payment_queue.S / self.current_time

        results["Total time\nspent in the system (s)"] = np.mean(self.total_time_spent_in_system)

        results["Number of customers served"] = self.number_of_customers_servered

        return results

    # -----------------------------------------Base simulation (with actual dataset)------------------------------------------#
    def base_simulation_impirical_data(self):

        # Loads the data
        data = pd.read_excel("assignments/three/gasstationdata33.xlsx")

        # for each row in the dataset, we create a customer instance
        # populate the customer instance with the data from the dataset
        for index, row in data.iterrows():
            opening_time = pd.to_datetime('2024-02-01 06:00:00')
            arrival_time = (row["Arrival Time"] - opening_time).total_seconds()
            self.customer_id + - 1
            customer = Customer(self.customer_id, arrival_time=arrival_time,
                                fuel_time=row["Service time Fuel"],
                                shop_time=row["Shop time"],
                                payment_time=row["Service time payment"],
                                parking_preference=str(row["Parking Preference"]),
                                shop_yes_no=row["Shop time"] > 0)

            # print the customer data
            # print(customer.__dict__)
            self.fes.add(Event(Event.ARRIVAL, customer, customer.arrival_time))

        while len(self.fes.events) > 0:
            # while self.current_time < self.max_time or self.non_arrival_events_left():
            event = self.fes.next()

            # customers that arrive after the closing time are not served (our policy)
            if self.current_time >= self.max_time and event.type == Event.ARRIVAL:
                # use the following print line only for testing purposes
                # print(f"Customer {event.customer.cust_id} arrived after closing time -> customer not served")                        
                continue

            # use the following print line only for testing purposes
            # print(repr(event))
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

            self.station_entry_queue.S += (len(self.station_entry_queue) + num_customers_at_fuel_pumps) * (
                    self.current_time - self.old_time)
            self.shop_queue.S += len(self.shop_queue) * (self.current_time - self.old_time)
            self.payment_queue.S += (len(self.payment_queue) + customer_at_cashier) * (
                    self.current_time - self.old_time)

            if event.type == Event.ARRIVAL:
                current_customer.system_entry_time = self.current_time
                self.station_entry_queue.join_queue(current_customer)

                # retrieve the preference of the first customer
                temp_preference = self.station_entry_queue.customers_in_queue[0].parking_preference

                # Try to assign the customer to a fuel pump
                if temp_preference == "nan":
                    status = self.handle_no_preference()

                elif temp_preference == "Left":
                    status = self.handle_left_preference()

                elif temp_preference == "Right":
                    status = self.handle_right_preference()

                if status != -1:
                    # add the time the customer spent in the entrance queue
                    self.waiting_time_entrance_queue.append(
                        self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                    # store the pump id the customer is assigned to
                    self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                    # assign the customer to the fuel pump
                    self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                    # customer was assigned to a fuel pump, hence we schedule a fuel departure event
                    self.fes.add(
                        Event(
                            Event.FUEL_DEPARTURE,
                            self.station_entry_queue.customers_in_queue[0],
                            self.current_time + self.station_entry_queue.customers_in_queue[0].fuel_time
                        )
                    )

                    # customer leaves the entrance queue
                    self.station_entry_queue.leave_queue(self.station_entry_queue.customers_in_queue[0])

            # ----------------------------------------------------------Handling the fuel departure event----------------------------------------------#
            if event.type == Event.FUEL_DEPARTURE:
                if current_customer.wants_to_shop:
                    # add customer to the shop queue and create a shop departure event
                    self.shop_queue.join_queue(current_customer)

                    self.fes.add(
                        Event(Event.SHOP_DEPARTURE, current_customer, self.current_time + current_customer.shop_time)
                    )

                elif not (current_customer.wants_to_shop):
                    # if cashier is idle, customer can go to pay straight away
                    if self.cashier.status == Server.IDLE:

                        # record the time the customer spent in the payment queue
                        self.waiting_time_payment_queue.append(0)

                        self.cashier.customer_arrive(current_customer)

                        # customer is with the cashier, hence we schedule a payment departure event
                        self.fes.add(
                            Event(Event.PAYMENT_DEPARTURE, current_customer,
                                  self.current_time + current_customer.payment_time)
                        )

                    else:
                        # if cahsier is busy, the customer joins the payment queue
                        # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                        current_customer.payment_queue_time = self.current_time
                        self.payment_queue.join_queue(current_customer)

            # ----------------------------------------------------------Handling the shop departure event----------------------------------------------#
            if event.type == Event.SHOP_DEPARTURE:
                # if cashier is idle, customer can go to pay straight away
                if self.cashier.status == Server.IDLE:

                    # record the time the customer spent in the payment queue
                    self.waiting_time_payment_queue.append(0)

                    self.cashier.customer_arrive(current_customer)

                    # schedule a payment departure event for the customer
                    self.fes.add(Event(Event.PAYMENT_DEPARTURE, current_customer,
                                       self.current_time + current_customer.payment_time))

                    # remove the customer from the shop queue
                    self.shop_queue.leave_queue(current_customer)

                else:
                    # if cahsier is busy, the customer joins the payment queue
                    # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                    current_customer.payment_queue_time = self.current_time
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

                    self.fes.add(
                        Event(Event.PAYMENT_DEPARTURE, next_customer, self.current_time + next_customer.payment_time)
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
                    if temp_preference == "nan":
                        status = self.handle_no_preference()

                    elif temp_preference == "Left":
                        status = self.handle_left_preference()

                    elif temp_preference == "Right":
                        status = self.handle_right_preference()

                    if status != -1:
                        # add the time the customer spent in the entrance queue
                        self.waiting_time_entrance_queue.append(
                            self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                        # store the pump id the customer is assigned to
                        self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                        # assign the customer to the fuel pump
                        self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                        # customer was assigned to a fuel pump, hence we schedule a fuel departure event
                        self.fes.add(
                            Event(
                                Event.FUEL_DEPARTURE,
                                self.station_entry_queue.customers_in_queue[0],
                                self.current_time + self.station_entry_queue.customers_in_queue[0].fuel_time)
                        )

                        # customer leaves the entrance queue
                        self.station_entry_queue.leave_queue(self.station_entry_queue.customers_in_queue[0])

        results = {}
        results["Waiting time\nFuel station (s)"] = np.mean(self.waiting_time_entrance_queue)
        results["Queue length\nFuel station (Customers)"] = self.station_entry_queue.S / self.current_time

        results["Queue length\nshop (Customers)"] = self.shop_queue.S / self.current_time

        results["Waiting time\nPayment queue (s)"] = np.mean(self.waiting_time_payment_queue)
        results["Queue length\nPayment queue (Customers)"] = self.payment_queue.S / self.current_time

        results["Total time\nspent in the system (s)"] = np.mean(self.total_time_spent_in_system)

        return results

    # -----------------------------------------Simulation without the shop------------------------------------------------------#
    def simulation_no_shop(self):

        # each fuel pump has its own payment terminal.
        # pump 0 has termnal 0, pump 1 has terminal 1 and so on
        payment_terminals = [Server(i) for i in range(4)]

        self.customer_id += 1
        arrival_time = self.interarrival_dist.rvs()
        current_customer = Customer(self.customer_id)
        current_customer = self.set_customer_data(current_customer)
        current_customer.system_entry_time = arrival_time

        self.fes.add(Event(Event.ARRIVAL, current_customer, arrival_time))

        while len(self.fes.events) > 0:
            # while self.current_time < self.max_time or self.non_arrival_events_left():
            event = self.fes.next()

            # customers that arrive after the closing time are not served (our policy)
            if self.current_time >= self.max_time and event.type == Event.ARRIVAL:
                # use the following print line only for testing purposes
                # print(f"Customer {event.customer.cust_id} arrived after closing time -> customer not served")                        
                continue

            # use the following print line only for testing purposes
            # print(repr(event))
            self.old_time = self.current_time
            self.current_time = event.time

            # retrieve the current customer
            current_customer = event.customer

            # update the queue length of the entrance queue
            num_customers_at_fuel_pumps = 0
            for pump in self.pump_stations:
                if pump.status == Server.BUSY:
                    num_customers_at_fuel_pumps += 1

            self.station_entry_queue.S += (len(self.station_entry_queue) + num_customers_at_fuel_pumps) * (
                    self.current_time - self.old_time)

            if event.type == Event.ARRIVAL:
                self.number_of_customers_servered += 1
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
                    self.waiting_time_entrance_queue.append(
                        self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                    # store the pump id the customer is assigned to
                    self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                    # assign the customer to the fuel pump
                    self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                    # generate the fuel departure event time
                    event_time = self.fuel_time_dist.rvs()

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

                self.fes.add(Event(Event.ARRIVAL, next_customer, self.current_time + next_arrival_time))

            # ----------------------------------------------------------Handling the fuel departure event----------------------------------------------#
            if event.type == Event.FUEL_DEPARTURE:
                # since each fuel pump has its own payment terminal, the customer can start paying straight away after fueling
                payment_terminals[current_customer.fuel_pump].customer_arrive(current_customer)
                event_time = self.payment_time_dist.rvs()

                # customer is at the payment terminal, hence we schedule a payment departure event
                self.fes.add(
                    Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + event_time)
                )
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
                    payment_terminals[0].customer_leave()

                    # check if any customers were blocked behind the current customer and allow them to leave if yes.
                    for cust in self.waiting_to_leave.customers_in_queue:
                        if cust.fuel_pump == 1:
                            # record the time the customer spent in the system
                            self.total_time_spent_in_system.append(self.current_time - cust.system_entry_time)
                            # customer leaves the system
                            self.waiting_to_leave.leave_queue(cust)
                            # fuel pump becomes idle
                            self.pump_stations[1].customer_leave()
                            payment_terminals[1].customer_leave()

                elif pump == 1:
                    if self.pump_stations[0].status == Server.IDLE:
                        self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                        self.pump_stations[1].customer_leave()
                        payment_terminals[1].customer_leave()

                    else:
                        # customer is blocked by the customer at pump 0, hence they join the waiting to leave queue
                        self.waiting_to_leave.join_queue(current_customer)

                elif pump == 2:
                    self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                    self.pump_stations[2].customer_leave()
                    payment_terminals[2].customer_leave()

                    # check if any customers were blocked behind the current customer and allow them to leave if yes.
                    for cust in self.waiting_to_leave.customers_in_queue:
                        if cust.fuel_pump == 3:
                            # record the time the customer spent in the system
                            self.total_time_spent_in_system.append(self.current_time - cust.system_entry_time)
                            # customer leaves the system
                            self.waiting_to_leave.leave_queue(cust)
                            # fuel pump becomes idle
                            self.pump_stations[3].customer_leave()
                            payment_terminals[3].customer_leave()

                elif pump == 3:
                    if self.pump_stations[2].status == Server.IDLE:
                        self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                        self.pump_stations[3].customer_leave()
                        payment_terminals[3].customer_leave()

                    else:
                        # customer is blocked by the customer at pump 2, hence they join the waiting to leave queue
                        self.waiting_to_leave.join_queue(current_customer)

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
                        self.waiting_time_entrance_queue.append(
                            self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                        # store the pump id the customer is assigned to
                        self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                        # assign the customer to the fuel pump
                        self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                        # generate the fuel departure event time
                        event_time = self.fuel_time_dist.rvs()

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
        results["Waiting time\nFuel station (s)"] = np.mean(self.waiting_time_entrance_queue)
        results["Queue length\nFuel station (Customers)"] = self.station_entry_queue.S / self.current_time

        results["Total time\nspent in the system (s)"] = np.mean(self.total_time_spent_in_system)
        results["Number of customers served"] = self.number_of_customers_servered

        return results

    # ----------------------------------------Simulation with four lines of pumps------------------------------------------------------#
    def simulation_four_lines_of_pumps(self):

        self.customer_id += 1
        arrival_time = self.interarrival_dist.rvs()
        current_customer = Customer(self.customer_id)
        current_customer = self.set_customer_data(current_customer)
        current_customer.system_entry_time = arrival_time

        self.fes.add(Event(Event.ARRIVAL, current_customer, arrival_time))

        while len(self.fes.events) > 0:
            # while self.current_time < self.max_time or self.non_arrival_events_left():
            event = self.fes.next()

            # customers that arrive after the closing time are not served (our policy)
            if self.current_time >= self.max_time and event.type == Event.ARRIVAL:
                # use the following print line only for testing purposes
                # print(f"Customer {event.customer.cust_id} arrived after closing time -> customer not served")                        
                continue

            # use the following print line only for testing purposes
            # print(repr(event))
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

            self.station_entry_queue.S += (len(self.station_entry_queue) + num_customers_at_fuel_pumps) * (
                    self.current_time - self.old_time)
            self.shop_queue.S += len(self.shop_queue) * (self.current_time - self.old_time)
            self.payment_queue.S += (len(self.payment_queue) + customer_at_cashier) * (
                    self.current_time - self.old_time)

            if event.type == Event.ARRIVAL:
                self.number_of_customers_servered += 1
                current_customer.system_entry_time = self.current_time
                self.station_entry_queue.join_queue(current_customer)

                # retrieve the preference of the first customer
                temp_preference = self.station_entry_queue.customers_in_queue[0].parking_preference

                # Try to assign the customer to a fuel pump
                status = self.handle_preferences_scenario_three(temp_preference)

                if status != -1:
                    # add the time the customer spent in the entrance queue
                    self.waiting_time_entrance_queue.append(
                        self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                    # store the pump id the customer is assigned to
                    self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                    # assign the customer to the fuel pump
                    self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                    # generate the fuel departure event time
                    event_time = self.fuel_time_dist.rvs()

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

                self.fes.add(Event(Event.ARRIVAL, next_customer, self.current_time + next_arrival_time))

            # ----------------------------------------------------------Handling the fuel departure event----------------------------------------------#
            if event.type == Event.FUEL_DEPARTURE:
                if current_customer.wants_to_shop:
                    # add customer to the shop queue and create a shop departure event
                    self.shop_queue.join_queue(current_customer)
                    event_time = self.shop_time_dist.rvs()

                    self.fes.add(
                        Event(Event.SHOP_DEPARTURE, current_customer, self.current_time + event_time)
                    )

                elif not (current_customer.wants_to_shop):
                    # if cashier is idle, customer can go to pay straight away
                    if self.cashier.status == Server.IDLE:

                        # record the time the customer spent in the payment queue
                        self.waiting_time_payment_queue.append(0)

                        self.cashier.customer_arrive(current_customer)
                        event_time = self.payment_time_dist.rvs()

                        # customer is with the cashier, hence we schedule a payment departure event
                        self.fes.add(
                            Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + event_time)
                        )

                    else:
                        # if cahsier is busy, the customer joins the payment queue
                        # save the time the customer joined the payment queue and used to calculate the time spent in the queue later
                        current_customer.payment_queue_time = self.current_time
                        self.payment_queue.join_queue(current_customer)

            # ----------------------------------------------------------Handling the shop departure event----------------------------------------------#
            if event.type == Event.SHOP_DEPARTURE:
                # if cashier is idle, customer can go to pay straight away
                if self.cashier.status == Server.IDLE:

                    # record the time the customer spent in the payment queue
                    self.waiting_time_payment_queue.append(0)

                    self.cashier.customer_arrive(current_customer)
                    event_time = self.payment_time_dist.rvs()

                    # schedule a payment departure event for the customer
                    self.fes.add(Event(Event.PAYMENT_DEPARTURE, current_customer, self.current_time + event_time))

                    # remove the customer from the shop queue
                    self.shop_queue.leave_queue(current_customer)

                else:
                    # if cahsier is busy, the customer joins the payment queue
                    current_customer.payment_queue_time = self.current_time
                    self.payment_queue.join_queue(current_customer)
                    self.shop_queue.leave_queue(current_customer)

            # ----------------------------------------------------------Handling the payment departure event----------------------------------------------#
            if event.type == Event.PAYMENT_DEPARTURE:

                # retieve the fuel pump the customer was being served at
                pump = current_customer.fuel_pump

                self.total_time_spent_in_system.append(self.current_time - current_customer.system_entry_time)
                self.pump_stations[pump].customer_leave()

                # if there are still customers in the payment queue, the cashier serves the next customer
                if self.payment_queue.get_queue_status() == Queue.NOT_EMPTY:
                    next_customer = self.payment_queue.customers_in_queue[0]
                    self.waiting_time_payment_queue.append(self.current_time - next_customer.payment_queue_time)
                    self.cashier.customer_arrive(next_customer)
                    event_time = self.payment_time_dist.rvs()

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

                    status = self.handle_preferences_scenario_three(temp_preference)

                    if status != -1:
                        # add the time the customer spent in the entrance queue
                        self.waiting_time_entrance_queue.append(
                            self.current_time - self.station_entry_queue.customers_in_queue[0].system_entry_time)

                        # store the pump id the customer is assigned to
                        self.station_entry_queue.customers_in_queue[0].fuel_pump = status

                        # assign the customer to the fuel pump
                        self.pump_stations[status].customer_arrive(self.station_entry_queue.customers_in_queue[0])

                        # generate the fuel departure event time
                        event_time = self.fuel_time_dist.rvs()

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
        results["Waiting time\nFuel station (s)"] = np.mean(self.waiting_time_entrance_queue)
        results["Queue length\nFuel station (Customers)"] = self.station_entry_queue.S / self.current_time

        results["Queue length\nshop (Customers)"] = self.shop_queue.S / self.current_time

        results["Waiting time\nPayment queue (s)"] = np.mean(self.waiting_time_payment_queue)
        results["Queue length\nPayment queue (Customers)"] = self.payment_queue.S / self.current_time

        results["Total time\nspent in the system (s)"] = np.mean(self.total_time_spent_in_system)
        results["Number of customers served"] = self.number_of_customers_servered

        return results


def confidence_interval(results, confidence=0.95):
    results = np.array(results)
    mean = results.mean(axis=0)
    std_dev = results.std(axis=0, ddof=1)

    low = mean - (1.96 * (std_dev / np.sqrt(len(results))))
    high = mean + (1.96 * (std_dev / np.sqrt(len(results))))

    # return low, high

    return (1.96 * (std_dev / np.sqrt(len(results))))


# ---------------------------------------------Main function------------------------------------------------------#
def main():
    # seed for reproducibility
    np.random.seed(420420)

    # NOTE: order of input parameters: [fuel_time_dist, shop_time_dist, service_time_payment_dist, interarrival_dist]
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_runs = 100

    # Perform n_run simulations
    sim_names = ["Base simulation with empirical data (benchmark)", "Base simulation fitted distributions",
                 "Simulation without the shop", "Simulation with four lines of pumps"]

    sim = Simulation(alphas, betas, mu=mu)

    # Clear the results file
    with open('results.txt', 'w') as f:
        f.write("-------------------Simulation setup-------------------\n")
        f.write(f"Number of runs: {n_runs}\n")
        f.write(f"Seed: 420420\n")
        f.write(f"Alpha values: {alphas}\n")
        f.write(f"Beta values: {betas}\n")
        f.write(f"Mu value: {mu}\n")

    for sim_name in range(4):

        simulation_results = [[] for i in range(n_runs)]

        if sim_name == 0:
            print("Base simulation with empirical data (benchmark)")
            for i in range(n_runs):
                simulation_results[i] = sim.base_simulation_impirical_data()
                sim.setup_simulation()
                print(f"Base Impirical Simulation {i} done")

        if sim_name == 1:
            print("Base simulation fitted distributions")
            for i in range(n_runs):
                simulation_results[i] = (sim.base_simulation_fitted())
                sim.setup_simulation()
                print(f"Base Simulation {i} done")

        elif sim_name == 2:
            print("Simulation without the shop")
            for i in range(n_runs):
                simulation_results[i] = (sim.simulation_no_shop())
                sim.setup_simulation()
                print(f"No shop Simulation {i} done")

        elif sim_name == 3:
            print("Simulation with four lines of pumps")
            for i in range(n_runs):
                simulation_results[i] = (sim.simulation_four_lines_of_pumps())
                sim.setup_simulation()
                print(f"Four line Simulation {i} done")

        # print(f"\n-------------------Results for {sim_names[sim_name]}-------------------")
        # # retreive the keys of the dictionary
        # keys = list(simulation_results[0].keys())
        # simulation_results = [pd.DataFrame(simulation_results[i], index=[f"Results for Runtime: {i}"]).T for i in range(n_runs)]
        # simulation_results= np.array(simulation_results)
        # mean = simulation_results.mean(axis=0)

        # lower_bound, upper_bound = confidence_interval(simulation_results)
        # for i in range(len(mean)):
        #     # print(f"{mean[i]} [{lower_bound[i]}, {upper_bound[i]}]")
        #     print(f"{keys[i]}: {mean[i][0]} [{lower_bound[i][0]}, {upper_bound[i][0]}]")

        # if sim_name == 0:
        #     with open('results.txt', 'a') as f:
        #         f.write(f"\n-------------------Results for {sim_names[sim_name]}-------------------\n")
        #         line = f"Waiting time Fuel station: {benchmark_sim_result["Waiting time\nFuel station (s)"]}\n"
        #         f.write(line)
        #         line = f"Queue length Fuel station: {benchmark_sim_result["Queue length\nFuel station (Customers)"]}\n"
        #         f.write(line)
        #         line = f"Queue length shop: {benchmark_sim_result["Queue length\nshop (Customers)"]}\n"
        #         f.write(line)
        #         line = f"Waiting time Payment queue: {benchmark_sim_result["Waiting time\nPayment queue (s)"]}\n"
        #         f.write(line)
        #         line = f"Queue length Payment queue: {benchmark_sim_result["Queue length\nPayment queue (Customers)"]}\n"
        #         f.write(line)
        #         line = f"Total time spent in the system: {benchmark_sim_result["Total time\nspent in the system (s)"]}\n"
        #         f.write(line)
        #         line = f"Number of customers served: 420 \n"
        #         f.write(line)

        # else:
        with open('results.txt', 'a') as f:
            f.write(f"\n-------------------Results for {sim_names[sim_name]}-------------------\n")
            keys = list(simulation_results[0].keys())
            simulation_results = [pd.DataFrame(simulation_results[i], index=[f"Results for Runtime: {i}"]).T for i in
                                  range(n_runs)]
            simulation_results = np.array(simulation_results)
            mean = simulation_results.mean(axis=0)
            std = simulation_results.std(axis=0, ddof=1)

            # lower_bound, upper_bound = confidence_interval(simulation_results)
            ci = confidence_interval(simulation_results)

            for i in range(len(mean)):
                # result_line = f"{keys[i]}: {mean[i][0]} [{lower_bound[i][0]}, {upper_bound[i][0]}] and std {std[i][0]} \n"
                result_line = f"{keys[i]}: has mean {mean[i][0]} (+- {ci[i][0]}) and std {std[i][0]} \n"
                f.write(result_line)

            f.write("\n")

    # print("Base simulation")
    # sim_basic = Simulation(alphas, betas)
    # results = sim_basic.base_simulation_fitted()

    # print(tabulate(results, headers='keys', tablefmt='pretty'))
    # print("")

    # print("Simulation with actual dataset")
    # sim_imperial = Simulation(alphas, betas)
    # results_imperial = sim_imperial.base_simulation_impreical_data()

    # print(tabulate(results_imperial, headers='keys', tablefmt='pretty'))
    # print("")    

    # print("Simulation without the shop")
    # sim_scenario_2 = Simulation(alphas, betas)
    # results2 = sim_scenario_2.simulation_no_shop()

    # print(tabulate(results2, headers='keys', tablefmt='pretty'))
    # print("")

    # print("Simulation with four lines of pumps")
    # sim_scenario_3 = Simulation(alphas, betas)
    # results3 = sim_scenario_3.simulation_four_lines_of_pumps()

    # print(tabulate(results3, headers='keys', tablefmt='pretty'))


def plotting_base_simulation_results():
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_runs = 1
    n_sims = 1001

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)
    # simulation_results = pd.DataFrame(
    #     columns=["Waiting time\nFuel station (s)", "Queue length\nFuel station (Customers)", "Queue length\nshop (Customers)", "Waiting time\nPayment queue (s)", "Queue length\nPayment queue (Customers)", "Total time\nspent in the system (s)", "Number of customers served"]
    # )

    means = []
    variances = []
    ci_low = []
    ci_high = []
    out_means = []

    simulation_results = []

    for u in tqdm(range(1, n_sims), desc="Simulation with base"):

        for i in range(n_runs):
            simulation_results.append(sim.base_simulation_fitted())
            sim.setup_simulation()

        dfs = [pd.DataFrame(res, index=[0]) for res in simulation_results]

        df = pd.concat(dfs, axis=0)

        means.append(df.mean(axis="index"))
        variances.append(df.var(axis="index"))
        # calculate the confidence interval
        temp_mean = pd.concat(means, axis=1)
        temp_var = pd.concat(variances, axis=1)
        # print(temp_mean.mean(axis=1).shape)
        out_means.append(temp_mean.mean(axis=1))
        ci_low.append(temp_mean.mean(axis=1) - 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))
        ci_high.append(temp_mean.mean(axis=1) + 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))

    # print(ci_low[0].shape)

    # means_over_times_df = pd.concat(means, axis=1)
    # variances_over_times_df = pd.concat(variances, axis=1)
    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)

    fig, ax = plt.subplots(len(df.columns), 1, figsize=(10, 30))
    #plt.subplots_adjust(hspace=1.5)




    for i, col in enumerate(df.columns):
        ax[i].plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax[i].fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax[i].set_xlabel("Number of simulations".title(), fontsize=16)
        ax[i].set_ylabel(f"{col}", fontsize=16)
        ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].tick_params(axis='both', which='minor', labelsize=12)
        ax[i].legend()

    # set the title to the figure
    fig.suptitle(f"Results Base Simulation (with fitted distributions) \nover {n_sims-1} simulations".title(), fontsize=24, y = 0.92)
    fig.savefig("base_simulation_results.png")
    plt.show()


def plotting_det_simulation_results():
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_runs = 1
    n_sims = 1001

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)
    # simulation_results = pd.DataFrame(
    #     columns=["Waiting time\nFuel station (s)", "Queue length\nFuel station (Customers)", "Queue length\nshop (Customers)", "Waiting time\nPayment queue (s)", "Queue length\nPayment queue (Customers)", "Total time\nspent in the system (s)", "Number of customers served"]
    # )

    means = []
    variances = []
    ci_low = []
    ci_high = []
    out_means = []

    simulation_results = []

    for u in tqdm(range(1, n_sims), desc="Simulation with empirical data"):

        for i in range(n_runs):
            simulation_results.append(sim.base_simulation_impirical_data())
            sim.setup_simulation()

        dfs = [pd.DataFrame(res, index=[0]) for res in simulation_results]

        df = pd.concat(dfs, axis=0)

        means.append(df.mean(axis="index"))
        variances.append(df.var(axis="index"))
        # calculate the confidence interval
        temp_mean = pd.concat(means, axis=1)
        temp_var = pd.concat(variances, axis=1)
        # print(temp_mean.mean(axis=1).shape)
        out_means.append(temp_mean.mean(axis=1))
        ci_low.append(temp_mean.mean(axis=1) - 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))
        ci_high.append(temp_mean.mean(axis=1) + 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))

    # print(ci_low[0].shape)

    # means_over_times_df = pd.concat(means, axis=1)
    # variances_over_times_df = pd.concat(variances, axis=1)
    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)

    fig, ax = plt.subplots(len(df.columns), 1, figsize=(10, 30))
    #plt.subplots_adjust(hspace=1.5)




    for i, col in enumerate(df.columns):
        ax[i].plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax[i].fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax[i].set_xlabel("Number of simulations".title(), fontsize=16)
        ax[i].set_ylabel(f"{col}", fontsize=16)
        ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].tick_params(axis='both', which='minor', labelsize=12)
        ax[i].legend()

    # set the title to the figure
    fig.suptitle(f"Results Base Simulation (with Empirical Data) \nover {n_sims-1} simulations".title(), fontsize=24, y = 0.92)
    fig.savefig("det_simulation_results.png")
    plt.show()

def plotting_no_shop_simulation_results():
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_runs = 1
    n_sims = 1001

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)
    # simulation_results = pd.DataFrame(
    #     columns=["Waiting time\nFuel station (s)", "Queue length\nFuel station (Customers)", "Queue length\nshop (Customers)", "Waiting time\nPayment queue (s)", "Queue length\nPayment queue (Customers)", "Total time\nspent in the system (s)", "Number of customers served"]
    # )

    means = []
    variances = []
    ci_low = []
    ci_high = []
    out_means = []

    simulation_results = []

    for u in tqdm(range(1, n_sims), desc="Simulation with no shop"):

        for i in range(n_runs):
            simulation_results.append(sim.simulation_no_shop())
            sim.setup_simulation()

        dfs = [pd.DataFrame(res, index=[0]) for res in simulation_results]

        df = pd.concat(dfs, axis=0)

        means.append(df.mean(axis="index"))
        variances.append(df.var(axis="index"))
        # calculate the confidence interval
        temp_mean = pd.concat(means, axis=1)
        temp_var = pd.concat(variances, axis=1)
        # print(temp_mean.mean(axis=1).shape)
        out_means.append(temp_mean.mean(axis=1))
        ci_low.append(temp_mean.mean(axis=1) - 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))
        ci_high.append(temp_mean.mean(axis=1) + 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))

    # print(ci_low[0].shape)

    # means_over_times_df = pd.concat(means, axis=1)
    # variances_over_times_df = pd.concat(variances, axis=1)
    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)

    fig, ax = plt.subplots(len(df.columns), 1, figsize=(10, 30))
    #plt.subplots_adjust(hspace=1.5)




    for i, col in enumerate(df.columns):
        ax[i].plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax[i].fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax[i].set_xlabel("Number of simulations".title(), fontsize=16)
        ax[i].set_ylabel(f"{col}", fontsize=16)
        ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].tick_params(axis='both', which='minor', labelsize=12)
        ax[i].legend()

    # set the title to the figure
    fig.suptitle(f"Results of Gas Station With No Shop \nover {n_sims-1} simulations".title(), fontsize=24, y = 0.92)
    fig.savefig("no_shop_simulation_results.png")
    plt.show()


def plotting_four_pumps_simulation_results():
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_runs = 1
    n_sims = 1001

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)
    # simulation_results = pd.DataFrame(
    #     columns=["Waiting time\nFuel station (s)", "Queue length\nFuel station (Customers)", "Queue length\nshop (Customers)", "Waiting time\nPayment queue (s)", "Queue length\nPayment queue (Customers)", "Total time\nspent in the system (s)", "Number of customers served"]
    # )

    means = []
    variances = []
    ci_low = []
    ci_high = []
    out_means = []

    simulation_results = []

    for u in tqdm(range(1, n_sims), desc="Simulation with four lines of pumps"):

        for i in range(n_runs):
            simulation_results.append(sim.simulation_four_lines_of_pumps())
            sim.setup_simulation()

        dfs = [pd.DataFrame(res, index=[0]) for res in simulation_results]

        df = pd.concat(dfs, axis=0)

        means.append(df.mean(axis="index"))
        variances.append(df.var(axis="index"))
        # calculate the confidence interval
        temp_mean = pd.concat(means, axis=1)
        temp_var = pd.concat(variances, axis=1)
        out_means.append(temp_mean.mean(axis=1))
        ci_low.append(temp_mean.mean(axis=1) - 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))
        ci_high.append(temp_mean.mean(axis=1) + 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))

    # print(ci_low[0].shape)

    # means_over_times_df = pd.concat(means, axis=1)
    # variances_over_times_df = pd.concat(variances, axis=1)
    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)

    fig, ax = plt.subplots(len(df.columns), 1, figsize=(10, 30))
    #plt.subplots_adjust(hspace=1.5)



    for i, col in enumerate(df.columns):
        ax[i].plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax[i].fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax[i].set_xlabel("Number of simulations".title(), fontsize=16)
        ax[i].set_ylabel(f"{col}", fontsize=16)
        ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].tick_params(axis='both', which='minor', labelsize=12)
        ax[i].legend()

    # set the title to the figure
    fig.suptitle(f"Results of Gas Station with Four lines of fuel pumps \nover {n_sims-1} Simulation".title(), fontsize=24, y = 0.92)
    fig.savefig("four_pumps_simulation_results.png")

    plt.show()


if __name__ == "__main__":
    # main()
    plotting_base_simulation_results()
    plotting_det_simulation_results()
    plotting_no_shop_simulation_results()
    plotting_four_pumps_simulation_results()
