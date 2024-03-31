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
# os.chdir(Path.cwd().parent.parent)
# np.random.seed(420420)


class Customer:
    NO_PREFERENCE = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self, cust_id, arrival_time=0, fuel_time=0, shop_time=0, payment_time=0,
                 parking_preference=NO_PREFERENCE,
                 shop_yes_no=False):
        """
        Initialize a Customer instance.

        Parameters:
        cust_id (int): The unique identifier for the customer.
        arrival_time (int, optional): The time the customer arrives. Defaults to 0.
        fuel_time (int, optional): The time the customer spends at the fuel pump. Defaults to 0.
        shop_time (int, optional): The time the customer spends in the shop. Defaults to 0.
        payment_time (int, optional): The time the customer spends at the payment queue. Defaults to 0.
        parking_preference (int, optional): The parking preference of the customer. Defaults to NO_PREFERENCE.
        shop_yes_no (bool, optional): Whether the customer wants to shop or not. Defaults to False.

        Attributes:
        system_entry_time (int): The time the customer entered the system. Initialized to 0.
        payment_queue_time (int): The entry time to the payment queue. Initialized to 0.
        entrance_queue_time (int): The entry time to the entrance queue. Initialized to 0.
        fuel_pump (None): The fuel pump the customer is assigned to. Initialized to None.
        """

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
    """
    A class used to represent a Future Event Scheduler (FES).

    Attributes
    ----------
    events : list
        a list of events to be scheduled

    Methods
    -------
    add(event)
        Adds an event to the events list in a heap queue format.
    next()
        Removes and returns the smallest event from the events list.
    isEmpty()
        Checks if the events list is empty.
    __repr__()
        Returns a string representation of the sorted events list.
    """
    def __init__(self):
        """
        Constructs the FES class with an empty events list.
        """
        self.events = []

    def add(self, event):
        """
        Adds an event to the events list in a heap queue format.

        Parameters:
        event : object
            The event to be added to the events list.
        """
        heapq.heappush(self.events, event)

    def next(self):
        """
        Removes and returns the smallest event from the events list.

        Returns:
        object
            The smallest event from the events list.
        """
        return heapq.heappop(self.events)

    def isEmpty(self):
        """
        Checks if the events list is empty.

        Returns:
        bool
            True if the events list is empty, False otherwise.
        """

        return len(self.events) == 0

    def __repr__(self):
        """
        Returns a string representation of the sorted events list.

        Returns:
        str
            A string representation of the sorted events list.
        """
        s = ''
        sortedEvents = sorted(self.events)
        for e in sortedEvents:
            s += f'{e}\n'
        return s


class Event:
    """
    A class used to represent an Event in the simulation.

    Attributes
    ----------
    ARRIVAL : int
        A constant representing the arrival event type.
    FUEL_DEPARTURE : int
        A constant representing the fuel departure event type.
    SHOP_DEPARTURE : int
        A constant representing the shop departure event type.
    PAYMENT_DEPARTURE : int
        A constant representing the payment departure event type.
    type : int
        The type of the event.
    customer : Customer
        The customer associated with the event.
    time : float
        The time of the event.

    Methods
    -------
    __init__(type, customer, time_of_event)
        Constructs the Event class.
    __lt__(other)
        Checks if the time of this event is less than the time of another event.
    __repr__()
        Returns a string representation of the event.
    """

    ARRIVAL = 0  # constant for arrival type
    FUEL_DEPARTURE = 1  # constant for fuel departure type
    SHOP_DEPARTURE = 2  # constant for shop departure type
    PAYMENT_DEPARTURE = 3  # constant for payment departure type

    def __init__(self, type, customer: Customer, time_of_event):
        """
        Constructs the Event class.

        Parameters:
        type (int): The type of the event.
        customer (Customer): The customer associated with the event.
        time_of_event (float): The time of the event.
        """
        self.type = type
        self.customer = customer
        self.time = time_of_event

    def __lt__(self, other):
        """
        Checks if the time of this event is less than the time of another event.

        Parameters:
        other (Event): Another event to compare with.

        Returns:
        bool: True if the time of this event is less than the time of the other event, False otherwise.
        """

        return self.time < other.time

    def __repr__(self):
        """
        Returns a string representation of the event.

        Returns:
        str: A string representation of the event.
        """

        s = ("Arrival", "Fuel Departure", "Shop Departure", "Payment Departure")
        return f"customer {self.customer.cust_id} has event type {s[self.type]} at time {self.time}"


class Queue:
    """
    A class used to represent a Queue in the simulation.

    Attributes
    ----------
    EMPTY : int
        A constant representing the empty queue status.
    NOT_EMPTY : int
        A constant representing the non-empty queue status.
    customers_in_queue : list
        A list of customers in the queue.
    S : int
        The total service time for all customers in the queue.

    Methods
    -------
    __init__()
        Constructs the Queue class.
    join_queue(customer)
        Adds a customer to the queue.
    leave_queue(customer)
        Removes a customer from the queue.
    get_queue_status()
        Returns the status of the queue.
    __len__()
        Returns the length of the queue.
    """

    # type of queue status
    EMPTY = 4  # constant for queue is empty
    NOT_EMPTY = 5  # constant for queue is not empty

    def __init__(self):
        """
        Constructs the Queue class with an empty customers list and zero total service time.
        """
        self.customers_in_queue = []
        self.S = 0

    def join_queue(self, customer: Customer):
        """
        Adds a customer to the queue.

        Parameters:
        customer (Customer): The customer to be added to the queue.
        """

        self.customers_in_queue.append(customer)

    def leave_queue(self, customer: Customer):
        """
        Removes a customer from the queue.

        Parameters:
        customer (Customer): The customer to be removed from the queue.
        """
        self.customers_in_queue.remove(customer)

    def get_queue_status(self):
        """
        Returns the status of the queue.

        Returns:
        int: The status of the queue. Returns EMPTY if the queue is empty, NOT_EMPTY otherwise.
        """

        return Queue.EMPTY if len(self.customers_in_queue) == 0 else Queue.NOT_EMPTY

    def __len__(self):
        """
        Returns the length of the queue.

        Returns:
        int: The length of the queue.
        """
        return len(self.customers_in_queue)


class Server:
    """
    A class used to represent a Server in the simulation.

    Attributes
    ----------
    IDLE : int
        A constant representing the idle server status.
    BUSY : int
        A constant representing the busy server status.
    server_id : str
        The identifier of the server.
    status : int
        The status of the server.
    current_customer : Customer
        The current customer being served by the server.

    Methods
    -------
    __init__(server_id)
        Constructs the Server class.
    customer_arrive(customer)
        Sets the server status to BUSY and assigns a customer to the server.
    customer_leave()
        Sets the server status to IDLE and removes the current customer from the server.
    """

    IDLE = 0
    BUSY = 1

    def __init__(self, server_id):
        """
        Constructs the Server class with a given server_id, sets the status to IDLE and current_customer to None.

        Parameters:
        server_id (str): The identifier of the server.
        """

        # cashier and fuel pumps are instances of the server class
        # cashier id will look like C1, C2, C3
        # fuel pump id will look like 0,1,2,3
        self.server_id = server_id
        self.status = Server.IDLE
        self.current_customer = None

    def customer_arrive(self, customer: Customer):
        """
        Sets the server status to BUSY and assigns a customer to the server.

        Parameters:
        customer (Customer): The customer to be served by the server.
        """
        self.current_customer = customer
        self.status = Server.BUSY

    def customer_leave(self):
        """
        Sets the server status to IDLE and removes the current customer from the server.
        """
        self.status = Server.IDLE
        self.current_customer = None


class Simulation:
    """
    A class used to represent a Simulation.

    Attributes
    ----------
    alphas : list
        A list of alpha parameters for the gamma distributions.
    betas : list
        A list of beta parameters for the gamma distributions.
    poisson_mean : float
        The mean parameter for the Poisson distribution.
    """

    def __init__(self,
                 alphas,
                 betas, mu
                 ):
        """
        Constructs the Simulation class with given alphas, betas, and mu.

        Parameters:
        alphas (list): A list of alpha parameters for the gamma distributions.
        betas (list): A list of beta parameters for the gamma distributions.
        mu (float): The mean parameter for the Poisson distribution.
        """


        self.alphas = alphas
        self.betas = betas
        self.poisson_mean = mu
        self.setup_simulation()

    def setup_simulation(self):
        """
        Sets up the simulation by initializing distributions, resetting simulation time and queues, 
        resetting counters and measurements, initializing servers, and resetting statistics.

        """

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
        self.number_of_customers_served = 0

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
        """
        Sets up the customer's data.

        Parameters:
        customer (Customer): The customer whose data is to be set up.

        Returns:
        Customer: The customer with the set up data.
        """

        temp_customer = customer  # simply another pointer to the same instance of the customer class instance

        temp_customer.parking_preference = np.random.choice([Customer.NO_PREFERENCE, Customer.LEFT, Customer.RIGHT],
                                                            p=[0.828978622327791, 0.09501187648456057,
                                                               0.07600950118764846])

        temp_customer.wants_to_shop = np.random.choice([True, False], p=[0.22327790973871733, 0.7767220902612827])

        return temp_customer

    def handle_no_preference(self):
        """
        Handles the scenario when the customer has no preference for the fuel pump.

        Returns:
        int: The id of the fuel pump if one is available, -1 otherwise.
        """


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
        """
        Handles the scenario when the customer has a preference for the left fuel pump.

        Returns:
        int: The id of the fuel pump if one is available, -1 otherwise.
        """


        # check pumps 3 and 4
        if self.pump_stations[3].status == Server.IDLE:
            if self.pump_stations[2].status == Server.IDLE:
                return 2
            else:
                return 3
        else:
            return -1  # cannot assign the customer to a pump

    def handle_right_preference(self):
        """
        Handles the scenario when the customer has a preference for the right fuel pump.

        Returns:
        int: The id of the fuel pump if one is available, -1 otherwise.
        """


        # check pumps 1 and 2
        if self.pump_stations[1].status == Server.IDLE:
            if self.pump_stations[0].status == Server.IDLE:
                return 0
            else:
                return 1
        else:
            return -1  # cannot assign the customer to a pump

    def handle_preferences_scenario_three(self, cust_preference):
        """
        Handles the scenario when the customer has a specific preference for the fuel pump.

        Parameters:
        cust_preference (int): The customer's preference for the fuel pump.

        Returns:
        int: The id of the fuel pump if one is available, -1 otherwise.
        """



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
        """
        Runs the base simulation of the gas station using fitted distributions for the interarrival times,
        fueling times, shopping times, and payment times.

        The simulation follows the process of a customer arriving at the gas station, choosing a fuel pump,
        possibly going to the shop, and then paying at the cashier. The simulation takes into account the
        customer's preferences for fuel pumps and whether they want to shop or not.

        The simulation also handles different events such as the arrival of a customer, the departure of a
        customer from the fuel pump, the departure of a customer from the shop, and the departure of a
        customer from the payment queue.

        The simulation continues until there are no more events in the future event scheduler (FES).

        The results of the simulation are returned as a dictionary containing the average waiting times,
        queue lengths, total time spent in the system, and the number of customers served.

        Returns:
        dict: A dictionary containing the results of the simulation. The keys of the dictionary are the
        names of the metrics and the values are the calculated values for these metrics.
        """
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
                continue

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
                self.number_of_customers_served += 1
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

        results["Number of customers served"] = self.number_of_customers_served

        return results

    # -----------------------------------------Base simulation (with actual dataset)------------------------------------------#
    def base_simulation_empirical_data(self):
        """
        Runs the base simulation of the gas station using empirical data for the interarrival times,
        fueling times, shopping times, and payment times.

        The simulation follows the process of a customer arriving at the gas station, choosing a fuel pump,
        possibly going to the shop, and then paying at the cashier. The simulation takes into account the
        customer's preferences for fuel pumps and whether they want to shop or not.

        The simulation also handles different events such as the arrival of a customer, the departure of a
        customer from the fuel pump, the departure of a customer from the shop, and the departure of a
        customer from the payment queue.

        The simulation continues until there are no more events in the future event scheduler (FES).

        The results of the simulation are returned as a dictionary containing the average waiting times,
        queue lengths, total time spent in the system, and the number of customers served.

        Returns:
        dict: A dictionary containing the results of the simulation. The keys of the dictionary are the
        names of the metrics and the values are the calculated values for these metrics.
        """
        # Loads the data
        data = pd.read_excel("assignments/three/gasstationdata33.xlsx")

        # for each row in the dataset, we create a customer instance
        # populate the customer instance with the data from the dataset
        for index, row in data.iterrows():
            opening_time = pd.to_datetime('2024-02-01 06:00:00')
            arrival_time = (row["Arrival Time"] - opening_time).total_seconds()
            self.customer_id += 1
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
                self.number_of_customers_served += 1
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

        results["Number of customers served"] = self.number_of_customers_served

        return results

    # -----------------------------------------Simulation without the shop------------------------------------------------------#
    def simulation_no_shop(self):
        """
        Runs the simulation of the gas station without a shop. In this scenario, 
        each fuel pump has its own payment terminal.

        The simulation follows the process of a customer arriving at the gas station, choosing a fuel pump, 
        and then paying at the pump's terminal. The simulation takes into account the customer's preferences 
        for fuel pumps.

        The simulation also handles different events such as the arrival of a customer, 
        the departure of a customer from the fuel pump, and the departure of a customer from the payment terminal.

        The simulation continues until there are no more events in the future event scheduler (FES).

        The results of the simulation are returned as a dictionary containing the average waiting times, 
        queue lengths, total time spent in the system, and the number of customers served.

        Returns:
        dict: A dictionary containing the results of the simulation.The keys of the dictionary are the 
        names of the metrics and the values are the calculated values for these metrics.
        """

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
                self.number_of_customers_served += 1
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
        results["Number of customers served"] = self.number_of_customers_served

        return results

    # ----------------------------------------Simulation with four lines of pumps------------------------------------------------------#
    def simulation_four_lines_of_pumps(self):
        """
        Runs the simulation of a gas station with four lines of fuel pumps.

        The simulation follows the process of a customer arriving at the gas station, choosing a fuel pump,
        possibly going to the shop, and then paying at the cashier. The simulation takes into account the
        customer's preferences for fuel pumps and whether they want to shop or not.

        The simulation also handles different events such as the arrival of a customer, the departure of a
        customer from the fuel pump, the departure of a customer from the shop, and the departure of a
        customer from the payment queue.

        The simulation continues until there are no more events in the future event scheduler (FES).

        The results of the simulation are returned as a dictionary containing the average waiting times,
        queue lengths, total time spent in the system, and the number of customers served.

        Returns:
        dict: A dictionary containing the results of the simulation. The keys of the dictionary are the
        names of the metrics and the values are the calculated values for these metrics.
        """


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
                self.number_of_customers_served += 1
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
        results["Number of customers served"] = self.number_of_customers_served

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
    """
    The main function to run the gas station simulation.

    This function sets up the parameters for the simulation, including the alpha and beta values 
    for the gamma distributions, the mean for the Poisson distribution, and the number of simulation runs. 
    It then creates an instance of the Simulation class and runs four different types of simulations: 
    a base simulation with empirical data, a base simulation with fitted distributions,
    a simulation without a shop, and a simulation with four lines of fuel pumps.

    The results of each simulation run are stored in a list and written to a text file named 'results.txt'. 
    The mean, standard deviation, and confidence interval of the results are also calculated and written 
    to the file.

    Note: This function uses a random seed for reproducibility.
    """


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
                simulation_results[i] = sim.base_simulation_empirical_data()
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

        with open('results.txt', 'a') as f:
            f.write(f"\n-------------------Results for {sim_names[sim_name]}-------------------\n")
            keys = list(simulation_results[0].keys())
            simulation_results = [pd.DataFrame(simulation_results[i], index=[f"Results for Runtime: {i}"]).T for i in
                                  range(n_runs)]
            simulation_results = np.array(simulation_results)
            mean = simulation_results.mean(axis=0)
            std = simulation_results.std(axis=0, ddof=1)

            ci = confidence_interval(simulation_results)

            for i in range(len(mean)):
                result_line = f"{keys[i]}: has mean {mean[i][0]} (+- {ci[i][0]}) and std {std[i][0]} \n"
                f.write(result_line)

            f.write("\n")


def plotting_base_simulation_results(n_runs =1, n_sims = 1000):
    """
    This function plots the evolution of the mean and confidence interval of the base simulation results over time
    it stores the plots in the graphs/base_fitted folder
    args:
    n_runs: int, the number of runs per simulation
    n_sims: int, the number of simulations to perform
    """
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677
    n_sims +=1

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)

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

    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)





    for i, col in enumerate(df.columns):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax.fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax.set_xlabel("Number of simulations".title(), fontsize=16)
        ax.set_ylabel(f"{col}", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.legend()
        # fig.suptitle(f"{col.title()}, Base Simulation (with Empirical Data) \nover {n_sims-1} simulations".title(), fontsize=20)
        fig.savefig(f"assignments/three/graphs/base_fitted/base_simulation_results_{col}.png")

    plt.show()


def plotting_det_simulation_results(n_runs =1, n_sims = 1000):
    """
    This function plots the evolution of the mean and confidence interval of the base simulation results over time
    it stores the plots in the graphs/base_empirical folder

    args:
    n_runs: int, the number of runs per simulation
    n_sims: int, the number of simulations to perform
    """
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_sims +=1

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)
    # simulation_results = pd.DataFrame(
    #     columns=["Waiting time\nFuel station (s)", 
                    #"Queue length\nFuel station (Customers)", "Queue length\nshop (Customers)", 
                    #"Waiting time\nPayment queue (s)", "Queue length\nPayment queue (Customers)", 
                    #"Total time\nspent in the system (s)", "Number of customers served"]
    # )

    means = []
    variances = []
    ci_low = []
    ci_high = []
    out_means = []

    simulation_results = []

    for u in tqdm(range(1, n_sims), desc="Simulation with empirical data"):

        for i in range(n_runs):
            simulation_results.append(sim.base_simulation_empirical_data())
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


    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)

    for i, col in enumerate(df.columns):
        fig, ax = plt.subplots(1, 1, figsize=(10.2, 6))
        ax.plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax.fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax.set_xlabel("Number of simulations".title(), fontsize=16)
        ax.set_ylabel(f"{col}", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.legend()
        fig.savefig(f"assignments/three/graphs/base_empirical/det_simulation_results_{col}.png")

    plt.show()

def plotting_no_shop_simulation_results(n_runs =1, n_sims = 1000):
    """
    This function plots the evolution of the mean and confidence interval of the base simulation results over time
    it stores the plots in the graphs/no_shop folder

    args:
    n_runs: int, the number of runs per simulation
    n_sims: int, the number of simulations to perform
    """
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_sims +=1

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)

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
        out_means.append(temp_mean.mean(axis=1))
        ci_low.append(temp_mean.mean(axis=1) - 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))
        ci_high.append(temp_mean.mean(axis=1) + 1.96 * np.sqrt(temp_var.mean(axis=1)) / np.sqrt(u * n_runs))


    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)





    for i, col in enumerate(df.columns):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax.fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax.set_xlabel("Number of simulations".title(), fontsize=16)
        ax.set_ylabel(f"{col}", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.legend()
        fig.savefig(f"assignments/three/graphs/no_shop/no_shop_simulation_results_{col}.png")

    plt.show()

def plotting_four_pumps_simulation_results(n_runs =1, n_sims = 1000):
    """
    This function plots the evolution of the mean and confidence interval of the base simulation results over time
    it saves the plots in the graphs/four_lines folder
    args:
    n_runs: int, the number of runs per simulation
    n_sims: int, the number of simulations to perform
    """
    alphas = [3.740741878984356, 0.9896321424751765, 64.16085452170083, 1.044611732553164]
    betas = [0.014062799908188449, 0.014062799908188449, 1.426471221627218, 0.007307573018982521]

    # for poission distribution of service time payment
    mu = 45.6603325415677

    n_sims +=1

    # Perform n_run simulations

    sim = Simulation(alphas, betas, mu=mu)

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


    ci_low_over_times_df = pd.concat(ci_low, axis=1)
    ci_high_over_times_df = pd.concat(ci_high, axis=1)
    out_means = pd.concat(out_means, axis=1)




    for i, col in enumerate(df.columns):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(out_means.loc[col], label=f"Mean {col}", color='orange')
        ax.fill_between(list(range(n_sims - 1)), ci_low_over_times_df.loc[col], ci_high_over_times_df.loc[col],
                           alpha=0.5, label=f"CI {col}")
        ax.set_xlabel("Number of simulations".title(), fontsize=16)
        ax.set_ylabel(f"{col}", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.legend()
        fig.savefig(f"assignments/three/graphs/four_lines/four_pumps_simulation_results_{col}.png")


    plt.show()


if __name__ == "__main__":
    main()
    # plotting_base_simulation_results()
    # plotting_det_simulation_results()
    # plotting_no_shop_simulation_results()
    # plotting_four_pumps_simulation_results()
