# Assignment 1 - README
by [Maxwell Litsios](mailto:m.l.h.litsios@student.tue.nl), [Mir](mailto:), and [Pranav](mailto:)

## Introduction
The goal of the assignment was to simulate different queueing systems and compare their performance. The systems we simulated were:

- Regular Queue: where groups of customers were placed in a single queue
- Split Singles Queue: a queue where single customers were placed in a separate queue and used to fill empty boats when 
- Complex Queue: TODO

## Code
We used a discrete event simulation to simulate the different queueing systems. The simulation was implemented in Python.
- The Boat Class was defined in the location ```aux_functions/Boat.py```
- The Queue Class was defined in the location ```aux_functions/Queue.py```
- The different queueing systems were implemented in the location ```QueueingSystems.py```
- The simulation of the different queueing systems was implemented in the location ```simulation.py```
- These were all evaluated in the location ```evaluation.py```