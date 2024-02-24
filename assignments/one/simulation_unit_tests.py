import unittest
import numpy as np
from simulation import simulate_boat_line

class TestSimulation(unittest.TestCase):

    def test_simulate_boat_line_BASE(self):
        # Test case 1: BASE queue type
        q_type = "BASE"
        len_q = 5
        max_group_size = 4
        min_group_size = 1
        boat_capacity = 10
        max_time_interval = 5
        min_queue_size = 1
        max_queue_size = 3

        result = simulate_boat_line(q_type, len_q, max_group_size, min_group_size, boat_capacity, max_time_interval,
                                    min_queue_size, max_queue_size)

        assert len(result) == 7
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert result[3] >= 0
        assert result[4] >= 0
        assert result[5] >= 0
        assert result[6] >= 0

    def test_simulate_boat_line_SINGLES(self):
        # Test case 2: SINGLES queue type
        q_type = "SINGLES"
        len_q = 5
        max_group_size = 4
        min_group_size = 1
        boat_capacity = 10
        max_time_interval = 5
        min_queue_size = 1
        max_queue_size = 3

        result = simulate_boat_line(q_type, len_q, max_group_size, min_group_size, boat_capacity, max_time_interval,
                                    min_queue_size, max_queue_size)

        assert len(result) == 7
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] >= 0
        assert result[3] >= 0
        assert np.isnan(result[4])
        assert result[5] >= 0
        assert result[6] >= 0
    
    def test_simulate_boat_line_DYNAMIC(self):
        # Test case 3: DYNAMIC queue type
        q_type = "DYNAMIC"
        len_q = 5
        max_group_size = 4
        min_group_size = 1
        boat_capacity = 10
        max_time_interval = 5
        min_queue_size = 1
        max_queue_size = 3

        result = simulate_boat_line(q_type, len_q, max_group_size, min_group_size, boat_capacity, max_time_interval,
                                    min_queue_size, max_queue_size)

        assert len(result) == 7
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert result[3] >= 0
        assert result[4] >= 0
        assert result[5] >= 0
        assert result[6] >= 0

if __name__ == "__main__":
    unittest.main()