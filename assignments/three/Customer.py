import numpy as np

class Customer:
    
    # ACCEPTED_SPOT = ["LEFT", "RIGHT",  "EITHER"]
    
    def __init__(self, F, p_s, P, S, spot):
        self.fueling_time = F
        self.buy_in_shop = np.random.choice(['BUY', 'PAY DIRECTLY'], p=[p_s, 1 - p_s])
        self.payment_time = P
        self.shopping_time = S
        self.preferred_spot = spot

    
        
    

         
        