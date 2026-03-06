def scale_between(a, a_min, a_max):
    # input between min and max
    # return scaled value between -1 and 1  
    return (2 *(a - a_min)/(a_max-a_min)) - 1

def scale_between_inv(a, a_min, a_max):
    # input -1 to 1 
    # return scaled value between -min and max  
    return (a + 1)*(a_max-a_min)*0.5 + a_min