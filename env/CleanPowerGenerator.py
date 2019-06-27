class ElectricityGenerator:
    """
    prod_profile: "radiation"/"windspeed" or "power" time series
    profile_type: indicate the type of prod_profile
    prod_function: the characteristic function of the generator
    nominal_power: the nominal power output of the generator
    """
    def __init__(self, prod_profile, profile_type, prod_function):
        self.prod_profile = prod_profile
        self.prod_function = prod_function
        self.profile_type = profile_type

    def output_power(self, time_posi):
        """
        Still have to implement the power output at specific temporal position
        """
        if self.profile_type == 'radiation' or self.profile_type == 'windspeed':
            output = self.prod_function(self.prod_profile, time_posi)
        else:
            assert self.profile_type == 'power'
            output = self.prod_profile[time_posi]
        return output
