import pandas as pd


class WaterSource:
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float

    def __init__(self, ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
        self.ph = ph
        self.hardness = hardness
        self.solids =  solids
        self.chloramines = chloramines
        self.sulfate = sulfate
        self.conductivity = conductivity
        self.organic_carbon = organic_carbon
        self.trihalomethanes = trihalomethanes
        self.turbidity = turbidity

    
    def mount(self):
        data = {
            'ph': self.ph,
            'Hardness': self.hardness,
            'Solids': self.solids,
            'Chloramines': self.chloramines,
            'Sulfate': self.sulfate,
            'Conductivity': self.conductivity,
            'Organic_carbon': self.organic_carbon,
            'Trihalomethanes': self.trihalomethanes,
            'Turbidity': self.turbidity
        }

        return pd.Series(data).values.reshape(1, -1)  # Single shape 1 row N col

