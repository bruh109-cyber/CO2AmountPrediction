import pandas as pd
import numpy as np
from sklearn import linear_model

class VehicleEmissionPredictor:
    def __init__(self):
        # Manually structuring the provided dataset matrix into an explicit dictionary
        self.dataset_matrix = {
            "CarModel": ["Toyota Aygo", "Mitsubishi Space Star", "Skoda Citigo", "Fiat 500", "Mini Cooper", 
                         "VW Up!", "Skoda Fabia", "Mercedes A-Class", "Ford Fiesta", "Audi A1", "Hyundai I20", 
                         "Suzuki Swift", "Ford Fiesta", "Honda Civic", "Hundai I30", "Opel Astra", "BMW 1", 
                         "Mazda 3", "Skoda Rapid", "Ford Focus", "Ford Mondeo", "Opel Insignia", "Mercedes C-Class", 
                         "Skoda Octavia", "Volvo S60", "Mercedes CLA", "Audi A4", "Audi A6", "Volvo V70", "BMW 5", 
                         "Mercedes E-Class", "Volvo XC70", "Ford B-Max", "BMW 2", "Opel Zafira", "Mercedes SLK"],
            "Volume": [1000, 1200, 1000, 900, 1500, 1000, 1400, 1500, 1500, 1600, 1100, 1300, 1000, 1600, 1600, 
                       1600, 1600, 2200, 1600, 2000, 1600, 2000, 2100, 1600, 2000, 1500, 2000, 2000, 1600, 2000, 
                       2100, 2000, 1600, 1600, 1600, 2500],
            "Weight": [790, 1160, 929, 865, 1140, 929, 1109, 1365, 1112, 1150, 980, 990, 1112, 1252, 1326, 1330, 
                       1365, 1280, 1119, 1328, 1584, 1428, 1365, 1415, 1415, 1465, 1490, 1725, 1523, 1705, 1605, 
                       1746, 1235, 1390, 1405, 1395],
            "CO2": [99, 95, 95, 90, 105, 105, 90, 92, 98, 99, 99, 101, 99, 94, 97, 97, 99, 104, 104, 105, 94, 
                    99, 99, 99, 99, 102, 104, 114, 109, 114, 115, 117, 104, 108, 109, 120]
        }
        
        self.df = pd.DataFrame(self.dataset_matrix)
        
    def train_regression_engine(self):
        
        self.X = self.df[['Weight', 'Volume']]
        self.y = self.df['CO2']
        
        self.regr = linear_model.LinearRegression()
        self.regr.fit(self.X, self.y)
        
        print("Multiple Regression Model Compiled Successfully.")
        print(f"Extracted Intercept (b) : {self.regr.intercept_:.6f}")
        print(f"Extracted Coefficients : Weight={self.regr.coef_[0]:.8f}, Volume={self.regr.coef_[1]:.8f}\n")
        
    def execute_prediction(self, target_weight, target_volume):
        input_data = np.array([[target_weight, target_volume]])
        prediction = self.regr.predict(input_data)
        return prediction[0]
    
    
predictor = VehicleEmissionPredictor()
predictor.train_regression_engine()

w_a, v_a = 2300, 1300
co2_a = predictor.execute_prediction(w_a, v_a)
print(f" Target A Prediction: A car weighing {w_a}kg with a {v_a}cc engine emits approx: {co2_a:.4f} g/km")


w_b, v_b = 3300, 1300
co2_b = predictor.execute_prediction(w_b, v_b)
print(f" Target B Prediction: A car weighing {w_b}kg with a {v_b}cc engine emits approx: {co2_b:.4f} g/km")
        
        




































