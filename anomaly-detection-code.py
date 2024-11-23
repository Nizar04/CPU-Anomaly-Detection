import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pyflux as pf
from sklearn.metrics import mean_squared_error

class CPUAnomalyDetector:
    def __init__(self, ar_order=11, ma_order=11, integration_order=0):
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.integration_order = integration_order
        self.model = None
        
    def load_data(self, train_path, test_path=None):
        train_data = pd.read_csv(train_path, 
                                parse_dates=['datetime'], 
                                infer_datetime_format=True)
        
        test_data = None
        if test_path:
            test_data = pd.read_csv(test_path, 
                                  parse_dates=['datetime'], 
                                  infer_datetime_format=True)
            
        return train_data, test_data
    
    def visualize_data(self, data, title="CPU Utilization"):
        plt.figure(figsize=(20, 8))
        plt.plot(data['datetime'], data['cpu'], color='black')
        plt.ylabel('CPU %')
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    def fit_model(self, data):
        self.model = pf.ARIMA(data=data,
                             ar=self.ar_order,
                             ma=self.ma_order,
                             integ=self.integration_order,
                             target='cpu')
        
        self.model.fit("M-H")
        return self.model
    
    def plot_model_fit(self):
        if self.model is None:
            raise Exception("Model not fitted yet!")
        
        self.model.plot_fit(figsize=(20, 8))
        plt.grid(True)
        plt.show()
    
    def predict_and_detect_anomalies(self, horizon=60, past_values=100):
        if self.model is None:
            raise Exception("Model not fitted yet!")
        
        self.model.plot_predict(h=horizon,
                              past_values=past_values,
                              figsize=(20, 8))
        plt.grid(True)
        plt.show()
    
    def calculate_anomaly_scores(self, actual, predicted, confidence_interval):
        lower_bound = predicted - confidence_interval
        upper_bound = predicted + confidence_interval
        
        anomaly_scores = np.zeros(len(actual))
        anomaly_scores[actual < lower_bound] = 1
        anomaly_scores[actual > upper_bound] = 1
        
        return anomaly_scores
    
    def evaluate_model(self, test_data):
        predictions = self.model.predict(h=len(test_data))
        mse = mean_squared_error(test_data['cpu'], predictions)
        rmse = np.sqrt(mse)
        return {
            'mse': mse,
            'rmse': rmse
        }

def main():
    detector = CPUAnomalyDetector()
    
    train_data, test_data = detector.load_data('cpu-train-a.csv', 
                                             'cpu-test-a.csv')
    
    print("Visualizing training data...")
    detector.visualize_data(train_data)
    
    print("Fitting ARIMA model...")
    model = detector.fit_model(train_data)
    
    print("Plotting model fit...")
    detector.plot_model_fit()
    
    print("Detecting anomalies...")
    detector.predict_and_detect_anomalies()
    
    if test_data is not None:
        print("Evaluating model performance...")
        metrics = detector.evaluate_model(test_data)
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
    
    print("Processing second dataset...")
    train_data_b, test_data_b = detector.load_data('cpu-train-b.csv',
                                                  'cpu-test-b.csv')
    
    detector.visualize_data(train_data_b, "CPU Utilization - Dataset B")
    detector.fit_model(train_data_b)
    detector.predict_and_detect_anomalies()

if __name__ == "__main__":
    main()
