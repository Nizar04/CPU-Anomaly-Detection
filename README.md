# CPU Utilization Anomaly Detection

A machine learning project that uses ARIMA (Autoregressive Integrated Moving Average) modeling to detect anomalies in CPU utilization patterns. The system analyzes time-series data to identify unusual spikes or patterns that could indicate system issues or security concerns.

## Features

- Time series analysis using ARIMA modeling
- Real-time anomaly detection capabilities
- Visual representation of CPU utilization patterns
- Confidence interval-based anomaly flagging
- Historical data analysis and forecasting

## Requirements

```
pandas>=1.2.0
numpy>=1.19.2
matplotlib>=3.3.0
pyflux>=0.4.17
datetime>=4.3
scikit-learn>=0.24.0
```

## Dataset Structure

The project expects CPU utilization data in CSV format with:
- `datetime`: Timestamp of the measurement
- `cpu`: CPU utilization percentage

Two sets of data are used:
- Training data: For model building
- Test data: For anomaly detection

## Model Details

The ARIMA model is configured with:
- Autoregressive (AR) order: 11
- Moving Average (MA) order: 11
- Integration order: 0

These parameters can be tuned based on your specific use case.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the anomaly detection:
```bash
python cpu_anomaly_detection.py
```

## Visualizations

The project generates three types of plots:
1. Raw CPU utilization
2. ARIMA model fit
3. Anomaly detection with confidence intervals

## Results

The model identifies anomalies by:
- Comparing actual values against predicted ranges
- Using confidence intervals for threshold determination
- Flagging points that fall outside expected bounds

## Future Improvements

- Add real-time monitoring capabilities
- Implement automated alerting system
- Add support for multiple metric types
- Include adaptive threshold adjustment
- Add API endpoints for integration

## License

MIT License
