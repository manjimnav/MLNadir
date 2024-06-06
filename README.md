# A Real-Time Machine Learning-Based Methodology for Short-Term Frequency Nadir Prediction in Low-Inertia Power Systems

This repository contains all the code needed to reproduce the experiments in the paper presented by Manuel Jes ́us Jim ́enez-Navarro, Jose Miguel Riquelme-Dominguez, Manuel Carranza-Garc ́ıa, Francisco M. Gonz ́alez-Longatt. 

> In the modern era, electricity is vital for societal advancement, driving economic growth and essential functions. However, the landscape of power systems is swiftly changing due to the integration of renewable energy sources and the decline of traditional synchronous generation, which reduces the total rotational inertia of the systems. This reduction in inertia leads to more frequent and severe frequency deviations, directly impacting power system behaviour. Therefore, there is a pressing need to anticipate frequency grid disturbances to maintain stability and prevent disruptions. A machine learning approach is proposed to address this issue, providing accurate and responsive frequency forecasting in power systems. This paper introduces a novel methodology that leverages machine learning for short-term minimum frequency prediction, emphasizing efficiency and rapid response. Results highlight the effectiveness of Decision Trees for one-hour forecasts, offering a balance of interpretability and predictive capability. Additionally, Decision Trees prove to be a practical choice for four-hour forecasts, demonstrating their versatility and efficiency. Validation was conducted using the SCADA of a Typhoon HIL real-time simulator, verifying that the proposed methodology is suitable for real-time applications.

## Prerequisites

In order to run the experimentation several dependencies must be installed. The requirements has been listed in the `requirements.txt` which can be installed as the following:

```
pip install -r requirements.txt
```

## Data

Original unprocessed data has been uploaded un two folders named: 1.InerciaGeneralizada-DisparoGenerador and 3.InerciaLocalizada-DisparoGenerador. Every folder contains a excel file for each scenario, case, subcase and results. In addition, the result of our preprocessing steps has been uploaded which is the results obtained after executing notebooks 0 and 1. 

The unprocessed and processed data can be found in the following link: https://uses0-my.sharepoint.com/:f:/g/personal/mjimenez3_us_es/EgcvgqQ8exxBoMglfD2ymTYBi2Bu2GvwwRvo4A173JfLQg?e=sUBwA4.


## Experimentation

In order to reproduce the results, the notebooks has been sortened in execution order.

* Notebooks 0 and 1 represents the preprocessing. Notebook 0 builds the dataset for every context, while notebook 1 produce the lagged version of every dataset.
* Notebook 2 launch the experimentation for every model in the comparison.
* Notebook 3 summarizes the results obtained.
* Notebook 4 and 5 imitates a real scenario producing and illustrating the obtained predictions.