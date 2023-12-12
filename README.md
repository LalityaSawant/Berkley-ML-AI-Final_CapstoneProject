# Repository: Berkeley-ML-AI-Final_CapstoneProject

## This repository contains a project completed for Berkeley AI ML course
#####                                                  Author - Lalitya Sawant
### Project notebook and presentation quick links:
- [CapstoneProject_Berkley.ipynb](https://github.com/LalityaSawant/Berkley-ML-AI-Final_CapstoneProject/blob/main/Final_CapstoneProject_Berkley.ipynb)

- [Project Presentation - PDF Format](https://github.com/LalityaSawant/Berkley-ML-AI-Final_CapstoneProject/blob/main/BerkeleyAIMLCapstoneProject-SalesForecasting.pdf)
- [Project Presentation - Slides Format](https://github.com/LalityaSawant/Berkley-ML-AI-Final_CapstoneProject/blob/main/BerkeleyAIMLCapstoneProject-SalesForecasting.pptx)

### Project Title:  **Sales Forecasting for Walmart dataset**

#### Executive summary:
In this application, we'll delve into a Kaggle dataset containing sales information from Walmart.

#### Rationale
**Why should anyone care about this question?**

Sales forecasting poses a common challenge for many organizations, resulting in potential revenue loss and diminished profits. 

#### Research Question
**What are you trying to answer?**

Understanding sales trends enables organizations to strategically order the necessary quantities of goods across various departments and locations.

#### Business Benefits:
By leveraging AI/ML models to predict sales forecasts, organizations can achieve:

1. **Optimized Inventory Management:**
   - Accurate predictions enable organizations to maintain optimal inventory levels, reducing excess stock or shortages.

2. **Improved Supply Chain Efficiency:**
   - Anticipating sales trends facilitates a more efficient and responsive supply chain, ensuring timely product availability.

3. **Enhanced Financial Planning:**
   - Reliable sales forecasts aid in developing robust financial plans, optimizing budget allocation and resource utilization.

4. **Maximized Revenue Generation:**
   - Strategic decision-making based on accurate sales predictions can lead to increased revenue and improved profitability.

5. **Customer Satisfaction:**
   - Meeting demand consistently enhances customer satisfaction by reducing out-of-stock instances and ensuring a positive shopping experience.

6. **Data-Driven Decision Making:**
   - Leveraging AI/ML models empowers organizations with data-driven insights, fostering informed decision-making across various business aspects.

7. **Competitive Edge:**
   - Proactive sales forecasting provides a competitive advantage by enabling organizations to stay ahead in the market with timely and precise responses to consumer demands.
  
#### Data Sources
**What data will you use to answer your question?**

As mentioned earlier we will be using the Walmart sales dataset from [Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/download?datasetVersionNumber=1)

#### Methodology
**What methods are you using to answer the question?**

To achieve this, a comprehensive analysis of the dataset is imperative. It includes below steps:

**Data Cleaning:** A meticulous review of the dataset to identify and rectify any inconsistencies or inaccuracies.

**Outlier Detection:** Identifying potential outliers and evaluating whether they should be excluded from the analysis.

**Bias Assessment:** Scrutinizing the dataset for any biases and implementing appropriate measures to address them.

**Data Transformation:** Converting textual/boolean data into a format understandable by the predictive model.

Once these preprocessing steps are accomplished, the subsequent task is to distribute the data in the training and testing set and then apply different algorithms to reach an accurate prediction of the forecasts.

#### Results
**What did your research find?**

#### **Some insights on Data:**
Original shape of data: (421570, 16)

Int64Index: 421570 entries, 0 to 421569
Data columns (total 16 columns):
| #   | Column        | Non-Null Count | Dtype    |
| --- | ------------- | -------------- | -------- |
| 0   | Store         | 421570         | int64    |
| 1   | Dept          | 421570         | int64    |
| 2   | Date          | 421570         | object   |
| 3   | Weekly_Sales  | 421570         | float64  |
| 4   | IsHoliday     | 421570         | bool     |
| 5   | Temperature   | 421570         | float64  |
| 6   | Fuel_Price    | 421570         | float64  |
| 7   | MarkDown1     | 150681         | float64  |
| 8   | MarkDown2     | 111248         | float64  |
| 9   | MarkDown3     | 137091         | float64  |
| 10  | MarkDown4     | 134967         | float64  |
| 11  | MarkDown5     | 151432         | float64  |
| 12  | CPI           | 421570         | float64  |
| 13  | Unemployment  | 421570         | float64  |
| 14  | Type          | 421570         | object   |
| 15  | Size          | 421570         | int64    |
dtypes: bool(1), float64(10), int64(3), object(2)
memory usage: 51.9+ MB

Shape of the data after data processing/cleanup: (420212, 11)


#### **Key Findings from Data Exploration:**

1. **Data Compilation:**
   - The data was initially provided in 4 separate CSV files.
   - We merged the store, features, and train CSVs to create a comprehensive dataset.

2. **Data Quality Enhancement:**
   - Identified and addressed null values in markdown columns by removing those columns.
   - Ensured better data quality for subsequent analysis.

3. **Sales Data Anomalies:**
   - Detected and addressed rows with negative sales values, likely data anomalies.
   - Removed such instances, maintaining the integrity of the dataset.

4. **Key Attributes Impacting Sales:**
   - Explored attributes like holidays, fuel price, unemployment, and temperature.

5. **Holiday Analysis:**
   - Categorized holidays into four types: Labor Day, Super Bowl, Thanksgiving, and Christmas.
   - Thanksgiving showed a strong positive impact on sales, while Super Bowl had a moderate impact.
   - Labor Day and Christmas did not exhibit a significant positive impact on sales.

6. **Other Sales Influencers:**
   - Explored factors beyond holidays, finding no clear positive or negative impact on sales.

7. **Yearly Sales Trend:**
   - Observed a consistent pattern of increased sales at the end of each year.

These insights provide a foundational understanding for our further analysis and decision-making processes.


#### **Further Data Processing Steps:**

1. **Feature Transformation:**
   - Addressed remaining categorical and ordinal fields requiring transformation for modeling.

2. **Correlation Analysis:**
   - Explored the correlation of features with weekly sales.


#### **Feature Selection:**
Upon executing Ridge Regression for feature selection, we obtained the following correlation coefficient data:
| #   | Features              | Coefs           |
| --- | --------------------- | --------------- |
| 3   | Size                  | 6111.455355     |
| 1   | Dept                  | 3272.028832     |
| 9   | Type_C                | 1379.949679     |
| 5   | Month                 | 1168.625192     |
| 2   | Fuel_Price            | 701.886808      |
| 17  | Thanksgiving_True     | 341.183575      |
| 18  | Christmas_False       | 206.217420      |
| 14  | Labor_Day_False       | 94.529108       |
| 13  | Super_Bowl_True        | 80.195029       |
| 11  | IsHoliday_True         | 62.867514       |
| 10  | IsHoliday_False        | -62.867514      |
| 12  | Super_Bowl_False       | -80.195029      |
| 15  | Labor_Day_True         | -94.529108      |
| 19  | Christmas_True         | -206.217420     |
| 16  | Thanksgiving_False     | -341.183575     |
| 7   | Type_A                | -410.515532     |
| 8   | Type_B                | -427.978958     |
| 4   | Week                 | -430.756689     |
| 6   | Year                 | -663.120183     |
| 0   | Store                | -1681.637899    |


Below is the output from the permutation_importance:
| Feature           | Mean           | Standard Deviation |
| ------------------ | -------------- | ------------------ |
| Size              | 73663404.289   | 675128.606         |
| Dept              | 23966202.892   | 296850.476         |
| Store             | 5440509.582    | 179566.791         |
| Type_C            | 4518730.235    | 145167.318         |
| Month             | 884192.429     | 110500.216         |
| Week              | 677551.526     | 40811.852          |
| Type_B            | 478886.728     | 44116.877          |
| Type_A            | 334070.412     | 46556.820          |
| Fuel_Price        | 235328.534     | 51785.721          |
| Super_Bowl_True   | 26413.466      | 10746.130          |
| Super_Bowl_False  | 26413.466      | 10746.130          |
| IsHoliday_False   | 14263.435      | 4812.539           |
| IsHoliday_True    | 14263.435      | 4812.539           |

#### Time Series Analysis and Modeling

After performing time series decomposition and the augmented Dickey-Fuller test, we concluded that the data is nonstationary. Subsequent decomposition at weekly and monthly intervals revealed a repetitive pattern in the data.

To address nonstationarity, we applied difference, shift, and log algorithms. The differential data emerged as the most effective in achieving stationarity.

For the final time series model, we utilized the auto_arima algorithm, identifying the following as the optimal model for predictions:

**Best model:** ARIMA(3,0,2)(0,0,0)[1] intercept

**Total fit time:** 10.236 seconds

### Part 1 project result Visualization:
![Auto-ARIMA prediction](https://github.com/LalityaSawant/Berkley-ML-AI-Final_CapstoneProject/blob/main/Images/Auto-ARIMA-prediction.png)

#### Next steps
**What suggestions do you have for the next steps?**

The predictions from the above model exhibit a slightly lower trend than the test data. Further tuning or exploring alternative algorithms may help achieve a closer alignment between the predictions and the test data.

# Finetuning on the previous project progress

## Overview
The previous iteration of the sales forecasting model has shown a slightly lower trend than the test data. To enhance the model's performance and achieve a closer alignment with the test data, I have undertaken further steps in model tuning and exploration of alternative algorithms.

## Model Exploration and Tuning - Current Iteration Highlights:

### Exponential Smoothing
The Exponential Smoothing model was applied to the dataset, revealing promising results in terms of prediction accuracy. This method leverages a weighted average of past observations, assigning exponentially decreasing weights to older data points. The adaptability of Exponential Smoothing makes it effective in capturing trends and seasonality in time-series data.

#### Result Visualization:
![Exponential-Smoothing prediction](https://github.com/LalityaSawant/Berkley-ML-AI-Final_CapstoneProject/blob/main/Images/Exponential_smoothing-prediction.png)


### LSTM and GRU Models
Two deep learning models, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), were implemented to further explore the dataset. Both LSTM and GRU exhibited notable improvements in prediction accuracy. Long Short-Term Memory Networks are known for their ability to capture long-term dependencies in sequential data, while Gated Recurrent Units, a more efficient variant of LSTM, also demonstrated competitive performance.

#### Result Visualization:
![CNN-LSTM prediction](https://github.com/LalityaSawant/Berkley-ML-AI-Final_CapstoneProject/blob/main/Images/LSTM-prediction.png)

![CNN-GRU prediction](https://github.com/LalityaSawant/Berkley-ML-AI-Final_CapstoneProject/blob/main/Images/GRU-prediction.png)


## Recommendations / Lessons Learned for Current Iteration:

1. **Fine-Tuning Parameters:** Ongoing efforts have been made to fine-tune the parameters of the current models, especially focusing on hyperparameter tuning for LSTM and GRU.

2. **Feature Engineering:** Some initial attempts at feature engineering have been made to enhance the representation of underlying patterns in the data. Additional features are being considered for future iterations.

3. **Data Augmentation:** Data augmentation techniques have been experimented with to artificially expand the dataset, providing the models with more diverse examples for improved generalization.

4. **Evaluation Metrics:** Evaluation metrics have been reassessed and fine-tuned to ensure alignment with the specific goals of the project. Iterative testing with various combinations of features, algorithms, and hyperparameters is ongoing.


## Further steps that can be tested:

**Ensemble Modeling:** Exploration of ensemble modeling can be conducted, combining the strengths of multiple models to enhance predictive performance.


## In conclusion:
The current iteration has seen progress in the exploration, tuning, and experimentation of models.
Exponential smoothing and LSTM models performed better than the previously used ARIMA model on the Walmart sales forecasting data.

###### [Reference](https://www.kaggle.com/code/aslanahmedov/walmart-sales-forecasting/notebook#Background)
