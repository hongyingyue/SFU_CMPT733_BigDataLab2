
## Dataset utilization
- We query CSI 500 Index composite stock data and company fundamentals from 2016
to 2023. The composite stocks comprise 500 small to mid-cap stocks traded on the
Shanghai and Shenzhen stock exchanges.
- We construct a dataset that includes the monthly return of each stock calculated from price data, along with valuation metrics, profitability metrics, momentum indicators etc. Each metrics category will involve multiple indicators, computed based on raw financial data. 

## Feature Preprocessing
We got stock factor data with monthly frequency from 2016 to 2023. The feature preprocessing mainly includes handling missing values, winsorizing, standardization, and industry neutralization processing(to make the feature value align within its own industry). The goal is to eliminate factor outliers, adjust the scale of the data, and remove the influence related to the market value factor.

## Data Label
We label the data based on its monthly return. Initially, we sort the monthly stock returns. Stocks in the top 30% are labeled as 1, while those in the bottom 30% are labeled as 0, distinguishing between good and poor stocks. Stocks with returns outside of the top 30% and bottom 30% are excluded from the dataset. This strategy enhances the distinction between high-return and low-return stock data points, ultimately improving the performance of final simulated investment returns.

## Feature selection
We employ a random forest classifier to automatically identify pertinent features, focusing on their significance in predicting the target variable. A selection threshold is set at an importance level greater than 0.05.

## Data Modelling
For model training and evaluation, we utilize Naive Bayes and XGBoost separately. Training data spans from 2016 to 2019, while testing data consists of records from 2020 to 2023.

## Evaluation
Based on predicted probabilities of good performance, stocks are categorized into five groups. As investors, we allocate investments to these groups at the start of each month and transition to the updated stock list within the corresponding group the following month. This process repeats monthly throughout the testing period from 2019 to 2023. Subsequently, we calculate various investment metrics by group, including cumulative return, annualized return, maximum drawdown, and Sharpe ratio, to evaluate investment performance comparing with the benchmark CSI 500. 