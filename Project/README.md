# Integrating AI and Quantitative Analysis for Equity Investment and Portfolio Optimization

Group: HappyCNY

Members: Shung Ho (Jonathan) Au, Hongying Yue, Sitong Zhai, Qin Duan

Interested in exploring Data Science application in Quantitative Finance, we aim to achieve the following goals:
- Develop a scalable yet effective machine learning framework that can identify high-return stocks to guide investment decisions
- Eliminate emotional bias and adopt effective features and algorithm to construct investment portfolio 
- Outperform the index benchmark in terms of % return and risk (maximum loss %)

## Report & Poster
Full report please see `report.pdf`

Poster please see `poster.pdf`

## Presentation Video
[CMPT 733 HappyCNY](https://youtu.be/YksgMYgVoBM)

## Data
We use JQData API, an open-source comprehensive quantitative financial data platform, which offers access to a wide array of financial information, enabling users to review and compute various financial metrics. Paid account is needed to download data.

Official repository: [jqdatasdk](https://github.com/JoinQuant/jqdatasdk)

This project uses CSI 500 China’s Mid & Small-cap Universe stocks from 2016 to 2023. 

All datasets are located under `/data/`

## Code
All codes are located under `/code/`
- ProcessData.py encompasses data acquisition and preprocessing tasks.
- Training.ipynb focuses on creating dataframes for features and labels, model training, evaluation, and result visualization.
- xgboost_shap.ipynb draws plot to show feature impact on model output.

## ML pipeline
Please see `pipeline.md`