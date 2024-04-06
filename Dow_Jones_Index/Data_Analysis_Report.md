# *Report on Data Analysis*

## *Table of Contents*
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analytic Techniques](#data-analytic-techniques)
- [Findings, Insights and Application](#findings-insights-and-application)
- [Reference](#reference)

### *Project Overview*
The objectives of this report is to discover any potential insights for the stock performance in 2011, acting as analysis for considering any trading strategy if any. 

### *Data Sources*
The dataset used in this report is extracted from the UCI repository, providing the weekly data of Dow Jones Industrial Index in 2011. [1]

### *Exploratory Data Analysis*
Before analysis, we need to figure out what kinds of questions is interested to be investigated as listed below:
1. What is the average price for each stock?
2. How is the correlation between two stocks?
3. What parameters can lead to higher pricing?
4. Develop model for stock IBM to predict pricing?

### *Data Analytic Techniques*
Data Format: structured time-series data, require module like Plotly <br/>
After identifying the data format, more advanced data preprocessing techniques are required:
1. Data formatting Skills <br/>
Aim: To remove the sign '$' in the specific columns showing the stock pricing.
```python
# find the '$' sign in below columns
df_replace = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']

# to replace the '$' sign in columns and convert them to be number type
for i in df_replace:
    df[i] = pd.to_numeric(df[i].map(lambda x: str(x).replace('$', '')))
df.head()
```
Aim: convert column 'date' into datetime type
```python
# import date type module
import datetime
df['date'] = pd.to_datetime(df['date'])
df.info()
```
2. Data Cleaning Skills <br/>
Aim: check missing value and confirm data filling
```python
# checking any missing data and relevance
df.isnull().sum()
```
Need to consider how to clean the missing data: delete its rows? number fill in the empty data? <br/>
In this case, it is not suggested to delete the rows with missing value because the rows contain valid data for other columns. <br/>
Data filling is also not recommended because the missing values is relate to previous week data. Filling with previous values can create contradiction as compared to the stock closing price. <br/>
Therefore, it is better to delete the two irrelevant columns with missing values for high accuracy in data analysis.
```python
# only missing data in two columns
# not recommended to just delete the rows of data because it lead to unaccurate analysis for other columns
# not recommended to fill missing values by previous values or interpolation because not make sense for the values compared to last weeks
# suggested to delete the two columns due to irrelevance of our analysis
df.drop(['percent_change_volume_over_last_wk', 'previous_weeks_volume'], axis=1, inplace=True)
```
3. Data Grouping <br/>
Aim: group the specific variables and provide clean table for analysis
```python
# let's see the average pricing for each stock 
df.groupby(['stock'])[['open', 'high', 'low', 'close']].mean()

# let's find the top 5 highest average price stock
top_five_close = df.groupby(['stock'])['close'].mean().sort_values(ascending=False)[0:5]
print('The top five stock with highest average price: {}'.format(top_five_close))
```
Aim: construct correlation matrix and heatmap visualization for analysis of correlation
```python
# apply heatmap to visualize the correlation
plt.figure(figsize=(25, 15))
sns.heatmap(df_close.corr(), linewidths=0.5)
plt.title('Correlation Heatmap for Stock Pairs')
plt.tight_layout()

# the correlation heatmap can help us to know the correlation
# for example, price of AA is highly relational to DD, DIS (above 0.7).
```
4. Machine Learning Model Training for Data Prediction
Aim: perform data standardlization and data splitting for training set and testing set. Models can be constructed with their corresponding accuracy.
```python
# Since all the variables have moderate correlation with 'close' target
# can be suitable for development of predictive model

from sklearn.preprocessing import StandardScaler

# set X and y data
X = df_IBM.drop('close', axis=1)
y = df_IBM['close']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)

# Data Standardlization to increase model performance in accuracy
sc = StandardScaler()
sc.fit(X_train)
X_train_st = sc.transform(X_train)
X_test_st = sc.transform(X_test)

# set multiple models, analyze which one have the highest score in accuracy
models = {
    'Linear_regression': LinearRegression(),
    'Ridge': Ridge(random_state=0),
    'DecisionTreeRegressor': DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=0),
    'KNegihborsRegreessor': KNeighborsRegressor(),
    'SVR': SVR(),
}

# record the score of each model
models_scores = {}
for name, model in models.items():
    model.fit(X_train_st, y_train)
    models_scores[(name, 'train')] = model.score(X_train_st, y_train)
    models_scores[(name, 'test')] = model.score(X_test_st, y_test)

# for visualization of models' scores
models_scores
```
### *Findings, Insights and Application* ✨
**Q1: What is the average price for each stock?** <br/>
Ans: can use describe() function to check each average price of the corresponding stock. <br/>
The top highest price stocks are IBM ($163.1), CAT ($103.2), CVX ($101.2), MMM ($91.7) and UTX ($84.0). <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/7fcd72c9-0907-425e-bb6d-0c1767329bce) <br/>
Therefore, for those who want to plan their trading strategy, the above high price stocks shall be considered with self-review of budget <br/>

**Q2: How is the correlation between two stocks?** <br/>
Ans: 
Correlation heatmap can be visualized as below graph:
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/c34a156f-680e-49c8-966a-5498eb8e6ebb) <br/>
Correlation matrix can be formed to study the top 10 correlated stock pairs <br/>
For example, (CSCO & MSFT), (KRFD, MCD), (BA, UTX) <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/42769fc4-16e3-4248-87aa-4c07c3a0d813) <br/>
Those information can be reported to data scientist or quant developer for designing pairs trading algorithm to gain potential profit.

**Q3: What parameters can lead to higher pricing?** <br/>
Ans: 
Relationship between closing price and volume have been checked. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/aca8e694-520f-4ad1-998f-bf986c57c30c) <br/>
According to the result, closing price is most likely to be high when volume is not in high level. <br/>
Thinking with financial knowledge, volume is actually counting the amount of buy and sell. <br/>
Therefore, the higher volume cannot garantee higher or lower closing price because individuals can buy and sell in the market.<br/>
After the whole day stock transaction, the volume of buy side and sell side can be balanced to keep the stock pricing with small fluctation. <br/>
For better idea of relationship analysis, volume may be better to be separated into buy side volume and sell side volume for analysis. <br/>
It may help to generate more impressive pattern of closing price with the change of buy side volume.

**Q4: Develop model for stock IBM to predict pricing?** <br/>
Ans: ML models are built for prediction of stock 'IBM' closing price after normalization. <br/>
Variance inflation factor can be checked for any multi-collinearity circumstance. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/ecbcf607-6148-4159-beef-983831a2921e) <br/>
After checking the corresponding models' accuracy, <br/> 
the coefficient of determination in Ridge model is the best among selected models.
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/3a206906-4c5e-473b-b7f7-e57a7097b84f) <br/>
The result may show that regularization with loss function in Ridge model can suitably componensate the seriousness of overfitting, especially as compared to linear regression. <br/>
However, the data may not large enough for the model testing. Therefore, the accuracy can also be influenced.

### *Reference*
[1] Uci repository. (https://archive.ics.uci.edu/dataset/312/dow+jones+index) <br/>
[2] Stack Overflow
