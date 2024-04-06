---
layout: default
---
[Back to Portfolio Page](./)

# *Report of Data Analysis on Bank Customer Churn*

## *Table of Contents*
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analytic Techniques](#data-analytic-techniques)
- [Findings, Insights and Application](#findings-insights-and-application)
- [Reference](#reference)

### *Project Overview*
The objectives of this report is to investigate any interesting relationships between different variables, then provide potential insights and suggestion for the bank services strategies. 

### *Data Sources*
The dataset used in this report is extracted from the Kaggle, providing the 10,000 customers information such as Age, Credit Score, Salary, Balance, Card Type, etc. for investigation. [1]

### *Exploratory Data Analysis*
Before analysis, we need to figure out what kinds of questions is interested to be investigated as listed below:
1. Any findings discovered?
2. What insights and suggestion can be made to prevent from customer churn?
3. Model development to predict the customer churn

### *Data Analytic Techniques*
Data Format: structured numerical data <br/>
After identifying the data format, more advanced data preprocessing techniques are required:
1. Data Cleaning Skills <br/>
Aim: although the dataset is quite completed with no missing values, the data shall also be checked with any duplicated values.
```python
# no missing data
# check any duplicated data
print('The Duplicates Count in Data: {}.'.format(df.duplicated().sum()))
```

2. Data Analytical Skills <br/>
Aim: To study the frequency in term of each variables for knowing the basic data information first, then define self-make function for simlifying the visualization process
```python
# no duplicated data

# function of graph plotting
def plot_pie_bar(dataset, title):
    # subplot for pie chart
    plt.subplot(2, 1, 1)
    dataset.plot(kind='pie', startangle=90, autopct='%1.1f%%', figsize=(10, 10))
    plt.title('Customer Distribution of {}'.format(title))
    plt.legend(loc = 'best')
    
    # subplot for bar chart
    plt.subplot(2, 1, 2)
    dataset.plot(kind='bar', align='center')
    plt.title('Customer Distribution of {}'.format(title))
    plt.ylabel('Customer')
    plt.grid(True)
    plt.tight_layout()
```
Aim: to create pivot table for calculation of the corresponding probabiility in term of the variables.
```python
# create pivot table for evaluation of exited rate
# for probability estimation
pd.pivot_table(df, index=['Gender'], values=['Exited'], margins=True)
```

3. Data Binning Skills <br/>
Aim: for specific columns like 'credit score' and 'age' may not be easily applied without preprocessing. To make the required information to be clean in analysis and visualization, it is suggested to perform data binning to add new features into the original dataset. <br/>
The below demostration for handling credit score data by FICO model can be a good example of data binning.
```python
# Customer Distribution of Credit Score
# For better understanding, it is suggested to implement data binning to consider credit score
# Let's apply the FICO 8 model for evaluation of model score

fico_model = [300, 580, 670, 740, 800, 851]
fico_rating = ['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
df_credit_score = pd.cut(df.CreditScore, fico_model, labels=fico_rating)
df_credit_score

# count the credit score in numerical format 
pd.value_counts(df_credit_score)

# Let's save the faeture of credit score (FICO model)
df['CreditScore_Class'] = df_credit_score
df
```

4. Data Grouping <br/>
Aim: construct correlation matrix and heatmap visualization for analysis of correlation
```python
# Attempt for correlation matrix to see any different insight as found
# Before correlation analysis, let's copy another set of data and remove some irrelvant data columns

df_irrel = ['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Card Type', 'CreditScore_Class', 'age_group', 'Balance_Group', 'Salary_Group', 'Gender']
df_backup = df
for i in range(len(df_irrel)):
    df_backup = df_backup.drop(df_irrel[i], axis=1)

df_corr = df_backup.corr()
df_corr

# visualization of correlation analysis
plt.figure(figsize=(25, 15))
sns.heatmap(df_corr, annot=True, fmt='.2f', linewidths=0.5)
plt.tight_layout()
plt.show()
# Let's check this correlation matrix for further development of predictive model
```

5. Machine Learning Model Training for Data Prediction <br/>
Aim: to convert category data into number by label encoder, preparing data for upcoming model training.
```python
# define function for category by label encoder
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def encode_cat(dataset, columns):
    encoder = LabelEncoder()
    
    for col in columns:
        dataset[col] = encoder.fit_transform(dataset[col])
    return dataset
        
df_ml = encode_cat(df_ml, df_category)
df_ml
```

Aim: check with multi-collinearity diagnosis to prevent the trained model to be extremely sensitive with the changes of high correlation variable.
```python
# check with multi-collinearity diagnosis
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

y, X = dmatrices('Exited ~ CreditScore+Geography+Gender+Age+Tenure+Balance+NumOfProducts+HasCrCard+IsActiveMember+EstimatedSalary+Satisfaction_Score+Card_Type+Point_Earned', data=df_ml, return_type='dataframe')
vid_df = pd.DataFrame()
vid_df['variables'] = X.columns
vid_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vid_df)
```
   
Aim: perform data standardlization and data splitting for training set and testing set. Models can be constructed with their corresponding accuracy.
```python
# data standardlization for more accurate result
# Classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 

# Set X and y
X = df_ml.drop('Exited', axis=1)
y = df_ml['Exited']

# split data into training set and testing set, ratio 4:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardlization
sc = MinMaxScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# define model dictionary type for recording scores of each model
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=0, max_depth=5, criterion='entropy'),
    'KNegihborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=0),
}

# train the model
model_scores = {}
for name, model in models.items():
    model.fit(X_train_std, y_train)
    model_scores[(name, 'train')] = model.score(X_train_std, y_train)
    model_scores[(name, 'test')] = model.score(X_test_std, y_test)

# for visualization of model accuracy
model_scores
```

### *Findings, Insights and Application* 
**Q1: Any findings discovered?** <br/>
Before in-depth analysis, the dataset have been checked with preprocessing. All the columns-count charts have been investigated for fair analysis without large bias. <br/>
Some findings can also be useful and meaningful for selecting the target customer group. <br/>

*Finding 1:* <br/>
Majority of the customers is Adult group (77.7%) (7768 out of 10000). <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/6b8f08c2-44ef-448a-8cf9-14abb54a0337)
<br/>
Adult group is the largest group in a city/country and more probably to generate high work productivity in society. <br/>
The second top customers are belong to young_adult. <br/>
Therefore, it is suggested that the bank can set up more strategic plan for these two target customer groups. <br/>

*Finding 2:* <br/>
After comparing the balance and salary chart, there are around 9.5% of customers' balance with 150,000 or above. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/d07a2b51-4c10-4db5-96e5-4422567b898b)
<br/>
At the same time, there are approximately 19.9% of customers' estimated salary with $120,000 to 160,000. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/b64d5179-f18e-48f4-861b-f8da6e8c1f1c)
<br/>
It may imply that although the customer have abiliy of gaining high salary, the balance may not be necessarily high. <br/>
Reason behind may be due to the extreme large expenditure in daily life (depend on individual's consumer behaviour). <br/>
Other important reason may be that the customers have high probability to have multiple bank accounts (depend on competition between other bank services, can also study why customers may high larger balance in bank A account but less balance in Bank B). <br/>

*Finding 3:* <br/>
It can be clearly observed that around 96.7% majority of customers have purchased 1 to 2 products of the bank. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/43e343f9-b994-4b92-8856-f4c4e419a6d6)
<br/>
The high number of product purchased is not common as shown in the sample distribution. <br/>

*Finding 4:* <br/>
Most of the customers (70.6%) in the sample are having credit card in the bank. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/a64007f1-29f8-4487-a408-4930957459d3)
<br/>

*Finding 5:* <br/>
The majority (79.6%) of customer have not exited. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/c7239c23-823e-4641-b17b-c4557ae19540)
<br/>

*Finding 6:* <br/>
The majority (79.6%) of customers have no complaint on the bank services. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/d324d152-5d27-4c92-a1a7-4aebbca76ece)
<br/>
It is an inspiring result for the bankers. <br/>

**Q2: What insights and suggestion can be made to prevent from customer churn?** <br/>
Combining the above findings, specific insights can be concluded: <br/>

*Insight 1:* <br/>
Insight for Relationship between Gender and Exited <br/>
For those who was not exited, the quantity of male customers is larger than that of female customers. <br/>
However, the quantity of female exited customers is larger than that of males customers. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/f904181a-3209-4b7e-b557-fcd66bc4b1d3)
<br/>
From the corresponding pivot table, there are around one-fourth of female customers have exited. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/a70adb9e-9b92-4ce0-b937-7804de3add62)
<br/>
We can say that the female customers may have higher probability than male customers to exit in the sample case. <br/>

Therefore, it is interesting to think why female customers have higher customer churn rate. <br/>
For example, related to salary? Maybe local culture have gender stereotypes to recruit more male employee for higher salary, leading to stability in the bank services? <br/>
Or maybe the different consumer behaviour of female and male have influence of the result? <br/>

To apply with the insight, it is suggested to provide credit card specific offers for female customers spending on accessories and clothing. <br/>
One reason is that buying accessories is more prevalent in female customer group than male customer group. <br/>
Credit card offers for buying accessories or facial treatment can not only retain the female customers as long as possible but also increase the transaction in accessories industry. <br/>
A win-win situation can be provided for the bank and other industry. <br/>

*Insight 2:* <br/>
Insight for Relationship between CreditScore_Class and Exited <br/>
It is obvious to see the left-skewed bar chart for both exited and not exited trend. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/f7dc650a-dd7a-446c-92c2-a318bda72c84)
<br/>
Furthermore, the probability of each CreditScore_Class is around 20% in the corresponding class, very close to the total probability of the 10,000 samples, 20.38%. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/508d6c33-f6f2-4912-9ce5-b5332b279dd4)
<br/>

It reflects that no matter which credit score the customers are having, there is still around one-fifth chance for the customers to exit. <br/>
As compared to gender-exited circumstance, it seems that gender have temporarily higher effect on churn rate. <br/>
Therefore, it is suggested to reduce the resources on how credit score impact on circumstance of exiting. <br/>

*Insight 3:* <br/>
Insight on Relationship between age_group and exited <br/>
It is clear to find that the largest age_group for exited and not exited customers is the Adult group. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/40982d2f-de56-4e78-b313-ab82cf69a232)
<br/>
The reason is that the samples take large number of adult data (highest productivity group), and it actually match with the realistic situation as mentioned. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/8a82dbf8-b344-466b-9171-8149e97f4f46)
<br/>
As shown in the pivot table, the probability for child, young_adult, elderly to exit is extremely low as compared to that of adult. <br/>
More interestingly, the probability of adult (23.88%) to exit is higher than the total probability of customer churn (20.38%). <br/>

It highlights the importance for a bank to focus on the adult group with more support or strategies. Retaining more adult customers can considerably reduce the customer churn rate. <br/>

*Insight 4:* <br/>
Insight on Relationship between tenure and exited <br/>
The exit and non-exit bar charts are similar in distribution. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/3f1882b2-7d72-496a-978c-29da4603e60e)
<br/>
No matter how long the tenure is, the percentage of exiting is still around 20%. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/3ada7899-c9ff-4aed-85b8-c01c3eeef533)
<br/>
Originally, I was thinking the customers with longer tenure may have lower churn rate, because I assume that older customers should be more loyal and stable in the bank. <br/>

As proved by the bar charts and pivot, the original hypothesis statement is not a must. <br/>
It is common that advanced bank services provided for the longer tenure customers is more attracting. <br/>

It helps us to understand that the current bank services may be old and insuitable to retain longer tenure. <br/>
The bank need to investigate the weakness of strategy and develop more attractive incentive for retaining adult customers. <br/>

*Insight 5:* <br/>
Insight on Relationship of Balance, Salary and Exited <br/>
The customers, who having balance under $50,000 and salary under $40,000, acoount for the lowest probability customers group to exit (13.48%). <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/7d2a8b09-ba7c-4753-9eab-94053b678a15)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/4033e7e0-36fd-4e80-8867-4134b757041a)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/b1b96ed8-52be-4371-8361-c2c653962e27)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/186a5c06-735e-4618-a217-dbec060c87ec)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/9307d62e-ce14-4a6e-9cfd-fc856a80cac3)
<br/>

The highest probability belong to those who have balance above $200,000. <br/>
However, it is non-sense to make conclusion with the highest probability due to the extremely low quantity of data with balance above 200,000. <br/>

In this analysis case, I would like to recognize the data with balance above $200,000 as outlier and ignore them for higher accuracy of analysis. <br/>

Interestingly, the probability of exited customers with balance of $100,000 to $150,000 is quite high (25.78%, higher than total churn probability of customers). <br/>
As we know, the customers with higher balance should be loyal to stay in the bank. <br/>

Combining this findings with metnioned tenure circumstance, there is several interesting insights. <br/>
<br/>
Customer balance may be one of the important parameter influencing the customer churn rate as compared to tenure. Long tenure may not be the main reason for staying and cannot guarantee low customer churn rate. These findings may suggest that customers in recent ceuntry may tend to be attracted by welcome gift offer to save little balance in their bank account, but may not consider loyality as important factors to stay in the bank. Once they have saved certain level of balance like above $100,000, customers may also have higher probability to consider exiting. The potential reason may be other attraction from other bank with new welcome gift offer or alternative incentives. <br/>
<br/>
Therefore, the data highlights the importance of keeping high tenure customers with middle level of balance. Retaining this target group customers can be one of the strengthness in SWOT analysis of a bank among market cmpetition. Higher discount offer for financial products may be suggested for those customers with high tenure and high balance to prevent high churn rate. <br/>

*Insight 6:* <br/>
Insight on relationship of number of products, credit card and exited <br/>
Although the customer data above 4 number of products may be a outlier, the trend of number of products data can also reflect that the more the number of products owned, the lower churn rate will be.
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/491ed8e3-ece9-404f-bdd8-848bc3f8d026)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/6bd07f54-c7ea-49d0-9c0f-3eeb19dfc082)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/6f130296-3b12-48c7-ad16-751937d02067)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/6bc6cb3a-afee-468c-adb2-135f978471b9)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/0089ba0d-dd40-48bd-a365-a06e6da89a1a)
<br/>
Having a credit card or not is not the major factor on customer churn because the corresponding probability is similar to the overall customer churn probability does. <br/>

Any suggestion: <br/>
One of the recommeded banking strategy is to strengthen the purchasing rate of products in order to retain customers with high balance and tenure. <br/>
For example, higher discounts of products <br/>
Set priority rule for those target customers with five tenure to purchase products <br/>
After having $100,000 balance or above, provide extra earn points or cashback for credit card. <br/>
Higher saving interest for those who having $150,000 or above. <br/>

*Insight 7:* <br/>
Insight on relationship of complain, satisfaction score and exited <br/>
It is very obvious to see that extremely high probability of customers (99.5%) with complaint experience to exit the bank. <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/e80ad898-8b94-4905-9f73-19bdecfd0252)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/3e4b1397-c8e7-4e26-a26e-58e6308d1ee3)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/4789edd8-4a5b-4f60-a599-246888a9ca2f)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/18165ad5-f098-4b60-b26c-b944a6ff792f)
<br/>

Comparing to the variable of complain, satisfaction score may not have high impact on exiting or not. <br/>

This findings suggest the insight that effort for complaint prevention is the key to retain customers, no matter how high the satisfaction score is. <br/>
Certain key performance index may be constructed with complaint number. <br/>
For example, keeping complaint number to be less than 20% of total customers in each month. <br/>
The reason of setting < 20% complaint cases is that the complaint probability in overall customers is around 20%. <br/>
In short term, the bank can set KPI with less than 20% complaint first. <br/>
In long term, the bank can also set KPI with lower number of complaint and the average satisfaction score to be above 4. <br/>

*Insight 8:* <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/81afcacf-fe04-47e1-877f-a2822ae14b19)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/5d37004c-32af-4391-8b19-5b270b3a01e5)
<br/>
Checking the row 'Exited', only 'Age', 'Balance' have positive and reasonable correlation with 'Exited'. <br/>
Saying that older age and high balance may be correlate to high churn rate. <br/>
Other variables' correlation with 'Exited' are negative. <br/>

The 'Complain' have a extremely high value of correlation with 'Exited'. <br/>
However, this extremely large value of correlation by 'Complain' is not recommended as input of the machine learning model. <br/>
Collinearity may occur, which means 'Complain' varies a little bit can lead to large change in prediction of exiting or not. <br/>
<br/>
Overall suggestion in bank strategy to prevent high customer churn rate: <br/>
1. Focus target customer group: Young_adult and adult, more offers for female customers <br/>
2. No need to dedicate much resources in term of CreditScore for retaining customers. <br/>
3. The existing strategy cannot be significantly powerful to retain customers with long tenure and high balance. <br/>
4. The customer churn rate is lower for those customers with higher number of bank products <br/>
5. Potential strategies (details shall be discussed with business team and customner relationship team) <br/>
    5.1. Set priority rule for those target customers with five tenure to purchase bank products <br/>
    5.2. After having $100,000 balance or above, provide extra earn points or cashback for credit card. <br/>
    5.3. Higher saving interest for those who having $150,000 or above. <br/>
    5.4. Higher discounts of bank products for customers who having $150,000 balance and five tenure. <br/>
    5.5. Special shopping credit card offers for female customers to consume in the specific accessories, 
         clothing and facial shops. <br/>
    5.6. Provide more advanced training for employees to increase service quality in order to reduce the 
         frequency of complaints. <br/>
    5.7. If trainers is not professional to teach, better to invite famous trainers from oversea. Maybe 
         interesting and innovative culture can be applicable in local city/country. <br/>
    5.8. Improve the digital bank mobile app with up-to-dated user-interface to increase the efficiency of bank 
         services, avoiding frequent manual mistake to minimize complaints. <br/>
    
**Q3: Model development to predict the customer exit or not** <br/>
Decision tree classifier model have been used and suitable for prediction with high accuracy (85%). <br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/28e8b399-d20a-4667-b2a8-f30002d4a676)
<br/>
![image](https://github.com/Andychan123456/Andy_Portfolio/assets/156527746/c41c2339-f069-4f1b-b5a2-c5f81b9bce62)
<br/>
In real application, the model shall be discussed with the data scientist team for advanced suggestion. <br/>
For example, data scientist team may suggest apply other non-supervisory model, deep learning model like MLP, NLP for further analysis. <br/>
After that, the model can be suggested for decision-making team to develop suitable strategies for reduce customer churn rate.

### *Reference*
[1] Kaggle Dataset. (https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data) <br/>
[2] Stack Overflow
