# PREDICTING HIGH PAYING DATA SCIENCE ROLES IN 2026
This is a Mini Project Repo
## Predictinng High Paying Data Science Roles in 2026

## PROBLEM STATEMENT
* In a competitive AI job market,companies struggle to benchmark salaries.Can we predict whether an AI/Data Science role will be "High Paying" > $150k or "Standard Paying" <= $150k based on job title, experience, location and remote status.
## AIMS OF THE PROJECT

1. **Analyse salary trends**:Explore how salary varies by job title, experience level, location, company size and remote status.
1. **Build predictive models**:Develop and compare classification models to predict salary level using role and profile features. 
1. **Identify key drivers**:Determine the top 3 factors that most influence whether a role is high paying.
1. **Provide actionable insights**:Offer recommendations for job seekers, HR teams and companies on how to position roles and careers for competitive compensation in 2026.
## DATA UNDERSTANDING
**Data Source**:[Kaggle](https://www.kaggle.com/datasets/uditjain13/ai-and-data-science-job-salaries-2026)

**Dataset Description**:This dataset contains 2026 AI and Data Science job postings with salary, job title, experience level, location, and remote status.Used to predict if a role is "High paying" > $150k.

**Key Columns**
1. 'job_title':Role name
1. 'experience_level':EN,MI,SE,EX
1. 'salary':Annual salary in USD
1. 'location': Job location
1. 'remote_ratio ':0,50,100
1. 'company_size':S,M,L
#Import the necessary libraries for cleaning and visualizing the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Load the dataset into a pandas DataFrame
df = pd.read_csv('AI AND DS.csv')
df.head(10)
df.info()
df.describe()
## DATA CLEANING
* Handling missing values
* Handling duplicates
* Handling outliers
* checking datatypes assigned to the columns

#### Handling missing values
#Check for duplicate rows in the dataset and remove them if necessary
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
#### Handling duplicates
#Check for duplicate rows in the dataset and remove them if necessary
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
#Check the data types of each column in the dataset
print(df.dtypes)
## EXPLORATORY DATA ANALYSIS

#### Univariate EDA
1. Histogram
1. Bar graph
1. pie Chart

#### Bivariate EDA
1. Box plot
1. Count plot
1. Scatter plot


#### Multivariate EDA
1. Pair plot
1. Heatmap
#Histogram of the 'Salary_USD' column
plt.figure(figsize=(10, 6))
plt.hist(df['salary_usd'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Salary in USD')
plt.xlabel('Salary in USD')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/323b14e5-160b-4e67-be25-644652393a28" />


#### Findings

1. **Most AI and Data Science jobs pay between $40k-$100k**:High paying jobs > $150k are uncommon and form a long right tail,making them a distinct minority class worth predicting.
1. **The market rate for most AI and Data Science roles sits between $50k-$100k**:Paying >$150k, puts the company in the top ~15-20% of payers which might end up making the company run bankrupt.

#Bar plot of the average salary by job title
sns.barplot(x='job_title', y='salary_usd', data=df, estimator=np.mean, ci=None)
plt.title('Average Salary by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Average Salary in USD')
plt.xticks(rotation=45)
plt.show()

<img width="621" height="593" alt="image" src="https://github.com/user-attachments/assets/ee8d7236-e3b6-4e79-9aa6-f6be770d5768" />

### Findings

1. **Job title matters a lot**:Research, ML, and BI roles average > $108k, while Analysts, and Analytics Engineer roles average > $80k.This means job_title will likely be a strong predictor for the salary given to the employees in 2026
1. **Business Intelligence Analysts**:Has the highest average salary, slightly above $110k.
1. **Research Scientists and Machine Learning Engineers**:Are also top earners, both averaging around $108k-$110k.

#Pie chart of the distribution of job titles in the dataset
plt.figure(figsize=(10, 8))
plt.pie(df['job_title'].value_counts(), labels=df['job_title'].value_counts().index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Job Titles')
plt.axis('equal')
plt.show()

<img width="841" height="662" alt="image" src="https://github.com/user-attachments/assets/679e0445-41ff-4bdb-8127-aa461fb08a15" />

### Findings

1. The job market in this dataset is dominated by Data Scientists, Data Engineers, Data Analysts, and ML Engineer roles, while specialized roles like CV Engineer, Manager, and LLM Engineer are the least common.
1. **Highly imbalanced distribution**:A few roles dominate, while most roles are small slices thus making them valuable in the future and this might make their pay be of higher.


#Box plot of the salary distribution by job title
plt.figure(figsize=(12, 6))
sns.boxplot(x='job_title', y='salary_usd', data=df)
plt.title('Salary Distribution by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Salary')
plt.xticks(rotation=45)
plt.show()

<img width="1037" height="686" alt="image" src="https://github.com/user-attachments/assets/7fd99b71-5330-4f64-bd5a-cecd6e89a936" />

### Findings
1. Management + Emerging AI roles have the highest typical pay.Analyst roles have the lowest typical pay,
1. Even though Data Engineer median is mid ~$95k, it has huge upside.Data Science Manager has both high median and highest max.

#Scatter plot of salary vs. years of experience
plt.figure(figsize=(10, 6))
plt.scatter(df['years_experience'], df['salary_usd'], alpha=0.5)
plt.title('Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary in USD')
plt.grid()

<img width="876" height="547" alt="image" src="https://github.com/user-attachments/assets/9ee37b05-bc05-454a-acbd-19c166dd783d" />


### Findings
#### 1. ***Overall Trend:Positive but Non-Linear***
* **As years of experience increase, salary tends to increase**.The cloud of points shifts upward from left to right.
* **0-3 years**:Most salaries are clustered between $20k-$120k .Very few points above $150k.
* **5-10 years**:Big jump .Salary range expands to $30k-$220k.Much higher density above $100k.
* **10-20+ years**:Highest salaries appear here .Points reach $250k-$360k.But also still many points in $60k-$120k range.

#### 1. ***High Paying >$150k Threshold***
* **Below 3 years experience**:Almost no dots above $150k.Rare outliers only.
* **3-5 years**:First dots starts crossing $150k.
* **5+ years**:Majority of >$150k salaries appear here density of dots above $150k increases a lot.
* **Peak High Paying Zone: 8-20 years experience**has the most concentration of dots above $200k and $300k.
