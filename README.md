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
### Findings
1. **Most AI and Data Science jobs pay between $40k-$100k**:High paying jobs > $150k are uncommon and form a long right tail,making them a distinct minority class worth predicting.
1. **The market rate for most AI and Data Science roles sits between $50k-$100k**:Paying >$150k, puts the company in the top ~15-20% of payers which might end up making the company run bankrupt.
