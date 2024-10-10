Data Importing and Preparation
import pandas as pd
​
# Load your dataset
data = pd.read_csv("C:/Users/colet/OneDrive/Documents/Jewelry Pricing Project/KaysLocationAnalysis/LongIslandKays.csv")
​
# Check for missing values
print(data.isnull().sum())
​
# Convert Store_Present to a categorical type
data['Store_Present'] = data['Store_Present'].astype('category')
​
placeDcid                                    0
placeName                                    0
Value:Median_Income_Household                5
Value:Count_Person_18OrMoreYears             2
Value:Count_Person_NeverMarried              4
Value:Count_Person_MarriedAndNotSeparated    4
Store_Present                                0
dtype: int64
# Drop rows with any null values
data_cleaned = data.dropna()
​
# Optional: Reset index after dropping rows
data_cleaned.reset_index(drop=True, inplace=True)
_cleaned
# Check for missing values
print(data_cleaned.isnull().sum())
placeDcid                                    0
placeName                                    0
Value:Median_Income_Household                0
Value:Count_Person_18OrMoreYears             0
Value:Count_Person_NeverMarried              0
Value:Count_Person_MarriedAndNotSeparated    0
Store_Present                                0
dtype: int64
# Features
features = data_cleaned[['Value:Median_Income_Household', 
                 'Value:Count_Person_18OrMoreYears', 
                 'Value:Count_Person_NeverMarried', 
                 'Value:Count_Person_MarriedAndNotSeparated']]
​
# Target variable (assuming you want to predict 'Store_Present' as a binary outcome)
target = data_cleaned['Store_Present']
​
Modeling the Data
from sklearn.model_selection import train_test_split
​
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
​
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
​
# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
​
# Fit the model
model.fit(X_train, y_train)
​
RandomForestClassifier(random_state=42)
y_pred = model.predict(X_test)
​
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
​
[[33  0]
 [ 1  0]]
              precision    recall  f1-score   support

           0       0.97      1.00      0.99        33
           1       0.00      0.00      0.00         1

    accuracy                           0.97        34
   macro avg       0.49      0.50      0.49        34
weighted avg       0.94      0.97      0.96        34

C:\Users\colet\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\colet\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\colet\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
New Store Likelihood Results
import pandas as pd
​
# Assuming your data is loaded into a DataFrame called data_cleaned
​
# Step 1: Check the data type of 'Store_Present'
print(data_cleaned['Store_Present'].dtype)
​
# Step 2: Convert 'Store_Present' to integer if needed
data_cleaned['Store_Present'] = data_cleaned['Store_Present'].astype(int)
​
# Step 3: Filter out places where a Kay Jewelers store already exists (Store_Present == 1)
new_locations = data_cleaned[data_cleaned['Store_Present'] == 0]
​
# Step 4: Feature columns used for prediction (customize based on what you want to use)
features = ['Value:Median_Income_Household', 'Value:Count_Person_18OrMoreYears',
            'Value:Count_Person_NeverMarried', 'Value:Count_Person_MarriedAndNotSeparated']
​
# Step 5: Make predictions on the new locations
new_locations['Predicted_Probability'] = model.predict_proba(new_locations[features])[:, 1]
​
# Step 6: Rank the new locations by predicted probability
ranked_new_locations = new_locations.sort_values(by='Predicted_Probability', ascending=False)
​
# Step 7: Show the top 10 places with the highest predicted probability of success
print(ranked_new_locations[['placeName', 'Predicted_Probability']].head(10))
​
category
     placeName  Predicted_Probability
74       11717                   0.29
52       11756                   0.25
24       11553                   0.25
132      11937                   0.21
53       11758                   0.20
101      11757                   0.17
84       11729                   0.16
44       11590                   0.15
42       11580                   0.14
31       11563                   0.14
C:\Users\colet\AppData\Local\Temp\ipykernel_9652\4228094580.py:9: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  data_cleaned['Store_Present'] = data_cleaned['Store_Present'].astype(int)
C:\Users\colet\AppData\Local\Temp\ipykernel_9652\4228094580.py:21: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  new_locations['Predicted_Probability'] = model.predict_proba(new_locations[features])[:, 1]
Due to data imbalances, will be rebalancing data and rerunning model
# Import necessary libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
​
​
# Separate features (X) and target (y)
X = data_cleaned[['Value:Median_Income_Household', 
                  'Value:Count_Person_18OrMoreYears', 
                  'Value:Count_Person_NeverMarried', 
                  'Value:Count_Person_MarriedAndNotSeparated']]
y = data_cleaned['Store_Present']
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
​
# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
​
# Check the distribution of the target after SMOTE
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_resampled.value_counts())
​
# Now you can train your model with the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)
​
# Predict and evaluate on the test set
y_pred = model.predict(X_test)
​
# Evaluate the model performance
print(classification_report(y_test, y_pred))
​
Before SMOTE: 0    109
1      7
Name: Store_Present, dtype: int64
After SMOTE: 0    109
1    109
Name: Store_Present, dtype: int64
              precision    recall  f1-score   support

           0       0.94      0.68      0.79        47
           1       0.06      0.33      0.11         3

    accuracy                           0.66        50
   macro avg       0.50      0.51      0.45        50
weighted avg       0.89      0.66      0.75        50

# Predict probabilities for the entire dataset
data_cleaned['Predicted_Probability'] = model.predict_proba(X)[:, 1]
​
# Create a new column for Final_Probability
data_cleaned['Final_Probability'] = data_cleaned.apply(
    lambda row: 1 if row['Store_Present'] == 1 else row['Predicted_Probability'], 
    axis=1
)
​
# Filter out existing stores (Store_Present == 1) for potential locations
potential_locations = data_cleaned[data_cleaned['Store_Present'] == 0]
​
# Sort by predicted probability in descending order
ranking = potential_locations.sort_values(by='Predicted_Probability', ascending=False)
​
# Select relevant columns for the final ranking
ranking_final = ranking[['placeName', 'Predicted_Probability', 'Final_Probability']]
​
# Display the ranking of potential store locations
ranking_final.reset_index(drop=True, inplace=True)
​
# Show the top 10 places with the highest predicted probability of success
top_10_places = ranking_final[['placeName', 'Final_Probability']].head(10).reset_index(drop=True)
print(top_10_places)
​
   placeName  Final_Probability
0      11742               0.92
1      11772               0.81
2      11542               0.78
3      11703               0.77
4      11550               0.76
5      11021               0.76
6      11795               0.72
7      11783               0.71
8      11780               0.70
9      11010               0.69
C:\Users\colet\AppData\Local\Temp\ipykernel_9652\1565174896.py:2: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  data_cleaned['Predicted_Probability'] = model.predict_proba(X)[:, 1]
C:\Users\colet\AppData\Local\Temp\ipykernel_9652\1565174896.py:5: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  data_cleaned['Final_Probability'] = data_cleaned.apply(
print(data_cleaned)
     placeDcid  placeName  Value:Median_Income_Household  \
0    zip/11735      11735                       127059.0   
1    zip/11010      11010                       136378.0   
2    zip/11020      11020                       173345.0   
3    zip/11021      11021                       111777.0   
4    zip/11023      11023                       146563.0   
..         ...        ...                            ...   
161  zip/11975      11975                       148958.0   
162  zip/11976      11976                       185000.0   
163  zip/11977      11977                       139375.0   
164  zip/11978      11978                       120909.0   
165  zip/11980      11980                       113004.0   

     Value:Count_Person_18OrMoreYears  Value:Count_Person_NeverMarried  \
0                              6900.0                           2200.0   
1                             20406.0                           6210.0   
2                              4879.0                           1643.0   
3                             15714.0                           4616.0   
4                              6514.0                           1451.0   
..                                ...                              ...   
161                             385.0                            135.0   
162                            2158.0                            387.0   
163                            2360.0                            461.0   
164                            3264.0                           1245.0   
165                            4363.0                           1649.0   

     Value:Count_Person_MarriedAndNotSeparated  Store_Present  \
0                                       3400.0              0   
1                                      12012.0              0   
2                                       3101.0              0   
3                                       8678.0              0   
4                                       4623.0              0   
..                                         ...            ...   
161                                      155.0              0   
162                                     1277.0              0   
163                                     1656.0              0   
164                                     1633.0              0   
165                                     2054.0              0   

     Predicted_Probability  Final_Probability  
0                     0.14               0.14  
1                     0.69               0.69  
2                     0.00               0.00  
3                     0.76               0.76  
4                     0.00               0.00  
..                     ...                ...  
161                   0.00               0.00  
162                   0.00               0.00  
163                   0.00               0.00  
164                   0.00               0.00  
165                   0.00               0.00  

[166 rows x 9 columns]
# Merge the cleaned data back into the original dataset
# Use a left merge to keep all original rows
complete_data = data.merge(data_cleaned[['placeName', 'Store_Present', 'Predicted_Probability', 'Final_Probability']], 
                                     on='placeName', 
                                     how='left', 
                                     suffixes=('', '_cleaned'))
​
# Fill NaN values in 'Predicted_Probability' and 'Final_Probability' with 0 or any other value as needed
complete_data['Predicted_Probability'] = complete_data['Predicted_Probability'].fillna(0)
complete_data['Final_Probability'] = complete_data['Final_Probability'].fillna(0)
​
# Save the complete DataFrame to a CSV file
complete_data.to_csv('complete_data_with_probabilities.csv', index=False)
​
print("Data saved to complete_data_with_probabilities.csv")
​
Data saved to complete_data_with_probabilities.csv
