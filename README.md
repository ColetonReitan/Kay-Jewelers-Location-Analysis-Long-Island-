# Kay Jewelers Long Island Location Analysis

## Project Overview  
This is a personal project I took on to further my understanding of mapping in Tableau. 
The goal is to create a dashboard that could give some insight as to possible locations for new Kay Jewelers stores on Long Island. 

In this project, Long Island demographic and Location data is sourced from the Stanford Data Commons, and Kay Jewelers store locations are taken from Google Maps. Machine learning modeling in python is employed on the data to determine locations on Long Island similar to existing stores. Lastly, the data and new information is uploaded into Tableau and a map-based dashboard is created.  

The dashboard link can be found here:   
https://public.tableau.com/app/profile/coleton.reitan7808/viz/KaysLongIslandStoreLocations/KayJewelersLocationAnalysis?publish=yes 

### Dashboard Screenshot
![](LocationDashSS.png)

The dashboard can be filtered by Never Married Population, 18 or Older Population, or Probability of Store.   
The big light blue dots represent existing stores. The dark blue dots represent towns on Long Island, and the bigger the dot, the higher probability of a new Kay Jewelers store location.   
The coloring of Long Island represents the median household income of a town - a gradient from dark red (lower income) to gold (higher income).  

## Creation Process
This was a multi-step process that involved downloading and cleaning data from the Stanford Data Commons and Google Maps, employing a RandomForestClassifier in python, and creating a dashboard in Tableau.   

Data Cleaning and Machine Learning Code

### 1) Downloading Data from Stanford Data Commons and Google Maps
https://datacommons.stanford.edu/

Nassau and Suffolk County Data was filtered for and downloaded from the Stanford Data Commons.   
Features included from the downloaded data included:
  - PlaceName  
    Zip Code of cities/towns on Long Island    
  - Median Income Household  
    The median household income per town on Long Island  
  - Person 18 or More Years  
    The count of people per town who are 18 years or older  
  - Person Never Married  
    The count of people per town who have never been married  
  - Person Married and Not Separated  
    The count of people per town who are married and not separated  

This was a quick analysis where only a few features were taken to be used in a classification model.  

Kay Jewelers store locations were taken from Google Maps.      
This was noted by an additional binary column. 

  - Store Present  
    Binary column noting if a Kay Jewelers store is present per town (binary, 1 yes, 0 no)  
    
