# Data 
## Alternate Names Database 
  - Characters list_GC.xlsx = Complete list of characters from across all editions/translations, including all alternate names
## Exluding Groups 
  - Carson.txt = Ciaran Carson's edition excluding groups
  - Kinsella.txt = Thomas Kinsella's edition excluding groups
  - Kinsella_NoPretales.txt = Thomas Kinsella's edition excluding pretales and groups
  - ORahilly_R1_English_GC.xlsx = Cecile O'Rahilly's English translation of Recension I excluding groups
  - ORahilly_R1_Irish.xlsx = Cecile O'Rahilly's Irish edition of Recension I excluding groups
  - ORahilly_R2_Irish.xlsx = Cecile O'Rahilly's Irish edition of Recension II excluding groups
## Including Groups 
  - Carson_Groups.txt = Ciaran Carson's edition including groups
  - Kinsella_Groups.txt = Thomas Kinsella's edition including groups
  - Kinsella_NoPretales_Groups.txt = Thomas Kinsella's edition excluding pretales
  - ORahilly_R1_English_Groups_GC.xlsx = Cecile O'Rahilly's English translation of Recension I excluding groups
  - ORahilly_R1_Irish_Groups.xlsx = Cecile O'Rahilly's Irish edition of Recension I excluding groups
  - ORahilly_R2_Groups.xlsx = Cecile O'Rahilly's Irish edition of Recension II excluding groups

# Python Notebook 
## Libraries
  - Install libraries for the desired analysis
## Data
  - Imports the data mentioned above
## Alternate Names 
  - Converts the datasets mentioned above into NetworkX graph objects, using the list of altername names to ensure there are no duplicate nodes and computes their network properties
## Combined Data 
  - Uses the networkx.compose() function to combine Cecile O'Rahilly's Recensions I and II into a single graph and computes it's network properties
## Grouped Data  
  - Performs similar actions to those mentioned in _Alternate Names_, but this time for the datasets that include the groups of characters
## Combine Networks 
  - Performs similar actions to those mentioned in  _Combined Data_,  but this time for the datasets that include the groups of characters
## MLE code (Forced PL)
  - Shane Mannion's code from https://github.com/Shaneul/MLE/blob/main/MLE_functions.py adjusted to force the degree distributions to be matched to a power law distribution, allowing for gamma comparisons
## Run MLE code (Forced PL)
  - Cell used to run the code from the previous section
## MLE code 
  - Shane Mannion's code from https://github.com/Shaneul/MLE/blob/main/MLE_functions.py
## Run MLE code 
  - Cell used to run the code from the previous section
## Truncation/IM&KS values 
  - Defines functions to compute the IM and KS values when comparing networks, as well as the function which randomly prunes one network to match the size of the other for additional analysis
## Run IM/KS truncation code
  - Cell used to run the code from the previous section
## Graph Edit Distance
  - Defines a function which computes graph edit distance using gklearn.ged.env library
## Run Graph Edit Distance
  - Cell used to run the code from the previous section
## NMI (Network Mutual Information)
  - Helcio Felippe's code from https://github.com/hfelippe/network-MI and https://www.nature.com/articles/s42005-024-01830-3 defines function which computes the network mutual information
## Run NMI Code
  - Cell used to run the code from the previous section
## Spearman's/Jaccard for Character Lists
  - Defines functions which compute the Spearman's rankd and Jaccard index for comparing the character lists in terms of degree and betweennness
## Run Spearman/Jaccard
  - Cell used to run the code from the previous section
