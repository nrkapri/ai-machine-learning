# Apriori

Apriori is an algorithm for frequent item set mining and association rule learning over relational databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database. 


#### support

support for Item I =  (# transactions containing I)/(# transactions)

#### confidence 

confidence (I1 --> I2) (who bought I1 also bought I2) =  (# transaction containing I1 and I2)/(# transactions containing I1)

#### lift

lift (I1 --> I2) () = confidence(I1 --> I2) / Support(I1 --> I2)

#### Pseudocode :

* Step1: set a minimum support and confidence 

* Step2: Take all the subsets in transactions having higher support than minimum support 

* Step3: Take all the rules of these subsets having higher confidence than minimum confidence 

* Step4: Sort the rules by decreaing lift 



 
