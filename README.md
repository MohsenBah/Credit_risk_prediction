# Credit_risk_prediction


The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes. The link to the original dataset can be found below.


## Content

1- Preprocessing: 

It is almost impossible to understand the original dataset due to its complicated system of categories and symbols. Thus, it is needed to do much preprocessing. Moreover, I plan to deploy the final model, so I chose these features.

    Age (numeric)
    Sex (text: male, female)
    Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
    Housing (text: own, rent, or free)
    Saving accounts (text - little, moderate, quite rich, rich,nan)
    Checking account (text - 'little', 'moderate', 'rich', nan)
    Credit amount (numeric)
    Duration (numeric, in month)
    Purpose(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)
    Risk (Value target - Good or Bad Risk)


2- Classification

As shown, LDA and Logistic Regeression the powerful models according to the AUC measure with startified cross validation.
I chose LR with l2 penalty as a final model.

<img src="static/img/models.png" width="600px">


3- web app by Flask
    
    To run tha web app please run app.py
    
<img src="static/img/webpage.png" width="600px">


4- Heroku 

Find the webapp here on Heroku:
    
https://credit-risk-webapp.herokuapp.com/

    
 ## Reference   


Professor Dr. Hans Hofmann
Institut f"ur Statistik und "Okonometrie
Universit"at Hamburg
FB Wirtschaftswissenschaften
Von-Melle-Park 5
2000 Hamburg 13 

https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
