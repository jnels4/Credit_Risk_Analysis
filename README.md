# Credit_Risk_Analysis
analyzing customer data to determine credit risk potential

## Overview

The purpose of this analysis is to use machine learning to determine the viability of predicting credit risk.  Most loans are "good" loans, meaning - "low risk" with little chance of the borrower defaulting; however, there are still some "high risk" loans, and we want to predict these loans based on a credit data set.

## Resources
- python
- pandas
- jupyter notebook
- sklearn
- imblearn
- numpy

## Results

- Model: Naive Random Over Sampling
- Balanced Accuracy: .65
- Precision: .99
- Recall (low risk): .71
- Recall (high risk): .58

![Screen Shot 2022-08-18 at 3 38 38 PM](https://user-images.githubusercontent.com/6634774/185480006-8bfd263c-c722-4967-8ba8-2f057c4db789.png)
![Screen Shot 2022-08-18 at 3 38 58 PM](https://user-images.githubusercontent.com/6634774/185480055-cd7ced10-b088-4399-a1f3-cae21c20214b.png)

- Model: SMOTE Oversampling
- Balanced Accuracy: .63
- Precision: .99
- Recall (low risk): .64
- Recall (high risk): .62
 
![Screen Shot 2022-08-18 at 3 41 45 PM](https://user-images.githubusercontent.com/6634774/185480538-bc320adc-8ead-40bd-922c-c703306ce3ab.png)
![Screen Shot 2022-08-18 at 3 42 06 PM](https://user-images.githubusercontent.com/6634774/185480600-79bb9ded-4e2f-47d5-acbd-748b94084512.png)

- Model: Undersampling (cluster centroids)
- Balanced Accuracy: .53
- Precision: .99
- Recall (low risk): .45
- Recall (high risk): .61

![Screen Shot 2022-08-18 at 3 43 01 PM](https://user-images.githubusercontent.com/6634774/185481090-d55cb401-bb59-4c12-bb7e-697792a03485.png)
![Screen Shot 2022-08-18 at 3 43 13 PM](https://user-images.githubusercontent.com/6634774/185481120-a7f5f434-b16d-4ea7-9c35-ef59120d1929.png)

- Model: Combination (over & under sampling) "SMOTEENN"
- Balanced Accuracy: .64
- Precision: .99
- Recall (low risk): .58
- Recall (high risk): .70

![Screen Shot 2022-08-18 at 3 44 38 PM](https://user-images.githubusercontent.com/6634774/185481336-d10d9177-72ef-4f2d-8733-2cefc486e6eb.png)
![Screen Shot 2022-08-18 at 3 44 51 PM](https://user-images.githubusercontent.com/6634774/185481369-a2f0be82-94b1-4e3e-b0d7-66b49b933dca.png)

- Model: Ensemble Learners - Balanced Random Forest Classifier
- Balanced Accuracy: .71
- Precision: .99
- Recall (low risk): .91
- Recall (high risk): .45

![Screen Shot 2022-08-18 at 3 49 53 PM](https://user-images.githubusercontent.com/6634774/185482199-6404c7ab-c5ae-4b85-bddb-98ec29b9138b.png)
![Screen Shot 2022-08-18 at 3 50 34 PM](https://user-images.githubusercontent.com/6634774/185482310-19da2492-629c-4245-bda3-1b6405240bea.png)

- Model: Ensemble Learners - ADABoost Classifier
- Balanced Accuracy: .93
- Precision: .99
- Recall (low risk): .92
- Recall (high risk): .94

![Screen Shot 2022-08-18 at 3 52 14 PM](https://user-images.githubusercontent.com/6634774/185482630-cbb3f238-e802-47fd-aad4-50b0aaf38645.png)
![Screen Shot 2022-08-18 at 3 52 26 PM](https://user-images.githubusercontent.com/6634774/185482659-6695e51b-6af3-4b8f-b8b1-e37afcaa22b3.png)

## Summary

Each of these models were extremly precise and able to easily predict the outcome of the dataset because a vast majority of the loans were low risk, thus it logically makes sense that if a model classifies most of the entries as low-risk it will reach a very high precision score; so if we made our decisions solely on precision, we could - theoretically - roll a 6 sided dice and have the same outcome.  This is not what we want.  We need a model that is great at predicting low risk loans, but is also great at predicting high risk loans; in this case, we need need not only precision and accuracy but high levels of recall or sensitivity in our model.  In other words, we want our model to be very good at identifying the lower percentatge high risk loans - the only model capable of this was the "ADABoost CLassifier", an ensemble model that recruits multiple different learning nodes and creates a final predictive outcome.

## Recommendation

ADABoost Classifier.

Amongst all the models that ran, this model was the best at predicting high-risk loans without losing the low-risk loans in the process.  It had a very good accuracy score of 93% and excellent recall for the low/high risk loan tipes at 92 and 94 respectivly.  

### Drabacks

It is possible that this model is overfit.  We would need to run a new set of data to determine if the outcomes managed to stay the same, or this model was only good at predicting this specific data set.  However, based on the data provided, we could suggest going with the ADABoost Classifier going forward.
