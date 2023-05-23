# Lung-PNet

![Lung-PNet](./assets/banner.png)

Lung-PNet is an end-to-end deep transfer learning network designed for the diagnosis of invasive adenocarcinomas (IAC) in pure ground-glass nodules (pGGNs) on non-enhancement high-resolution CT (HRCT). 

## Objectives
The objective of Lung-PNet is to develop and validate an end-to-end 3D deep learning model pipeline for the detection of IAC histologic subtypes of pGGNs. 

## Methods
Lung-PNet was developed in a retrospective study to automatically segment pGGN lesions and classify IAC subtypes in a single-center lung CT image of patients who underwent surgical resection for pulmonary nodules between January 2019 and June 2022. The model’s performance was evaluated using a Dice score to measure its ability to detect and segment pGGNs, and the area under the receiver operating characteristic curve (AUC) to measure its ability to classify pGGNs.

## Datasets
We worked with datasets of 448 patients, separated into training, test, and holdout datasets. The datasets were distributed as follows:
- Training Dataset: 327 pGGNs
- Test Dataset: 82 pGGNs
- Holdout Dataset: 39 pGGNs

## Results
The performance metrics of the Lung-PNet model were as follows:

- Dice Score for pGGN Segmentation:
  - Internal Test Dataset: 0.868 (95% CI: 0.85-0.90)
  - Holdout Test Dataset: 0.860 (95% CI: 0.81-0.89)
  
- AUC for IAC Diagnosis:
  - Internal Test Dataset: 0.925 (95% CI: 0.85-1.00)
  - Holdout Test Dataset: 0.911 (95% CI: 0.82-0.99)

- Sensitivity: 54.8%
- Specificity: 95.6%
- Accuracy: 85.1%

Comparatively, the average values for four human readers (three radiologists and one surgeon) were 65.3% sensitivity, 74.5% specificity, and 72.1% accuracy. 

## Conclusions
Lung-PNet provides a non-invasive and user-friendly tool for detecting IAC from pGGN on HRCT. This 3D deep learning model outperformed three radiologists and one surgeon in predicting pGGN-like IAC and non-IAC, significantly improving the predictive performance of these physicians. This tool could serve as an essential aid for clinicians, enabling them to automatically segment candidate lesions such as pGGNs and subsequently detect IAC subtypes for surgical intervention, enhancing patient management and informed treatment decisions.

## Keywords
pure ground-glass nodule (pGGN), invasive adenocarcinoma (IAC), CT, deep learning
