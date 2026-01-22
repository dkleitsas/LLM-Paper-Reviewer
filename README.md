# A LLM-based paper evaluation pipeline

This repository contains the implementation of a modular, machine learning-based architecture for the automatic evaluation of scientific research papers, using Natural Language Processing (NLP) techniques and Large Language Models (LLMs). 

The system performs:

- **Document parsing** from PDF to raw text  
- **Section segmentation** using a hybrid LSTM–BERT model  
- **Post-processing and correction** of segmentation inconsistencies  
- **Section-level classification** into accepted/rejected labels  
- **Document-level decision aggregation** based on section predictions


## Achieved performance

TASK | ACCURACY
--- | --- | 
| Section Segmentation         | 80%      |
| Section-Level Classification | 77%      |
| Paper-Level Decision (with CWA)   | **92%**  |

## Visualization of performance of different aggregation techinques
![image](https://github.com/user-attachments/assets/cb658ba3-3a72-491d-8e11-d93fa9d48150)
