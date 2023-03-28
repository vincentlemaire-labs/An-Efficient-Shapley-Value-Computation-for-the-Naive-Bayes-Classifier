# Installation

Requirements for python 3.8.6 and Windows 10

Note : Code for Two Class Datasets
1. First login to [www.khiops.com](http://www.khiops.com)
2. Go then to the "Install" Pane of this site and download the windows installer (from the "Downloads" section)
3. Execute the Installer (only for windows) : double click on 'khiops-10.1.1-setup.exe' and follow all the steps
4. Install the library pykhiops (provided here) using pip install:  `pip install pykhiops.10.1.1.tar.gz`
5. Install the Python librairies if not present for you:  pandas, numpy, scipy, sklearn, shap, matplotlib


***

# Usage

Please execute the code by following theses steps in order.
The code is here given for the Adult dataset, the eighth in the list of 10 
(but you may change it to another one in the loop in each step)


### STEP 1: Training the model

> python 01_Train_Classifiers_Based_On_Khiops.py

This script trains the ANB using Khiops on the 10 datasets.

Input:
* The dataset (for example "experiments\Adult\Adult.txt").
* A dictionary file which indicates the nature of the variables (numerical or categorical), (for example "experiments\Adult\Adult.kdic").

Output:
* In each folder, this step produces new files which are the training reports and the trained model (for example "experiments\Adult\Modeling.kdic").


### STEP 2: Computation of our Shapley values and WoE values

> 02_Compute_WoE_And_OurShapley.py

This script computes the WoE values and our Shapley values.

Input: 
* The dataset (for example "experiments\Adult\Adult.txt").
* The trained model (for example "experiments\Adult\Modeling.kdic").

Output: two files 
* For example "experiments\Adult\Interpretation_SHAP.txt" (our Shapley values).
* For example "experiments\Adult\Interpretation_WOE.txt" (WoE values).

These two files contains J*C columns (J=number of variables, C=number of classes) plus additional columns.


### STEP 3: Computation of Pearson coefficient of the Table 2 in the paper

>  03_Compute_Correlation_Coefficient_Between_OurShapley_And_WoE.py

This script computes Pearson correlation coefficients between our Shapley values and WoE values.

Input: 
* For example "experiments\Adult\Interpretation_SHAP.txt".
* For example "experiments\Adult\Interpretation_WOE.txt".

Output:
* In each folder, a file containing the correlation coefficients. For example "experiments\Adult\Correlations.txt".
    This file contains:
        (i) The correlation values per variable (JxC lines) , 
        (ii) The mean and the variance of this vector.
        (iii) Our global Shapley values (average of the sum of the absolute values of all examples).
        (iv) The Global WoE values (average of the sum of the absolute values of all examples).
        Note: the variable names are not indicated but they may be find in the "Modeling.kdic" file.
    
* A file to summarize the 10 datasets: "experiments\ALL_Correlations.txt".
    This file contains the Pearson correlation coefficients for all the datasets (Table 2 in the paper) and their standard deviations.


### STEP 3bis: Computation of Kendall coefficient of the Table 2 in the paper

> 03bis_Compute_Correlation_CoefficientKendall _Between_OurShapley_And_WoE.py

This step is similar to step 3 but for the Kendall correlation coefficient.

Input: 
* For example "experiments\Adult\Interpretation_SHAP.txt".
* For example "experiments\Adult\Interpretation_WOE.txt".

Output:
* In each folder, a file containing the correlation coefficients. For example "experiments\Adult\CorrelationsKendall.txt".
    This output file contains:
        (i) N lines (number of examples) of correlation coefficients (since the analysis is per individual).
        (ii) The mean and standard deviation of this vector of N components.
        (ii) The Global values of our Shapley values and the WoE values.

* A file to summarize the 10 dataset: "experiments\ALL_CorrelationsKendall.txt".
    This file contains the Pearson correlation coefficients for all the datasets (Table 2 in the paper) and their standard deviations.


### STEP 4: Computation of the Global Kernel Shap values

> 04_ComputeKernelShap20Datasets.py

This code computes the global KenelShap values for each dataset with a budget of **two hours** per dataset.

Input: 
* the dataset (for example "experiments\Adult\Adult.txt").

Output: 
* In each folder, a file containing the global KenelShap values. For example "experiments\Adult\KernelShap_Adult.txt".
  This file contains:
    (i) The value of N_k (see Table 3 in the paper) and the time spend to compute the values.
    (ii) A line with the name of variables used by the classifier.
    (iii) C lines (the classes) and J columns (the variables). The C*J values are the KernelShap Values.


### STEP 5: Computation of our Global Shapley values

> 05_ComputeOurGlobalShapley.py

This code computes the Global Shapley values for each dataset.

Input: 
* For example "experiments\Adult\Interpretation_SHAP.txt".

Output:
* In each folder, a file containing the global Shapley values. For eaxmple "experiments\Adult\OurGlobalShap_Adult.txt".
    This output file contans C lines (the classes) and J columns (the variables).
    The  C*J values are the global Shapley values.


### STEP 6: Computation of the correlation coefficients values in Table 3

After step 4:

Using for example "experiments\Adult\KernelShap_Adult.txt" and "experiments\Adult\OurGlobalShap_Adult.txt".

We were able to compute the values in Table 3 of the paper.
This computation has been simply done using:
* Excel for the Pearson coefficients.
* The scipy.stats.kendalltau with the default parameter, i.e $\tau$-b, for the Kendall coefficients.
