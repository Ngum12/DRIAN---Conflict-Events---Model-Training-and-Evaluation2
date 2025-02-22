# Conflict Events Classification with Optimization Techniques

**Problem Statement:**  
This project investigates the classification of conflict events using various machine learning approaches. We explore classical supervised learning algorithms (Logistic Regression, SVM, XGBoost) and neural networks (both simple and optimized). Our goal is to improve model performance, convergence speed, and efficiency through techniques such as regularization (L2), different optimizers, early stopping, dropout, and hyperparameter tuning.

## Dataset Overview
- **Name**: Conflict Events Dataset  
- **Description**: This dataset contains records of conflict events (e.g., riots, protests, violence against civilians) with features such as location, date, actors involved, fatalities, etc.  
- **Features**: 
  - **Categorical**: EVENT_TYPE, ACTOR1, COUNTRY, etc.  
  - **Numerical**: YEAR, FATALITIES, LATITUDE, LONGITUDE, etc.  
- **Target**: `EVENT_TYPE` (multi-class)

## Project Structure
```
Project_Name/
├── notebook.ipynb        # Main Jupyter notebook with all code and analysis
├── saved_models/         # Directory containing saved model files
│   ├── xgb_model.pkl
│   └── optimized_nn_model.h5
└── README.md             # This file
```

## Implementation Summary
We implemented:
1. **Classical ML Models**: Logistic Regression, SVM, XGBoost (with hyperparameter tuning).  
2. **Simple Neural Network**: Baseline network without optimization.  
3. **Optimized Neural Network**: With various optimization techniques (dropout, L2 regularization, different optimizers, early stopping, etc.).  
4. **Five Training Instances**: Each with distinct hyperparameters and optimization settings.

---

## Training Instances Table

Below is the table detailing the **five** training instances for the optimized neural network, each with different combinations of optimizer, regularization, learning rate, dropout, and early stopping. The metrics (Accuracy, F1-score, Recall, Precision, Loss) are examples; replace them with your actual results.

| **Training Instance** | **Optimizer** | **Regularizer**  | **Epochs** | **Early Stopping** | **Number of Layers** | **Learning Rate** | **Accuracy** | **F1-Score** | **Recall** | **Precision** | **Loss** |
|-----------------------|---------------|------------------|------------|---------------------|-----------------------|-------------------|-------------|-------------|----------|------------|---------|
| **Instance 1**       | Adam (default)| None             | 20         | No                  | 2                     | 0.001 (default)   | 0.85        | 0.83        | 0.81     | 0.84       | 0.57    |
| **Instance 2**       | RMSProp       | L2 (0.001)       | 50         | Yes                 | 3                     | 0.0005            | 0.88        | 0.86        | 0.85     | 0.87       | 0.45    |
| **Instance 3**       | RMSProp       | L2 (0.001)       | 50         | Yes                 | 3                     | 0.0005            | 0.87        | 0.85        | 0.84     | 0.85       | 0.47    |
| **Instance 4**       | Adam          | L2 (0.001) + Dropout (0.5) | 50 | Yes | 4 | 0.001 | 0.90 | 0.89 | 0.88 | 0.90 | 0.40 |
| **Instance 5**       | Adam          | L2 (0.0005)      | 60         | Yes                 | 3                     | 0.0001            | 0.92        | 0.90        | 0.91     | 0.89       | 0.35    |

---

## Discussion of Findings

1. **Which Combination Worked Better?**  
   - Based on the table, **Instance 5** (with a lower learning rate of 0.0001, dropout 0.4, L2=0.0005, and early stopping) achieved the highest accuracy and F1-score. This suggests that a combination of **lower learning rate**, **moderate dropout**, and **L2 regularization** helps the network generalize better to the validation set.

2. **Which Implementation Worked Better Between the ML Algorithm and Neural Network?**  
   - The optimized neural network outperformed the classical ML algorithms in terms of accuracy and F1-score, indicating that the deeper architecture and regularization techniques were effective. However, **XGBoost** also performed well, especially if hyperparameters (like `max_depth` and `learning_rate`) were tuned carefully.

3. **Making Predictions Using Test Data**  
   - We split our data into training, validation, and test sets (60%, 20%, 20%). After selecting the best model based on validation metrics, we evaluated on the **test set** to ensure an unbiased estimate of performance.  
   - The final test accuracy and confusion matrices for each model are available in the notebook. Predictions can also be made interactively by inputting feature values, which the model transforms and classifies.

---

## How to Run This Notebook and Load the Best Saved Model

1. **Clone the Repository:**  
   ```bash
   git clone <your-github-repo-link>
   cd Project_Name
   ```
2. **Open the Notebook:**  
   - Open `notebook.ipynb` in Google Colab or Jupyter Notebook.  
3. **Run All Cells:**  
   - Execute the cells sequentially to load the data, preprocess, train models, and evaluate them.  
4. **Load the Best Saved Model:**  
   - For classical models (e.g., XGBoost):  
     ```python
     import joblib
     best_model = joblib.load('saved_models/xgb_model.pkl')
     ```
   - For the optimized neural network:  
     ```python
     from tensorflow.keras.models import load_model
     best_nn_model = load_model('saved_models/optimized_nn_model.h5')
     ```
5. **Video Presentation:**  
   - Provide a short (5-minute) video showing the table above, the final results, and your explanation of why certain hyperparameter combinations worked best.

---

## Conclusion
This project demonstrates how **regularization, optimizer choice, dropout, and early stopping** can significantly impact neural network performance on a multi-class classification task (conflict event classification). By systematically experimenting with different hyperparameters (as shown in the table), we identified **Instance 5** as the most effective configuration. Additionally, our optimized neural network outperformed the classical ML models in most metrics, showcasing the value of deeper architectures and careful tuning.

**Next Steps:**
- Further **hyperparameter tuning** for XGBoost or other classical models.
- **Data augmentation** or advanced techniques if the dataset has imbalanced classes.
- **Model ensembling** (combining multiple models) for potentially better performance.

---

_Use this README as a template, adjusting the metrics, descriptions, and final conclusions to match your actual project results._
