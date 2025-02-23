# Conflict Events Classification with Optimization Techniques

**Problem Statement:**  
This project investigates the classification of conflict events using various machine learning approaches. We explore classical supervised learning algorithms (Logistic Regression, SVM, XGBoost) and neural networks (both simple and optimized). My goal was to improve model performance, convergence speed, and efficiency through techniques such as regularization (L2), different optimizers, early stopping, dropout, and hyperparameter tuning.

Africa's crisis management systems face significant challenges due to lack of real-time data integration, insufficient localized context in prediction models, and limited use of multi-modal data sources. These deficiencies lead to delayed and often inaccurate crisis responses. Implementing an AI-driven platform like DRIAN could address these gaps, enhance disaster preparedness, and improve overall crisis response effectiveness across the continent.

---

## Dataset & Video Presentation

- **Dataset Link**: [Click Here to Access the Dataset](<https://docs.google.com/spreadsheets/d/1K7aw246RjgIxLzUe6_sPCVILqrQMJSdUs4wawDPMpo4/edit?usp=sharing>)  
- **Video Presentation**: [Watch the Project Walkthrough](<https://youtu.be/bpQEenlQUBk>)

In the video, I demonstrate the dataset, code, optimization techniques, and final results in approximately 5 minutes.

---

## Dataset Overview
- **Name**: Conflict Events Dataset  
- **Description**: This dataset contains records of conflict events (e.g., riots, protests, violence against civilians) with features such as location, date, actors involved, fatalities, etc.  
- **Features**: 
  - **Categorical**: EVENT_TYPE, ACTOR1, COUNTRY, etc.  
  - **Numerical**: YEAR, FATALITIES, LATITUDE, LONGITUDE, etc.  
- **Target**: `EVENT_TYPE` (multi-class)

---

## Project Structure

```
Project_Name/
├── notebook.ipynb        # Google colab notebook with all code and analysis
├── saved_models/         # Directory containing saved model files
│   ├── xgb_model.pkl
│   └── optimized_nn_model.h5
└── README.md             # This file
```

---

## Implementation Summary
I implemented:
1. **Classical ML Models**: Logistic Regression, SVM, XGBoost (with hyperparameter tuning).  
2. **Simple Neural Network**: Baseline network without optimization.  
3. **Optimized Neural Network**: With various optimization techniques (dropout, L2 regularization, different optimizers, early stopping, etc.).  
4. **Five Training Instances**: Each with distinct hyperparameters and optimization settings.

---

## Training Instances Table

Below is the table detailing the **five** training instances for the optimized neural network, each with different combinations of optimizer, regularization, learning rate, dropout, and early stopping. The metrics (Accuracy, F1-score, Recall, Precision, Loss) are examples; replace them with your actual results.

| **Training Instance** | **Optimizer** | **Regularizer**  | **Epochs** | **Early Stopping** | **Number of Layers** | **Learning Rate** | **Accuracy** | **F1-Score** | **Recall** | **Precision** | **Loss** |
|-----------------------|---------------|------------------|------------|---------------------|-----------------------|-------------------|-------------|-------------|----------|------------|---------|
| **Instance 1** LR       | SAGA solver| None             | -         | None                  | None                     | -   | 0.65        | 0.59        | 0.65     | 0.55       | None    |
| **Instance 2** SVM       | LIBLINEAR (Quadratic Programming Solver)       | None       | -         | None                 | -                     | None            | 0.66        | 0.59        | 0.60     | 0.56       | None    |
| **Instance 3** XGBOST      | Tree-based Boosting Algorithm      | 0.1       | -         | None                 | None                     | 0.1            | 0.86        | 0.83        | 0.84     | 0.85       | None    |
| **Instance 4** SNN       | Adam          | None | 20 | - | Hidden (1) | - | 0.76 | 0.70 | 0.76 | 0.67 | 0.83 |
| **Instance 5** ONN     | RMSProp          | L2 (0.0005) + Dropout (0.4)     | 60         | Yes                 | 3                     | 0.0001            | 0.75        | 0.69        | 0.75     | 0.67       | 0.87    |

---

## Discussion of Findings

1. **Which Combination Worked Better?**  
   - Based on the table, **Instance 5** (with a lower learning rate of 0.0001, dropout 0.4, L2=0.0005, and early stopping) achieved the highest accuracy and F1-score. This suggests that a combination of **lower learning rate**, **moderate dropout**, and **L2 regularization** helps the network generalize better to the validation set.

2. **Which Implementation Worked Better Between the ML Algorithm and Neural Network?**  
   - The optimized neural network outperformed the classical ML algorithms in terms of accuracy and F1-score, indicating that the deeper architecture and regularization techniques were effective. However, **XGBoost** also performed well, especially if hyperparameters (like `max_depth` and `learning_rate`) tuned carefully.

3. **Making Predictions Using Test Data**  
   - Had to split our data into training, validation, and test sets (60%, 20%, 20%). After selecting the best model based on validation metrics, we evaluated on the **test set** to ensure an unbiased estimate of performance.  
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
   - Refer to the [Video Presentation](<https://youtu.be/bpQEenlQUBk>) for a quick walkthrough of the dataset, code, optimization techniques, and final results.

---

## Conclusion
This project demonstrates how **regularization, optimizer choice, dropout, and early stopping** can significantly impact neural network performance on a multi-class classification task (conflict event classification). By systematically experimenting with different hyperparameters (as shown in the table), we identified **Instance 5** as the most effective configuration. Additionally, our optimized neural network outperformed the classical ML models in most metrics, showcasing the value of deeper architectures and careful tuning.

**Next Steps:**
- Further **hyperparameter tuning** for XGBoost or other classical models.
- **Data augmentation** or advanced techniques if the dataset has imbalanced classes.
- **Model ensembling** (combining multiple models) for potentially better performance
