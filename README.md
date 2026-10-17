
# AI Job Market & Salary Predictor

**Author:** Aditya Prasad Das  
**Programme:** MSc Artificial Intelligence, University of Bristol

## Project Overview

This project demonstrates an end-to-end machine learning
workflow for estimating technology salaries from job-related
features such as role, location, education, experience and
skills.

The current version uses synthetic demonstration data.
Its predictions are not verified UK salary statistics.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regression
- Streamlit
- Joblib

## Key Features

- Generate and prepare a structured dataset
- Encode categorical variables
- Scale numerical features
- Train a Random Forest regression model
- Evaluate predictions using MAE and R-squared
- Explore an interactive salary prediction dashboard

## Project Structure

    ai-job-market-salary-predictor/
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    ├── README.md
    └── models/

## How to Run

Install the dependencies:

    pip install -r requirements.txt

Train and save the model:

    python train_model.py

Launch the dashboard:

    streamlit run app.py

## Evaluation

The training script reports:

- Mean Absolute Error (MAE)
- R-squared score

These metrics evaluate performance on a held-out portion
of the synthetic dataset. They do not demonstrate accuracy
on real-world salaries.

## Future Improvements

- Use a credible, properly licensed salary dataset
- Add exploratory data visualisations
- Compare multiple regression models
- Improve validation and error analysis
- Deploy the dashboard online

## Disclaimer

This is an educational portfolio project. Salary predictions
are illustrative and should not be used for financial or
career decisions.

## Author

Aditya Prasad Das
