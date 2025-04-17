# 🏡 Linear Regression – California Housing Price Prediction

This project uses a Linear Regression model to predict house prices based on the California housing dataset from the 1990 U.S. Census. It's part of my hands-on machine learning learning journey.

---

## 🔍 Problem Statement

> Predict the `median_house_value` of a house using features such as:
- Median income
- Number of rooms
- Population
- Households
- Housing age

---

## 📊 Dataset

- **Source**: [California Housing Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- Includes ~20,000 entries
- Real-world, tabular data with some missing values

---

## 🛠️ Steps Performed

1. ✅ Data cleaning (handled missing values using median)
2. ✅ Feature selection (used only numerical features)
3. ✅ Train-test split (80/20)
4. ✅ Trained `LinearRegression` model from scikit-learn
5. ✅ Evaluated performance using:
   - MAE
   - MSE
   - RMSE
6. ✅ Visualized predictions using scatter plot

---

## 📈 Results

- **MAE**: ~$56,714
- **RMSE**: ~$77,258
- The model learned basic trends but under/overpredicted in some ranges

---

## 📌 Key Learnings

- Hands-on with `model.fit()`, weights, bias
- Importance of feature selection
- How to interpret error metrics (MAE vs MSE vs RMSE)
- Visual inspection of model accuracy

---

## 💡 Next Steps

- Add `ocean_proximity` using one-hot encoding
- Try more powerful models (e.g., Random Forest, XGBoost)
- Build a Streamlit web app to deploy the model interactively

---

🧠 _Part of my [Machine Learning Journey](https://github.com/anurag1210/anurag-ml-journey)_
