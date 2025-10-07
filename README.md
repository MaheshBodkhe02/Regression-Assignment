# Regression-Assignment
Regression Assignment Question and Coding 


```python
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")
print(f"Difference: {r2 - adjusted_r2:.4f}")
print("\nLarge difference suggests overfitting with too many predictors")

# Compare with fewer features
X_reduced = X[:, :3]  # Use only first 3 features
model_reduced = LinearRegression()
model_reduced.fit(X_reduced, y)
y_pred_reduced = model_reduced.predict(X_reduced)

r2_reduced = r2_score(y, y_pred_reduced)
adjusted_r2_reduced = 1 - (1 - r2_reduced) * (n - 1) / (n - 3 - 1)

print(f"\nWith only 3 features:")
print(f"R²: {r2_reduced:.4f}")
print(f"Adjusted R²: {adjusted_r2_reduced:.4f}")
print(f"Difference: {r2_reduced - adjusted_r2_reduced:.4f}")
```

## 22. Why is it important to scale variables in Multiple Linear Regression?

Scaling variables in Multiple Linear Regression is important for several practical and computational reasons, though it doesn't affect the model's predictive performance when using ordinary least squares. When predictors are on vastly different scales (e.g., income in thousands versus age in years), their coefficients become incomparable—you cannot directly assess which variable has stronger influence because the units are different. Standardization (mean=0, std=1) or normalization allows fair comparison of coefficient magnitudes. For regularized regression methods like Ridge or LASSO, scaling is crucial because penalties are applied to coefficient magnitudes, and unscaled features would be penalized disproportionately based on their original units rather than their true importance. Gradient descent algorithms converge faster with scaled features, as the optimization surface becomes more spherical rather than elongated. Scaling also helps numerical stability in computations and makes it easier to set appropriate hyperparameters. However, scaling makes interpretation less intuitive since coefficients represent changes per standard deviation rather than per original unit.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np

# Create data with different scales
np.random.seed(42)
X = np.column_stack([
    np.random.randn(100) * 1000,  # Feature 1: large scale (income)
    np.random.randn(100) * 10,    # Feature 2: medium scale (age)
    np.random.randn(100)           # Feature 3: small scale (rating)
])
y = 2*X[:, 0]/1000 + 3*X[:, 1]/10 + 4*X[:, 2] + np.random.randn(100)

# Without scaling
model_unscaled = Ridge(alpha=1.0)
model_unscaled.fit(X, y)

# With scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = Ridge(alpha=1.0)
model_scaled.fit(X_scaled, y)

print("Unscaled coefficients:", model_unscaled.coef_)
print("Scaled coefficients:", model_scaled.coef_)
print("\nScaled coefficients are now comparable in magnitude!")
print("This shows the relative importance of each feature")
```

## 23. What is polynomial regression?

Polynomial regression is a form of regression analysis that models the relationship between variables as an nth-degree polynomial, allowing for curved, nonlinear relationships while still using linear regression techniques. Instead of fitting a straight line, it fits curves to the data by including polynomial terms like X², X³, and so on. The equation takes the form Y = β₀ + β₁X + β₂X² + β₃X³ + ... + βₙXⁿ + ε. Despite the nonlinear relationship between X and Y, the model remains "linear" in terms of its parameters (the β coefficients), which means standard linear regression methods can estimate them. Polynomial regression is particularly useful when data shows curvature that simple linear regression cannot capture. It provides flexibility to model complex patterns while maintaining interpretability and computational simplicity. The degree of the polynomial determines the model's flexibility—higher degrees can fit more complex patterns but risk overfitting. Common applications include growth curves, dose-response relationships, and any phenomena where effects accelerate or decelerate nonlinearly.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate nonlinear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.ravel()**2 + X.ravel() + 2 + np.random.randn(100) * 0.5

# Fit linear model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear = linear_model.predict(X)

# Fit polynomial model (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly = poly_model.predict(X_poly)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_linear, 'r-', label='Linear Regression', linewidth=2)
plt.plot(X, y_poly, 'g-', label='Polynomial Regression (degree=2)', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.show()

print("Polynomial captures the curved relationship better than linear")
```

## 24. How does polynomial regression differ from linear regression?

The fundamental difference between polynomial and linear regression lies in the functional form used to model the relationship between variables. Linear regression assumes a straight-line relationship where the effect of X on Y is constant across all values of X. Polynomial regression allows for curved relationships where the effect of X on Y changes depending on X's value, capturing acceleration, deceleration, or oscillating patterns. While linear regression has a single slope parameter, polynomial regression includes multiple terms (X², X³, etc.) with separate coefficients, each contributing to the overall shape. Despite these differences, polynomial regression is technically still a linear model because it's linear in its parameters—the coefficients can be estimated using the same ordinary least squares method. Linear regression is simpler, more interpretable, and less prone to overfitting, making it preferable when relationships are truly linear. Polynomial regression offers greater flexibility but requires careful selection of polynomial degree to avoid overfitting, and interpretation becomes more complex as you must consider combined effects of multiple terms.

## 25. When is polynomial regression used?

Polynomial regression is used when the relationship between variables exhibits clear nonlinear patterns that cannot be adequately captured by a straight line. It's particularly appropriate when data shows curvature, such as parabolic shapes, S-curves, or U-shaped patterns. Common scenarios include modeling growth rates that accelerate or slow over time, such as population growth or learning curves. It's useful in dose-response studies where effects initially increase with dosage but may plateau or decline at high levels. Economic applications include modeling diminishing returns, where benefits increase with investment but at a decreasing rate. Polynomial regression works well when you have theoretical reasons to expect polynomial relationships or when exploratory data analysis reveals curved patterns. It's preferred over more complex nonlinear methods when the relationship, while curved, isn't extremely complex and when interpretability matters. However, it should be used cautiously with appropriate degree selection, cross-validation, and awareness of its limitations at the boundaries of the data range where predictions can become unreliable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Example: Diminishing returns scenario
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
# Returns increase but at diminishing rate
y = 20 * np.sqrt(X.ravel()) + np.random.randn(50) * 2

# Try different polynomial degrees
degrees = [1, 2, 3]
plt.figure(figsize=(15, 4))

for idx, degree in enumerate(degrees, 1):
    plt.subplot(1, 3, idx)
    
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    plt.scatter(X, y, alpha=0.5, label='Actual data')
    plt.plot(X, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
    plt.xlabel('Investment')
    plt.ylabel('Returns')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()

plt.tight_layout()
plt.show()
```

## 26. What is the general equation for polynomial regression?

The general equation for polynomial regression of degree n is: Y = β₀ + β₁X + β₂X² + β₃X³ + ... + βₙXⁿ + ε, where Y is the dependent variable, X is the independent variable, β₀ is the intercept, β₁ through βₙ are the coefficients for each polynomial term, n is the degree of the polynomial, and ε represents the error term. Each coefficient captures different aspects of the relationship: β₁ represents the linear component, β₂ captures the quadratic curvature, β₃ captures the cubic inflection points, and so on. The degree n determines the model's flexibility and complexity—higher degrees allow more complex curves but increase the risk of overfitting. Despite involving powers of X, the equation remains linear in its parameters (the β values), which allows standard linear regression techniques to estimate the coefficients through ordinary least squares. The choice of degree should balance model fit with generalization, typically validated through cross-validation and consideration of the underlying phenomenon being modeled.

## 27. Can polynomial regression be applied to multiple variables?

Yes, polynomial regression can absolutely be extended to multiple variables, creating what's called multivariate polynomial regression. In this case, the model includes polynomial terms for each variable individually as well as interaction terms between variables. For two variables X₁ and X₂ with degree 2, the equation becomes: Y = β₀ + β₁X₁ + β₂X₂ + β₃X₁² + β₄X₂² + β₅X₁X₂ + ε. The model captures nonlinear effects of each variable and their combined interactions. As you increase the number of variables and the polynomial degree, the model complexity grows rapidly—the number of terms increases exponentially. While this provides tremendous flexibility to model complex, multidimensional relationships, it comes with significant challenges. The curse of dimensionality means you need exponentially more data to reliably estimate all parameters. Overfitting becomes a serious concern, requiring careful regularization and validation. Interpretation becomes increasingly difficult with many interaction terms. Despite these challenges, multivariate polynomial regression is valuable when phenomena truly involve complex, interacting nonlinear effects, particularly in fields like chemistry, physics, and response surface methodology.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate data with two variables
np.random.seed(42)
X1 = np.random.uniform(0, 5, 100)
X2 = np.random.uniform(0, 5, 100)
X = np.column_stack([X1, X2])

# Create nonlinear relationship with interaction
y = 2*X1 + 3*X2 + 0.5*X1**2 - 0.3*X2**2 + 0.4*X1*X2 + np.random.randn(100)

# Apply polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original features:", X.shape[1])
print("Polynomial features:", X_poly.shape[1])
print("\nFeature names:", poly.get_feature_names_out(['X1', 'X2']))

# Fit model
model = LinearRegression()
model.fit(X_poly, y)

print("\nCoefficients for each term:")
for name, coef in zip(poly.get_feature_names_out(['X1', 'X2']), model.coef_):
    print(f"{name}: {coef:.4f}")
```

## 28. What are the limitations of polynomial regression?

Polynomial regression has several important limitations that restrict its applicability and reliability. Overfitting is a primary concern, especially with high-degree polynomials that can perfectly fit training data by capturing noise rather than true patterns, leading to poor generalization on new data. The model exhibits extreme sensitivity at the boundaries of the data range, where polynomial curves can diverge wildly, producing unrealistic extrapolations. Multicollinearity naturally occurs between polynomial terms (X, X², X³, etc.) since they're mathematically related, causing coefficient instability and interpretation difficulties. The curse of dimensionality rapidly increases model complexity when extending to multiple variables—a second-degree polynomial with just five variables creates 20 terms. Choosing the appropriate polynomial degree is challenging, requiring domain knowledge and careful validation. Interpretability decreases significantly with higher degrees as relationships become less intuitive. Polynomial models lack theoretical justification in many real-world scenarios where other nonlinear functions (exponential, logarithmic) might be more appropriate. They're computationally sensitive to scaling and can suffer from numerical instability with very high degrees.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Demonstrate overfitting with high-degree polynomial
np.random.seed(42)
X_train = np.linspace(0, 1, 20).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train).ravel() + np.random.randn(20) * 0.1

# Test on broader range to show boundary issues
X_test = np.linspace(-0.5, 1.5, 100).reshape(-1, 1)
y_test = np.sin(2 * np.pi * X_test).ravel()

plt.figure(figsize=(15, 5))

for idx, degree in enumerate([2, 5, 15], 1):
    plt.subplot(1, 3, idx)
    
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.plot(X_test, y_test, 'g--', label='True function', alpha=0.5)
    plt.plot(X_test, y_test_pred, 'r-', label=f'Degree {degree}')
    plt.ylim(-2, 2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Degree {degree}\nTrain MSE: {train_mse:.4f}')
    plt.legend()
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
print("Notice: High degree polynomial overfits and shows wild behavior at boundaries")
```

## 29. What methods can be used to evaluate model fit when selecting the degree of a polynomial?

Several methods help evaluate model fit and select the appropriate polynomial degree, balancing complexity with predictive performance. Cross-validation, particularly k-fold cross-validation, assesses how well the model generalizes to unseen data by training on subsets and testing on held-out folds—the degree with best average performance is preferred. Train-test split provides a simpler approach where the model trains on one portion and evaluates on another, preventing overfitting assessment. Information criteria like AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) penalize model complexity while rewarding fit, naturally favoring parsimonious models. Adjusted R² accounts for the number of parameters, unlike regular R² which always increases with more terms. Learning curves plot training and validation errors against polynomial degree, revealing where overfitting begins—typically when validation error starts increasing while training error continues decreasing. Residual analysis checks for patterns suggesting inadequate fit or excessive complexity. Domain knowledge and theoretical considerations should guide the selection, as purely data-driven approaches might miss important context about the true underlying relationship.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 1, 50).reshape(-1, 1)
y = 2 * X.ravel()**2 + X.ravel() + np.random.randn(50) * 0.1

# Evaluate different polynomial degrees
degrees = range(1, 10)
train_scores = []
cv_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Training score
    train_score = model.score(X_poly, y)
    train_scores.append(train_score)
    
    # Cross-validation score
    cv_score = cross_val_score(model, X_poly, y, cv=5, 
                               scoring='r2').mean()
    cv_scores.append(cv_score)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(degrees, cv_scores, 's-', label='CV Score', linewidth=2)
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Selection: Training vs Cross-Validation Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

best_degree = degrees[np.argmax(cv_scores)]
print(f"Best polynomial degree based on CV: {best_degree}")
print(f"CV Score at best degree: {max(cv_scores):.4f}")
```

## 30. Why is visualization important in polynomial regression?

Visualization is critically important in polynomial regression because it provides intuitive understanding of model behavior that statistics alone cannot fully convey. Plotting the fitted curve against actual data immediately reveals whether the polynomial captures the relationship appropriately—you can see if the curve follows the data's natural pattern or exhibits unrealistic oscillations indicating overfitting. Visualizations expose boundary behavior, showing how predictions behave outside the training data range, where polynomials often diverge dramatically. Residual plots reveal patterns in prediction errors, indicating whether higher or lower degree polynomials are needed. Visual comparison of different polynomial degrees helps stakeholders understand trade-offs between fit and complexity without requiring deep statistical knowledge. For multivariate polynomial regression, 3D surface plots or contour plots illustrate interaction effects between variables. Visualization also aids in detecting outliers or influential points that disproportionately affect the polynomial fit. Learning curves visually demonstrate overfitting by showing divergence between training and validation performance. Ultimately, visualization bridges the gap between mathematical models and practical understanding, making polynomial regression results accessible and actionable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate data with clear nonlinear pattern
np.random.seed(42)
X = np.linspace(-2, 2, 30).reshape(-1, 1)
y = X.ravel()**3 - 2*X.ravel()**2 + X.ravel() + np.random.randn(30) * 0.5

# Create fine grid for smooth curve visualization
X_plot = np.linspace(-2.5, 2.5, 300).reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

degrees = [1, 3, 5, 10]
for idx, degree in enumerate(degrees):
    row, col = idx // 2, idx % 2
    
    # Fit polynomial
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_plot_poly = poly.transform(X_plot)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    y_plot = model.predict(X_plot_poly)
    y_pred = model.predict(X_poly)
    
    # Calculate R²
    r2 = r2_score(y, y_pred)
    
    # Plot
    axes[row, col].scatter(X, y, alpha=0.6, s=50, label='Data points')
    axes[row, col].plot(X_plot, y_plot, 'r-', linewidth=2, 
                        label=f'Polynomial fit (degree={degree})')
    axes[row, col].set_xlabel('X', fontsize=11)
    axes[row, col].set_ylabel('Y', fontsize=11)
    axes[row, col].set_title(f'Degree {degree} Polynomial (R²={r2:.3f})', 
                             fontsize=12, fontweight='bold')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_ylim(-8, 8)

plt.tight_layout()
plt.show()

print("Visualization shows:")
print("- Degree 1: Underfitting (too simple)")
print("- Degree 3: Good fit (appropriate complexity)")
print("- Degree 5: Slight overfitting (unnecessary complexity)")
print("- Degree 10: Severe overfitting (wild oscillations)")
```

## 31. How is polynomial regression implemented in Python?

Polynomial regression in Python is implemented using scikit-learn's `PolynomialFeatures` class combined with `LinearRegression`. The process involves three main steps: first, creating polynomial features from the original data using `PolynomialFeatures`, which transforms input features into polynomial terms. Second, fitting a standard `LinearRegression` model on these transformed features. Third, using the fitted model to make predictions. The `PolynomialFeatures` class handles the mathematical transformation, generating all polynomial combinations up to the specified degree, including interaction terms if desired. The `degree` parameter controls polynomial complexity, `include_bias` determines whether to include the intercept term, and `interaction_only` can limit features to only interaction terms without individual polynomial powers. After transformation, the data is treated as a standard linear regression problem, leveraging all the familiar methods and metrics. This implementation is elegant because it separates feature engineering from model fitting, making the code modular and easy to experiment with different degrees. The approach extends naturally to pipelines, cross-validation, and regularization techniques.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3*X.ravel() - 0.5*X.ravel()**2 + 0.1*X.ravel()**3 + np.random.randn(100) * 3

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: Step-by-step implementation
print("Method 1: Step-by-step")
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Method 2: Using Pipeline (recommended)
print("\nMethod 2: Using Pipeline")
polynomial_regression = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear_regression', LinearRegression())
])

polynomial_regression.fit(X_train, y_train)
y_pred_pipeline = polynomial_regression.predict(X_test)

print(f"Test R²: {r2_score(y_test, y_pred_pipeline):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred_pipeline):.4f}")

# Visualization
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
y_plot = polynomial_regression.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test data')
plt.plot(X_plot, y_plot, 'g-', linewidth=2, label='Polynomial fit (degree=3)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression Implementation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nFeature names:", poly_features.get_feature_names_out(['X']))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

---

This comprehensive guide covers all aspects of linear and polynomial regression, from fundamental concepts to practical implementation. Each answer provides clear explanations with supporting code examples, making the material accessible for both learning and practical application in data science projects.
