# Final-Project-Data-Wrangling

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load dataset
df = pd.read_csv('C:/Users/clark/Downloads/Housing.csv')

# Scatterplot of area vs price
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='area', y='price')
plt.title('Square Footage vs Sale Price')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Sale Price')
plt.show()

# Correlation
corr_value = df['area'].corr(df['price'])
print("Correlation between area and price:", corr_value)

# Mainroad vs Non-Mainroad Average Prices
avg_prices = df.groupby('mainroad')['price'].mean().reset_index()
avg_prices['mainroad'] = avg_prices['mainroad'].map({0: 'No', 1: 'Yes'})

plt.figure(figsize=(7,5))
sns.barplot(data=avg_prices, x='mainroad', y='price')
plt.title('Average House Price: Mainroad vs Non-Mainroad')
plt.xlabel('Located on Main Road?')
plt.ylabel('Average Price')
plt.tight_layout()
plt.show()

# Correlation again
corr_value2 = df['mainroad'].corr(df['price'])
print("Correlation between mainroad and price:", corr_value2)

# Multiple Regression
X = df[['area', 'mainroad', 'basement', 'stories']]
y = df['price']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())

# Additional Correlation Check
corr_value = df['parking'].corr(df['price'])
print("Correlation between area and price:", corr_value)
