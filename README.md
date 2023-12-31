# Price elasticity Project 📈

![](imgs/fundo.avif)

Price elasticity is a fundamental concept in economics that measures the sensitivity of the quantity demanded of a good or service in response to a change in its price. It helps to understand how consumers react to price changes and is a valuable tool for businesses and governments in making economic decisions. It is calculated by dividing the percentage change in the quantity demanded by the product by the percentage change in the price of the product.

This elasticity can be classified as elastic, inelastic and unitary price elasticity:
- Elastic: Elastic price elasticity is a measure that indicates that the demand for a product is very sensitive to changes in its price. It is calculated when the elasticity value is greater than 1. This means that consumers are very price sensitive and have many substitute products on the market. They can easily find similar products at lower prices, which encourages them to switch to these products instead of paying a higher price for the original product.
- Inelastic: Inelastic price elasticity is a measure that indicates that the demand for a product is not very sensitive to changes in its price. It is calculated when the elasticity value is less than 1. This means that consumers do not have many options for substitute products in the market or cannot easily find these products. They may be willing to pay a higher price for the original product, even if the price increases, because they don't have many alternatives available.
- Unitary: Unit price elasticity is a measure that indicates that the demand for a product is proportional to changes in its price. It is calculated when the elasticity value is exactly equal to 1. This means that consumers are moderately price sensitive and have few substitute product options in the market. They may be willing to switch to similar products if the original product's price increases significantly, but they won't switch to those products easily.

# 1.0 Project Idea

The idea of ​​the project is to study the concept of price elasticity and, therefore, the deep relationship between demand and price of products. In addition, it seeks to predict how much it is acceptable to increase/decrease the value of products, impacting demand, to try to find out if we would be able to increase revenue.

The results can be viewed on a page created in streamlit.

The dataset can be found in the "data" folder.

# 2.0 Business Assumptions

- Many columns were useless for the analysis and were removed
- Some NA data weren't removed, because it wouldn't make a difference

## 2.1 Data Description

After removing useless columns for analysis:

| Atributos                          | Descrição                                                                                                                                             |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| date_imp_d | Product posting date  |
| category_name | Pruduct category |
| name | Product name |
| price | Product price |
| disc_price | Product price |
| merchant | Product merchant |
| brand | Product brand |
| day_n | Product posting week day |
| month | Product posting month (number) |
| month_n | Product posting month |
| day | Product posting day (number) |
| week_number | Product posting week (number) |

# 3.0 Solution Strategy

![](imgs/mind_map_elastic.png)

# 4.0 Insights

1. Which merchant sold the most?
2. What is the best selling category?
3. What is the best selling brand?
4. Which days sell the most?
5. Which months sell the most?
6. Which weeks sell the most?

## 4.1 Top 3 Insights

Which merchant sold the most?

![](imgs/insight_1.png)

What is the best selling brand?

![](imgs/insight_3.png)

Which months sell the most?

![](imgs/insight_5.png)

# 5.0 Machine Learning

In the project, the method of least squares (OLS) was used, through the statsmodels library. 

The method of least squares is a statistical technique used to find the best straight line that describes the relationship between two variables. He is commonly used to perform linear regression, which is the process of finding the linear relationship between an independent variable (x) and a dependent variable (y). The objective of the least squares method is to minimize the sum of squares of the differences between the observed values ​​of y and the values ​​estimated by the regression line. In other words, it seeks to find the line that best fits the observed data, minimizing the sum of squares of forecast errors.

Through this method, we got the price_elasticity, price_mean, quantity_mean, intercept, slope, rsquared	and p_value of the products.

![](imgs/results_all.png)

Statistics summary for unique example:

![](imgs/results_unique.png)

# 6.0 Business results

Here are the business results for each product:

![](imgs/buss_results.png)

# 7.0 Streamlit

Development of a streamlit page capable of predicting the possible scenarios for each product after applying a discount or price increase, informing whether it was worth it or not

The analysis can be accessed through this link: https://elastic-preco-project.streamlit.app/

# 8.0 Conclusions

We studied the relationship between demand and price for each product, using minimum squares method, and informed the CEO of possible outcomes following a discount or price increase.

# 9.0 Lessons Learned

- Development of the price elasticity concept and visualization in practice
- Prediction data using Minimum squares method 
- Data visualization in different scenarios (discount and price increase)
- Possibility of agile and professional data query via Cloud Streamlit
