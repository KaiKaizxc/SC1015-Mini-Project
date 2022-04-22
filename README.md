# SC1015-Mini-Project

<section>
  <h2> DSAI Data Science and Aritificial Intelligence Project </h2>
  <section>
   <p> Hello! We are Lin Kai, Yee Hao and Dhairya from Nanyang Technological University. Our project will be on Bitcoin, inspired by the relatively new rise of blockchain technology. There is a general notion that investment in Bitcoin is a high risk affair, due to the volatile nature of cryptocurrency, as well as the fact that it is 'not backed by anything', unlike most currencies. Our group finds the world of mining to be a truly fascinating one. </p>
   <img src = "Assets/Bitcoin.jpg" style = "width: 400px;"/>
   
   <h3>Problem Statement</h3>
   <p> With the recent boom in mining activity over the course of the pandemic, we could not help but wonder: how does mining activity affect Bitcoin pricing? Is bitcoin actually an asset class worth considering? Or is it just an asset whose price is driven by tweets and memes and influencers? 
    <b>For simplicity sake, we will assume bitcoin’s price is reflective of its state in the future.</b>
  
   </p>
    <h3>Bitcoin's price over the years</h3>
    <img src = "MatplotLibGraphs/PriceYears.png" style = "width: 1000px;"/>
    
   
    
  </section>
  
</section>

<section>
  <h2>Exploratory Data Analysis</h2>
  




<section> 
  <h2> Analysis of Model Building </h2>
  <section>
    <p> The 4 types of Machine Learning models that we have decided to build are as follows:
      <p>
      -Multi-Variate Linear Regression
      </p>
      <p>
      -Random Forest Regression
      </p>
      <p>
      -K Nearest Neighbors Regression
      </p>
      <p>
      -Neural Networks
      </p>
   Our group shall proceed to analyze each of the 4 models we have chosen to see their effectiveness in predicting the price of Bitcoin. In general, we will be looking at 4 metrics(where applicable) for each of the models. These are namely: Mean Absolute Error(MAE), Mean Squared Error(MSE), Root Mean Square Error(RMSE), as well as the Explained Variance(R^2). However, there are notable differences in characteristics between the 4 models, which our group will address. These differences might explain certain behavior or predictions by the models.
    <h3> Machine Learning: Multi-Variate Linear Regression </h3>
    <p> Linear Regression is a method of "finding the best fit linear line between the independent and dependent variable:. It "fits the model with coefficients to minimze the residual sum of squares" between the set of variables. Linear Regression models are extremely common in various Machine Learning applications as they are easy to implement and interpret. As such, our group chose this model to obtain a baseline of what to expect from our other models. </p>
    <p> Bitcoin mining has many parameters associated with it. Given the volatile nature of Bitcoin, it is more likely that a myriad of factors regarding mining affects the price of Bitcoin, rather than one primary factor. As such, we decided to attempt multi-variate Linear Regression in hopes of obtaining a more accurate prediction model for the pricing of Bitcoin. </p>
    Below shows the code for our Linear Regression model:
    
    ```
    # Import train_test_split from sklearn
    from sklearn.model_selection import train_test_split

    # Split the Dataset into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # Check the sample sizes
    print("Train Set :", y_train.shape, X_train.shape)
    print("Test Set  :", y_test.shape, X_test.shape)
    
    #Import LinearRegression model from Scikit-Learn
    from sklearn.linear_model import LinearRegression
    
    #Linear Regression using Train Data
    linreg = LinearRegression()         # create the linear regression object
    linreg.fit(X_train, y_train)        # train the linear regression model
    
    #Coefficients of the Linear Regression line
    print('Intercept of Regression \t: b = ', linreg.intercept_)
    print('Coefficients of Regression \t: a = ', linreg.coef_)
    print()
    
    #Print the Coefficients against Predictors
    pd.DataFrame(list(zip(X_train.columns, linreg.coef_[0])), columns = ["Predictors", "Coefficients"])
    
    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)

    Import mean_squared_error from sklearn
    from sklearn.metrics import mean_squared_error

    #Check the Goodness of Fit (on Train Data)
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    print()

    #Check the Goodness of Fit (on Test Data)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
    print()
    ```
    
   <p> The independent variables we chose were Average Difficulty, Cumulative Total Number of Coins, Number of Active Addresses, and Daily Hash Rate, measured in trillions of hashes per second. Of course, our dependent variable was price of Bitcoin. </p>
    <p> Based on our model, we obtained the various coefficients of regression: 1.23x10^-10, -1.32x10^-3, 1.36x10^-2 and 1.53x10^-4, for Average Difficulty, Cumulative Total Number of Coins, Number of Active Addresses and Daily Hash Rate respectively. TO COMPLETE</p>
    <p>From our Linear Regression Model, we obtained an R^2 of 0.760 on our train data set, and 0.752 on our test data set, while MSE value was 2864396 on the train data set and 2796655 on our test data set. Solely based off these numbers, we see that our model has a rather high accuracy, with about 75% of all the actual data being predicted by our model is correct.</p>
    <p>For the MSE values, since there are no values for us to compare against yet, we will use these values later on once we have built our other models. In general however, a smaller MSE value would indicate that the model is more accurate. The results obtained from this Multi-Variate Linear Regression will be used as a baseline for what to expect for our 3 remaining unbuilt models.</p>
    <h3>Machine Learning:Random Forest Regression</h3>
    <p> Random Forest is a "meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting". Interestingly, our group found during our research, that Random Forest models, although typically used to perform classification, can also double as a regression model. As such, we decided to try it out, to see if it produces a better model than our Linear Regression model. </p>
    <p> The process of Random Forest is as follows: Out of all the rows within the dataset, the same number of them will be selected at random, with repetition being allowed. This becomes the 'Train' data. Then, out of the columns that contain the independent variables Average Difficulty, Cumulative Total Number of Coins, Number of Active Addresses, and Daily Hash Rate, a number of columns will be randomly selected, which allows for the splitting at each parent node. This process is known as bootstrapping, and it is repeated multiple times to create multiple smaller trees, each tree with a different set of 'Train' data and variables chosen for regression. The price predicted by the model is an average of all the various predictions created by the smaller trees.</p>
    <p> Random Forest Regression is effective because it combines the simplicity of a decision tree with flexibility in the multiple repetitions, creating a more accurate model. Our code allowed the scikit-learn algorithm to determine the number of trees it would create, as well as the number of independent variables sampled for each decision tree created. However, fine tuning can be done to have control over all these randomized aspects, which can help to prevent issues like overfitting, helping to bring up the accuracy of the model even further.</p>
    Here is our code for the Random Forest Regression:
    
    ```
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    
    tree_model = DecisionTreeRegressor()
    rf_model = RandomForestRegressor()
    
    tree_model.fit(train_scaled, y_train)
    rf_model.fit(train_scaled, y_train)
    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    
    tree_mse = mean_squared_error(y_train, tree_model.predict(train_scaled))
    tree_mae = mean_absolute_error(y_train, tree_model.predict(train_scaled))
    
    rf_mse = mean_squared_error(y_train, rf_model.predict(train_scaled))
    rf_mae = mean_absolute_error(y_train, rf_model.predict(train_scaled))

    print("Decision Tree training mse = ",tree_mse," & mae = ",tree_mae," & rmse = ", sqrt(tree_mse))
    print("Random Forest training mse = ",rf_mse," & mae = ",rf_mae," & rmse = ", sqrt(rf_mse))
    
    tree_test_mse = mean_squared_error(y_test, tree_model.predict(test_scaled))
    tree_test_mae = mean_absolute_error(y_test, tree_model.predict(test_scaled))
    
    rf_test_mse = mean_squared_error(y_test, rf_model.predict(test_scaled))
    rf_test_mae = mean_absolute_error(y_test, rf_model.predict(test_scaled))
    
    print("Decision Tree test mse = ",tree_test_mse," & mae = ",tree_test_mae," & rmse = ", sqrt(tree_test_mse))
    print("Random Forest test mse = ",rf_test_mse," & mae = ",rf_test_mae," & rmse = ", sqrt(rf_test_mse))
    ```

   <p>Our group included code for if a singular Decision Tree was used for regression only as a comparison to the Random Forest model. Out of the multiple times that we ran the model, RMSE, MSE and MAE values were all generally lower in the Random Forest model, showing that the Random Forest process does help improve the accuracy of the model.</p>
    <h3>Machine Learning:K Nearest Neighbors</h3>
    <p></p>
    <h3>Machine Learning:Neural Networks</h3>
    <p></p>
    <h2>Analysis and Insights</h2>
    <p>From our model building, we can see that the models we have built are all reliable at providing us a prediction of the price of Bitcoin, as all of them have a R^2 value of at least 0.75. This indicates to us that, from our preliminary data analysis, the 4 independent variables that we selected were good candidates for our model building. This tells us that out of the 17 variables available to us in the data set, knowledge of only Average Difficulty, Cumulative Total Number of Coins, Number of Active Addresses and Daily Hash Rate will allow us to obtain a largely accurate prediction for the price of Bitcoin. It is important to note that each of the 4 models have their own pros and cons, and any one of the models can be better than the rest under the right circumstances. </p>
    <p>Being able to obtain the price prediction of Bitcoin for a given set of data allows us to analyze a few things by comparing it to the actual price of Bitcoin. It helps us determine whether, based on our model’s prediction, if Bitcoin was over-valued, or under-valued at any given point in time. If the predicted price exceeds the actual price of Bitcoin for a given time, it could be seen as an indication that it is a good time to purchase Bitcoin, as it is currently under-valued. Additionally, if our price prediction is observed to be consistently and noticeable lower or higher than the actual price, we can conclude that, while mining activity affects the price of Bitcoin, there are other external influences that can affect Bitcoin pricing to a much larger extent, such as tweets from famous personalities like Elon Musk, who has enough influence to significantly impact market demand. </p>
    <p>All of our models pointed us to the conclusion that mining activity does have a profound impact on the price of Bitcoin. Generally, as mining activity increases, the price of Bitcoin rises along with it. This is because difficulty in mining rises as the mining space becomes more saturated. With the need for more sophisticated, higher cost computer components to remain competitive within the mining space, it naturally leads to an increase in Bitcoin price. Increasing Bitcoin prices would lead to rising interest in mining, highlighting greater trust and belief of the profitability of Bitcoin. If this trajectory continues for Bitcoin, the success of Bitcoin will only continue to grow for many more years to come. This also helps cement the fact that Bitcoin is an asset worth consideration. The stigma around cryptocurrencies being volatile and valueless has led people to believe that it is an ‘inferior’ investment relative to fiat currency. It is important to note that our analysis was focused on the mining scene of Bitcoin; there are many other factors that could indicate the growth in interest and success in Bitcoin. For instance, the adoption of payment in cryptocurrency by some organizations highlights how Bitcoin has evolved from being the center of tweets and memes, to being the fundamentally sound, massive giant that it is today. While our analysis was focused around Bitcoin, the insights to be gained remain largely the same for the various other types of cryptocurrencies.</p>
    <h2>Conclusion</h2>
    <p>All in all, our group has learned how to utilize various methods to obtain price prediction models for the price of Bitcoin. The price data that we were able to obtain allows us to confidently conclude that Bitcoin is an asset that is grounded in reality, and not just fueled by tweets and memes, like many other people might suggest. There is a real value and backing to Bitcoin, making it no less different from fiat currency or conventional stocks and bonds. Bitcoin is an asset that has a good system and strong fundamentals backing it up. As such, it is an asset that is worth investing in.</p>
