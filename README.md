# SC1015-Mini-Project

<section>
  <h2> DSAI Data Science and Aritificial Intelligence Project </h2>
  <section>
    <p> Hello! We are Lin Kai, Yee Hao and Dhairya from Nanyang Technological University. Our project will be on bitcoin, inspired by the relatively new rise of blockchain technology. Is bitcoin actually an asset class worth considering? Or is it just an asset whose price is driven by tweets and memes and influencers? </p>
    <img src = "Assets/Bitcoin.jpg" style = "width: 400px;">

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
    <p> Linear Regression is a method of "finding the best fit linear line between the independent and dependent variable:. It "fits the model with coefficients to minimze the residual sum of squares" between the set of variables. Linear Regression models are extremely common in various Machine Learning applications as they are easy to implement and interpret. As such, our group chose this model to allow us to obtain a baseline of what to expect from our other model choices. </p>
    <p> Bitcoin mining has many parameters associated with it. Our group believes that a myriad of factors regarding mining can affect the price of Bitcoin. As such, we decided to attempt multi-variate Linear Regression in hopes of obtaining a more accurate prediction model for the pricing of Bitcoin. </p>
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
    <p> Based on our model, we obtained the various coefficients of regression: 1.23x10^-10, -1.32x10^-3, 1.36x10^-2 and 1.53x10^-4, for Average Difficulty, Cumulative Total Number of Coins, Number of Active Addresses and Daily Hash Rate respectively. Out of the 4 variables, the highest coefficient of regression belonged to Number of Active Addresses, which tells us that it had the biggest impact on the price of Bitcoin. This piece of information will help us in our analysis for all subsequent model building.</p>
    <p>From our Linear Regression Model, we obtained an R^2 of 0.760 on our train data set, and 0.752 on our test data set, while MSE value was 2864396 on the train data set and 2796655 on our test data set. Solely based off these numbers, we see that our model has a rather high accuracy, with about 75% of all the actual data being predicted by our model is correct.</p>
    <p>For the MSE values, since there are no values for us to compare against yet, we will use these values later on once we have built our other models. In general however, a smaller MSE value would indicate that the model is more accurate. The results obtained from this Multi-Variate Linear Regression will be used as a baseline for what to expect for our 3 remaining unbuilt models.</p>
    <h3>Machine Learning:Random Forest Regression</h3>
    <p>
