==================
Final Verdict
==================
--->    The issue is similar to predicting the outcome of a coin toss.
--->    Predicting the probability of heads or tails can't be done with a predictive model if the coin is fair.

--->    If your target is independent of your features and has an equal probability distribution:
           #    Building a predictive model isn't recommended.

--->    It's essential to determine if there's a pattern between the response and any features.
--->    In this case, there's no noticeable pattern or connection.


==================
Book Analysis
==================

1. **Null Value Check**:
   - No null values detected.

2. **Duplicate Record Check**:
   - Excluding CustomerID & Name (which are unique), no duplicates found.

3. **Data Balance**:
   - The dataset is balanced.

4. **Statistical Analysis**:
   - The mean and median values are closely aligned, indicating a well-distributed dataset.

5. **Visualization**:
   - Pie Chart: Showed gender, churn, and other metrics in a balanced distribution.
   - Location-Gender Distribution:
     - Los Angeles, Chicago: Predominantly Male
     - Miami: Predominantly Male
     - New York, Houston: Equal Male and Female distribution
   - Subscription Length & Monthly Bill: Unique counts plotted to visualize frequency distribution.
   - Age, Monthly Bill, Total Usage (in GB): Displayed similar user frequencies across groups.
   - Heatmap: Collinearity in the data is less than 0.05%, indicating minimal correlation between target (Y) and input variables (X).
   - Pair Plot: No significant insights due to lack of collinearity.






==================
Churn Modelling via Machine Learning
==================

1. **Data Encoding**:
   - **Gender**: Encoded as 0 (Male) & 1 (Female) using a Lambda function for machine readability.
   - **Location**: Used ordinal encoding. Tested one-hot encoding due to the limited number of features, but results were consistent due to the balanced nature of the data.

2. **Data Segregation**:
   - Segregated and split data into X (features) and Y (target). Excluded 'CustomerID' & 'Name' as they are unique and don't contribute to prediction.

3. **Data Scaling**:
   - Applied a standard scaler to normalize values. Scaling helps to keep the input values within a specific range and can potentially speed up computations. However, this step is optional.

4. **Data Splitting**:
   - Used `train_test_split` with a test size of 0.3.

5. **Model Definition**:
   - Default model function named `CalModel`.
   - Evaluation function named `Evaluations_store`.
   - A variable named `Master_Evaluations` is used to store metric details for comparison across models.

6. **Model Training Approach**:
   - Initially, a default model is trained.
   - Models are then hyper-tuned with multiple parameters.
   - After successful hyper-tuning, models with specific parameters are trained.
   - Models trained include: Logistic Regression, SVC, KNeighbors Classifier, Decision Tree Classifier, Stratified KFold, and RandomForest Classifier.

7. **Evaluation Dataframe**:
   - Created a dataframe from `Master_Evaluations` and processed it for visualization.

8. **Model Performance**:
   - Across all models, accuracy remained consistent.
   - Metrics like F1 score, recall, and precision varied from model to model.






==================
Churn Modelling via Deep Learning (ANN)
==================

1. **Data Encoding**:
   - **Gender**: Encoded as 0 (Male) & 1 (Female) using a Lambda function for machine readability.
   - **Location**: Used ordinal encoding. One-hot encoding was tested due to the limited number of features, but results were consistent given the balanced data.

2. **Data Segregation**:
   - Segregated and split data into X (features) and Y (target). Excluded 'CustomerID' & 'Name' since they are unique and don't contribute to prediction.

3. **Data Scaling**:
   - Applied a standard scaler to normalize values. This ensures input values are within a specific range and can speed up computations, although it's optional.

4. **Data Splitting**:
   - Utilized `train_test_split` with a test size of 0.3.

5. **Early Stopping**:
   - Created an early stopping object to monitor the minimum validation loss and save computational resources.

6. **Model Definition**:
   - Defined a sequential model.
   - Added input, hidden, and dropout layers with neurons and the 'relu' activation function.
   - The output layer consists of a single neuron with a 'sigmoid' activation.
   - Compiled the ANN using optimizer='adam', loss='binary_crossentropy', and metrics=['accuracy'].
   - Fitted the model with a batch size of 32 and 200 epochs.

7. **Predictions**:
   - Predicted the test set using the trained model and applied a threshold for classification.

8. **Evaluation and Metrics**:
   - Visualized Loss and Accuracy with respect to Epochs.
   - Evaluated model predictions against the original test set.
   - Generated a classification report to obtain all evaluation metrics.






==================
Prior Deployment
==================

- Dumped a model using pickle:	
import pickle
with open("LogisticRegression_Model.pkl", "wb") as output:
    pickle.dump(model, output)

- Dumped an ordinal encoder using pickle:
import pickle
with open("OrdinalEncoder_Locations.pkl", "wb") as output:
    pickle.dump(encoder, output) 




==================
Deployment
==================
- Load the pickled model for prediction:
with open('LogisticRegression_Model.pkl', 'rb') as file:
    model = pickle.load(file)


- Load the pickled ordinal encoder for location encoding:
with open('OrdinalEncoder_Locations.pkl', 'rb') as file:
    OE = pickle.load(file)


- Using Streamlit, I created an interface that:
- Takes input from the user.
- Encodes the gender and location upon clicking the "Predict" button.
- Predicts whether the customer will churn or not, i.e., if the customer will stay or exit.



==================
TO RUN:
==================
- Extract all files in loaction
- open a terminal/shell/cmd at that location
- run command "streamlit run .\churn_customer_app.py"



NOTE: "Flask can also be used, but it takes time to design the pages."



I am attaching some links and a copy of my resume for your perusal.
- Portfolio : http://omkarbs23.pythonanywhere.com/
- Resume :  http://omkarbs23.pythonanywhere.com/resume
- LinkedIn : https://www.linkedin.com/in/omkarbs23/
- GitHub : https://github.com/omkarbs23
- Email : omkarbs23@gmail.com




