# SUP-4: Classification
```AI for Industry (AI4I)®  SUP-4: Classification```

Classification is a type of Supervised Learning problem where the objective is to identify to which of a set of categories(sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known. Examples are: assigning a given email to the [“spam” or “non-spam”](https://en.wikipedia.org/wiki/Anti-spam_techniques) class, and assigning a diagnosis to a given patient based on observed characteristics of the patient (sex, blood pressure, presence or absence of certain symptoms, etc.). Classification is an example of pattern recognition.

In this section, we will be applying what we learnt previously to look at a problem that most companies are concerned with: Will my customer leave me?

#### Prerequisites
* Complete [“Supervised Learning with scikit-learn” course](https://www.datacamp.com/users/sign_in?redirect=https%3A%2F%2Flearn.datacamp.com%2Fcourses%2Fsupervised-learning-with-scikit-learn) on Datacamp
* Complete “Setup your Machine Learning Environment” course on this LMS.
Requirements
To complete this course, you will need a desktop/laptop system with Python installed as described in the “Setup your Machine Learning Environment” section.

#### Instructions
This section consists of 5 sections which may include quizzes test your understanding of the material. Proceed to the next section to begin.

#### Materials
Use the link below to download a zip file containing the template notebooks for the following sections:

* SLCL-2 Exploratory Data Analysis
* SLCL-3 Classification using KNN
Solutions are also provided for all exercises but you are encouraged to attempt the activities on your own before referring to them.

learnai_classification2  [Download](https://mk0learnfx6ydjq1jij.kinstacdn.com/wp-content/uploads/2020/04/learnai_classification2.zip)

## Exploratory Data Analysis
In this lesson, we will be building models to predict customer churn. What is customer churn? This can be simply defined as the loss or attrition of customers.

Please read these short articles to give you a quick overview of this problem and why it is important to solve.

* [Blog article from Hubspot](https://blog.hubspot.com/service/what-is-customer-churn)
* [Wikipedia entry for Customer Churn](https://en.wikipedia.org/wiki/Customer_attrition)

Once you are done, you may begin exploring your data.

#### Instructions
1. The data is available as part of IBM’s sample datasets. Click on this link to download the file directly onto your machine.
2. The description for the data can be found here. However, please note that this description is for a newer version of the dataset which has many more features. Focus of the features in the downloaded file.
3. For this course, all of the exercises will be done in Jupyter as notebooks. Unzip the provided file (at the end of the previous section) in your project folder.
4. Place the files in the same folder as the downloaded dataset. Launch an instance of the Jupyter server and open __Ex_EDA_start.ipynb__.
5. Explore your data as guided by the notebook.
6. Once you are ready, you may proceed to the next section.

#### Questions to ask your data
* What is the shape of your data? Number of rows and columns.
* How many of the columns are numerical and how many are categorical?
* For the numerical columns, what does the distributions look like?
* What is the name of the column to be predicted?
* How are the various attributes correlated to the outcome variable?
* For the numerical columns, how many missing values are there for each column?
* For the categorical columns, how many missing values are there for each column?
* What visualizations can you use to highlight outliers in the data?

## Classification using KNN
In this section, we will build a simple classification model to predict customer churn. We will use an algorithm which you have probably seen before, K-nearest Neighbors. You can read this tutorial from Datacamp if you need a refresher.

We will build the model using Jupyter. The notebook created in this section can then be used for all subsequent sections in courses and beyond.

As a reminder, our objective is to build a customer churn model i.e. a model that will predict if a customer will stop using the services of the telco company.

#### Exercise
Find Ex_KnnClassification_start.ipynb file. Place this file in the same location as the data file from the previous section.

Start an instance of the Jupyter server if it is not already running and load the notebook.

Follow the instructions on the notebook and complete the tasks. You can compare your notebook with the completed on in the Solutions folder (Ex_KnnClassification_soln.ipynb). You may now proceed to the next section.

#### References
[Scikit-Learn Documentation](https://scikit-learn.org/stable/index.html)

## Handling Data Imbalance
In real projects, having data that is imbalanced is quite common. For classification problems, data imbalance refers to the situation where proportion of data for the various classes heavily favor a few classes over others. This will lead to a inherently biased model that will more easily predict the majority class. One characteristic of such a model is a seeming high accuracy score. However when the metric is looked at carefully, we can see that this accuracy is due to the proportion of the data.

In this section, we will look at the imbalance in our data, study its impact on model building and apply a simple process to handle it.

Before doing the exercise, read through this [tutorial](https://www.datacamp.com/community/tutorials/diving-deep-imbalanced-data) from Datacamp to understand Data Imbalance and general methods to solve it.

### Exercise
#### Initial Analysis
For this exercise, begin by making a copy of your solution notebook from the previous exercise. If you have issues with your notebook, you can use the provided Exercise solution instead. We will be modifying parts of this notebook to handle the data imbalance. __Instructions will be described here with code snippets provided where necessary.__

First, let look at outcome classes to verify that we have data imbalance. In the previous exercise, we calculated counts on our outcome variable ChurnLabel.

``` 
print("Row count for each outcome")
print(output_var.value_counts())

Row count for each outcome
0    5163
1    1869 
```

By adding an additional parameter, we can also calculate the proportion.

```
print(output_var.value_counts(normalize=True))

0    0.734215
1    0.265785
```

We can definitely see that there is an imbalance in the outcome with more customers not churned compared to those who do. The churned customers represent slightly more than 25 % of the total number of customers and there is 1:3 ratio with the unchurned customers.

Let’s have a look at how this effected the model training. Refer back to the confusion matrix generated when the model was used on the test data.


![alt text](https://mk0learnfx6ydjq1jij.kinstacdn.com/wp-content/uploads/2019/09/confusion.png "Confusion Matrix 1")

If we calculate the recall for each outcome class, we can see that we are getting 87% for the ‘No’ outcome (customers who did not churn) and about 57.8 % for the ‘Yes’ outcome. There is an obvious disparity of results between the two. There could be a few reasons for this, let’s see if the data imbalance could be one of them.

### Balancing the data
To process our data, we will be using the imbalanced-learn library. Follow the instructions under the ‘Getting Started’ section to install this library on your system.

For an initial attempt, let’s try the simplest, naive method to solve our data imbalance, random over-sampling. Once the library is installed, add the following import to your notebook.

``` from imblearn.over_sampling import RandomOverSampler ```

In this exercise, we will need to run the pre-processing pipeline and modelling algorithm separately. Replace the definition of the model pipeline from the previous exercise to execute the pre-processing pipeline on the input data i.e:

``` 
# Replace this original code
model = make_pipeline(
     preprocess,
     KNeighborsClassifier(n_neighbors=5)
 )

# With this
preprocessed_data = preprocess.fit_transform(input_data)
```
Then, apply the train/test split on this data.

```
x_train, x_test, y_train, y_test = train_test_split(preprocessed_data, output_var, test_size=0.3, random_state=42) 
```

We will applying data balancing on the training data only. Let’s add in a couple lines of code to display the class counts.

```
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} n".format(sum(y_train==0)))
```

To do the over sampling, we first create an instance of the RandomOverSampler class and fit to our training data (refer to the User guide from the library for more information). You can then print out the class counts afterwards.

``` 
over_sample = RandomOverSampler(random_state=0)
x_train_res, y_train_res = over_sample.fit_resample(x_train, y_train.ravel())
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {} n".format(sum(y_train_res==0))) 
```

You should get the following printouts. Notice how by default, the oversampling will match the quantity for the minority class (churned customers) to the majority class.

```
After OverSampling, counts of label '1': 3641
After OverSampling, counts of label '0': 3641 
```
We can now create and train the model, being careful to use the resampled data. You should be able to execute the rest of the notebook to predict on the test data and calculate the metrics.

```
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_res, y_train_res)
```

### Evaluating the results
Let’s take a look at the confusion matrix for this new model.


![alt text](https://mk0learnfx6ydjq1jij.kinstacdn.com/wp-content/uploads/2019/09/confusion_rand.png "Confusion Matrix 2")

Recall for the ‘No’ outcome has dropped to 67.6% but it has increased significantly for the ‘Yes’ outcome to 76.7%.

Let’s look at the other metrics. The table below compares the values from the 2 models (unbalanced vs balanced data) with the values in the parentheses coming from the balanced data model.



|   | Precision            | Recall            | f1-score          |
|---| -------------        |:-------------:    | --------          |
|0  | 0.84234 (0.88250)    | 0.87057 (0.67608) | 0.85622 (0.76562) |
|1  | 0.63315 (0.47775)    |0.57823 (0.76701)  | 0.60444 (0.58877) |



In addition to the recall for the churned customers, only the precision for the ‘No’ outcome has improved (albeit slightly). All other measurements have become worse. So, how do we evaluate this new model? Remember that the metrics we choose is dependent on the problem we are trying to solve. In this scenario, one can imagine that the metrics related to the ‘Yes’ outcome is more important: We want to target all customers that might churn (i.e. recall) through a retention campaign and we want to minimize the number of false positives to reduce any unnecessary resources (precision). From there we can evaluate the costs/benefits associated with the 2 models and see which gives us a better return.

Let’s try an alternative method for oversampling, Synthetic Minority Oversampling Technique (SMOTE). Begin by importing the class in your notebook. A smote instance is initialized replacing the RandomOverSample from previously. Use the same random seed as shown below. Your code should look like this:

```
# Include SMOTE in the library import
from imblearn.over_sampling import SMOTE

...

# Replace the previous oversampling method
over_sample = SMOTE(random_state=2)
```

Run the rest of the notebook as is. The resulting confusion matrix and calculated metrics should look like this.

![alt text](https://mk0learnfx6ydjq1jij.kinstacdn.com/wp-content/uploads/2019/09/confusion_rand_smote.png "Confusion Matrix 3")

|   | Precision            | Recall            | f1-score          |
|---| -------------        |:-------------:    | --------          |
|0  | 0.84234 (0.88433)    | 0.87057 (0.69317) | 0.85622 (0.77716) |
|1  | 0.63315 (0.49073)    |0.57823 (0.76531)  | 0.60444 (0.59801) |


There are no significant differences between the 2 models which seems to suggest the data is not sensitive to the oversampling method. The key decision therefore is decide whether to apply data balancing at all.

As typical of real projects, selecting between models can sometimes be neither straightforward nor obvious. Arguments can be made on both sides to apply data balancing. One argument against could be that while the data is imbalanced, it is not significant enough to impact the quality of the model. Common wisdom on this topic indicate that data balancing is typically needed when the majority to minority class proportion is 4:1 or higher.

The best practice should be to select the metrics __before__ any calculations are ever made. In that way, we will not be tempted to select the model based on our biases or preferences and then justify our choice by selecting only the favorable metrics.

You can compare your notebook with the solution in the zip file (__Ex_DataImbalance_soln.ipynb__)

For the rest of the course, we will not be applying data balancing. However, you are encouraged to test it out in the other lessons as we try different algorithms.

## Decision Trees and Ensemble Methods
In this section, you will be using what you have learnt previously in Datacamp to build a Decision Tree and a Random Forest model for our telco churn data.

#### Exercise
Begin by making a copy of your solution from the “Classification using KNN” section. You will be modifying this notebook for this exercise. Remember to modify the markdown text and code comments to reflect changes made.

Our objective is to build two models: A Decision Tree and its ensembled version, Random Forest. We will then evaluate these models to see if any of them provide any improvement on our previous models.

Modify the library imports by included the Decision Tree and Random Forest algorithms from scikit-learn. Also include GridSearchCV as we will be using it to find the best parameters for each of these algorithms. Your imports should look something like this:

```
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

### Decision Tree
Let’s modify this notebook to replace the KNN algorithm with Decision Tree. Go to the cell under Model Building and rename the variable model to pipeline. Replace the KNN algorithm with DecisionTreeClassifier using the random_state of 42.

```
pipeline = make_pipeline(
     preprocess,
     DecisionTreeClassifier(random_state=42)
)
```
Next, we want to define the values for the parameter search grid. With our ML algorithm defined in a pipeline, we will need to know the name of the step to access its parameters. If you are using make_pipeline as shown above, the step name defaults to the lower case name of the function. To verify this, we can print out the steps in the pipeline from its parameter.

```
pipeline.named_steps
```
You should see that the name is ‘decisiontreeclassifier’. To create the parameter search values, we will need to append the parameter name to this (separated by a double underscore). Let’s define the values for three parameter as shown below. You should refer to the documentation if you require any explanation of these parameters.

```
params = {
    'decisiontreeclassifier__max_depth': [3,4,5,6],
    'decisiontreeclassifier__min_samples_leaf': [0.04, 0.06, 0.08],
    'decisiontreeclassifier__max_features': [0.2, 0.4, 0.6, 0.8]
}
```
We can now define the model variable using GridSearchCV. Here, we use a 5 fold cross-validation and setting n_jobs=-1 tells the program to use all the available CPU cores on your machine. For this exercise, we will be using balanced_accuracy scorer because of our unbalanced data. You should replace this with the suitable scorer based on your objectives. By redefining the variables (pipeline and model) this way, we do not need to modify any other code to run the notebook successfully.

```
model = GridSearchCV(estimator=pipeline,
                         param_grid=params,
                         scoring='balanced_accuracy',
                         cv=5,
                         n_jobs=-1)
```

Execute the notebook until the training of the model (i.e. model.fit() ). We can then have a look at the best parameters as well as the best score based on the cross-validation.

```
print("Best hyperparameters:n{}".format(model.best_params_))
print("Best Accuracy:n{}".format(model.best_score_))
```

Continue executing the rest of the notebook and compare the results with KNN model from the previous section. Is the result better, worse or the same as before? Test out different values for the grid search to see if you can find more optimum values.


## Random Forest
We will now build a Random Forest model in the same notebook. Create new cells at the bottom of the notebook and define a new model, renaming the variables where appropriate, using the same steps as above i.e.


* Define the pipeline
* Define the parameter search grid values
* Creating a model using GridSearchCV

```
# Define a new pipeline for Random Forest
pipeline2 = make_pipeline(
     preprocess,
     RandomForestClassifier(random_state=42)
 )
```
```
# Define the parameter search grid. Refer to scikit documentation for explanation
params2 = {
     'randomforestclassifier__n_estimators': [200,300,400,500],
     'randomforestclassifier__max_depth': [3,4,5,6],
     'randomforestclassifier__min_samples_leaf': [0.04, 0.06, 0.08] ,
     'randomforestclassifier__max_features': [0.2, 0.4, 0.6, 0.8] 
 }
```
```
model2 = GridSearchCV(estimator=pipeline2,
                         param_grid=params2,
                         scoring='balanced_accuracy',
                         cv=5,
                         n_jobs=-1)
```
Train the model and display the best parameters. Also calculate the metrics for this model.

Surprisingly, this random forest model does not perform as well as the non-ensemble version. Why do you think this is the case? Try out other parameter values to see if you can improve on the results further.

If you face any issues with any of the tasks here, compare your work against the solution in the provided zip file (__Ex_Trees_soln.ipynb__)

You have successfully built 2 additional models for our data. Proceed to the next section and don’t give up!

## Logistic Regression and building your own Ensemble
In this section, we will be building our own ensemble model by combining previously built models.

#### Exercise
Begin by making a copy of your solution from the “Classification using KNN” section. You will be modifying this notebook for this exercise. Unlike previous lessons, we will have multiple base models which we will then combine into our final model.

Let’s define the pipelines for the individual algorithms. We will using 2 algorithms from previous sections (KNN and Decision Tree) and add Logistic regression as a third.

First, modify the library imports to include KNN, Decision Tree and Logistic Regression from scikit learn. Also include VotingClassifier which will be used to combine our models.

```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
```

Define the pipelines for KNN and Decision Tree using parameters discovered in previous sections. For Logistic Regression, we can use the default parameters.

```
model_knn = make_pipeline(
     preprocess,
     KNeighborsClassifier(n_neighbors=5)
 )

model_dt = make_pipeline(
     preprocess,
     DecisionTreeClassifier(random_state=42,
                           max_depth=3,
                           max_features=0.4,
                           min_samples_leaf=0.08)
)

model_lr = make_pipeline(
     preprocess,
     LogisticRegression()
)
```

Next, we define a list of tuples for each of our base classifiers.

```
clfs = [
    ('KNN', model_knn),
    ('Decision Tree', model_dt),
    ('Logistic Regression', model_lr)
]
```

After the train/test data split, insert the following loop in a new cell which will train each of our classifiers, calculate a metric and display the value. We can use these values to compare against the final model.

```
for clf_name, clf in clfs:
        clf.fit(x_train, y_train)
         pred_test = clf.predict(x_test)

    # Calculate the metrics used to evaluate previous models
         f1 = f1_score(y_test, pred_test)
         print('{:s} : {:.3f}'.format(clf_name, f1)) 
```

Let’s now create our final, ensemble model. We will use VotingClassifier which by default using hard voting (majority voting). Define the algorithm, passing in our list of base classifiers and name the variable model (so that we can run the rest of the notebook without modifications). Train this model.

```
model = VotingClassifier(estimators=clfs)
model.fit(x_train, y_train)
```

Execute the rest of the notebook and compare the results with those from the individual base classifiers. You should be able to get a model that performs slightly better than any of the models.

You have now successfully built your own ensemble model. You can compare your solution with __Ex_CustomEnsm_soln.ipynb__ from the provided zip file. Complete the quiz to complete this lesson.

