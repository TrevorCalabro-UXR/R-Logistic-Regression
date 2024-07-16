# R for UX Researchers: Predicting Purchase Behavior with Logistic Regression

Welcome to the R for UX Researchers series! This tutorial walks you through the process of using R and RStudio to predict user purchase behavior using logistic regression. By the end of this tutorial, you will have performed data analysis, built a predictive model, and created compelling visualizations to support your stakeholders with actionable insights.

## Summary

This repository provides a step-by-step guide for UX researchers to predict purchase behavior using R and RStudio. It includes:
- Data analysis
- Predictive modeling
- Data visualization

## Scenario

I used to work for an e-commerce company that operates an online shopping platform. There, we tracked basic user interactions such as page views, product clicks, session duration, and purchase behavior. One day, the Director of UX came up to me and asked,

"Can you figure out what factors influence whether users make a purchase during their session via our site analytics?"

Knowing how our site was instrumented, I immediately knew the data could answer her question, but I'd have to do some analysis first. That's why, for this business question, I chose to use R and RStudio rather than Excel or Google Sheets.

>**Disclaimer:** For the record, I'm not a quant genius. I am a true mixed-methods researcher who gravitates toward qualitative methods by nature. I've been described as a people person and have always enjoyed the qual side of the profession. But my nine years working at Minitab, the world's most sophisticated desktop statistical software package, really gave me an appreciation and understanding of the power and necessity of quant. When telling my career story, I refer to my time at Minitab as being immersed in the X-Men school for stats, which I hope provides some context around where I'm coming from. Learn more about why crafting a career narrative is important in my article "Experienced UX Research Resume Writing: A Storytelling Approach."

That said, my intention here is to provide a more real-world and practical scenario that applies 1-to-1 to questions asked of us as UX Researchers in our everyday work. In this article, we'll walk through how to do all of this in RStudio step by step. I hope you work alongside me and actually follow the steps on your own computer.

Before we dive into the step-by-step tutorial, let's talk more about why I choose R and RStudio instead of more familiar tools like Excel or Google Sheets.

## Prerequisites

Before you start, ensure you have the following:
- R and RStudio installed
- Basic understanding of R programming
- [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets) from Kaggle

## Step-by-Step Tutorial

### Step 1: Download the Dataset

Download and save the dataset as a CSV file.
>✏️ **NOTE:** Make a note of the full path to the online_shoppers_intention.csv file on your computer. You'll need this path to import the dataset in step 5 below.

### Step 2: Start a New RStudio Project

1. Open RStudio
2. Go to `File > New Project > New Directory > New Project`
3. Name your project (e.g., "User_Purchase_Analysis") and choose a location
4. Click `Create Project`

![RStudio Screen](https://tinyurl.com/24nll93j)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*You’ll know you have done this correctly when you see a new R file item in the Files tab at the bottom right of the screen.*

### Step 3: Install Necessary Packages
Install the required packages by copy/pasting the code snippet below into the RStudio Console:

```r
install.packages("tidyverse")
install.packages("corrplot")
install.packages("caret")
install.packages("e1071")
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7f6543a3-0f7e-49e1-8679-65d88f475896_1100x685.jpeg)

### Step 4: Load the Libraries
Next, in your R script, load the necessary libraries:

```r
library(tidyverse)
library(corrplot)
library(caret)
library(e1071)
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7536cef1-ade5-4123-a5f5-767ad3e9a81d_1100x685.jpeg)
>✏️ **NOTE:** Disregard the Conflicts section shown in the Console.

### Step 5: Import the Dataset
Now, import the dataset into RStudio using the code snippet below.

```r
data <- read.csv("path/to/your/dataset/online_shoppers_intention.csv")
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa60a9349-8ce6-49a5-acc6-f9cea855489b_1100x685.jpeg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>You’ll know you have done this correctly when you see a new data item in the Environment tab at the top right of the screen.</em>

>✏️ **NOTE:** You will have to replace the "path/to/your/dataset" with the full path on your own machine to the CSV file.

### Step 6: Clean the Data
Now, let's prepare the data for analysis by handling missing values and converting categorical variables into factors.

```r
# Check for missing values
sum(is.na(data))

# Convert categorical variables to factors
data$Month <- as.factor(data$Month)
data$VisitorType <- as.factor(data$VisitorType)
data$Weekend <- as.factor(data$Weekend)
data$Revenue <- as.factor(data$Revenue)
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F88d95c93-e2e0-4a3c-8eb3-4d0b384090cc_1100x685.jpeg)
> 💡 <strong>TIP:</strong> If at any time during these steps you get an error message, simply copy/paste that message verbatim into a Google search. You’ll likely find the fix in the first few result links. Also, checking forums like Stack Overflow can be very helpful for resolving common issues.

### Step 7: Perform Exploratory Data Analysis (EDA)
EDA stands for Exploratory Data Analysis. It is a step in the data analysis process that involves summarizing and visualizing the main characteristics of a dataset. The goal of EDA is to understand the data's structure, spot outliers, identify patterns, and gain insights for further analysis and modeling. I usually run this code, or something similar, at the start of most projects when I'm using R.

Copy and paste the following EDA code snippet into your R script and run it section by section.

```r
# Summary statistics
summary(data)

# Histograms
ggplot(data, aes(x = ProductRelated_Duration)) + geom_histogram(binwidth = 30)
ggplot(data, aes(x = PageValues)) + geom_histogram(binwidth = 10)

# Density plots
ggplot(data, aes(x = ProductRelated_Duration)) + geom_density()
ggplot(data, aes(x = PageValues)) + geom_density()

# Boxplots
ggplot(data, aes(x = Revenue, y = ProductRelated_Duration)) + geom_boxplot()
ggplot(data, aes(x = Revenue, y = PageValues)) + geom_boxplot()

# Scatter plot
ggplot(data, aes(x = ProductRelated_Duration, y = PageValues)) + geom_point()

# Bar chart
ggplot(data, aes(x = Month, fill = Revenue)) + geom_bar(position = "fill")

# Correlation matrix and heatmap
cor_matrix <- cor(data %>% select_if(is.numeric))
corrplot(cor_matrix, method = "circle")

# Contingency table
table(data$VisitorType, data$Revenue)
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb033195f-ac5b-4f3e-867e-9d2c3fe73a87_1100x684.jpeg)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*If you get an error message, copy/paste it verbatim into a Google search for help.*

### Step 8: Interpret the EDA Results

Let's take a closer look at the EDA results. Review the summary statistics, histograms, density plots, boxplots, scatter plots, bar charts, correlation matrix, and contingency table. You'll start to see patterns emerge, like how ProductRelated_Duration and PageValues might influence whether a user makes a purchase. These insights are crucial for understanding our data's story.
![RStudio Plots](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff217d28e-f4e1-4338-8d40-aece026eb1b5_918x1957.jpeg)

### Step 9: Decide on a Model Type

Seeing the binary nature of the Revenue variable (purchase or no purchase), it became clear that a logistic regression would be the best fit for our analysis. This model will help us predict the probability of a purchase based on various predictors. Let's get our dataset ready for the logistic regression to ensure everything runs smoothly.

### Step 10: Split the Data into Sets
Next, we'll have to split the data into training and testing sets so we can build and validate our model, ensuring its predictions are accurate and generalizable to new data. (I did a quick Google search to find a template for this code snippet.) 

```r
set.seed(123)
trainIndex <- createDataPartition(data$Revenue, p = .8, list = FALSE, times = 1)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe55f7511-10a8-455d-a910-ab7e124208bf_1100x684.jpeg)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*You’ll know you have done this correctly when you see new data items in the Environment tab at the top right of the screen.*

### Step 11: Build the Logistic Regression Model
Now, we need to build a logistic regression model to predict whether a user will make a purchase, helping us understand the influence of various factors. Copy/paste this code snippet to build the model.

```r
model <- train(Revenue ~ ., data = trainData, method = "glm", family = "binomial")
summary(model)
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff197e525-0ad8-47ea-80ea-4e40d521a5a5_1100x685.jpeg)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*You’ve done this correctly if you see a new item titled model in the Environment tab at the top right.*

### Step 12: Evaluate the Model
Now, we'll evaluate the model to see how well it predicts purchases and identify any areas where it may need improvement. (I did a quick Google search to find this code snippet as well.) 

```r
predictions <- predict(model, newdata = testData)
confusionMatrix(predictions, testData$Revenue)
```
![RStudio Screen](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0b0f27a6-b287-4f41-a429-7cbf6fefed59_1100x684.jpeg)

Since the output has been verified, we have no need for further improvement; let's move on to creating visuals—the fun stuff! 

### Step 13: Visualize Predicted vs. Actual Values
This plot will show us how well the model's predictions match the actual values so we can assess its accuracy.

```r
ggplot(testData, aes(x = Revenue, y = predictions)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  labs(title = "Predicted vs Actual Revenue", x = "Actual Revenue", y = "Predicted Revenue") +
  theme_minimal()
```
![RStudio Plot](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F04f30c0b-2b05-45c6-aaec-503453bd8182_575x404.jpeg)

###### Points on the Line: ######
The points that lie on the red diagonal line represent instances where the model's predictions perfectly match the actual outcomes. In your case, there are points at (False, False) and (True, True), indicating that the model correctly predicted some instances of both non-purchases and purchases.

###### Points off the Line: ######
The two points that do not lie on the red diagonal line represent instances where the model's predictions do not match the actual outcomes.

This means the model is good enough to explore more visuals. Let's create enhanced visualizations to help us communicate our findings to the Director of UX.


### Step 14: Enhanced Interpretation and Density Plot of Predictions
I created two additional plots, a feature importance plot and a density plot of predictions, to explore more persuasive data visualizations.

```r
importance <- varImp(model, scale = FALSE)
ggplot(importance, aes(x = reorder(Overall, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Importance") +
  theme_minimal()

ggplot() +
  geom_density(aes(x = predictions, fill = "Predicted"), alpha = 0.5) +
  geom_density(aes(x = testData$Revenue, fill = "Actual"), alpha = 0.5) +
  labs(title = "Density Plot of Predicted vs Actual Revenue", x = "Revenue", fill = "Legend") +
  theme_minimal()
```

### Plot 1: Feature Importance Plot ###
This plot shows the importance of each feature in predicting the target variable. It helps in understanding which features are most influential in determining whether a user makes a purchase during their session.

![RStudio Plot](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb7369ff1-6505-4558-b7c3-6e0d7e9259a7_1400x900.jpeg)

###### Observation: ######
The PageValues feature has the highest importance, meaning it has the biggest impact on the model's predictions. The MonthJul, MonthSep, and SpecialDay features had the least importance.

###### Insight: ######
Knowing which features are most important helps us understand what affects revenue predictions the most. For example, focusing on the PageValues feature can have a significant impact on our predictions.

### Plot 2: Density Plot of Predictions ###
This plot compares the distribution of predicted and actual values, illustrating how well the model's predictions align with the actual outcomes.

![RStudio Plot](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F94b83cb2-3d62-4d2d-a30f-599ea844ceee_916x564.jpeg)

###### Observation: ######
The overlap between the density curves of predicted and actual values shows how well the distributions match.&nbsp;

###### Insight: ######
The density curves indicates good model performance.

### Step 15: Finalizing Impactful Insights
This model is highly likely to give the Director of UX confident answers to her original question: "Can you figure out what factors influence whether users make a purchase during their session via our site analytics?"

I included the Boxplots from Step 8 and the two enhanced plots, the Density Plot and the Feature Importance Plot, in my report to the Director of UX. These plots clearly showed how ProductRelated_Duration and PageValues might influence whether a user makes a purchase, emphasizing the importance of these features in our analysis.

These visualizations do a better job of telling the story of the predictive analysis. After seeing these, I used them in my report because they provided clearer insights into the factors influencing user purchases.

The main insight is that concentrating the UX team's efforts on the PageValues feature will impact user purchase behavior most.

This data gave the Director of UX a clear direction as to where to allocate the team's efforts to increase conversions.

### Step 16: Using R Markdown and Persuasive Reporting

Creating detailed and interactive visualizations in RStudio is a powerful way to communicate your findings, but doing this is its own challenge and requires a whole article itself. Because of this, I've decided to cover how to connect your analysis to a dynamic dashboard or create an HTML report in a future article.

In the meantime, let's quickly go over how to create a shareable HTML report for the analysis we've just completed:

First, make sure you have the R Markdown package installed. If not, you can install it by running:

```r
install.packages("rmarkdown")
```

1. In RStudio, go to `File > New File > R Markdown...`
2. Give your document a title, author name, and choose HTML as the output format.
3. Copy and paste your analysis into the R Markdown file.
4. Click the `Knit` button to generate the HTML report.
5. Share the HTML file with stakeholders.

## Conclusion

By following these steps, you have successfully analyzed factors influencing user purchases using the Online Shoppers Purchasing Intention Dataset. Woot, woot! You can celebrate. This tutorial not only demonstrated how to handle and clean data but also how to build a predictive model and create compelling visualizations in RStudio. These skills are crucial for any UX researcher aiming to leverage data to improve digital products. I hope that, as you went through this tutorial, you were envisioning how your own data would work in the real world.

I also hope this tutorial demonstrated how, with R and RStudio, you can perform complex analyses and create powerful visualizations that are hard to achieve with Excel or Google Sheets. I've found that this kind of thing goes a long way when persuading stakeholders and helping them make data-driven and user-centered decisions.

I also hope this tutorial demonstrated how, with R and RStudio, you can perform complex analyses and create powerful visualizations that are hard to achieve with Excel or Google Sheet

Feel free to [explore the full tutorial here](https://trevorcalabro.substack.com/p/r-for-ux-researchers-series-article) for additional insights and context.


