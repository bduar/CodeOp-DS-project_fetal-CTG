"""
# "Fetal Health Classification from Cardiotocographic Data using Machine Learning Models"
# A streamlit dashboard for data visualization
# Berta Duran Arqué (https://github.com/bduar)
# October 2022
# Python script 1/1

"""

# Load dependencies
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold

st.title("Fetal Health Classification from Cardiotocographic Data using Machine Learning Models")
st.header("Berta Duran Arqué · October 2022 · React-DS-01 @ CodeOp")

st.write("The dataset hereby presented consists of measurements of fetal heart rate (FHR) and uterine contraction (UC) features on cardiotocograms classified by expert obstetricians.")
st.write("The original publication that produced the data is: Ayres-de Campos et al. SisPorto 2.0: a program for automated analysis of cardiotocograms. J Matern Fetal Med. 2000 Sep-Oct;9(5):311-8. [DOI](https://doi.org/10.1002/1520-6661(200009/10)9:5<311::AID-MFM12>3.0.CO;2-9). PMID: 11132590. The dataset was donated to the UCI Machine Learning Repository, and is freely available at [link](https://archive-beta.ics.uci.edu/ml/datasets/cardiotocography#Abstract).")

# Load pre-processed dataset
data = pd.read_csv("./Data/ctg_pp_02.csv")

# Describe the dataset and the add a table with variable names
st.subheader("Variables description")
variables_description = pd.DataFrame({"Abbreviation": data.columns, 
                                     "Description": ["FHR baseline (beats per minute)", 
                                                     "# of accelerations per second", 
                                                     "# of fetal movements per second", 
                                                     "# of uterine contractions per second", 
                                                     "# of light decelerations per second", 
                                                     "# of severe decelerations per second", 
                                                     "# of prolongued decelerations per second", 
                                                     "percentage of time with abnormal short term variability", 
                                                     "mean value of short term variability", 
                                                     "percentage of time with abnormal long term variability", 
                                                     "mean value of long term variability", 
                                                     "width of FHR histogram", 
                                                     "minimum of FHR histogram", 
                                                     "Maximum of FHR histogram", 
                                                     "# of histogram peaks", 
                                                     "# of histogram zeros", 
                                                     "histogram median", 
                                                     "histogram variance", 
                                                     "histogram tendency", 
                                                     "FHR pattern class code (1 to 10)", 
                                                     "fetal state class code (N=normal; S=suspect; P=pathologic)"]})

obs = data.shape[0]
caption_str = f"The dataset has 19 predictive variables and 2 classification variables. There are a total of {obs} observations."

st.table(variables_description)
st.caption(caption_str)

#### EDA
st.subheader("Exploratory Data Analysis")

# Correlation plot
corr = data.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, annot=True, cmap='RdBu', ax = ax)
st.write(fig)
st.caption("Correlation matrix of all the variables in the dataset. Measure of correlation: Pearson correlation coefficient.")

# Boxplots of raw and normalized datasets
# Load the raw and normalized datasets dictionary
datasets_dict = joblib.load("./Notebooks/models/datasets_dict.sav")

# Plot raw data (can be NSP or CLASS, they should be identical), we will leave the classifier variable out
fig, axs = plt.subplots(5, 4, figsize = (12, 12))
axs = axs.flatten()

for col_name, ax in zip(datasets_dict["raw_NSP"][0], axs):
    ax.boxplot(datasets_dict["raw_NSP"][0][col_name])
    ax.set_title(col_name)

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
st.write(fig)
st.caption("Distribution of all the predictor variables in the raw data.") 

# Repeat now plotting normalized data
fig, axs = plt.subplots(5, 4, figsize = (12, 12))
axs = axs.flatten()

for col_name, ax in zip(datasets_dict["std_NSP"][0], axs):
    ax.boxplot(datasets_dict["std_NSP"][0][col_name])
    ax.set_title(col_name)

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
st.write(fig)
st.caption("Distribution of all the predictor variables in the standardized data.") 

##### Modeling
st.subheader("Modeling")

# Add a slider and pick either NSP or CLASS
classifier = st.selectbox(
    'This dataset has two classifiers, CLASS and NSP. Please, select one.',
    ('CLASS', 'NSP'))

# Do a bar plot
df = data[classifier].value_counts().rename_axis('Value').reset_index(name='Occurrences')
df = df.sort_values(by=['Value'])

fig = plt.figure()
ax = fig.add_axes([1, 1, 1, 1])
ax.barh(df["Value"], df["Occurrences"], height = .7)
plt.title(f"Weights of the categories in {classifier}")
plt.yticks(np.arange(0, df.shape[0]), df["Value"])
plt.xlabel('Occurrences')
plt.ylabel('Categories')
st.write(fig)

st.markdown(
"""
A 7:3 train:test split was performed on the dataset.
The following algorithms were trained using a Random Search Grid with five-fold cross-validation:
- Logistic Regression (LG) 
- k-Nearest Neighbors (kNN)
- Support Vector Machine Classifier (SVM) 
- Decision Tree (DT) 
- Random Forest (RF)
**Random Forest Classifiers** displayed the best performance according to the F1 macro-averaged metric.
Therefore, Random Forests were further refined with two consecutive Grid Searches, tuning the following hyperparameters, with criterion set to 'entropy': 
- max_depth 
- min_samples_leaf
- min_samples_split
- n_estimators
"""
)

# The best performing datasets were the Raw ones:
ds = "raw_"+classifier

# Load saved models and datasets
best_models = joblib.load("./Notebooks/models/fitted_grids_third.sav")
datasets_dict = joblib.load("./Notebooks/models/datasets_dict.sav")

rf_model = best_models[ds]["model"]
X_train, X_test, y_train, y_test = datasets_dict[ds][0], datasets_dict[ds][1], datasets_dict[ds][2], datasets_dict[ds][3]

model_details_str = f"The best-performing model for the raw {classifier} dataset is: {rf_model}"
st.write(model_details_str)

y_pred = rf_model.predict(X_test)
st.text('Model Report:\n ' + classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.title("Class categories")
st.pyplot(plt)
st.caption("Confusion matrix of the Random Forest best model.")