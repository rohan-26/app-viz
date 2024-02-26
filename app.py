# Streamlit is an open-source app framework for Machine Learning and Data Science projects.
import streamlit as st

# Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and data manipulation Python library.
import pandas as pd

# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions.
import numpy as np

# Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
import matplotlib.pyplot as plt

# Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns

# Plotly Express is a terse, consistent, high-level API for creating figures with Plotly.py.
import plotly.express as px

# (Note: Streamlit is imported twice in the provided code, which is redundant.)
import streamlit as st

# Python's built-in library for generating random numbers.
import random

# PIL (Python Imaging Library) is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
from PIL import Image

# Altair is a declarative statistical visualization library for Python.
import altair as alt

# Set the title for the Streamlit app >>>>>>>>>>>>
st.title("Cats")


# Load the NYU logo image >>>>>>>>>>>>
image_cats = Image.open('cat.jpg')
# Display the NYU logo on the Streamlit app
st.image(image_cats, width=500)


# Create a sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

# Dropdown menu for selecting the page mode (Introduction, Visualization, Prediction)
app_mode = st.sidebar.selectbox('🔎 Select Page',['Visualization'])

# Dropdown menu for selecting the dataset (currently only "The Cat Dataset" is available)
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["The Cat Dataset"])

# Load the wine quality dataset
df = pd.read_csv("cats.csv")

hwt_min, hwt_max = st.sidebar.slider('Select Height Range', min_value=int(df['Hwt'].min()), max_value=int(df['Hwt'].max()), value=(int(df['Hwt'].min()), int(df['Hwt'].max())))
bwt_min, bwt_max = st.sidebar.slider('Select Weight Range', min_value=float(df['Bwt'].min()), max_value=float(df['Bwt'].max()), value=(float(df['Bwt'].min()), float(df['Bwt'].max())))

# Filtering the dataframe based on the slider values
filtered_df = df[(df['Hwt'] >= hwt_min) & (df['Hwt'] <= hwt_max) & (df['Bwt'] >= bwt_min) & (df['Bwt'] <= bwt_max)]

# Dropdown menu for selecting which variable from the dataset to predict
list_variables = df.columns
#select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

# Display a header for the Visualization section
st.markdown("## Visualization")

# Allow users to select two variables from the dataset for visualization
symbols = st.multiselect("Select two variables", list_variables, ["Hwt", "Bwt"])

# Create a slider in the sidebar for users to adjust the plot width
width1 = st.sidebar.slider("plot width", 1, 25, 10)

# Create tabs for different types of visualizations
tab1, tab2, tab3, tab4= st.tabs(["Line and Bar Charts ", "📈 Correlation", "A Description", "Predict Values"])

DF = df[["Sex", "Hwt", "Bwt"]]

Map = {"M":0, "F":1}
DF["Sex_Int"] = DF["Sex"].map(Map)

X = DF[["Sex_Int", "Hwt"]]
y = DF["Bwt"]

# Content for the "Line and Bar Charts" tab
with tab1:
  tab1.subheader("Line and Bar Charts")
  # Display a line chart for the selected variables
  tab1.line_chart(data=filtered_df, x=symbols[0], y=symbols[1], width=0, height=0, use_container_width=True)
  # Display a bar chart for the selected variables
  tab1.bar_chart(data=filtered_df, x=symbols[0], y=symbols[1], use_container_width=True)

# Content for the "Correlation" tab
with tab2:
  tab2.subheader("Correlation Tab 📉")
  # Create a heatmap to show correlations between variables in the dataset
  fig, ax = plt.subplots(figsize=(width1, width1))
  sns.heatmap(DF.corr(), cmap=sns.cubehelix_palette(8), annot=True, ax=ax)
  tab2.write(fig)
  df2 = df
  st.markdown("### Pairplot")
  fig3 = sns.pairplot(df2)
  st.pyplot(fig3)

# Content for the "A Description" tab
with tab3:
  tab3.subheader("Correlation Tab 📉")
  df = pd.read_csv("cats.csv")
  df.describe(include='all')
  st.markdown("## A description of the dataset")
  st.dataframe(df.describe(include='all'))

  import sweetviz as sv
  report = sv.analyze(df)
  report.show_html("report.html")

  #Display the Sweetviz report
  if st.button("Generate Report"):
    import streamlit.components.v1 as components
    st.title("Sweetviz Report of the Data")
    report_path = "report.html"
    HtmlFile = open(report_path, "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height = 1000, width = 1000)


# Content for the "Predict Values" tab
with tab4:

  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

  from sklearn.linear_model import LinearRegression

  lr = LinearRegression()
  lr.fit(X_train, y_train)

  pred = lr.predict(X_test)

  coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])

  st.set_option('deprecation.showPyplotGlobalUse', False)
  plt.figure(figsize=(10,7))
  plt.title("Actual vs. predicted weights",fontsize=25)
  plt.xlabel("Actual test set weights",fontsize=18)
  plt.ylabel("Predicted weights", fontsize=18)
  plt.scatter(x=y_test,y=pred)
  st.pyplot()

  from sklearn import metrics

  st.write("The mean absolute error between the predicted weights and the actual weights is:", metrics.mean_absolute_error(y_test,pred))
  st.write("The mean squared error between the predicted weights and the actual weights is:", metrics.mean_squared_error(y_test,pred))

  from sklearn.metrics import r2_score

  st.write("The model predicts the relationship between the heights and the weights to an accuracy of about : ", end=" ")
  r2_score(y_test,pred,multioutput='variance_weighted') * 100
  st.write("%")
