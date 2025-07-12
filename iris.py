# Iris Flower Classification Project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("iris.csv")

# Explore the data
print("Rows and Columns:", df.shape)
print("Column Names:", df.columns)
print("Missing Values:\n", df.isnull().sum())
print("Unique Species:", df['species'].unique())

# Visualize relationships between features
sns.pairplot(df, hue='species')
plt.show()

# Features and labels
X = df.drop("species", axis=1)
y = df["species"]

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (you can change classifier here)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# Make predictions
prediction = model.predict(x_test)

# Accuracy score
print("Accuracy:", accuracy_score(y_test, prediction))

# Show predicted vs actual values
print("Predicted:", prediction)
print("Actual:   ", list(y_test))

# Confusion matrix
cm = confusion_matrix(y_test, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()