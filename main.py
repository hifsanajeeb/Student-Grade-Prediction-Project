import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Data Collection
data = pd.read_csv("G.csv")

# Step 2: Data Preprocessing
# Remove any missing or inconsistent data
data.dropna(inplace=True)

# Convert categorical variables into numerical variables
encoder = LabelEncoder()
data["Subject"] = encoder.fit_transform(data["Subject"])
data["Grade"] = encoder.fit_transform(data["Grade"])

# Normalize the numerical variables
scaler = MinMaxScaler()
data[["Attendance", "Mid I", "Mid II", "Quiz 1", "Quiz 2", "Quiz 3", "Final Exam"]] = scaler.fit_transform(
    data[["Attendance", "Mid I", "Mid II", "Quiz 1", "Quiz 2", "Quiz 3", "Final Exam"]])

# Split the dataset into training and testing sets
X = data.drop(["ID", "Name", "Total Marks", "Grade", "Subject"], axis=1)
y = data["Grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the feature names
feature_names = X.columns.tolist()
X_test = X_test[X_train.columns]

# Step 3: Building and Tuning the Model
# Linear Regression
linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)

# Decision Tree Classification
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# Hyperparameter tuning for Decision Tree Classification
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}
grid_search = GridSearchCV(tree_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_tree_model = grid_search.best_estimator_

# Step 4: Evaluating the Models
# Linear Regression
y_pred_linear = linear_model.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
report_linear = classification_report(y_test, y_pred_linear)

print("Linear Regression Metrics:")
print("Accuracy:", accuracy_linear)
print("Classification Report:")
print(report_linear)
print()

# Decision Tree Classification
# Select the corresponding columns in the test data
X_test = X_test[X_train.columns]

y_pred_tree = best_tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
report_tree = classification_report(y_test, y_pred_tree)

print("Decision Tree Classification Metrics:")
print("Accuracy:", accuracy_tree)
print("Classification Report:")
print(report_tree)
print()
# Step 5: Save the best model and preprocessing objects to files
if accuracy_tree > accuracy_linear:
    joblib.dump(best_tree_model, "final_model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
else:
    joblib.dump(linear_model, "final_model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")


from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load the trained model and preprocessing objects
model = joblib.load("final_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form inputs
    attendance = float(request.form['attendance'])
    mid1 = float(request.form['mid1'])
    mid2 = float(request.form['mid2'])
    quiz1 = float(request.form['quiz1'])
    quiz2 = float(request.form['quiz2'])
    quiz3 = float(request.form['quiz3'])
    final = float(request.form['final'])

    # Preprocess the inputs
    data = pd.DataFrame([[attendance, mid1, mid2, quiz1, quiz2, quiz3, final]], columns=['Attendance', 'Mid I', 'Mid II', 'Quiz 1', 'Quiz 2', 'Quiz 3', 'Final Exam'])
    data = scaler.transform(data)

    # Make the grade prediction
    prediction = model.predict(data)
    predicted_grade = encoder.inverse_transform(prediction)[0]

    # Render the result template with the predicted grade
    return render_template('result.html', predicted_grade=predicted_grade)

if __name__ == '__main__':
    # Set up the feature names for the inputs
    X_train_columns = ['Attendance', 'Mid I', 'Mid II', 'Quiz 1', 'Quiz 2', 'Quiz 3', 'Final Exam', 'Grade']
    app.run(host='127.0.0.1', port=5000)


