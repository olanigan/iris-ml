from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import hopsworks
import os

IRIS = "iris"

project = hopsworks.login()
fs = project.get_feature_store()

try:
    print("Fetching feature view...")
    feature_view = fs.get_feature_view(name=IRIS, version=1)
except:
    print("Creating feature view...")
    iris_fg = fs.get_feature_group(name=IRIS, version=1)
    query = iris_fg.select_all()
    feature_view = fs.create_feature_view(name=IRIS,
                                      version=1,
                                      description="Read from Iris flower dataset",
                                      labels=["variety"],
                                      query=query)

X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

metrics = classification_report(y_test, y_pred, output_dict=False)
print(metrics)

from sklearn.metrics import confusion_matrix

results = confusion_matrix(y_test, y_pred)
print(results)

ASSET_DIR = "../assets/"
# Check if assets folder exists
if os.path.isdir(ASSET_DIR) == False:
    os.mkdir(ASSET_DIR)

from matplotlib import pyplot

df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                     ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

cm = sns.heatmap(df_cm, annot=True)

fig = cm.get_figure()
fig.savefig(ASSET_DIR + "confusion_matrix.png") 
fig.show()

from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import os
import joblib
import hopsworks
import shutil

# Login and Fetch Model Registry
# project =  hopsworks.login()
mr = project.get_model_registry()
print("Save Model to Hopworks and locally")

# The 'iris_model' directory will be saved to the model registry
model_dir="iris_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(model, model_dir + "/iris_model.pkl")
shutil.copyfile(ASSET_DIR + "confusion_matrix.png", model_dir + "/confusion_matrix.png")

input_example = X_train.sample()
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

iris_model = mr.python.create_model(
    version=2,
    name=IRIS, 
    # metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Iris Flower Predictor")

iris_model.save(model_dir)