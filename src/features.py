import random
import pandas as pd
import hopsworks
from flower import get_random_iris_flower

BACKFILL=True

if BACKFILL == True:
    iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
else:
    iris_df = get_random_iris_flower()

# Login and Fetch Feature Store
project = hopsworks.login()
fs = project.get_feature_store()


iris_fg = fs.get_or_create_feature_group(name="iris",
                                  version=1,
                                  primary_key=["sepal_length","sepal_width","petal_length","petal_width"],
                                  description="Iris flower dataset"
                                 )

try:
    print("Inserting Features")
    iris_fg.insert(iris_df, write_options={"wait_for_job": False})
except:
    print("Feature(s) already exists")
    iris_fg.insert(get_random_iris_flower(), write_options={"wait_for_job": False})