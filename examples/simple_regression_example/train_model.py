import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from examples.simple_regression_example.experiment_utils import create_experiment

from dotenv import load_dotenv
load_dotenv()

# Uncomment the line below or add MLFLOW_TRACKING_URI in .env. 
# If you are using .env, you should load the environment variables by calling dotenv.load_dotenv()
# mlflow.set_tracking_uri(f"http://127.0.0.1:{os.environ.get('PORT')}")

print(mlflow.get_tracking_uri())

# Create plot root directory if not exists
plot_root_path = os.path.join('outputs', 'simple_regression', 'plots')
if not os.path.exists(plot_root_path):
    os.makedirs(plot_root_path)

# Load Dataset
dataset_df = pd.read_csv('./examples/datasets/Salary_Data.csv')

# Check dataset
dataset_df.describe()
dataset_df.isna().sum()
dataset_df.info()

# Plot Age x Salary
sns.scatterplot(x=dataset_df['Age'], y=dataset_df['Salary'])
plt.savefig(os.path.join(plot_root_path, 'Age_x_Salary.jpg'))

# Plot YearsExperience x Salary
sns.scatterplot(x=dataset_df['YearsExperience'], y=dataset_df['Salary'])
plt.savefig(os.path.join(plot_root_path, 'YearsExperience_x_Salary.jpg'))

# Check correlation
sns.heatmap(dataset_df.corr(), annot=True)
plt.savefig(os.path.join(plot_root_path, 'corr.jpg'))

# Create new experiment if not exists
experiment = create_experiment('simple_regression_example')

# Start new run
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

    # Build pipeline
    pipeline = Pipeline([
        ('min-max', MinMaxScaler()),
        ('lr', LinearRegression())
    ])

    # Fit model
    pipeline.fit(
        dataset_df[['Age', 'YearsExperience']],
        dataset_df[['Salary']]
    )

    # Make predictions and create new columns with them
    dataset_df['Y_Pred'] = pipeline.predict(
        dataset_df[['Age', 'YearsExperience']]
    )
    
    # Log the model error
    mse = mean_squared_error(dataset_df['Salary'], dataset_df['Y_Pred'])
    mlflow.log_metrics({
        'mae': mean_absolute_error(dataset_df['Salary'], dataset_df['Y_Pred']),
        'mape': mean_absolute_percentage_error(dataset_df['Salary'], dataset_df['Y_Pred']),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(dataset_df['Salary'], dataset_df['Y_Pred']),
    })

    # Get the linear regretion to log the coefs and intercept
    lr: LinearRegression = pipeline.steps[1][1]
    mlflow.log_params({
        'c1': lr.coef_[0][0],
        'c1': lr.coef_[0][1],
        'intercept': lr.intercept_,
    })

    # Log the model
    mlflow.sklearn.log_model(pipeline, 'model')

# Get the models that are still in production, and archive them
model_name = 'simple_regression_example_model'
client = mlflow.MlflowClient()

# client.get_latest_versions don't return all versions. 
# We will need to fetch  models while response bringing models
try:
    old_models = client.get_latest_versions(model_name, ['Production'])
    while len(old_models) > 0:
        for model in old_models:
            client.transition_model_version_stage(model_name, model.version, 'Archived')    
        old_models = client.get_latest_versions(model_name, ['Production'])
except mlflow.MlflowException as ex:
    print(ex.message)

# Register new model creating new version       
model_version = mlflow.register_model(
    f'runs:/{run.info.run_uuid}/model',
    model_name
)

# Make transition of the new model to production
client.transition_model_version_stage(model_name, model_version.version, 'Production')