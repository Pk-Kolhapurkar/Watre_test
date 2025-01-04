import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/prathamesh.khade20/Watre_test.mlflow")

dagshub.init(repo_owner='prathamesh.khade20', repo_name='Watre_test', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)