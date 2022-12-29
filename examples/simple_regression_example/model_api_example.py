from flask import Flask, request
import mlflow
from mlflow.exceptions import MlflowException

app = Flask(__name__)


@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    try:
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/simple_regression_example_model/Production"
        )
    except MlflowException as ex:
        return {'success': False, 'message': ex.message}, 500
    except:
         return {'success': False, 'message': 'server internal error'}, 500

    if request.is_json:
        req = request.json
        res = model.predict([[req['Age'], req['YearsExperience']]])

        return {'success': True, 'body': {'Salary': res[0][0]}}, 200

    else:
         return {'success': False, 'message': 'API only accept JSON body'}, 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
