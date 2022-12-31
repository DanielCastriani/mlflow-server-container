from flask import Flask, request, render_template
import mlflow
from mlflow.exceptions import MlflowException

app = Flask(__name__, template_folder='static')

@app.route('/')
def index():
    return render_template('./html/index.html')


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

        errors = {}
        age = req.get('age', False)
        years_experience = req.get('yearsExperience', False)

        if age == False:
            errors['Age'] = "The field 'Age' is required"
        elif type(years_experience) != float and type(years_experience) != int:
            errors['Age'] = "The field 'Age' need to be int or float"        
        elif age < 0:
            errors['Age'] = "The field 'Age' need to be greater then 0"



        if years_experience == False:
            errors['YearsExperience'] = "The field 'YearsExperience' is required"
        elif type(years_experience) != float and type(years_experience) != int:
            errors['YearsExperience'] = "The field 'YearsExperience' need to be int or float"                
        elif years_experience < 0:
            errors['YearsExperience'] = "The field 'YearsExperience' need to be greater then 0"

        if len(errors.keys()) > 0:
            return errors, 400



        res = model.predict([[age, years_experience]])

        return {'success': True, 'body': {'Salary': res[0][0]}}, 200

    else:
         return {'success': False, 'message': 'API only accept JSON body'}, 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
