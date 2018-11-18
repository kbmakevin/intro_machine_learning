from flask import (
    Flask,
    render_template,
)

app = Flask(__name__, template_folder='templates')

# run code from another python file
from iris import data_set
from iris import main as iris_main


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/iris')
def iris():
    iris_main()
    return render_template('iris.html',
                           feature_name_colspan=len(data_set.feature_names),
                           data_set=data_set)


@app.route('/breastcancer')
def breast_cancer():
    import breast_cancer
    return 'executing breast_cancer!'


if __name__ == '__main__':
    app.run(debug=True)
