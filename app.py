from flask import (
    Flask,
    render_template,
)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/iris')
def iris():
    from iris import data_set as iris_dataset
    from iris import main as iris_main
    iris_main()
    return render_template('iris.html',
                           feature_name_colspan=len(iris_dataset.feature_names),
                           data_set=iris_dataset)


@app.route('/breastcancer')
def breast_cancer():
    from breast_cancer import data_set_feature_names as breast_cancer_feature_names
    from breast_cancer import collect_data_out as breast_cancer_collect_data_out
    from breast_cancer import cleanse_data_out as breast_cancer_cleanse_data_out
    from breast_cancer import test_clf_out as breast_cancer_test_clf_out
    return render_template('breast_cancer.html',
                           collect_data_out=breast_cancer_collect_data_out,
                           cleanse_data_out=breast_cancer_cleanse_data_out,
                           test_clf_out=breast_cancer_test_clf_out,
                           feature_name_colspan=len(breast_cancer_feature_names),
                           breast_cancer_feature_names=breast_cancer_feature_names)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
