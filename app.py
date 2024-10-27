from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route("/")
def home_page():
    return render_template('home_page.html')

@app.route('/predictdata', methods=['GET','POST'])
def predictdata():
    if request.method == 'GET':
        return render_template('data_input.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethinicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score'),

        )
        pred_df=data.get_html_as_df()
        print(pred_df)

        pred_pipeline=PredictPipeline()
        results=pred_pipeline.predict(pred_df)
        return render_template('data_input.html',results[0])


if __name__=="__main__":
    app.run(debug=True)