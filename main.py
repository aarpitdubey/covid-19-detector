from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')
clf = pickle.load(file);

file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():

    if request.method == "POST":
        #saving the data which was fetched from index.html into a dictionary called report
        report = request.form

        # Assigning the respective values for further predicting process
        fever       = int(report['fever'])
        age         = int(report['age'])
        pain        = int(report['pain'])
        fever       = int(report['fever'])
        runnyNose   = int(report['runnyNose'])
        diffBreath  = int(report['diffBreath'])

        # Code for interference
        inputFeatures = [fever, pain, age, runnyNose, diffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        results = clf.predict([inputFeatures])[0]
        #print(infProb)
        #print(results)

        # Logic to declare whether the person is Covid-19 Positive or Negative.
        if results == 1:
            results = "Covid-19 Positive. Immediately need isolation under the supervision of expert's and doctor's."
        else:
            results = "Covid-19 Negative. Can take normal doctor's prescription"
        
        # rendering the  show.html page to show the Predicted data
        return render_template('show.html', inf=int(round(infProb * 100)), result=results);

    # rendering the index.html page to show the user interface for user.
    return render_template('index.html');



# main  for run the application
if __name__ == "__main__":
    app.run(debug=True)