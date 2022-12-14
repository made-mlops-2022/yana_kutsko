## Project setup:

<pre><code>python3 -m venv venv
source venv/bin/activate
pip install .</code></pre>

And then run:

<pre><code>python3 -m pip install -e .</code></pre>

## To train model via console utility:

<em>Train KNeighborsClassifier model:</em>

<pre><code>python ml_project/src/model_train_utility.py 'ml_project/configs/knn.yaml'</code></pre>

<em>Train LogisticRegression model:</em>

<pre><code>python ml_project/src/model_train_utility.py 'ml_project/configs/logreg.yaml'</code></pre>

## To predict via previous artifacts:

<pre><code>python ml_project/src/model_predict_utility.py 'ml_project/models/knn.pkl' 'ml_project/data/heart_cleveland.csv' 'ml_project/predictions/predictions.csv'</code></pre>
