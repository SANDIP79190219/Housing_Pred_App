import flask
from flask import request, jsonify
import psycopg2
import sqlalchemy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import re
from collections import defaultdict
from collections import Counter 
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

app = flask.Flask(__name__)
app.config["DEBUG"] = True



def freq_dist(train):
    i=0
    data1 = pd.DataFrame(np.random.rand(0, 3) * 0, columns=['Feature_Name', 'Cat_Name', 'Cat_Count'])
    for feature in list(train.select_dtypes(include = ["object"]).columns):
        f = list(train.groupby(by=feature)[feature].count().index)
        c = list(train.groupby(by=feature)[feature].count())
        fc = zip(f, c)
        for idx, item in enumerate(fc):
            cat , co = item
            data1.loc[i, 'Feature_Name']     = feature
            data1.loc[i, 'Cat_Name']         = cat
            data1.loc[i, 'Cat_Count']        = co
            i+=1
    return data1



def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.route("/")
@app.route("/home")
@app.route("/index")
@app.route('/', methods=['GET'])
def home():
    data = pd.read_csv('train.csv')
    return '''<h1>Sales Price Prediction Application</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''



@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    #conn = sqlite3.connect('books.db')
    #conn.row_factory = dict_factory
    #cur = conn.cursor()
    #all_books = cur.execute('SELECT * FROM books;').fetchall()
    
    conn = psycopg2.connect(database = "postgres", user = "postgres", password = "bolu@2019", host = "127.0.0.1", port = "5432")
    cur = conn.cursor()
    cur.execute('select * from books')
    all_books = cur.fetchall()
    return jsonify(all_books)

@app.route('/api/v1/resources/books/data', methods=['GET'])
def api_data():
    train = pd.read_csv("train.csv")
    data = freq_dist(train)
    data['Percent'] = data.Cat_Count/train.shape[0] * 100
    data = data.set_index('Feature_Name')
    data = data[data['Percent']<10]
    
    for i in range(len(data)):
        col = data.iloc[i].name
        val = data.iloc[i].Cat_Name
        train[col] = train[col].replace(val,'Other', method='ffill')
        
    train = pd.get_dummies(train, drop_first=True)

    X = train.drop('SalePrice', axis=1).values
    y = train['SalePrice'].values

    from sklearn.preprocessing import Imputer

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    
    X = imp.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
    
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=100)
    gbrt.fit(X_train, y_train)
    #X_train.fillna(X_train.mean())
    errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt.staged_predict(X_test)]
    #
    bst_n_estimators = np.argmin(errors)
    gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
    gbrt_best.fit(X_train, y_train)
    
    #print("Tuned Model Parameters: {}".format(gbrt_best.best_params_))
    #print("Best score is {}".format(gbrt_best.score(X_test, y_test)))
    return jsonify("GradientBoostingRegressor Best score is {}".format(gbrt_best.score(X_test, y_test)))

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


@app.route('/api/v1/resources/books', methods=['GET'])
def api_filter():
    query_parameters = request.args

    id = query_parameters.get('id')
    published = query_parameters.get('published')
    author = query_parameters.get('author')

    query = "SELECT * FROM books WHERE id="
    to_filter = []

    if id:
    #    query += ' id=? AND'
        to_filter.append(id)
    #if published:
    #    query += ' published=? AND'
    #    to_filter.append(published)
    #if author:
    #    query += ' author=? AND'
    #    to_filter.append(author)
    if not (id or published or author):
        return page_not_found(404)

    query += id

    conn = psycopg2.connect(database = "postgres", user = "postgres", password = "bolu@2019", host = "127.0.0.1", port = "5432")
    cur = conn.cursor()

    results = cur.execute(query)
    results = cur.fetchall()

    return jsonify(results)

app.run()