import flask
from flask import request, jsonify, render_template
import requests
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
    i = 0
    data1 = pd.DataFrame(np.random.rand(0, 3) * 0, columns=['Feature_Name', 'Cat_Name', 'Cat_Count'])
    for feature in list(train.select_dtypes(include=["object"]).columns):
        f = list(train.groupby(by=feature)[feature].count().index)
        c = list(train.groupby(by=feature)[feature].count())
        fc = zip(f, c)
        for idx, item in enumerate(fc):
            cat, co = item
            data1.loc[i, 'Feature_Name'] = feature
            data1.loc[i, 'Cat_Name'] = cat
            data1.loc[i, 'Cat_Count'] = co
            i += 1
    return data1


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


@app.route("/")
@app.route("/home")
@app.route("/index")
@app.route('/', methods=['GET', 'POST'])
def home():
    # data = pd.read_csv('train.csv')
    # return render_template('Home.html')
    return '''<h1>Sales Price Prediction Application</h1>
<!DOCTYPE HTML>
<html lang = "en">
  <head>
    <title>formDemo.html</title>
    <meta charset = "UTF-8" />
  </head>
  <body>
    <p>Sales Price Prediction parameters</p>
    <form>
      <fieldset>
        <legend>Enter Model Parameters</legend>
        <p>
          <label>KitchenAbvGr   </label>
          <input type = "number"
                 id = "a"
                 name = "KitchenAbvGr"
                 value = "" />
          <t></t>       
          <label>SalesPrice   </label>
          <input type = "number"
                 id = "f"
                 name = "SalesPrice"
                 value = "" />		        
        </p>
        <p>
          <label>OverallQual    </label>
          <input type = "number"
                 id = "b"
                 name = "OverallQual"
                 value = "" />
        </p>
        <p>
          <label>Fireplaces     </label>
          <input type = "number"
                 id = "c"
                 name = "Fireplaces"
                 value = "" />
        <p>
          <label>BedroomAbvGr   </label>
          <input type = "number"
                 id = "d"
                 name = "BedroomAbvGr"
                 value = "" />
        </p>
        <p>
          <label>BsmtFullBath   </label>
          <input type = "number"
                 id = "e"
                 name = "BsmtFullBath"
                 value = "" />
        </p>
        <p>
          <label></label>
        <script type="text/python">
        def generateFullName():
             return '/api/v1'      
        </script>
          <button type="submit" formmethod="POST" formaction="/api/v1/resources/books/data">Predict Price</button>
        </p>
      </fieldset>
    </form>
  </body>
</html>'''


@app.route('/api/v1', methods=['POST'])
def data_link():
    KitchenAbvGr = request.form.get('KitchenAbvGr')
    OverallQual = request.form.get('OverallQual')
    Fireplaces = request.form.get('Fireplaces')
    BedroomAbvGr = request.form.get('BedroomAbvGr')
    BsmtFullBath = request.form.get('BsmtFullBath')
    return jsonify({'KitchenAbvGr': KitchenAbvGr, 'OverallQual': OverallQual, 'Fireplaces': Fireplaces,
                    'BedroomAbvGr': BedroomAbvGr, 'BsmtFullBath': BsmtFullBath})


@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    # conn = sqlite3.connect('books.db')
    # conn.row_factory = dict_factory
    # cur = conn.cursor()
    # all_books = cur.execute('SELECT * FROM books;').fetchall()

    conn = psycopg2.connect(database="postgres", user="postgres", password="bolu@2019", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    cur.execute('select * from books')
    all_books = cur.fetchall()
    return jsonify(all_books)


@app.route('/api/v1/resources/books/data', methods=['POST'])
def api_data():
    KitchenAbvGr = request.form.get('KitchenAbvGr')
    OverallQual = request.form.get('OverallQual')
    Fireplaces = request.form.get('Fireplaces')
    BedroomAbvGr = request.form.get('BedroomAbvGr')
    BsmtFullBath = request.form.get('BsmtFullBath')

    train = pd.read_csv("train.csv")
    data = freq_dist(train)
    data['Percent'] = data.Cat_Count / train.shape[0] * 100
    data = data.set_index('Feature_Name')
    data = data[data['Percent'] < 10]

    for i in range(len(data)):
        col = data.iloc[i].name
        val = data.iloc[i].Cat_Name
        train[col] = train[col].replace(val, 'Other', method='ffill')
    train = pd.get_dummies(train, drop_first=True)
    y = train['SalePrice'].values

    # OverallQual = int(OverallQual)
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import Lasso

    lm = LinearRegression()
    lm.fit(train[['KitchenAbvGr', 'OverallQual', 'Fireplaces', 'BedroomAbvGr', 'BsmtFullBath']], y)
    result = lm.predict([[int(KitchenAbvGr), int(OverallQual), int(Fireplaces), int(BedroomAbvGr), int(BsmtFullBath)]])

    GBM = GradientBoostingRegressor(max_depth=2, n_estimators=100)
    GBM.fit(train[['KitchenAbvGr', 'OverallQual', 'Fireplaces', 'BedroomAbvGr', 'BsmtFullBath']], y)
    result1 = GBM.predict(
        [[int(KitchenAbvGr), int(OverallQual), int(Fireplaces), int(BedroomAbvGr), int(BsmtFullBath)]])

    ridge = Ridge(alpha=0.01, normalize=True)
    ridge.fit(train[['KitchenAbvGr', 'OverallQual', 'Fireplaces', 'BedroomAbvGr', 'BsmtFullBath']], y)
    result2 = ridge.predict(
        [[int(KitchenAbvGr), int(OverallQual), int(Fireplaces), int(BedroomAbvGr), int(BsmtFullBath)]])

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(train[['KitchenAbvGr', 'OverallQual', 'Fireplaces', 'BedroomAbvGr', 'BsmtFullBath']], y)
    result3 = regressor.predict(
        [[int(KitchenAbvGr), int(OverallQual), int(Fireplaces), int(BedroomAbvGr), int(BsmtFullBath)]])

    tree_reg = DecisionTreeRegressor(max_depth=2)
    tree_reg.fit(train[['KitchenAbvGr', 'OverallQual', 'Fireplaces', 'BedroomAbvGr', 'BsmtFullBath']], y)
    result4 = tree_reg.predict(
        [[int(KitchenAbvGr), int(OverallQual), int(Fireplaces), int(BedroomAbvGr), int(BsmtFullBath)]])

    lasso = Lasso(alpha=0.4, normalize=True)
    lasso.fit(train[['KitchenAbvGr', 'OverallQual', 'Fireplaces', 'BedroomAbvGr', 'BsmtFullBath']], y)
    result5 = lasso.predict(
        [[int(KitchenAbvGr), int(OverallQual), int(Fireplaces), int(BedroomAbvGr), int(BsmtFullBath)]])
    # return jsonify({'Predicted House Sale Price is': result[0]})
    # return jsonify({'Predicted House Sale Price is': OverallQual})
    html = '''<h1>Sales Price Prediction Application</h1>
<!DOCTYPE HTML>
<html lang = "en">
  <head>
    <title>formDemo.html</title>
    <meta charset = "UTF-8" />
  </head>
  <body>
    <p>Sales Price Prediction parameters</p>
    <form>
      <fieldset>
        <legend>Enter Model Parameters</legend>
        <p>
          <label>&nbsp &nbsp &nbsp &nbsp KitchenAbvGr &nbsp &nbsp</label>
          <input type = "number"
                 id = "a"
                 name = "KitchenAbvGr"
                 value = %s />
          <t></t>       
          <label &nbsp;> &nbsp &nbsp &nbsp &nbsp House Price [ LinearRegression ] &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp</label>
          <input type = "number"
                 id = "f"
                 name = "SalesPrice"
                 value = %s />		        
        </p>
        <p>
          <label>&nbsp &nbsp &nbsp &nbsp OverallQual &nbsp &nbsp &nbsp &nbsp </label>
          <input type = "number"
                 id = "b"
                 name = "OverallQual"
                 value = %s />
          <label &nbsp;> &nbsp &nbsp &nbsp &nbsp House [ GradientBoostingRegressor ] &nbsp &nbsp &nbsp</label>
          <input type = "number"
                 id = "f"
                 name = "SalesPrice"
                 value = %s />	
        </p>
        <p>
          <label>&nbsp &nbsp &nbsp &nbsp Fireplaces &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp</label>
          <input type = "number"
                 id = "c"
                 name = "Fireplaces"
                 value = %s />
          <label &nbsp;> &nbsp &nbsp &nbsp &nbsp House Price [ Ridge ] &nbsp &nbsp &nbsp  &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp  &nbsp &nbsp &nbsp  &nbsp &nbsp &nbsp</label>
          <input type = "number"
                 id = "f"
                 name = "SalesPrice"
                 value = %s />	
        <p>
          <label>&nbsp &nbsp &nbsp &nbsp BedroomAbvGr &nbsp</label>
          <input type = "number"
                 id = "d"
                 name = "BedroomAbvGr"
                 value = %s />
          <label &nbsp;> &nbsp &nbsp &nbsp &nbsp House Price [ RandomForestRegressor] &nbsp</label>
          <input type = "number"
                 id = "f"
                 name = "SalesPrice"
                 value = %s />	
        </p>
        <p>
          <label>&nbsp &nbsp &nbsp &nbsp BsmtFullBath &nbsp &nbsp &nbsp</label>
          <input type = "number"
                 id = "e"
                 name = "BsmtFullBath"
                 value = %s />
                 
          <label &nbsp;> &nbsp &nbsp &nbsp &nbsp House Price [ DecisionTreeRegressor ] &nbsp</label>
          <input type = "number"
                 id = "f"
                 name = "SalesPrice"
                 value = %s />	
                 
          <label &nbsp;> &nbsp &nbsp &nbsp &nbsp House Price [ Lasso ] &nbsp</label>
          <input type = "number"
                 id = "f"
                 name = "SalesPrice"
                 value = %s />	
        </p>
        <p>
          <label></label>
        <script type="text/python">
        def generateFullName():
             return '/api/v1'      
        </script>
          <button type="submit" formmethod="POST" formaction="/api/v1/resources/books/data">Predict Price</button>
        </p>
      </fieldset>
    </form>
  </body>
</html>''' % (
    KitchenAbvGr, round(result[0]), OverallQual, round(result1[0]), Fireplaces, round(result2[0]), BedroomAbvGr,
    round(result3[0]), BsmtFullBath, round(result4[0]), round(result5[0]))
    # html1 = '''<label>&nbsp &nbsp &nbsp &nbsp BsmtFullBath   </label>'''
    return html


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


@app.route('/api/templates/')
def student():
    return render_template('student.html')


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
    # if published:
    #    query += ' published=? AND'
    #    to_filter.append(published)
    # if author:
    #    query += ' author=? AND'
    #    to_filter.append(author)
    if not (id or published or author):
        return page_not_found(404)

    query += id

    conn = psycopg2.connect(database="postgres", user="postgres", password="bolu@2019", host="127.0.0.1", port="5432")
    cur = conn.cursor()

    results = cur.execute(query)
    results = cur.fetchall()

    return jsonify(results)


app.run()
