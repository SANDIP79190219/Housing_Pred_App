import flask
from flask import request, jsonify, render_template
import requests
import psycopg2
import sqlalchemy

app = flask.Flask('__name__')
app.config["DEBUG"] = True

@app.route("/")
def welcome():
    return render_template("Algo_Test.html")

#if __name__ == __main__
app.run()