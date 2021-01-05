from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://shiv:bolu@2019@127.0.0.1:5432'

#'postgresql://shiv:bolu@2019@127.0.0.1:5432/idm_srini_team'
#'postgresql://shiv:bolu@2019@127.0.0.1:5432'

db = SQLAlchemy(app)
migrate = Migrate(app, db)
manager=Manager(app)

manager.add_command('db', MigrateCommand)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128))
  
if __name__ == '__main__':
    manager.run()