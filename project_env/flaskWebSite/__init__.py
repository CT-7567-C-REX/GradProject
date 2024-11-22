from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask (__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = "tde"
db = SQLAlchemy(app)

from flaskWebSite.frontend.routes import frontend
from flaskWebSite.processAPI.routes import pep

app.register_blueprint(frontend)
app.register_blueprint(pep)


