from flask import Flask

app = Flask (__name__)

from flaskWebSite.frontend.routes import frontend
from flaskWebSite.processAPI.routes import pep

app.register_blueprint(frontend)
app.register_blueprint(pep)


