from flask import Blueprint

pep = Blueprint('pep', __name__)


@pep.route("/testforblueprint", methods=['GET', 'POST'])
def test():

    return "test"