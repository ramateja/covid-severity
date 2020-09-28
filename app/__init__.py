from flask import Blueprint
from flask_restful import Api
from flask import Flask

app = Flask(__name__)
api = Api(app)




from resources.ping import ping
from app.views import HomePage, DisplayImage
from resources.covid_severity import XrayAPI

api.add_resource(ping, '/ping')
api.add_resource(HomePage, '/home')
api.add_resource(XrayAPI, '/covid')
api.add_resource(DisplayImage, '/<path:filename>')




