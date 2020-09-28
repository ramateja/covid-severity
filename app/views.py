from flask_restful import Resource,request
from flask import render_template, make_response, send_from_directory
import os, config

class HomePage(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'),200,headers)

class DisplayImage(Resource):
    def get(self,filename):
        return send_from_directory(os.path.join(config.BASE_DIR,config.STATIC_FILES_PATH),filename)