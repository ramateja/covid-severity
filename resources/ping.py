from flask_restful import Resource,request



class ping(Resource):
    def get(self):
        
        return {
            "message" : "server ping successful"
        }
