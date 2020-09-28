import torchxrayvision as xrv
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import skimage , skimage.filters
import cv2
import numpy as np
from flask_restful import Resource,request
import matplotlib.pyplot as plt
from flask import render_template, make_response, send_from_directory
import random
import config
import os
import math

theta_bias_geographic_extent = torch.from_numpy(np.asarray((0.8705248236656189, 3.4137437)))

class PneumoniaSeverityNet(torch.nn.Module):
    def __init__(self):
        super(PneumoniaSeverityNet, self).__init__()
        self.model = xrv.models.DenseNet(weights="all")
        self.model.op_threshs = None
        self.theta_bias_geographic_extent = torch.from_numpy(np.asarray((0.8705248236656189, 3.4137437)))
        self.theta_bias_opacity = torch.from_numpy(np.asarray((0.5484423041343689, 2.5535977)))

    def forward(self, x):
        preds = self.model(x)
        preds = preds[0,xrv.datasets.default_pathologies.index("Lung Opacity")]
        geographic_extent = preds*self.theta_bias_geographic_extent[0]+self.theta_bias_geographic_extent[1]
        opacity = preds*self.theta_bias_opacity[0]+self.theta_bias_opacity[1]
        geographic_extent = torch.clamp(geographic_extent,0,8)
        opacity = torch.clamp(opacity,0,6)
        return {"geographic_extent":geographic_extent,"opacity":opacity}



def transform_image(img):
    #img = skimage.io.imread(file_path)
    img = xrv.datasets.normalize(img, 255)  

    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]                    


    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    
    img = transform(img)
    img = torchvision.transforms.Compose([])(img)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.requires_grad_()
    return img

def load_model():
    model   = PneumoniaSeverityNet()
    return model

def get_predictions(img, model):
    
    preds = model(img)
    return preds

def get_overlay(opt, img):
    grads = torch.autograd.grad(opt["geographic_extent"], img, retain_graph=True)[0][0][0]
    blurred = skimage.filters.gaussian(grads**2, sigma=(5, 5), truncate=3.5)
    return blurred

def convert_image(content):
    cv2_format = cv2.imdecode(
        np.fromstring(content.read(), np.uint8), cv2.IMREAD_COLOR
    )
    # print(np.fromstring(content.read(), np.uint8).shape)
    return cv2_format

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)

model = load_model()

class XrayAPI(Resource):
    def post(self):
        try:
            files = request.files.getlist('scan')
        except Exception as e:
            return {
                "message":"scan field missing"
            }, 401
        random_file_id = str(random.randint(100000,999999))
        if len(files)>1:
            return {
                "message":"Cannot process multiple files"
            }, 402
        content = files[0] 
        img_cv2 = convert_image(content)
        
        
        img = transform_image(img_cv2)
        file_save_loc = os.path.join(config.BASE_DIR, config.APP_FOLDER, config.STATIC_FILES_PATH,config.INCOMING_FILES_PATH, random_file_id+".jpg")
       
        global model
        outputs = get_predictions(img, model)
        grads = torch.autograd.grad(outputs["geographic_extent"], img)[0][0][0]
        blurred = skimage.filters.gaussian(grads**2, sigma=(5, 5), truncate=3.5)
        
        render = False
        if "viz" in request.form:
            render = int(request.form["viz"]) == 1
        
        file_names = []

        full_frame()
        my_dpi = 100
        fig = plt.figure(frameon=False, figsize=(224/my_dpi, 224/my_dpi), dpi=my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img[0][0].detach(), cmap="gray", aspect='auto')
        plt.savefig(file_save_loc)
        ax.imshow(blurred, alpha=0.5)
        viz_save_loc = os.path.join(config.BASE_DIR, config.APP_FOLDER,config.STATIC_FILES_PATH, config.VISUALISATION_FILES_PATH, 
                        random_file_id+"_vis.jpg")
        plt.savefig(viz_save_loc)
        file_names.append((config.INCOMING_FILES_PATH+"/"+random_file_id+".jpg",
                            config.VISUALISATION_FILES_PATH+"/"+random_file_id+"_vis.jpg",
                            truncate(outputs["geographic_extent"].detach().numpy(),2),
                            truncate(outputs["opacity"].detach().numpy(),2))
                        )
        # output["visualisation"][pathalogy] = random_file_id+".jpg"
        d = file_names[0]
        if render:
            headers = {'Content-Type': 'text/html'}
            return make_response(render_template(
                    'Results.html', input=d[0], vis=d[1],geo=d[2],opac=d[3]
                ),200,headers)
        else:
            return outputs
            
    