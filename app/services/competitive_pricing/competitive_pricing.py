from app.utils import catch_exceptions,download_from_s3
import os
import requests
from PIL import Image
from io import BytesIO
import re
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import preprocess_input,VGG16
from tensorflow.keras.models import load_model
from scipy.sparse import hstack
from constants import channel_id_name_mapping,upcoming_marketplace,not_trained
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from keras import backend as K
from urllib.parse import urlparse
import base64
import json
from config import Config
import warnings
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, MaxPool2D, GlobalAveragePooling2D, Lambda, Conv2D, concatenate, ZeroPadding2D, Layer, MaxPooling2D
import cv2

import pandas as pd

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType
from pyspark.sql.types import *
from pyspark.sql.functions import first, collect_list, mean, min, max
import pyspark.sql.functions as func
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.regression import RandomForestRegressionModel

warnings.filterwarnings("ignore")

logger = logging.getLogger(name=__name__)

class Pricing(object):
    trained_marketplaces = ["Flipkart", "Amazon_USA", "Bonanza", "Etsy", "Amazon_India", "eBay"]
    def __init__(self):
        pass

    @catch_exceptions
    def check_file_exists(self,request_data,file_name):
        try:
            if request_data["marketplace"] in self.trained_marketplaces:
                s3_path = 'pricing_features/2021/2'
            else:
                s3_path = 'pricing_features'

            folder_name = request_data["category_name"]+'/'
            if(file_name == 'price_stats.json' or file_name =='similar_product_model.h5' or file_name == 'mape_values.json'):
                folder_name = ''
            if request_data["marketplace"] == "Amazon_India" and request_data["category_name"] == "Home_and_Kitchen" and folder_name:
                folder_name = folder_name + request_data["sub_category_name"] + '/'

            if request_data["marketplace"] == "Flipkart" and request_data["category_name"] in ["Clothing", "Kitchen__Cookware_and_Serveware"] and folder_name:
                folder_name = folder_name + request_data["sub_category_name"] + '/'
            if request_data["marketplace"] == "Amazon_USA" and request_data["category_name"] in ["Clothing", "Tools_and_Home_Improvement", "Automotive"] and folder_name:
                folder_name = folder_name + request_data["sub_category_name"] + '/'
            if request_data["marketplace"] == "Bonanza" and request_data["category_name"] in ["Collectibles", "Home_and_Garden"] and folder_name:
                folder_name = folder_name + request_data["sub_category_name"] + '/'
            if request_data["marketplace"] == "Etsy" and request_data["category_name"] in ["Craft_Supplies_and_Tools"] and folder_name:
                folder_name = folder_name + request_data["sub_category_name"] + '/'
            if request_data["marketplace"] == "Amazon_UK" and request_data["category_name"] in ["Books"] and folder_name:
                folder_name = folder_name + request_data["sub_category_name"] + '/'
            if request_data["marketplace"] == "eBay" and request_data["category_name"] in ["Business_and_Industrial", "Collectibles_and_Art"] and folder_name:
                folder_name = folder_name + request_data["sub_category_name"] + '/'


            abs_file_path = request_data["marketplace"]+'/'+folder_name+file_name
            print("abs_file_path==>",str(abs_file_path))
            print("s3_path==>",str(s3_path+'/'+abs_file_path))
            if not os.path.exists(abs_file_path):
                os.makedirs("/".join(abs_file_path.split('/')[:-1]), exist_ok=True)
                download_from_s3(s3_path+'/'+abs_file_path,abs_file_path)
            return abs_file_path
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def get_features(self,request_data,file_path,input_data):
        try:
            request_data = request_data["data"]
            if (isinstance(input_data, str)) and (input_data == 'NA' or input_data == "not_available" or input_data == "Not_Available"):
                input_data = ''
            abs_file_path = self.check_file_exists(request_data,file_path)
            model = self.load_pickle(abs_file_path)
            feature = model.transform(input_data)
            return feature
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def clean_text(self,text):
        # https://gist.github.com/sebleier/554280
        # we are removing the words from the stop words list: 'no', 'nor', 'not'
        stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
                    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
                    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
                    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
                    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
                    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
                    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
                    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
                    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
                    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
                    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
                    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
                    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
                    'won', "won't", 'wouldn', "wouldn't"]
        
        # calling the decontracted function
        sent = self.decontracted(text)
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        text = sent.lower().strip()  
        
        return text

    @catch_exceptions
    def decontracted(self,phrase):
        # https://stackoverflow.com/a/47091490/4084039
        # specific
        phrase = str(phrase)
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    @catch_exceptions
    def get_image(self,image):
        try:
            parsed_image = urlparse(image)
            if all([parsed_image.scheme, parsed_image.netloc, parsed_image.path]):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                bytes_string = base64.b64decode(image)
                img = Image.open(BytesIO(bytes_string))
            return img
        except Exception as e:
            logger.error(e,exc_info=True)
            return ''

    @catch_exceptions
    def get_image_features(self,image):
        try:
            image_width, image_height = 224, 224
            image = image.resize((image_width,image_height)) 
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            model = VGG16(include_top=False, weights='imagenet')   
            image_features = model.predict(image)
            image_features = image_features.reshape((1,25088))
            return image_features
        except Exception as e:
            logger.error(e,exc_info=True)
    
    @catch_exceptions
    def similar_product_model(self):
        """
        This function return a custom CNN similar product model.
        """
        vgg_model = VGG19(weights="imagenet", include_top=False, input_shape=(224,224,3))
        convnet_output = GlobalAveragePooling2D()(vgg_model.output)
        convnet_output = Dense(4096, activation='relu')(convnet_output)
        convnet_output = Dropout(0.5)(convnet_output)
        convnet_output = Dense(4096, activation='relu')(convnet_output)
        convnet_output = Dropout(0.5)(convnet_output)
        convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
      
        s1_inp = Input(shape=(224,224,3))    
        s1 = MaxPool2D(pool_size=(4,4),strides = (4,4),padding='valid')(s1_inp)
        s1 = ZeroPadding2D(padding=(4, 4), data_format=None)(s1)
        s1 = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='valid')(s1)
        s1 = ZeroPadding2D(padding=(2, 2), data_format=None)(s1)
        s1 = MaxPool2D(pool_size=(7,7),strides = (4,4),padding='valid')(s1)
        s1 = Flatten()(s1)

        s2_inp = Input(shape=(224,224,3))    
        s2 = MaxPool2D(pool_size=(8,8),strides = (8,8),padding='valid')(s2_inp)
        s2 = ZeroPadding2D(padding=(4, 4), data_format=None)(s2)
        s2 = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='valid')(s2)
        s2 = ZeroPadding2D(padding=(1, 1), data_format=None)(s2)
        s2 = MaxPool2D(pool_size=(3,3),strides = (2,2),padding='valid')(s2)
        s2 = Flatten()(s2)
    
        merge_one = concatenate([s1, s2])
        merge_one_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(merge_one)
        merge_two = concatenate([merge_one_norm, convnet_output], axis=1)
        emb = Dense(4096)(merge_two)
        l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)
    
        final_model = tf.keras.models.Model(inputs=[s1_inp, s2_inp, vgg_model.input], outputs=l2_norm_final)
        return final_model
    
    @catch_exceptions
    def get_image_features_flipkart(self,image,request_data):
        """
        This function is used to extract the image features.
        """
        try:
            image_width, image_height = 224, 224
            image = image.resize((image_width,image_height))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            model = VGG16(include_top=False, weights='imagenet')
            image_features = model.predict(image)
            image_features = image_features.reshape((1,25088))
            return image_features
        except Exception as e:
            logger.error(e,exc_info=True)
    
    @catch_exceptions
    def get_best_selling_rank(self,best_selling_rank):
        try:
            if best_selling_rank and best_selling_rank != 'NA':
                best_selling_rank = (best_selling_rank.split('#')[1]).split(' ')[0]
                return int("".join(re.findall('\d+', best_selling_rank)))
            return 0
        except Exception as e:
            logger.error(e,exc_info=True)
    
    @catch_exceptions
    def get_category_sub_category(self,request_data):
        try:
            get_marketplace_hierarchy = json.loads(requests.post(url = Config.GET_MARKETPLACE_HIERARCHY,json= request_data).text)
            if get_marketplace_hierarchy["status"]:
                return get_marketplace_hierarchy["data"]
        except Exception as e:
            logger.error(e,exc_info=True)
    
    @catch_exceptions    
    def get_price_details(self, request_data, price='', price_range=''):
        product_details = {}
        try:
            product_details["marketplace"] = request_data["marketplace"]

            if request_data["marketplace"] in upcoming_marketplace:
                product_details["price"] = "Coming Soon"
            else:
                product_details["price"] = price
                product_details["price_range"] = price_range
            return product_details
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def tokenize(self,request_data):
        if request_data["marketplace"] in self.trained_marketplaces:
            with open(self.check_file_exists(request_data,'tokenizer_for_text.pk'), 'rb') as file:
                text_tokenizer = pickle.load(file)
        
        tokenized_text = text_tokenizer.texts_to_sequences(request_data['combined_text'])
        tokenized_text = pad_sequences(tokenized_text, maxlen=256)
        return tokenized_text

    @catch_exceptions
    def load_pickle(self,abs_file_path):
        with open(abs_file_path,'rb') as f:
            model = pickle.load(f)
        return model

    @catch_exceptions
    def get_price_3_pyspark(self, request_data, final_features):
        sc = SparkContext.getOrCreate()
        sql = SQLContext(sc)
        
        df = pd.DataFrame()
        df['features'] = final_features.tolist()
        df['label'] = 0
        final_train = sql.createDataFrame(df)

        list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
        final_train = final_train.select(
            final_train["label"], 
            list_to_vector_udf(final_train["features"]).alias("finalFeatures"))

        if request_data["marketplace"] == "Flipkart" and request_data["category_name"] in ["Clothing", "Kitchen__Cookware_and_Serveware"]:
            model_path = "/home/ubuntu/vdezi_ai_competitive_price_predictor/"+request_data['marketplace']+'/'+request_data["category_name"]+'/'+request_data["sub_category_name"]+"/RF_model"
        elif request_data["marketplace"] == "Etsy" and request_data["category_name"] in ["Craft_Supplies_and_Tools"]:
            model_path = "/home/ubuntu/vdezi_ai_competitive_price_predictor/"+request_data['marketplace']+'/'+request_data["category_name"]+'/'+request_data["sub_category_name"]+"/RF_model"
        else:
            model_path = "/home/ubuntu/vdezi_ai_competitive_price_predictor/"+request_data['marketplace']+'/'+request_data["category_name"]+"/RF_model"
        
        model_3 = RandomForestRegressionModel.load(model_path)

        pred = model_3.transform(final_train)
        price_3 = pred.select('prediction').collect()
        return price_3

    @catch_exceptions
    def get_price_3_sklearn(self, request_data, final_features):
        model_3 = self.load_pickle(self.check_file_exists(request_data,'RF_model.pk'))
        price_3 = model_3.predict(final_features)
        return price_3.reshape(-1, 1)

    @catch_exceptions
    def calculate_price(self,request_data,final_features):
        if request_data["marketplace"] in self.trained_marketplaces:
            model_1 = load_model(self.check_file_exists(request_data,'DL_model_1.h5'),compile=False)
            price_1 = model_1.predict(final_features)
            final_features = np.append(final_features, price_1, axis=1)
            model_2 = load_model(self.check_file_exists(request_data,'DL_model_2.h5'),compile=False)
            price_2 = model_2.predict([self.tokenize(request_data),final_features])
            if request_data["marketplace"] == "Flipkart" and request_data["category_name"] == "Pet_Supplies":
                return str(abs(round(price_2[0][0])))
            final_features = np.append(final_features, price_2, axis=1)

            if request_data["marketplace"] in ["Amazon_USA", "Amazon_India", "Amazon_UK", "eBay"]:
                price_3 = self.get_price_3_sklearn(request_data, final_features)
            elif request_data["marketplace"] == "Bonanza" and request_data["category_name"] in ["Specialty_Services", "Collectibles", "Home_and_Garden"]:
                price_3 = self.get_price_3_sklearn(request_data, final_features)
            else:
                price_3 = self.get_price_3_pyspark(request_data, final_features)

            final_features = np.append(final_features, price_3, axis=1)
            model_4 = self.load_pickle(self.check_file_exists(request_data,'XGB_model.pk'))
            price_4 = int(model_4.predict(final_features).tolist()[0])
            price = round(price_4)
        else:
            model = load_model(self.check_file_exists(request_data,'model_1.h5'),compile=False)
            price = round(model.predict(final_features).tolist()[0][0], 2)

        if request_data["marketplace"] == "Amazon_USA":
            price = price/100

        return abs(price)

    @catch_exceptions
    def get_price_stats(self,request_data):
        abs_path_price_stats = self.check_file_exists(request_data,'price_stats.json')
        with open(abs_path_price_stats,'rb') as file:
            price_stats = json.load(file)
        try:
            minimum = price_stats.get(request_data["category_name"]).get(request_data['sub_category_name']).get('minimum')
            maximum = price_stats.get(request_data["category_name"]).get(request_data['sub_category_name']).get('maximum')
            mean = price_stats.get(request_data["category_name"]).get(request_data['sub_category_name']).get('mean')
            median = price_stats.get(request_data["category_name"]).get(request_data['sub_category_name']).get('median')
        except:
            minimum = maximum = mean = median = 0
        return minimum,maximum,mean,median   

    @catch_exceptions
    def get_price_range(self, request_data, price):
        abs_path_mape_values = self.check_file_exists(request_data,'mape_values.json')
        with open(abs_path_mape_values,'rb') as file:
            mape_values = json.load(file)
        mape_value = 0
        category_name = request_data["category_name"]
        sub_category_name = request_data["sub_category_name"]
        if mape_value > price or mape_value == 0:
            percent_value = price/100 * 10
            range_min = str(round(abs(price - percent_value), 2))
            range_max = str(round(abs(price + percent_value), 2))
            total_range = range_min+' - '+range_max
            return total_range

        if request_data["marketplace"] == "Amazon_India" and request_data["category_name"] == "Home_and_Kitchen":
            mape_value = mape_values[category_name][sub_category_name]
        elif request_data["marketplace"] == "Flipkart" and request_data["category_name"] in ["Clothing", "Kitchen__Cookware_and_Serveware"]:
            mape_value = mape_values[category_name][sub_category_name]
        elif request_data["marketplace"] == "Amazon_USA" and request_data["category_name"] in ["Clothing", "Tools_and_Home_Improvement", "Automotive"]:
            mape_value = mape_values["category_name"]["sub_category_name"]
        elif request_data["marketplace"] == "Bonanza" and request_data["category_name"] in ["Collectibles", "Home_and_Garden"]:
            mape_value = mape_values[category_name][sub_category_name]
        elif request_data["marketplace"] == "Etsy" and request_data["category_name"] in ["Craft_Supplies_and_Tools"]:
            mape_value = mape_values[category_name][sub_category_name]
        elif request_data["marketplace"] == "Amazon_UK" and request_data["category_name"] in ["Books"]:
            mape_value = mape_values[category_name][sub_category_name]
        elif request_data["marketplace"] == "eBay" and request_data["category_name"] in ["Business_and_Industrial", "Collectibles_and_Art"]:
            mape_value = mape_values[category_name][sub_category_name]
        else:
            mape_value = mape_values[category_name]

        range_min = str(round(abs(price - mape_value), 2))
        range_max = str(round(abs(price + mape_value), 2))
        total_range = range_min+' - '+range_max
        return total_range

    def get_currency(self, request_data, price, price_range):
        if request_data["marketplace"] in ["Amazon_India", "Flipkart"]:
            return "INR "+price, "(INR "+price_range+")"
        elif request_data["marketplace"] in ["Amazon_USA", "Etsy", "eBay"]:
            return "USD "+price, "(USD "+price_range+")"
        elif request_data["marketplace"] in ["Amazon_UK", "Bonanza"]:
            return "GBP "+price, "(GBP "+price_range+")"


    @catch_exceptions
    def predict_price(self,request_data, image):
        try:  
            request_data['data']['combined_text'] = [self.clean_text(10*(request_data["data"]['brand']+' ')+5*(request_data["data"]['color']+' ')+3*(request_data["data"]['product_title']+' ')+request_data["data"]['description'])]
            price_details_list = []
            channel_id_list = request_data["data"]["channel_id"]
            for channel_id in channel_id_list:
                request_data["data"]["channel_id"]  = channel_id
                if not channel_id_name_mapping.get(channel_id):
                    continue
                request_data["data"]["marketplace"] = channel_id_name_mapping.get(channel_id)
                if request_data["data"]["marketplace"] not in upcoming_marketplace:
                    request_data["data"].update(self.get_category_sub_category(request_data))
                    print("\n====================>", request_data["data"]["marketplace"])
                    print("====================>", request_data["data"]["category_name"])

                    # mapping of bonanza categories
                    if request_data["data"]["marketplace"] == "Bonanza" and request_data["data"]["category_name"] == "Antiquities":
                        request_data["data"]["category_name"] = "Antiques"
                    if request_data["data"]["marketplace"] == "Bonanza" and request_data["data"]["category_name"] == "Specialty_Services":
                        request_data["data"]["category_name"] = "Everything_Else"
                    if request_data["data"]["marketplace"] == "Bonanza" and request_data["data"]["category_name"] in ["DVDs_and_Movies", "Music"]:
                        request_data["data"]["category_name"] = "Entertainment_Memorabilia"
                    
                    if request_data["data"]["marketplace"] == "Amazon_UK" and request_data["data"]["category_name"] == "Books" and request_data["data"]["sub_category_name"] == "Horror":
                        request_data["data"]["sub_category_name"] = "Fiction"
                    if request_data["data"]["marketplace"] == "Amazon_UK" and request_data["data"]["category_name"] == "Books" and request_data["data"]["sub_category_name"] == "School_Books":
                        request_data["data"]["sub_category_name"] = "Childrens_Books"
                    
                    if request_data["data"]["marketplace"] == "Flipkart" and request_data["data"]["category_name"] not in not_trained:
                        image_features = self.get_image_features_flipkart(image,request_data)
                    else:
                        image_features = self.get_image_features(image)

                    best_selling_rank = self.get_best_selling_rank(request_data["data"]["best_selling_rank"])

                    if request_data["data"]["marketplace"] in self.trained_marketplaces:
                        text_features = self.get_features(request_data,'vectorizer_for_text.pk',request_data['data']['combined_text'])
                        image_features = self.get_features(request_data,'tsvd_vectorizer_for_image_features.pk',image_features)
                        sub_category_features = self.get_features(request_data,'vectorizer_for_sub_category.pk',[request_data["data"]['sub_category_name']])

                        minimum,maximum,mean,median = self.get_price_stats(request_data['data'])

                        combined_features = hstack((text_features,sub_category_features,best_selling_rank,image_features,minimum,maximum,mean,median)).toarray()
                        
                        scaled_features = self.get_features(request_data,'vectorizer_for_scaling_of_data.pk',combined_features)
                        final_features = self.get_features(request_data,'vectorizer_for_selection_of_data.pk',scaled_features)

                    price = self.calculate_price(request_data["data"],final_features)
                    price_range = self.get_price_range(request_data["data"], price)
                    price, price_range = self.get_currency(request_data["data"], str(price), price_range)
                    price_details = self.get_price_details(request_data["data"], price, price_range)
                else:
                    price_details = self.get_price_details(request_data["data"])
                price_details_list.append(price_details)
                K.clear_session()
            return price_details_list
        except Exception as e:
            logger.error(e,exc_info=True)

    @catch_exceptions
    def get_price(self,request_data):
        try:
            response_data = {}
            mandatory_fields = ["sku_id","account_id","channel_id","image","product_title","brand","description","color"]
            for field in mandatory_fields:
                if not field in request_data["data"]:
                    response_data = {
                        "status":False,
                        "message":"Required field is missing",
                        "error_obj":{
                            "description":"{} is missing".format(field),
                            "error_code":"REQUIRED_FIELD_IS_MISSING"
                        }
                    }
                if field in request_data["data"]:
                    if len(request_data["data"][field])== 0:
                        response_data = {
                            "status":False,
                            "message":"Required field is missing",
                            "error_obj":{
                                "description":"{} is missing".format(field),
                                "error_code":"REQUIRED_FIELD_IS_MISSING"
                            }
                        }
            optional_fields = ["best_selling_rank"]
            for field in optional_fields:
                if field not in request_data["data"]:
                    request_data["data"]["best_selling_rank"] = "#1"

            if not response_data:
                image = self.get_image(request_data["data"]["image"])
                if image:
                    price_details = self.predict_price(request_data, image)
                    response_data = {
                        "status":True,
                        "data":price_details,
                        "columns": [
                            {
                                "column_key": "marketplace",
                                "column_name": "Marketplace",
                                "column_position": 1,
                            },
                            {
                                "column_key": "price",
                                "column_name": "Price",
                                "column_position": 2,
                            },
                            {
                                "column_key": "price_range",
                                "column_name": "Price Range",
                                "column_position": 3,
                            },
                        ]
                    }
                else:
                    response_data = {
                    "status":False,
                    "message":"Image is not valid",
                    "error_obj":{
                        "description":"Image is not valid",
                        "error_code":"INVALID_IMAGE"
                    }
                }
            
            return response_data
        except Exception as e:
            logger.error(e,exc_info=True)



CompetitivePricing = Pricing()

