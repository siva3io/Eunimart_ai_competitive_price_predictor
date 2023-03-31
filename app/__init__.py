# third-party imports
import os
import logging.config
from flask import Flask
from boto3.session import Session

from flask_redis import FlaskRedis
from flask_sqlalchemy import SQLAlchemy


# local imports
from app.models import *
from config import Config

redis = FlaskRedis()
db = SQLAlchemy()

DEBUG_LEVELV_NUM = 35

logging.addLevelName(DEBUG_LEVELV_NUM, "REQUEST_DEBUG")


def custom_debug(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_LEVELV_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(DEBUG_LEVELV_NUM, message, args, **kws)
     

logging.Logger.request_debug = custom_debug


def create_app():
    app = Flask(__name__, instance_relative_config=True, root_path=os.path.join(os.getcwd(), 'app'))
    app.config.from_object(Config)
    
    redis.init_app(app)
    db.init_app(app)
    
    boto3_session = Session(
            aws_access_key_id = Config.BOTO3_ACCESS_KEY, 
            aws_secret_access_key = Config.BOTO3_SECRET_KEY,
            region_name = Config.BOTO3_REGION
        )

    if Config.DEPLOY_ENV != 'dev':    
        for handler_type in [ "custom_handler", "info_file_handler", "debug_file_handler", "error_file_handler" ]:
            Config.LOGGING_CONFIG["handlers"][handler_type]["boto3_session"] = boto3_session
        
    logging.config.dictConfig(Config.LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.request_debug("Logs Setuped in Region us-east-2")

    from app.routes import blueprints

    for route_blueprint in blueprints:
        app.register_blueprint(route_blueprint, url_prefix='/api')  #replace acl with your service-path 
        
    return app
