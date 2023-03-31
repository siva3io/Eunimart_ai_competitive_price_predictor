import logging
from flask import Blueprint, jsonify, request
from app.services.competitive_pricing import CompetitivePricing

competitive_pricing = Blueprint('competitive_pricing', __name__)

logger = logging.getLogger(__name__)


@competitive_pricing.route('/predict_price', methods=['POST'])
def get_competitive_pricing():
    request_data = request.get_json()
    data = CompetitivePricing.get_price(request_data)
    if not data:
        data = {}
    return jsonify(data)
    
