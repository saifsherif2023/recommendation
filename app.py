import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from lightfm import LightFM
from lightfm.data import Dataset
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

class RecommendationService:
    def __init__(self):
        self.model = None
        self.user_mapping = None
        self.item_mapping = None
        self.item_features = None
        self.df_products = None
        self.load_data()
        self.load_model()

    def load_data(self):
        """Load product data from CSV and ratings from MongoDB"""
        try:
            # Load data from MongoDB
            client = MongoClient(os.getenv('DB_URI2'))
            db = client['handMade']
            
            self.df_products = pd.DataFrame(list(db.products.find()))
            if '_id' not in self.df_products:
                raise ValueError("Missing '_id' in products collection.")
            self.df_products['_id'] = self.df_products['_id'].astype(str)
            
            self.df_products['category_name'] = self.df_products['category'].apply(
                lambda x: x['name'] if isinstance(x, dict) and 'name' in x else None
            )

            self.ratings_df = pd.DataFrame(list(db.ratings.find()))
            if 'product_id' not in self.ratings_df:
                raise ValueError("Missing 'product_id' in ratings collection.")
            self.ratings_df['product_id'] = self.ratings_df['product_id'].astype(str)

            logger.info("Successfully loaded products and ratings from MongoDB")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_model(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), "models", "trained_model.pkl")
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.user_mapping = model_data['user_mapping']
                self.item_mapping = model_data['item_mapping']
                self.item_features = model_data['item_features']
        except FileNotFoundError:
            logger.warning("Trained model not found. Some features will be limited.")
            self.model = None

    def train_model(self):
        """Train the recommendation model"""
        try:
            valid_product_ids = set(self.df_products['_id'])
            self.ratings_df = self.ratings_df[self.ratings_df['product_id'].isin(valid_product_ids)]
            
            # Prepare dataset
            dataset = Dataset()
            dataset.fit(
                users=self.ratings_df['user_id'].unique(),
                items=self.ratings_df['product_id'].unique(),
                item_features=self.df_products['category_name'].unique()
            )

            # Create mappings
            (self.user_mapping, self.item_mapping, self.item_features) = dataset.mapping()

            # Build interactions matrix
            (interactions, weights) = dataset.build_interactions(
                (row['user_id'], row['product_id'], row['rating']) 
                 for _, row in self.ratings_df.iterrows()
            )

            # Build item features matrix
            item_features = dataset.build_item_features(
                ((row['_id'], [row['category_name']]) 
                 for _, row in self.df_products.iterrows()
                 if pd.notna(row['category_name']))
            )

            # Train model
            self.model = LightFM(loss='warp', learning_rate=0.05)
            self.model.fit(
                interactions=interactions,
                item_features=item_features,
                sample_weight=weights,
                epochs=30,
                num_threads=4
            )

            # Save model
            model_path = os.path.join(os.path.dirname(__file__), "models", "trained_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, "wb") as f:
                pickle.dump({
                    'model': self.model,
                    'user_mapping': self.user_mapping,
                    'item_mapping': self.item_mapping,
                    'item_features': item_features
                }, f)

            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def get_recommendations(self, user_id, n_recommendations=5):
        try:
            if not self.model or user_id not in self.user_mapping[0]:
                return self.get_popular_products(n_recommendations)

            internal_user_id = self.user_mapping[0][user_id]
            n_items = len(self.item_mapping[0])
            scores = self.model.predict(
                user_ids=internal_user_id,
                item_ids=np.arange(n_items),
                item_features=self.item_features
            )

            top_items = np.argsort(-scores)[:n_recommendations]
            original_item_ids = [self.item_mapping[1][item_id] for item_id in top_items]

            recommendations = self.df_products[self.df_products['_id'].isin(original_item_ids)][
                ['_id', 'title', 'price', 'priceAfterDiscount', 'ratingsAverage', 'ratingsQuantity', 'category_name', 'description']
            ]

            return recommendations.to_dict(orient='records')

        except Exception as e:
            logger.error(f"Error in get_recommendations: {str(e)}")
            return self.get_popular_products(n_recommendations)

    def get_popular_products(self, n=5, min_ratings=5):
        popular_df = self.df_products[self.df_products['ratingsQuantity'] >= min_ratings]
        top = popular_df.sort_values(by='ratingsAverage', ascending=False).head(n)[
            ['_id', 'title', 'price', 'priceAfterDiscount', 'ratingsAverage', 'ratingsQuantity', 'category_name', 'description']
        ]
        return top.to_dict(orient='records')

try:
    recommendation_service = RecommendationService()
except Exception as e:
    logger.critical(f"Failed to initialize RecommendationService: {e}")
    recommendation_service = None

@app.route('/')
def home():
    status_msg = "Recommendation service running."
    if recommendation_service is None:
        status_msg += " Waiting for database setup."

    return jsonify({
        'status': 'running',
        'message': status_msg,
        'service': 'recommendation-service',
        'endpoints': {
            'health': '/health',
            'recommend': '/recommend/<user_id>',
            'popular': '/popular',
            'train': '/train'
        }
    }), 200

@app.route('/train', methods=['POST'])
def train():
    if recommendation_service is None:
        return jsonify({"error": "Recommendation service not ready"}), 503
    
    """Train the recommendation model"""
    try:
        logger.info("Starting model training process...")
        
        # Check if data is loaded
        if recommendation_service.df_products is None or recommendation_service.ratings_df is None:
            logger.error("Data not loaded properly")
            return jsonify({
                "error": "Data not loaded properly",
                "details": {
                    "products_loaded": recommendation_service.df_products is not None,
                    "ratings_loaded": recommendation_service.ratings_df is not None
                }
            }), 500

        # Log data statistics
        logger.info(f"Training data stats - Products: {len(recommendation_service.df_products)}, Ratings: {len(recommendation_service.ratings_df)}")
        
        # Attempt training
        success = recommendation_service.train_model()
        
        if success:
            logger.info("Model training completed successfully")
            return jsonify({
                "message": "Model successfully trained and saved",
                "details": {
                    "products_count": len(recommendation_service.df_products),
                    "ratings_count": len(recommendation_service.ratings_df)
                }
            })
        else:
            logger.error("Model training failed")
            return jsonify({
                "error": "Failed to train model",
                "details": "Check application logs for more information"
            }), 500
    except Exception as e:
        logger.error(f"Error in train endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Training failed",
            "details": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    if recommendation_service is None:
        return jsonify({"error": "Recommendation service not ready"}), 503
    
    try:
        recommendations = recommendation_service.get_recommendations(user_id)
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/popular')
def popular():
    if recommendation_service is None:
        return jsonify({"error": "Recommendation service not ready"}), 503
    
    try:
        return jsonify(recommendation_service.get_popular_products())
    except Exception as e:
        logger.error(f"Error in popular endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check MongoDB connection
        try:
            client = MongoClient(os.getenv('DB_URI2'))
            client.admin.command('ping')
            mongo_status = "connected"
            products_count = client['handmade'].products.count_documents({})
            ratings_count = client['handmade'].ratings.count_documents({})
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            mongo_status = "disconnected"
            products_count = 0
            ratings_count = 0

        # Check model status
        model_status = "loaded" if recommendation_service and recommendation_service.model else "not loaded"

        app_status = "waiting_for_data" if products_count == 0 or ratings_count == 0 else "healthy"

        return jsonify({
            'status': app_status,
            'service': 'recommendation-service',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'mongodb': mongo_status,
                'model': model_status
            },
            'data_statistics': {
                'products_count': products_count,
                'ratings_count': ratings_count,
                'model_trained': model_status == "loaded"
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 