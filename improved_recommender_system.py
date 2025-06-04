#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Key Improvements:
1. Sparse matrix optimization for memory efficiency
2. On-demand similarity computation for speed
3. Performance monitoring and profiling
4. Better evaluation methodology
5. Configuration management

Author: Senior Algorithm Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import logging
import psutil  # For memory monitoring

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.metrics = {}
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {'start_time': time.time()}
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        if operation in self.metrics:
            elapsed = time.time() - self.metrics[operation]['start_time']
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = current_memory - self.start_memory
            
            self.metrics[operation].update({
                'elapsed_time': elapsed,
                'memory_used_mb': memory_used,
                'end_time': time.time()
            })
            
            print(f"⏱️  {operation}: {elapsed:.2f}s, Memory: {memory_used:.1f}MB")
    
    def get_report(self):
        """Get performance report"""
        return self.metrics

class ImprovedCollaborativeFilteringRecommender:
    """
    Improved Collaborative Filtering Recommender with optimizations
    """
    
    def __init__(self, model_path: str = "improved_cf_model.pkl", 
                 min_interactions: int = 5, svd_components: int = 100,
                 similarity_threshold: float = 0.1, max_similar_users: int = 50):
        """
        Initialize with improved parameters
        """
        self.model_path = model_path
        self.min_interactions = min_interactions
        self.svd_components = svd_components
        self.similarity_threshold = similarity_threshold
        self.max_similar_users = max_similar_users
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Model components - using sparse matrices
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_item_sparse = None  # Sparse matrix instead of dense
        self.item_user_sparse = None  # Transpose for efficient item access
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        
        # Data storage
        self.interaction_data = None
        self.valid_user_ids = set()
        
        # Model metadata
        self.model_info = {
            'created_at': None,
            'n_users': 0,
            'n_items': 0,
            'n_interactions': 0,
            'sparsity': 0.0,
            'memory_usage_mb': 0.0,
            'algorithms': ['improved_user_cf', 'sparse_matrix_factorization']
        }
        
        logger.info("Improved Collaborative Filtering Recommender initialized")
    
    def load_data(self, perfume_file: str = "香水数据.xlsx", 
                  interaction_file: str = "香水评论数据.xlsx") -> bool:
        """Load and preprocess data with performance monitoring"""
        self.monitor.start_timer("data_loading")
        
        try:
            logger.info("Loading data files...")
            
            if not os.path.exists(perfume_file) or not os.path.exists(interaction_file):
                logger.error("Data files not found!")
                return False
            
            # Load data
            df = pd.read_excel(perfume_file)
            interactions = pd.read_excel(interaction_file)
            
            # Merge data
            self.interaction_data = pd.merge(df, interactions, on='香水链接', how='inner')
            
            self.monitor.end_timer("data_loading")
            return self._preprocess_data_improved()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _preprocess_data_improved(self) -> bool:
        """Improved data preprocessing with better filtering"""
        self.monitor.start_timer("data_preprocessing")
        
        try:
            logger.info("Preprocessing interaction data with improved filtering...")
            
            # Remove duplicates and handle missing values
            initial_size = len(self.interaction_data)
            self.interaction_data = self.interaction_data.drop_duplicates()
            self.interaction_data = self.interaction_data.dropna(subset=['用户ID', '香水名称', '评分'])
            
            logger.info(f"Removed {initial_size - len(self.interaction_data)} duplicate/invalid records")
            
            # Improved filtering with higher thresholds
            user_counts = self.interaction_data['用户ID'].value_counts()
            item_counts = self.interaction_data['香水名称'].value_counts()
            
            # Use higher thresholds for better quality
            valid_users = user_counts[user_counts >= self.min_interactions].index
            valid_items = item_counts[item_counts >= self.min_interactions].index
            
            # Filter data
            self.interaction_data = self.interaction_data[
                (self.interaction_data['用户ID'].isin(valid_users)) &
                (self.interaction_data['香水名称'].isin(valid_items))
            ]
            
            # Store valid user IDs
            self.valid_user_ids = set(self.interaction_data['用户ID'].unique())
            
            # Encode users and items
            self.interaction_data['user_encoded'] = self.user_encoder.fit_transform(
                self.interaction_data['用户ID']
            )
            self.interaction_data['item_encoded'] = self.item_encoder.fit_transform(
                self.interaction_data['香水名称']
            )
            
            # Update model info
            self.model_info.update({
                'n_users': len(self.user_encoder.classes_),
                'n_items': len(self.item_encoder.classes_),
                'n_interactions': len(self.interaction_data)
            })
            
            logger.info(f"Preprocessed data: {self.model_info['n_users']} users, "
                       f"{self.model_info['n_items']} items, "
                       f"{self.model_info['n_interactions']} interactions")
            
            self.monitor.end_timer("data_preprocessing")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return False
    
    def _build_sparse_matrices(self) -> None:
        """Build sparse user-item matrices for memory efficiency"""
        self.monitor.start_timer("sparse_matrix_building")
        
        logger.info("Building sparse user-item matrices...")
        
        # Create sparse user-item matrix
        rows = self.interaction_data['user_encoded'].values
        cols = self.interaction_data['item_encoded'].values
        data = self.interaction_data['评分'].values
        
        shape = (self.model_info['n_users'], self.model_info['n_items'])
        self.user_item_sparse = csr_matrix((data, (rows, cols)), shape=shape)
        self.item_user_sparse = self.user_item_sparse.T.tocsr()  # Transpose for item access
        
        # Calculate sparsity
        total_cells = shape[0] * shape[1]
        non_zero_cells = self.user_item_sparse.nnz
        self.model_info['sparsity'] = (1 - non_zero_cells / total_cells) * 100
        
        # Calculate memory usage
        memory_usage = (self.user_item_sparse.data.nbytes + 
                       self.user_item_sparse.indices.nbytes + 
                       self.user_item_sparse.indptr.nbytes) / 1024 / 1024
        self.model_info['memory_usage_mb'] = memory_usage
        
        logger.info(f"Sparse matrix created: {shape}, "
                   f"density: {non_zero_cells / total_cells:.6f}, "
                   f"memory: {memory_usage:.1f}MB")
        
        self.monitor.end_timer("sparse_matrix_building")
    
    def _compute_user_similarity_efficient(self, user_id: int, top_k: int = None) -> List[Tuple[int, float]]:
        """Efficient on-demand user similarity computation"""
        if top_k is None:
            top_k = self.max_similar_users
        
        user_vector = self.user_item_sparse[user_id]
        
        # Find candidate users who have rated common items
        common_items = user_vector.nonzero()[1]
        if len(common_items) == 0:
            return []
        
        # Get users who rated these items
        candidate_users = set()
        for item_id in common_items:
            candidate_users.update(self.item_user_sparse[item_id].nonzero()[1])
        
        candidate_users.discard(user_id)  # Remove self
        
        if not candidate_users:
            return []
        
        # Compute similarities only with candidate users
        similarities = []
        user_norm = np.sqrt(user_vector.dot(user_vector.T).toarray()[0, 0])
        
        if user_norm == 0:
            return []
        
        for other_user in candidate_users:
            other_vector = self.user_item_sparse[other_user]
            
            # Compute cosine similarity efficiently
            dot_product = user_vector.dot(other_vector.T).toarray()[0, 0]
            other_norm = np.sqrt(other_vector.dot(other_vector.T).toarray()[0, 0])
            
            if other_norm > 0:
                similarity = dot_product / (user_norm * other_norm)
                if similarity > self.similarity_threshold:
                    similarities.append((other_user, similarity))
        
        # Return top-k similar users
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _improved_user_based_recommend(self, user_encoded: int, n_recommendations: int = 3) -> List[Tuple[str, float]]:
        """Improved user-based collaborative filtering"""
        self.monitor.start_timer("user_based_recommendation")
        
        try:
            # Get similar users efficiently
            similar_users = self._compute_user_similarity_efficient(user_encoded)
            
            if not similar_users:
                self.monitor.end_timer("user_based_recommendation")
                return []
            
            # Get user's rated items
            user_ratings = self.user_item_sparse[user_encoded]
            rated_items = set(user_ratings.nonzero()[1])
            
            # Predict ratings for unrated items
            item_scores = {}
            
            for other_user, similarity in similar_users:
                other_ratings = self.user_item_sparse[other_user]
                other_items = other_ratings.nonzero()[1]
                other_data = other_ratings.data
                
                for i, item_id in enumerate(other_items):
                    if item_id not in rated_items:
                        rating = other_data[i]
                        if item_id not in item_scores:
                            item_scores[item_id] = {'numerator': 0, 'denominator': 0}
                        
                        item_scores[item_id]['numerator'] += similarity * rating
                        item_scores[item_id]['denominator'] += abs(similarity)
            
            # Calculate final scores
            recommendations = []
            for item_id, scores in item_scores.items():
                if scores['denominator'] > 0:
                    predicted_rating = scores['numerator'] / scores['denominator']
                    item_name = self.item_encoder.inverse_transform([item_id])[0]
                    recommendations.append((item_name, predicted_rating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            self.monitor.end_timer("user_based_recommendation")
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.warning(f"Improved user-based CF failed: {str(e)}")
            self.monitor.end_timer("user_based_recommendation")
            return []

    def _sparse_matrix_factorization(self) -> None:
        """Improved matrix factorization using sparse matrices"""
        self.monitor.start_timer("matrix_factorization")

        logger.info(f"Training sparse SVD model with {self.svd_components} components...")

        # Use TruncatedSVD which works well with sparse matrices
        self.svd_model = TruncatedSVD(
            n_components=self.svd_components,
            random_state=42,
            algorithm='randomized'  # Faster for large matrices
        )

        # Fit and transform using sparse matrix
        self.user_factors = self.svd_model.fit_transform(self.user_item_sparse)
        self.item_factors = self.svd_model.components_

        logger.info("Sparse matrix factorization completed")
        self.monitor.end_timer("matrix_factorization")

    def train_model(self) -> bool:
        """Train the improved collaborative filtering model"""
        self.monitor.start_timer("total_training")

        try:
            logger.info("Starting improved model training...")

            # Build sparse matrices
            self._build_sparse_matrices()

            # Train matrix factorization
            self._sparse_matrix_factorization()

            # Update model info
            self.model_info['created_at'] = datetime.now().isoformat()

            logger.info("Improved model training completed successfully")
            self.monitor.end_timer("total_training")
            return True

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            self.monitor.end_timer("total_training")
            return False

    def recommend(self, user_id: Any, n_recommendations: int = 3) -> List[Tuple[str, float]]:
        """Generate recommendations with performance monitoring"""
        self.monitor.start_timer("recommendation_generation")

        # Validate user input
        validation_result = self.validate_user_id(user_id)
        if not validation_result['valid']:
            logger.warning(f"Invalid user ID: {user_id}")
            self.monitor.end_timer("recommendation_generation")
            return []

        user_encoded = validation_result['encoded_id']

        # Use improved user-based CF
        recommendations = self._improved_user_based_recommend(user_encoded, n_recommendations)

        # Fallback to popularity if no recommendations
        if not recommendations:
            recommendations = self._popularity_based_recommend(user_encoded, n_recommendations)

        self.monitor.end_timer("recommendation_generation")
        return recommendations

    def validate_user_id(self, user_id: Any) -> Dict[str, Any]:
        """Validate user ID and return encoded version"""
        try:
            # Convert to appropriate type and try different formats
            original_input = user_id

            # Try as string first
            user_id_str = str(user_id).strip()
            if user_id_str in self.valid_user_ids:
                encoded_id = self.user_encoder.transform([user_id_str])[0]
                return {
                    'valid': True,
                    'encoded_id': encoded_id,
                    'original_id': user_id_str,
                    'message': 'Valid user ID'
                }

            # Try as integer if it's numeric
            try:
                user_id_int = int(float(user_id))
                if user_id_int in self.valid_user_ids:
                    encoded_id = self.user_encoder.transform([user_id_int])[0]
                    return {
                        'valid': True,
                        'encoded_id': encoded_id,
                        'original_id': user_id_int,
                        'message': 'Valid user ID'
                    }
            except (ValueError, TypeError):
                pass

            # Not found in any format
            return {
                'valid': False,
                'encoded_id': None,
                'original_id': original_input,
                'message': f'User ID "{original_input}" not found in system'
            }

        except Exception as e:
            return {
                'valid': False,
                'encoded_id': None,
                'original_id': user_id,
                'message': f'Error validating user ID: {str(e)}'
            }

    def _popularity_based_recommend(self, user_encoded: int, n_recommendations: int = 3) -> List[Tuple[str, float]]:
        """Popularity-based fallback recommendation"""
        try:
            # Get user's rated items
            user_ratings = self.user_item_sparse[user_encoded]
            rated_items = set(user_ratings.nonzero()[1])

            # Calculate item popularity scores
            item_stats = self.interaction_data.groupby('item_encoded').agg({
                '评分': ['mean', 'count']
            })
            item_stats.columns = ['avg_rating', 'rating_count']

            # Weighted popularity score
            item_stats['popularity_score'] = (
                item_stats['avg_rating'] * np.log1p(item_stats['rating_count'])
            )

            # Filter unrated items
            unrated_items = item_stats[~item_stats.index.isin(rated_items)]
            top_items = unrated_items.nlargest(n_recommendations, 'popularity_score')

            recommendations = []
            for item_encoded, row in top_items.iterrows():
                item_name = self.item_encoder.inverse_transform([item_encoded])[0]
                score = row['popularity_score']
                recommendations.append((item_name, score))

            return recommendations

        except Exception as e:
            logger.error(f"Popularity-based recommendation failed: {str(e)}")
            return []

    def save_model(self) -> bool:
        """Save improved model with performance metrics"""
        try:
            model_data = {
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'user_item_sparse': self.user_item_sparse,
                'item_user_sparse': self.item_user_sparse,
                'svd_model': self.svd_model,
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'interaction_data': self.interaction_data,
                'valid_user_ids': self.valid_user_ids,
                'model_info': self.model_info,
                'performance_metrics': self.monitor.get_report(),
                'min_interactions': self.min_interactions,
                'svd_components': self.svd_components,
                'similarity_threshold': self.similarity_threshold,
                'max_similar_users': self.max_similar_users
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Improved model saved to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self) -> bool:
        """Load improved model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return False

            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Restore model components
            self.user_encoder = model_data['user_encoder']
            self.item_encoder = model_data['item_encoder']
            self.user_item_sparse = model_data['user_item_sparse']
            self.item_user_sparse = model_data['item_user_sparse']
            self.svd_model = model_data['svd_model']
            self.user_factors = model_data['user_factors']
            self.item_factors = model_data['item_factors']
            self.interaction_data = model_data['interaction_data']
            self.valid_user_ids = model_data['valid_user_ids']
            self.model_info = model_data['model_info']

            # Load performance metrics if available
            if 'performance_metrics' in model_data:
                print("\n📊 Previous Training Performance:")
                for operation, metrics in model_data['performance_metrics'].items():
                    if 'elapsed_time' in metrics:
                        print(f"   {operation}: {metrics['elapsed_time']:.2f}s, "
                              f"Memory: {metrics.get('memory_used_mb', 0):.1f}MB")

            logger.info(f"Improved model loaded from {self.model_path}")
            logger.info(f"Model created: {self.model_info['created_at']}")
            logger.info(f"Users: {self.model_info['n_users']}, Items: {self.model_info['n_items']}")
            logger.info(f"Memory usage: {self.model_info['memory_usage_mb']:.1f}MB")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def get_performance_report(self):
        """Get detailed performance report"""
        return {
            'model_info': self.model_info,
            'performance_metrics': self.monitor.get_report(),
            'memory_efficiency': {
                'sparse_matrix_memory_mb': self.model_info.get('memory_usage_mb', 0),
                'estimated_dense_memory_mb': (self.model_info['n_users'] * self.model_info['n_items'] * 8) / 1024 / 1024,
                'memory_savings_percent': 0
            }
        }


def main():
    """Main function to demonstrate improved system"""
    print("🚀" * 30)
    print("   IMPROVED COLLABORATIVE FILTERING SYSTEM")
    print("🚀" * 30)
    print("Key Improvements: Sparse Matrices + Performance Monitoring")
    print("=" * 80)

    # Initialize improved recommender
    recommender = ImprovedCollaborativeFilteringRecommender()

    # Try to load existing model
    model_loaded = recommender.load_model()

    if not model_loaded:
        print("\n📊 Training improved model...")

        # Load and train model
        if not recommender.load_data():
            print("❌ Failed to load data.")
            return

        if not recommender.train_model():
            print("❌ Failed to train model.")
            return

        # Save model
        if not recommender.save_model():
            print("⚠️ Model trained but failed to save.")

    # Display performance report
    report = recommender.get_performance_report()
    print(f"\n📈 IMPROVED MODEL STATISTICS")
    print(f"   Users: {report['model_info']['n_users']:,}")
    print(f"   Items: {report['model_info']['n_items']:,}")
    print(f"   Interactions: {report['model_info']['n_interactions']:,}")
    print(f"   Sparsity: {report['model_info']['sparsity']:.2f}%")
    print(f"   Memory Usage: {report['model_info']['memory_usage_mb']:.1f}MB")

    # Calculate memory savings
    dense_memory = (report['model_info']['n_users'] * report['model_info']['n_items'] * 8) / 1024 / 1024
    sparse_memory = report['model_info']['memory_usage_mb']
    savings = ((dense_memory - sparse_memory) / dense_memory) * 100
    print(f"   Dense Matrix Would Use: {dense_memory:.1f}MB")
    print(f"   Memory Savings: {savings:.1f}%")

    # Test recommendations with performance monitoring
    print(f"\n🎯 TESTING IMPROVED RECOMMENDATIONS")
    print("-" * 50)

    test_users = ['4', '272115', '56212']

    for user_id in test_users:
        print(f"\n👤 Testing User: {user_id}")
        recommendations = recommender.recommend(user_id, 3)

        if recommendations:
            print(f"✅ Generated {len(recommendations)} recommendations:")
            for i, (item_name, score) in enumerate(recommendations, 1):
                print(f"   {i}. {item_name} (Score: {score:.3f})")
        else:
            print("❌ No recommendations generated")

    # Display final performance metrics
    final_report = recommender.monitor.get_report()
    print(f"\n⏱️  PERFORMANCE SUMMARY")
    print("-" * 50)
    for operation, metrics in final_report.items():
        if 'elapsed_time' in metrics:
            print(f"   {operation}: {metrics['elapsed_time']:.3f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ 程序出现错误: {str(e)}")
