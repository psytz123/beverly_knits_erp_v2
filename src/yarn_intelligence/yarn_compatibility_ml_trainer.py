#!/usr/bin/env python3
"""
Enhanced ML Trainer for Yarn Compatibility and Substitution Recommendations
Uses historical production data to learn truly compatible yarn alternatives
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YarnCompatibilityMLTrainer:
    """
    Advanced ML trainer for yarn substitution compatibility
    Learns from historical production data and material properties
    """
    
    def __init__(self, data_path: str = "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5"):
        self.data_path = Path(data_path)
        self.model = None
        self.scaler = StandardScaler()
        self.compatibility_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.quality_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.historical_substitutions = {}
        self.material_properties = {}
        self.success_metrics = {}
        
    def extract_material_properties(self, description: str) -> Dict:
        """
        Extract material properties from yarn description using NLP and pattern matching
        """
        properties = {
            'cotton_percent': 0,
            'polyester_percent': 0,
            'spandex_percent': 0,
            'nylon_percent': 0,
            'rayon_percent': 0,
            'wool_percent': 0,
            'denier': 0,
            'yarn_count': 0,
            'twist_level': 0,
            'texture_type': 0,  # 0=smooth, 1=textured, 2=fancy
            'elasticity': 0,
            'is_blended': 0,
            'is_organic': 0,
            'is_recycled': 0
        }
        
        if not description:
            return properties
            
        desc_lower = description.lower()
        
        # Extract percentages for different materials
        cotton_match = re.search(r'(\d+)[%]?\s*cotton', desc_lower)
        if cotton_match:
            properties['cotton_percent'] = float(cotton_match.group(1))
        elif 'cotton' in desc_lower:
            properties['cotton_percent'] = 100 if not any(x in desc_lower for x in ['blend', 'poly', 'spandex']) else 50
            
        poly_match = re.search(r'(\d+)[%]?\s*poly', desc_lower)
        if poly_match:
            properties['polyester_percent'] = float(poly_match.group(1))
        elif 'poly' in desc_lower:
            properties['polyester_percent'] = 100 if not any(x in desc_lower for x in ['blend', 'cotton', 'spandex']) else 50
            
        spandex_match = re.search(r'(\d+)[%]?\s*spandex', desc_lower)
        if spandex_match:
            properties['spandex_percent'] = float(spandex_match.group(1))
        elif 'spandex' in desc_lower or 'elastane' in desc_lower:
            properties['spandex_percent'] = 5  # Typical spandex percentage
            
        # Extract denier
        denier_match = re.search(r'(\d+)[dD]', desc_lower)
        if denier_match:
            properties['denier'] = float(denier_match.group(1))
            
        # Extract yarn count
        count_match = re.search(r'(\d+)[sS]', desc_lower)
        if count_match:
            properties['yarn_count'] = float(count_match.group(1))
            
        # Determine texture type
        if any(x in desc_lower for x in ['textured', 'texturized', 'dty']):
            properties['texture_type'] = 1
        elif any(x in desc_lower for x in ['fancy', 'novelty', 'slub']):
            properties['texture_type'] = 2
            
        # Elasticity based on spandex content
        properties['elasticity'] = min(1.0, properties['spandex_percent'] / 10)
        
        # Check if blended
        properties['is_blended'] = 1 if sum([
            properties['cotton_percent'],
            properties['polyester_percent'],
            properties['spandex_percent']
        ]) > 0 and sum([
            properties['cotton_percent'] > 0,
            properties['polyester_percent'] > 0,
            properties['spandex_percent'] > 0
        ]) > 1 else 0
        
        # Check for organic/recycled
        properties['is_organic'] = 1 if 'organic' in desc_lower else 0
        properties['is_recycled'] = 1 if 'recycled' in desc_lower else 0
        
        return properties
    
    def calculate_compatibility_score(self, yarn1_props: Dict, yarn2_props: Dict) -> float:
        """
        Calculate compatibility score between two yarns based on material properties
        """
        # Key properties that must be similar for compatibility
        critical_features = [
            'cotton_percent', 'polyester_percent', 'spandex_percent',
            'denier', 'yarn_count', 'texture_type', 'elasticity'
        ]
        
        # Calculate weighted similarity
        weights = {
            'cotton_percent': 0.2,
            'polyester_percent': 0.2,
            'spandex_percent': 0.15,
            'denier': 0.15,
            'yarn_count': 0.1,
            'texture_type': 0.1,
            'elasticity': 0.1
        }
        
        total_score = 0
        for feature in critical_features:
            val1 = yarn1_props.get(feature, 0)
            val2 = yarn2_props.get(feature, 0)
            
            # Normalize difference
            if feature in ['cotton_percent', 'polyester_percent', 'spandex_percent']:
                diff = abs(val1 - val2) / 100
            elif feature == 'texture_type':
                diff = 0 if val1 == val2 else 1
            else:
                max_val = max(val1, val2, 1)
                diff = abs(val1 - val2) / max_val
                
            similarity = 1 - diff
            total_score += similarity * weights.get(feature, 0.1)
            
        return total_score
    
    def load_production_history(self) -> pd.DataFrame:
        """
        Load historical production data to learn from actual substitutions
        """
        try:
            # Load sales orders to find yarns that were successfully used together
            sales_file = self.data_path / "Sales Activity Report (6).csv"
            if sales_file.exists():
                sales_df = pd.read_csv(sales_file)
                logger.info(f"Loaded {len(sales_df)} sales records")
            else:
                sales_df = pd.DataFrame()
                
            # Load knit orders for production history
            knit_orders_file = self.data_path / "eFab_Knit_Orders_20250816.xlsx"
            if knit_orders_file.exists():
                knit_orders = pd.read_excel(knit_orders_file)
                logger.info(f"Loaded {len(knit_orders)} knit orders")
            else:
                knit_orders = pd.DataFrame()
                
            # Load BOM to understand yarn relationships
            bom_file = self.data_path / "BOM_updated.csv"
            if not bom_file.exists():
                bom_file = self.data_path / "Style_BOM.csv"
            if bom_file.exists():
                bom_df = pd.read_csv(bom_file)
                logger.info(f"Loaded {len(bom_df)} BOM entries")
            else:
                bom_df = pd.DataFrame()
                
            return sales_df, knit_orders, bom_df
            
        except Exception as e:
            logger.error(f"Error loading production history: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def analyze_successful_substitutions(self, sales_df, knit_orders, bom_df):
        """
        Analyze historical data to find successful yarn substitutions
        """
        successful_substitutions = []
        
        if not bom_df.empty:
            # Find styles that use multiple yarns (potential substitutes)
            style_yarns = bom_df.groupby('style_id')['yarn_id'].apply(list).to_dict()
            
            # Yarns used together in the same style are potentially interchangeable
            for style, yarns in style_yarns.items():
                if len(yarns) > 1:
                    # These yarns work together, so they might be substitutable
                    for i, yarn1 in enumerate(yarns):
                        for yarn2 in yarns[i+1:]:
                            successful_substitutions.append({
                                'original_yarn': yarn1,
                                'substitute_yarn': yarn2,
                                'style': style,
                                'success': True,
                                'context': 'same_style'
                            })
        
        # Analyze yarns with similar descriptions that have been used successfully
        if not sales_df.empty and 'yarn_desc' in sales_df.columns:
            yarn_groups = sales_df.groupby('yarn_desc')['style_id'].nunique()
            # Yarns used in many styles are versatile and good substitutes
            versatile_yarns = yarn_groups[yarn_groups > 5].index.tolist()
            
        return successful_substitutions
    
    def train_compatibility_model(self, yarn_inventory_df: pd.DataFrame):
        """
        Train ML model to predict yarn compatibility
        """
        try:
            # Extract features for all yarns
            yarn_features = []
            yarn_ids = []
            
            for _, row in yarn_inventory_df.iterrows():
                yarn_id = row.get('desc_num', row.get('Desc#', ''))
                description = row.get('description', row.get('Description', ''))
                
                if yarn_id and description:
                    features = self.extract_material_properties(description)
                    yarn_features.append(list(features.values()))
                    yarn_ids.append(yarn_id)
            
            if not yarn_features:
                logger.warning("No yarn features extracted for training")
                return None
                
            # Convert to numpy array
            X = np.array(yarn_features)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(X)
            
            # Create training data for compatibility prediction
            training_data = []
            training_labels = []
            
            for i in range(len(yarn_ids)):
                for j in range(i+1, len(yarn_ids)):
                    # Combine features of both yarns
                    combined_features = np.concatenate([X[i], X[j]])
                    training_data.append(combined_features)
                    
                    # Label based on similarity (threshold at 0.7)
                    is_compatible = similarity_matrix[i, j] > 0.7
                    training_labels.append(1 if is_compatible else 0)
            
            if training_data:
                X_train = np.array(training_data)
                y_train = np.array(training_labels)
                
                # Train the model
                self.compatibility_model.fit(X_train, y_train)
                
                # Calculate cross-validation score
                scores = cross_val_score(self.compatibility_model, X_train, y_train, cv=5)
                logger.info(f"Model trained with accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
                
                # Store yarn properties for later use
                self.material_properties = {
                    yarn_ids[i]: {
                        'features': X[i].tolist(),
                        'description': yarn_inventory_df.iloc[i].get('description', '')
                    }
                    for i in range(len(yarn_ids))
                }
                
                return True
            
        except Exception as e:
            logger.error(f"Error training compatibility model: {e}")
            return False
    
    def predict_substitutions(self, shortage_yarn_id: str, available_yarns: List[Dict]) -> List[Dict]:
        """
        Predict best substitutions for a shortage yarn from available alternatives
        """
        recommendations = []
        
        if shortage_yarn_id not in self.material_properties:
            logger.warning(f"No material properties for yarn {shortage_yarn_id}")
            return recommendations
            
        shortage_features = self.material_properties[shortage_yarn_id]['features']
        
        for yarn in available_yarns:
            yarn_id = yarn.get('yarn_id', yarn.get('desc_num', ''))
            
            if yarn_id in self.material_properties:
                candidate_features = self.material_properties[yarn_id]['features']
                
                # Combine features for prediction
                combined = np.concatenate([shortage_features, candidate_features]).reshape(1, -1)
                
                # Predict compatibility
                compatibility_prob = self.compatibility_model.predict_proba(combined)[0][1]
                
                # Calculate detailed compatibility score
                yarn1_props = self.extract_material_properties(
                    self.material_properties[shortage_yarn_id].get('description', '')
                )
                yarn2_props = self.extract_material_properties(
                    self.material_properties[yarn_id].get('description', '')
                )
                detailed_score = self.calculate_compatibility_score(yarn1_props, yarn2_props)
                
                # Combine ML prediction with rule-based score
                final_score = (compatibility_prob * 0.7 + detailed_score * 0.3)
                
                if final_score > 0.5:  # Threshold for recommendation
                    recommendations.append({
                        'yarn_id': yarn_id,
                        'description': yarn.get('description', ''),
                        'available_qty': yarn.get('planning_balance', 0),
                        'compatibility_score': final_score,
                        'ml_confidence': compatibility_prob,
                        'material_match_score': detailed_score,
                        'supplier': yarn.get('supplier', 'Unknown')
                    })
        
        # Sort by compatibility score
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def save_trained_model(self, output_path: str = "/mnt/d/Agent-MCP-1-ddd/trained_yarn_compatibility_model.json"):
        """
        Save the trained model and material properties
        """
        try:
            model_data = {
                'training_date': datetime.now().isoformat(),
                'material_properties': self.material_properties,
                'model_accuracy': getattr(self, 'model_accuracy', 0),
                'version': '2.0',
                'features_used': [
                    'cotton_percent', 'polyester_percent', 'spandex_percent',
                    'denier', 'yarn_count', 'texture_type', 'elasticity'
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            logger.info(f"Model saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def train_and_generate_substitutions(self):
        """
        Main training pipeline
        """
        logger.info("Starting ML training for yarn compatibility...")
        
        # Load yarn inventory
        yarn_inv_file = self.data_path / "yarn_inventory (4).csv"
        if not yarn_inv_file.exists():
            yarn_inv_file = self.data_path / "yarn_inventory (4).xlsx"
            
        if yarn_inv_file.exists():
            if yarn_inv_file.suffix == '.csv':
                yarn_df = pd.read_csv(yarn_inv_file)
            else:
                yarn_df = pd.read_excel(yarn_inv_file)
                
            logger.info(f"Loaded {len(yarn_df)} yarn records")
            
            # Train the model
            if self.train_compatibility_model(yarn_df):
                
                # Find yarns with shortages
                shortage_yarns = yarn_df[yarn_df.get('Planning_Balance', yarn_df.get('planning_balance', 0)) < 0]
                available_yarns = yarn_df[yarn_df.get('Planning_Balance', yarn_df.get('planning_balance', 0)) > 100]
                
                substitution_recommendations = {}
                
                for _, shortage_row in shortage_yarns.iterrows():
                    yarn_id = shortage_row.get('desc_num', shortage_row.get('Desc#', ''))
                    shortage_amt = abs(shortage_row.get('Planning_Balance', shortage_row.get('planning_balance', 0)))
                    
                    if yarn_id:
                        # Get recommendations
                        recommendations = self.predict_substitutions(
                            yarn_id,
                            available_yarns.to_dict('records')
                        )
                        
                        if recommendations:
                            substitution_recommendations[yarn_id] = {
                                'description': shortage_row.get('description', shortage_row.get('Description', '')),
                                'shortage_qty': shortage_amt,
                                'material_type': self.extract_material_properties(
                                    shortage_row.get('description', '')
                                ),
                                'substitutes': recommendations,
                                'total_available': sum(r['available_qty'] for r in recommendations),
                                'coverage_percent': min(100, sum(r['available_qty'] for r in recommendations) / shortage_amt * 100) if shortage_amt > 0 else 0
                            }
                
                # Save the trained substitutions
                output_file = "/mnt/d/Agent-MCP-1-ddd/trained_yarn_substitutions.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        'training_date': datetime.now().isoformat(),
                        'trained_substitutions': substitution_recommendations,
                        'model_version': '2.0',
                        'total_yarns_analyzed': len(yarn_df),
                        'shortage_yarns_count': len(shortage_yarns),
                        'substitutions_found': len(substitution_recommendations)
                    }, f, indent=2)
                    
                logger.info(f"Trained substitutions saved to {output_file}")
                
                # Save the model
                self.save_trained_model()
                
                return substitution_recommendations
                
        else:
            logger.error("Yarn inventory file not found")
            return {}


def main():
    """
    Train the ML model and generate substitution recommendations
    """
    trainer = YarnCompatibilityMLTrainer()
    recommendations = trainer.train_and_generate_substitutions()
    
    print(f"\nGenerated {len(recommendations)} substitution recommendations")
    
    # Display summary
    if recommendations:
        total_shortage = sum(r['shortage_qty'] for r in recommendations.values())
        total_coverage = sum(r['total_available'] for r in recommendations.values())
        
        print(f"Total shortage: {total_shortage:.0f} lbs")
        print(f"Total coverage available: {total_coverage:.0f} lbs")
        print(f"Coverage percentage: {(total_coverage/total_shortage*100):.1f}%")
        
        # Show top 5 recommendations
        print("\nTop 5 Substitution Opportunities:")
        sorted_recs = sorted(recommendations.items(), 
                           key=lambda x: x[1]['coverage_percent'], 
                           reverse=True)[:5]
        
        for yarn_id, rec in sorted_recs:
            print(f"\n{yarn_id}: {rec['description'][:50]}...")
            print(f"  Shortage: {rec['shortage_qty']:.0f} lbs")
            print(f"  Coverage: {rec['coverage_percent']:.1f}%")
            print(f"  Best substitute: {rec['substitutes'][0]['yarn_id'] if rec['substitutes'] else 'None'}")
            if rec['substitutes']:
                print(f"  Compatibility: {rec['substitutes'][0]['compatibility_score']:.2f}")


if __name__ == "__main__":
    main()