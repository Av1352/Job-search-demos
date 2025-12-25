import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class HealthcarePricePredictor:
    """ML model for predicting healthcare prices"""
    
    def __init__(self):
        self.weights = None
        self.training_losses = []
        self.feature_names = ['Procedure Type', 'Location', 'Insurance Type', 'Facility Type']
        
        # Mappings
        self.procedures = {
            'MRI - Brain (CPT 70553)': 'mri_brain',
            'CT Scan - Chest (CPT 71260)': 'ct_chest',
            'Knee Arthroscopy (CPT 29881)': 'knee_surgery',
            'Colonoscopy (CPT 45378)': 'colonoscopy',
            'Blood Panel (CPT 80053)': 'blood_panel',
            'X-Ray - Chest (CPT 71046)': 'xray_chest'
        }
        
        self.locations = {
            'Boston, MA': 'boston_ma',
            'Houston, TX': 'houston_tx',
            'San Francisco, CA': 'sf_ca',
            'Miami, FL': 'miami_fl',
            'Chicago, IL': 'chicago_il'
        }
        
        self.insurances = {
            'PPO Insurance': 'ppo',
            'HMO Insurance': 'hmo',
            'High Deductible Plan': 'high_deductible',
            'Medicare': 'medicare',
            'Uninsured (Cash)': 'uninsured'
        }
        
        self.facility_types = ['academic', 'community', 'outpatient', 'premium']
        
        # Base prices and multipliers
        self.base_prices = {
            'mri_brain': 2500, 'ct_chest': 1800, 'knee_surgery': 8500,
            'colonoscopy': 3200, 'blood_panel': 250, 'xray_chest': 350
        }
        
        self.location_multipliers = {
            'boston_ma': 1.25, 'houston_tx': 0.95, 'sf_ca': 1.45,
            'miami_fl': 1.05, 'chicago_il': 1.10
        }
        
        self.insurance_discounts = {
            'ppo': 0.85, 'hmo': 0.80, 'high_deductible': 0.90,
            'medicare': 0.65, 'uninsured': 1.0
        }
        
        self.insurance_coverage = {
            'ppo': 0.70, 'hmo': 0.75, 'high_deductible': 0.50,
            'medicare': 0.80, 'uninsured': 0.0
        }
        
        self.facility_multipliers = {
            'academic': 1.15, 'community': 0.90,
            'outpatient': 0.70, 'premium': 1.35
        }
        
    def generate_training_data(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data"""
        X = []
        y = []
        
        procedures = list(self.base_prices.keys())
        locations = list(self.location_multipliers.keys())
        insurances = list(self.insurance_discounts.keys())
        
        for _ in range(n_samples):
            proc = np.random.choice(procedures)
            loc = np.random.choice(locations)
            ins = np.random.choice(insurances)
            fac = np.random.choice(self.facility_types)
            
            # Encode features
            proc_idx = procedures.index(proc)
            loc_idx = locations.index(loc)
            ins_idx = insurances.index(ins)
            fac_idx = self.facility_types.index(fac)
            
            # Calculate price
            base_price = self.base_prices[proc]
            loc_mult = self.location_multipliers[loc]
            ins_mult = self.insurance_discounts[ins]
            fac_mult = self.facility_multipliers[fac]
            
            noise = 0.85 + np.random.random() * 0.3
            price = base_price * loc_mult * ins_mult * fac_mult * noise
            
            X.append([proc_idx, loc_idx, ins_idx, fac_idx])
            y.append(price)
        
        return np.array(X), np.array(y)
    
    def train(self, learning_rate: float = 0.001, iterations: int = 1000):
        """Train the linear regression model"""
        X, y = self.generate_training_data()
        n, m = X.shape
        
        # Initialize weights (including bias)
        self.weights = np.zeros(m + 1)
        
        # Gradient descent
        for iter_num in range(iterations):
            # Predictions
            predictions = self.weights[0] + X @ self.weights[1:]
            
            # Loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            
            if iter_num % 100 == 0:
                self.training_losses.append({
                    'iteration': iter_num,
                    'loss': float(loss)
                })
            
            # Gradients
            error = predictions - y
            grad_bias = np.mean(error)
            grad_weights = (X.T @ error) / n
            
            # Update weights
            self.weights[0] -= learning_rate * grad_bias
            self.weights[1:] -= learning_rate * grad_weights
        
        # Calculate metrics
        final_predictions = self.weights[0] + X @ self.weights[1:]
        self.r2 = 1 - (np.sum((y - final_predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        self.mae = np.mean(np.abs(y - final_predictions))
        self.rmse = np.sqrt(np.mean((y - final_predictions) ** 2))
        
    def predict(self, procedure: str, location: str, insurance: str, facility_type: str) -> Dict:
        """Make a prediction"""
        procedures = list(self.base_prices.keys())
        locations = list(self.location_multipliers.keys())
        insurances = list(self.insurance_discounts.keys())
        
        # Map inputs
        proc_key = self.procedures[procedure]
        loc_key = self.locations[location]
        ins_key = self.insurances[insurance]
        
        # Encode
        proc_idx = procedures.index(proc_key)
        loc_idx = locations.index(loc_key)
        ins_idx = insurances.index(ins_key)
        fac_idx = self.facility_types.index(facility_type)
        
        features = np.array([proc_idx, loc_idx, ins_idx, fac_idx])
        
        # Predict
        prediction = self.weights[0] + features @ self.weights[1:]
        
        # Calculate contributions
        contributions = [
            {
                'feature': self.feature_names[i],
                'contribution': float(self.weights[i + 1] * features[i]),
                'value': float(features[i])
            }
            for i in range(len(features))
        ]
        
        return {
            'price': max(0, float(prediction)),
            'contributions': contributions
        }
    
    def predict_all_facilities(self, procedure: str, location: str, insurance: str) -> Dict:
        """Predict for all facility types"""
        
        facilities_info = {
            'academic': {
                'name': 'University Medical Center',
                'quality': 4.5,
                'distance': '2.3 mi',
                'wait_time': '3-5 days'
            },
            'community': {
                'name': 'Community Hospital',
                'quality': 4.0,
                'distance': '4.1 mi',
                'wait_time': '1-3 days'
            },
            'outpatient': {
                'name': 'QuickCare Imaging Center',
                'quality': 4.2,
                'distance': '1.8 mi',
                'wait_time': 'Same day'
            },
            'premium': {
                'name': 'Premium Medical Group',
                'quality': 4.8,
                'distance': '5.6 mi',
                'wait_time': '1 week'
            }
        }
        
        ins_key = self.insurances[insurance]
        coverage = self.insurance_coverage[ins_key]
        
        facilities = []
        for fac_type in self.facility_types:
            pred = self.predict(procedure, location, insurance, fac_type)
            info = facilities_info[fac_type]
            
            patient_pays = pred['price'] * (1 - coverage)
            
            facilities.append({
                **info,
                'type': fac_type.capitalize(),
                'price': pred['price'],
                'patient_pays': patient_pays,
                'contributions': pred['contributions']
            })
        
        # Sort by patient cost
        facilities.sort(key=lambda x: x['patient_pays'])
        
        # Calculate savings
        avg_patient_pays = np.mean([f['patient_pays'] for f in facilities])
        best_patient_pays = facilities[0]['patient_pays']
        savings = avg_patient_pays - best_patient_pays
        savings_percent = (savings / avg_patient_pays) * 100
        
        # Feature importance
        total_weight = np.sum(np.abs(self.weights[1:]))
        feature_importance = [
            {
                'feature': self.feature_names[i],
                'importance': float(np.abs(self.weights[i + 1]) / total_weight)
            }
            for i in range(len(self.feature_names))
        ]
        
        return {
            'facilities': facilities,
            'best_facility': facilities[0],
            'savings': savings,
            'savings_percent': savings_percent,
            'model_metrics': {
                'r2': self.r2,
                'mae': self.mae,
                'rmse': self.rmse
            },
            'feature_importance': feature_importance,
            'training_losses': self.training_losses,
            'training_size': 500
        }