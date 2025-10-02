# ================================================================
# Telco Package Recommendation Demo - Synthetic Data + Model + Scoring
# Author: you
# Purpose: Generate data, train binary model, score test + all packages
# ================================================================

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# ----------------------------
# 1) Synthetic data generator
# ----------------------------
class TelcoSyntheticDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        
        # --- Telco package catalog ---
        self.packages = {
            'BASIC_10': {'price': 425, 'data_gb': 5,  'voice': 1000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 0},
            'BASIC_20': {'price': 565, 'data_gb': 10, 'voice': 1000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 0},
            'BASIC_30': {'price': 665, 'data_gb': 15, 'voice': 1000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 0},
            'BASIC_40': {'price': 725, 'data_gb': 20, 'voice': 750,  'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 0},
            'BASIC_60': {'price': 880, 'data_gb': 30, 'voice': 1000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 0},
            'BASIC_UNLIMITED': {'price': 940, 'data_gb': 50, 'voice': 1000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 0},
            'PREMIUM_25': {'price': 740, 'data_gb': 25, 'voice': 1000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'PREMIUM_30': {'price': 805, 'data_gb': 30, 'voice': 3000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'PREMIUM_40': {'price': 865, 'data_gb': 40, 'voice': 3000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'PREMIUM_SOCIAL': {'price': 980, 'data_gb': 50, 'voice': 3000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 0},
            'PREMIUM_STREAMING_30': {'price': 700, 'data_gb': 30, 'voice': 3000, 'sms': 250, 'has_social': 0, 'has_youtube': 1, 'has_netflix': 1},
            'PREMIUM_STREAMING_40': {'price': 800, 'data_gb': 40, 'voice': 3000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 1},
            'PREMIUM_STREAMING_60': {'price': 900, 'data_gb': 60, 'voice': 3000, 'sms': 250, 'has_social': 1, 'has_youtube': 1, 'has_netflix': 1},
            'ECONOMY_1':  {'price': 350, 'data_gb': 1,  'voice': 1000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'ECONOMY_3':  {'price': 385, 'data_gb': 3,  'voice': 1000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'ECONOMY_6':  {'price': 420, 'data_gb': 6,  'voice': 1000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'ECONOMY_10': {'price': 470, 'data_gb': 10, 'voice': 1000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'ECONOMY_15': {'price': 565, 'data_gb': 15, 'voice': 1000, 'sms': 250, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'STARTER_2':  {'price': 185, 'data_gb': 2,  'voice': 500,  'sms': 500, 'has_social': 0, 'has_youtube': 0, 'has_netflix': 0},
            'STARTER_10': {'price': 235, 'data_gb': 10, 'voice': 500,  'sms': 500, 'has_social': 1, 'has_youtube': 0, 'has_netflix': 0}
        }
        
        # --- Microsegments (behavioral + economic) ---
        self.microsegments = {
            'MS_001': {'profile': 'Young_Social_Heavy',        'age_range': (18, 26), 'income_range': (6000, 15000), 'data_heavy': True,  'price_sensitive': True},
            'MS_002': {'profile': 'Young_Moderate_Usage',      'age_range': (18, 26), 'income_range': (5000, 12000), 'data_heavy': False, 'price_sensitive': True},
            'MS_003': {'profile': 'Young_Professional',        'age_range': (25, 35), 'income_range': (12000, 25000),'data_heavy': True,  'price_sensitive': False},
            'MS_004': {'profile': 'Adult_Family_Economic',  'age_range': (30, 45), 'income_range': (8000, 18000), 'data_heavy': False, 'price_sensitive': True},
            'MS_005': {'profile': 'Adult_Premium',        'age_range': (35, 50), 'income_range': (20000, 50000),'data_heavy': True,  'price_sensitive': False},
            'MS_006': {'profile': 'Middle_Age_Work_Focused',      'age_range': (40, 55), 'income_range': (15000, 35000),'data_heavy': False, 'price_sensitive': False},
            'MS_007': {'profile': 'Senior_Basic',             'age_range': (55, 70), 'income_range': (5000, 15000), 'data_heavy': False, 'price_sensitive': True},
            'MS_008': {'profile': 'Child_Parent_Control',   'age_range': (8, 15),  'income_range': (0, 0),        'data_heavy': False, 'price_sensitive': True},
            'MS_009': {'profile': 'Student',                 'age_range': (18, 25), 'income_range': (3000, 8000),  'data_heavy': True,  'price_sensitive': True},
            'MS_010': {'profile': 'Business_Professional',    'age_range': (28, 45), 'income_range': (18000, 40000),'data_heavy': True,  'price_sensitive': False}
        }

    # --- Helpers to build a customer snapshot ---
    def get_suitable_current_packages(self, microsegment_id):
        ms_profile = self.microsegments[microsegment_id]['profile']
        if 'Young_Social' in ms_profile:         return ['BASIC_20', 'BASIC_30', 'BASIC_40', 'BASIC_60']
        elif 'Young_Moderate' in ms_profile:         return ['BASIC_10', 'BASIC_20', 'ECONOMY_6', 'ECONOMY_10']
        elif 'Young_Professional' in ms_profile:  return ['BASIC_40', 'BASIC_60', 'PREMIUM_25', 'PREMIUM_30']
        elif 'Adult_Family_Economic' in ms_profile: return ['ECONOMY_6', 'ECONOMY_10', 'ECONOMY_15', 'BASIC_20']
        elif 'Adult_Premium' in ms_profile:  return ['PREMIUM_30', 'PREMIUM_40', 'PREMIUM_SOCIAL', 'PREMIUM_STREAMING_40']
        elif 'Middle_Age_Work' in ms_profile:       return ['PREMIUM_25', 'PREMIUM_30', 'BASIC_30', 'BASIC_40']
        elif 'Senior_Basic' in ms_profile:       return ['ECONOMY_1', 'ECONOMY_3', 'ECONOMY_6', 'BASIC_10']
        elif 'Child' in ms_profile:             return ['STARTER_2', 'STARTER_10']
        elif 'Student' in ms_profile:           return ['ECONOMY_3', 'ECONOMY_6', 'BASIC_10', 'BASIC_20']
        else:                                   return ['PREMIUM_30', 'PREMIUM_40', 'PREMIUM_SOCIAL', 'BASIC_60']

    def generate_customer_profile(self, microsegment_id):
        ms = self.microsegments[microsegment_id]
        age = np.random.randint(ms['age_range'][0], ms['age_range'][1] + 1)
        income = 0 if ms['profile']=='Child_Parent_Control' else np.random.randint(ms['income_range'][0], ms['income_range'][1] + 1)
        current_pkg = np.random.choice(self.get_suitable_current_packages(microsegment_id))
        cur = self.packages[current_pkg]

        if ms['data_heavy']:
            data_usage_ratio = np.random.uniform(0.8, 1.5)
            data_overage_freq = np.random.uniform(0.2, 0.8)
        else:
            data_usage_ratio = np.random.uniform(0.3, 0.8)
            data_overage_freq = np.random.uniform(0.0, 0.3)
        voice_usage_ratio = np.random.uniform(0.3, 0.9)

        if ms['price_sensitive']:
            price_sens = np.random.uniform(0.6, 0.9)
            pay_punct  = np.random.uniform(0.7, 0.95)
        else:
            price_sens = np.random.uniform(0.2, 0.6)
            pay_punct  = np.random.uniform(0.85, 1.0)

        return {
            'age': age,
            'income': income,
            'current_package': current_pkg,
            'current_monthly_fee': cur['price'],
            'current_data_gb': cur['data_gb'],
            'current_voice_minutes': cur['voice'],
            'current_contract_remaining_months': np.random.randint(0, 13),
            'months_on_current_package': np.random.randint(1, 25),
            'tenure_total_months': np.random.randint(6, 60),
            'avg_monthly_data_usage': cur['data_gb'] * data_usage_ratio,
            'avg_monthly_voice_usage': cur['voice'] * voice_usage_ratio,
            'data_overage_frequency': data_overage_freq,
            'payment_punctuality_score': pay_punct,
            'customer_service_contacts_6m': np.random.poisson(2),
            'complaint_history_score': np.random.uniform(0.0, 0.5),
            'churn_risk_score': np.random.uniform(0.1, 0.8),
            'price_sensitivity_score': price_sens,  # renamed for clarity
            'tech_adoption_level': np.random.uniform(0.3, 0.95),
            'family_size': np.random.randint(1, 5),
            'location_type': np.random.choice(['urban', 'suburban', 'rural'], p=[0.6, 0.3, 0.1]),
            'social_media_hours_daily': np.random.uniform(1, 8),
            'video_streaming_freq': np.random.uniform(0.2, 0.9),
            'music_streaming': np.random.choice([0, 1], p=[0.3, 0.7]),
            'gaming_mobile': np.random.uniform(0.0, 0.8),
            'work_related_calls': np.random.uniform(0.2, 0.8),
            'international_calls': np.random.uniform(0.0, 0.3),
            'peak_hour_pattern': np.random.choice(['morning', 'afternoon', 'evening', 'mixed']),
            'weekend_vs_weekday_ratio': np.random.uniform(0.8, 1.5)
        }

    # Business-rule probability (for ground truth generation only)
    def calculate_acceptance_probability(self, customer_profile, offered_package, microsegment_id):
        offered = self.packages[offered_package]
        ms = self.microsegments[microsegment_id]
        base_prob = 0.6 if not ms['price_sensitive'] else 0.3

        # Price effect (ratio vs current)
        price_ratio = offered['price'] / customer_profile['current_monthly_fee']
        if price_ratio > 2.0:   base_prob *= 0.4
        elif price_ratio > 1.5: base_prob *= 0.6
        elif price_ratio > 1.2: base_prob *= 0.8
        elif price_ratio < 0.8: base_prob *= 1.3

        # Data need
        if customer_profile['data_overage_frequency'] > 0.5:
            data_ratio = offered['data_gb'] / max(customer_profile['current_data_gb'], 1)
            if data_ratio > 2.0:   base_prob *= 1.5
            elif data_ratio > 1.5: base_prob *= 1.2

        # Content alignment
        if offered['has_netflix'] and customer_profile['video_streaming_freq'] > 0.7: base_prob *= 1.3
        if offered['has_social']  and customer_profile['social_media_hours_daily'] > 4: base_prob *= 1.2

        # Contract timing
        if customer_profile['current_contract_remaining_months'] <= 2: base_prob *= 1.4
        elif customer_profile['current_contract_remaining_months'] > 6: base_prob *= 0.7

        # Risk & payment
        if customer_profile['churn_risk_score'] > 0.7: base_prob *= 0.6
        elif customer_profile['churn_risk_score'] < 0.3: base_prob *= 1.1
        if customer_profile['payment_punctuality_score'] < 0.8: base_prob *= 0.7

        return float(min(max(base_prob, 0.02), 0.95))

    # Build dataset: each row = (customer, offered_package) with binary target
    def generate_dataset(self, n_samples=5000, train_ratio=0.8):
        print(f"Generating {n_samples} synthetic samples...")
        rows = []
        for i in range(n_samples):
            if i % 1000 == 0: print(f"  progress: {i}/{n_samples}")
            ms_id = random.choice(list(self.microsegments.keys()))
            cust = self.generate_customer_profile(ms_id)
            cust_id = f"CUST_{i+10000:06d}"
            offered_pkg = random.choice([k for k in self.packages.keys() if k != cust['current_package']])
            offered = self.packages[offered_pkg]

            prob = self.calculate_acceptance_probability(cust, offered_pkg, ms_id)
            accepted = 1 if random.random() < prob else 0

            price_diff = offered['price'] - cust['current_monthly_fee']
            data_ratio = offered['data_gb'] / max(cust['current_data_gb'], 1)
            voice_ratio = offered['voice'] / max(cust['current_voice_minutes'], 1)

            rows.append({
                'customer_id': cust_id,
                'microsegment_id': ms_id,
                'offer_date': datetime.now() - timedelta(days=random.randint(1, 365)),
                'offered_package_id': offered_pkg,
                'accepted_offer': accepted,

                # current state
                'current_package_id': cust['current_package'],
                'current_monthly_fee': cust['current_monthly_fee'],
                'current_data_gb': cust['current_data_gb'],
                'current_voice_minutes': cust['current_voice_minutes'],
                'current_contract_remaining_mths': cust['current_contract_remaining_months'],
                'months_on_current_package': cust['months_on_current_package'],

                # customer profile
                'cust_age': cust['age'],
                'cust_income_estimated': cust['income'],
                'cust_tenure_total_months': cust['tenure_total_months'],
                'cust_avg_monthly_data_usage': cust['avg_monthly_data_usage'],
                'cust_avg_monthly_voice_usage': cust['avg_monthly_voice_usage'],
                'cust_data_overage_frequency': cust['data_overage_frequency'],
                'cust_payment_punct_score': cust['payment_punctuality_score'],
                'cust_cs_contacts_6m': cust['customer_service_contacts_6m'],
                'cust_complaint_score': cust['complaint_history_score'],
                'cust_churn_risk_score': cust['churn_risk_score'],
                'cust_price_sens_score': cust['price_sensitivity_score'],
                'cust_tech_adopt_level': cust['tech_adoption_level'],
                'cust_family_size': cust['family_size'],
                'cust_location_type': cust['location_type'],

                # usage behavior
                'usage_social_heavy': 1 if cust['social_media_hours_daily'] > 4 else 0,
                'usage_video_stream_freq': cust['video_streaming_freq'],
                'usage_music_streaming': cust['music_streaming'],
                'usage_gaming_mobile': cust['gaming_mobile'],
                'usage_work_related_calls': cust['work_related_calls'],
                'usage_international_calls': cust['international_calls'],
                'usage_peak_hour_pattern': cust['peak_hour_pattern'],
                'usage_weekend_weekday_ratio': cust['weekend_vs_weekday_ratio'],

                # offered package features
                'offer_package_price': offered['price'],
                'offer_data_gb': offered['data_gb'],
                'offer_voice_minutes': offered['voice'],
                'offer_contract_months': 12,
                'offer_has_netflix': offered['has_netflix'],
                'offer_has_unlim_social': offered['has_social'],
                'offer_has_youtube_prem': offered['has_youtube'],
                'offer_special_content_cnt': offered['has_social'] + offered['has_youtube'] + offered['has_netflix'],

                # deltas / context
                'price_increase_amount': price_diff,
                'price_increase_pct': price_diff / cust['current_monthly_fee'],
                'data_inc_ratio': data_ratio,
                'voice_inc_ratio': voice_ratio,
                'upgrade_magnitude': (data_ratio + voice_ratio) / 2.0,
                'contract_ext_mths': 12 - cust['current_contract_remaining_months'],
                'offer_timing_contract_renew': 1 if cust['current_contract_remaining_months'] <= 2 else 0,
                'offer_seasonal_camp': np.random.choice([0, 1], p=[0.7, 0.3]),
                'offer_retention_camp': 1 if cust['churn_risk_score'] > 0.6 else 0,
                'offer_competitor_resp': np.random.choice([0, 1], p=[0.8, 0.2]),
                'offer_channel': np.random.choice(['call_center', 'sms', 'app', 'email'], p=[0.4, 0.3, 0.2, 0.1]),
                'offer_personalization_lvl': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.4, 0.3])
            })

        df = pd.DataFrame(rows)

        # --- microsegment aggregates (vectorized) ---
        ms_grp = df.groupby('microsegment_id').agg(
            ms_avg_age=('cust_age','mean'),
            ms_avg_income=('cust_income_estimated','mean'),
            ms_avg_tenure=('cust_tenure_total_months','mean'),
            ms_avg_acceptance_rate=('accepted_offer','mean'),
            ms_price_sensitivity_avg=('cust_price_sens_score','mean'),
            ms_upgrade_propensity=('upgrade_magnitude','mean'),
            ms_churn_rate_12m=('cust_churn_risk_score','mean')
        ).reset_index()

        # conditional rates with safe defaults
        ms_netflix = (df[df['offer_has_netflix']==1]
                      .groupby('microsegment_id')['accepted_offer']
                      .mean().rename('ms_netflix_adoption_rate'))
        ms_renewal = (df[df['offer_timing_contract_renew']==1]
                      .groupby('microsegment_id')['accepted_offer']
                      .mean().rename('ms_contract_renewal_rate'))

        ms_grp = (ms_grp.merge(ms_netflix, on='microsegment_id', how='left')
                        .merge(ms_renewal, on='microsegment_id', how='left'))
        ms_grp['ms_netflix_adoption_rate'] = ms_grp['ms_netflix_adoption_rate'].fillna(0.5)
        ms_grp['ms_contract_renewal_rate'] = ms_grp['ms_contract_renewal_rate'].fillna(0.7)
        ms_grp['ms_avg_package_satisfaction'] = 4.0

        final_df = df.merge(ms_grp, on='microsegment_id', how='left')

        # --- split (stratify by target only to keep 1D) ---
        train_df, test_df = train_test_split(
            final_df, test_size=(1-train_ratio),
            random_state=self.seed,
            stratify=final_df['accepted_offer']
        )

        # persist base files
        train_df.to_csv('telco_training_data.csv', index=False)
        test_df.to_csv('telco_test_data.csv', index=False)
        final_df.to_csv('telco_full_data.csv', index=False)

        print(f"Done. Total={len(final_df)}, Train={len(train_df)}, Test={len(test_df)}, AcceptanceRate={final_df['accepted_offer'].mean():.3f}")
        return train_df, test_df, final_df

    # Build an all-packages grid for a set of customers (for ranking demo)
    def build_scoring_grid_for_customers(self, customers_df):
        # Get available columns that exist in the dataframe
        available_cols = customers_df.columns.tolist()
        cust_keys = ['customer_id','microsegment_id','current_package_id','current_monthly_fee',
                     'current_data_gb','current_voice_minutes','current_contract_remaining_mths',
                     'months_on_current_package','cust_age','cust_income_estimated',
                     'cust_tenure_total_months','cust_avg_monthly_data_usage',
                     'cust_avg_monthly_voice_usage','cust_data_overage_frequency',
                     'cust_payment_punct_score','cust_churn_risk_score']
        
        # Add optional columns if they exist
        optional_cols = ['social_media_hours_daily','video_streaming_freq']
        for col in optional_cols:
            if col in available_cols:
                cust_keys.append(col)
        
        base = (customers_df
                .sort_values('offer_date')
                .drop_duplicates('customer_id', keep='last')[cust_keys])

        all_rows = []
        for _, r in base.iterrows():
            customer_profile = {
                'age': r.cust_age,
                'income': r.cust_income_estimated,
                'current_package': r.current_package_id,
                'current_monthly_fee': r.current_monthly_fee,
                'current_data_gb': r.current_data_gb,
                'current_voice_minutes': r.current_voice_minutes,
                'current_contract_remaining_months': r.current_contract_remaining_mths,
                'months_on_current_package': r.months_on_current_package,
                'tenure_total_months': r.cust_tenure_total_months,
                'avg_monthly_data_usage': r.cust_avg_monthly_data_usage,
                'avg_monthly_voice_usage': r.cust_avg_monthly_voice_usage,
                'data_overage_frequency': r.cust_data_overage_frequency,
                'payment_punctuality_score': r.cust_payment_punct_score,
                'customer_service_contacts_6m': 0.0,
                'complaint_history_score': 0.0,
                'churn_risk_score': r.cust_churn_risk_score,
                'price_sensitivity_score': 0.5,
                'tech_adoption_level': 0.5,
                'family_size': 1,
                'location_type': 'urban',
                'social_media_hours_daily': getattr(r, 'social_media_hours_daily', 4.0),
                'video_streaming_freq': getattr(r, 'video_streaming_freq', 0.5),
                'music_streaming': 0,
                'gaming_mobile': 0.0,
                'work_related_calls': 0.0,
                'international_calls': 0.0,
                'peak_hour_pattern': 'evening',
                'weekend_vs_weekday_ratio': 1.0
            }
            ms_id = r.microsegment_id
            for pkg_id, pkg in self.packages.items():
                prob = self.calculate_acceptance_probability(customer_profile, pkg_id, ms_id)
                all_rows.append({
                    'customer_id': r.customer_id,
                    'microsegment_id': ms_id,
                    'offered_package_id': pkg_id,
                    # package features (same columns as training 'offered' fields)
                    'offer_package_price': pkg['price'],
                    'offer_data_gb': pkg['data_gb'],
                    'offer_voice_minutes': pkg['voice'],
                    'offer_contract_months': 12,
                    'offer_has_netflix': pkg['has_netflix'],
                    'offer_has_unlim_social': pkg['has_social'],
                    'offer_has_youtube_prem': pkg['has_youtube'],
                    'offer_special_content_cnt': pkg['has_social'] + pkg['has_youtube'] + pkg['has_netflix'],
                    # context-like fields used by model
                    'current_monthly_fee': customer_profile['current_monthly_fee'],
                    'current_data_gb': customer_profile['current_data_gb'],
                    'current_voice_minutes': customer_profile['current_voice_minutes'],
                    'price_increase_amount': pkg['price'] - customer_profile['current_monthly_fee'],
                    'price_increase_pct': (pkg['price'] - customer_profile['current_monthly_fee']) / max(customer_profile['current_monthly_fee'],1),
                    'data_inc_ratio': pkg['data_gb'] / max(customer_profile['current_data_gb'],1),
                    'voice_inc_ratio': pkg['voice'] / max(customer_profile['current_voice_minutes'],1),
                    'upgrade_magnitude': (pkg['data_gb'] / max(customer_profile['current_data_gb'],1) + 
                                                pkg['voice'] / max(customer_profile['current_voice_minutes'],1)) / 2.0,
                    'offer_timing_contract_renew': 1 if customer_profile['current_contract_remaining_months'] <= 2 else 0,
                    # minimal customer features to let model generalize
                    'cust_age': customer_profile['age'],
                    'cust_income_estimated': customer_profile['income'],
                    'cust_tenure_total_months': customer_profile['tenure_total_months'],
                    'cust_data_overage_frequency': customer_profile['data_overage_frequency'],
                    'cust_churn_risk_score': customer_profile['churn_risk_score'],
                    'usage_social_heavy': 1 if customer_profile['social_media_hours_daily'] > 4 else 0,
                    'usage_video_stream_freq': customer_profile['video_streaming_freq'],
                    # target is unknown in scoring grid; will be predicted
                })
        grid_df = pd.DataFrame(all_rows)
        return grid_df

# ----------------------------
# 2) Train model & score
# ----------------------------
def build_training_pipeline(train_df):
    """Builds a simple, robust classification pipeline (OneHot + Logistic)."""
    target_col = 'accepted_offer'

    # Columns to drop from inputs (pure IDs or leakage)
    drop_cols = {
        target_col,
        'offer_date',
        'customer_id'
    }

    X = train_df.drop(columns=list(drop_cols))
    y = train_df[target_col].astype(int)

    # Identify categorical vs numeric by dtype
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encode', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols),
            ('num', SimpleImputer(strategy='median'), num_cols)
        ],
        remainder='drop'
    )

    clf = LogisticRegression(max_iter=3000, n_jobs=None, solver='lbfgs')

    pipe = Pipeline(steps=[
        ('prep', pre),
        ('model', clf)
    ])
    pipe.fit(X, y)

    # quick train AUC
    try:
        train_pred = pipe.predict_proba(X)[:,1]
        print(f"Train AUC: {roc_auc_score(y, train_pred):.3f}")
    except Exception:
        pass

    return pipe, cat_cols + num_cols  # feature order after prep is handled by pipeline

def score_dataframe(trained_pipe, df, out_path):
    """Scores a dataframe that has the same feature columns; writes CSV."""
    df_in = df.copy()
    target_col = 'accepted_offer'
    existed_target = target_col in df_in.columns

    if existed_target:
        y_true = df_in[target_col].astype(int)
    else:
        y_true = None

    # Inputs for pipe: drop IDs & non-features (same as training)
    drop_cols = {'offer_date','customer_id'}
    if existed_target:
        drop_cols.add(target_col)
    
    # Only drop columns that actually exist in the dataframe
    existing_drop_cols = [col for col in drop_cols if col in df_in.columns]
    X = df_in.drop(columns=existing_drop_cols)

    proba = trained_pipe.predict_proba(X)[:,1]
    pred = (proba >= 0.5).astype(int)

    scored = df_in.copy()
    scored['acceptance_probability'] = proba
    scored['prediction'] = pred

    scored.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(scored)} rows)")

    if y_true is not None:
        print(f"Test AUC: {roc_auc_score(y_true, proba):.3f} | Acc: {accuracy_score(y_true, pred):.3f}")
        print(classification_report(y_true, pred, digits=3))
    return scored

# ----------------------------
# 3) Run everything
# ----------------------------
if __name__ == "__main__":
    gen = TelcoSyntheticDataGenerator(seed=42)

    # Generate data
    train_df, test_df, full_df = gen.generate_dataset(n_samples=7000, train_ratio=0.8)

    # Train model
    pipe, feature_cols = build_training_pipeline(train_df)

    # Score test (the offered package in each test row)
    _ = score_dataframe(pipe, test_df, 'telco_test_scored.csv')

    # Build per-customer × all-packages grid for test customers
    grid = gen.build_scoring_grid_for_customers(test_df)
    # The grid lacks some columns present in training; we’ll add safe defaults
    grid['microsegment_id'] = grid['microsegment_id'].astype(str)
    grid['current_package_id'] = grid.get('current_package_id', pd.Series(['UNKNOWN']*len(grid)))
    grid['offer_channel'] = 'app'
    grid['offer_personalization_lvl'] = 'medium'
    grid['offer_retention_camp'] = 0
    grid['offer_competitor_resp'] = 0
    grid['offer_seasonal_camp'] = 0
    grid['usage_music_streaming'] = grid.get('usage_music_streaming', 0)
    grid['usage_gaming_mobile'] = grid.get('usage_gaming_mobile', 0.0)
    grid['usage_work_related_calls'] = grid.get('usage_work_related_calls', 0.0)
    grid['usage_international_calls'] = grid.get('usage_international_calls', 0.0)
    grid['usage_peak_hour_pattern'] = 'evening'
    grid['usage_weekend_weekday_ratio'] = 1.0
    grid['cust_payment_punct_score'] = grid.get('cust_payment_punct_score', 0.9)

    # Align columns with training set (missing columns become NaN → handled by OneHotEncoder/passthrough)
    needed = [c for c in train_df.columns if c not in ['accepted_offer','offer_date','customer_id']]
    for c in needed:
        if c not in grid.columns:
            grid[c] = np.nan
    grid = grid[needed + ['customer_id']] if 'customer_id' in grid.columns else grid[needed]

    # Score the grid
    grid_scored = score_dataframe(pipe, grid, 'telco_test_grid_scored.csv')

    # Top-1 package per customer (rank by probability)
    if 'customer_id' in test_df.columns:
        # Reattach customer_id for ranking (if dropped earlier)
        if 'customer_id' not in grid_scored.columns and 'customer_id' in test_df.columns:
            # ensure we have customer_id; if not available, skip top1
            pass
        if 'customer_id' in grid_scored.columns:
            top1 = (grid_scored
                    .assign(rank_prob=grid_scored.groupby('customer_id')['acceptance_probability']
                            .rank(method='first', ascending=False))
                    .query('rank_prob == 1')
                    .drop(columns=['rank_prob']))
            top1.to_csv('telco_test_top1_reco.csv', index=False)
            print(f"Wrote: telco_test_top1_reco.csv ({len(top1)} rows)")

    # Create cleaned version for SAS Model Studio training (remove probability/prediction columns)
    print("\nCreating cleaned version for SAS Model Studio...")
    grid_cleaned = grid_scored.drop(columns=['acceptance_probability', 'prediction'], errors='ignore')
    
    # Fix column order and missing values to match training data structure
    print("Fixing column order and data types...")
    
    # Reorder columns to match training data (customer_id first, then others)
    if 'customer_id' in grid_cleaned.columns:
        cols = ['customer_id'] + [col for col in train_df.columns if col not in ['customer_id', 'accepted_offer', 'offer_date']]
    else:
        cols = [col for col in train_df.columns if col not in ['customer_id', 'accepted_offer', 'offer_date']]
    
    # Only keep columns that exist in both datasets
    available_cols = [col for col in cols if col in grid_cleaned.columns]
    grid_cleaned = grid_cleaned[available_cols]
    
    # Fill missing values with appropriate defaults based on training data
    for col in grid_cleaned.columns:
        if col in train_df.columns:
            if train_df[col].dtype in ['int64', 'int32']:
                # Fill integer columns with median or 0
                grid_cleaned[col] = grid_cleaned[col].fillna(0).astype(int)
            elif train_df[col].dtype in ['float64', 'float32']:
                # Fill float columns with median
                median_val = train_df[col].median()
                grid_cleaned[col] = grid_cleaned[col].fillna(median_val)
            elif train_df[col].dtype == 'object':
                # Fill categorical columns with mode
                mode_val = train_df[col].mode().iloc[0] if not train_df[col].mode().empty else 'unknown'
                grid_cleaned[col] = grid_cleaned[col].fillna(mode_val)
    
    grid_cleaned.to_csv('telco_test_grid_cleaned_for_sas.csv', index=False)
    print(f"Wrote: telco_test_grid_cleaned_for_sas.csv ({len(grid_cleaned)} rows)")
    print(f"Columns: {len(grid_cleaned.columns)} (should match training: {len(train_df.columns)-3})")

    print("\nAll done. Files ready for SAS:")
    print(" - telco_training_data.csv")
    print(" - telco_test_data.csv")
    print(" - telco_full_data.csv")
    print(" - telco_test_scored.csv            # one row per test offer (probability)")
    print(" - telco_test_grid_scored.csv       # one row per (test customer × each package)")
    print(" - telco_test_grid_cleaned_for_sas.csv  # cleaned version for SAS Model Studio training")
    print(" - telco_test_top1_reco.csv         # top package per test customer")
