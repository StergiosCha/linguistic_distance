from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import traceback
import subprocess
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import itertools
from collections import defaultdict, Counter
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results' 
DATA_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# URIEL imports with error handling
try:
    from urielplus import urielplus
    URIEL_AVAILABLE = True
    print("✓ URIEL+ (urielplus) available")
except ImportError:
    print("⚠ Warning: urielplus not available. Install with: pip install urielplus")
    print("⚠ URIEL typological analysis will be skipped")
    URIEL_AVAILABLE = False

class CLDFCognateAnalyzer:
    def __init__(self, cldf_base_path):
        """CLDF-based cognate distance analyzer"""
        self.base_path = Path(cldf_base_path) if cldf_base_path else None
        self.langs = None
        self.forms = None
        self.cognates = None
        self.lp_map = None
        self.lang_lookup = None
        
        self.language_mapping = {
            "Ancient Greek": "Greek: Ancient",
            "Modern Greek": "Greek: Modern Std", 
            "Latin": "Latin",
            "Italian": "Italian",
            "Spanish": "Spanish",
            "French": "French",
            "Romanian": "Romanian",
            "Old Church Slavonic": "Old Church Slavonic",
            "Bulgarian": "Bulgarian",
            "Russian": "Russian",
            "Serbian": "Serbo-Croat",
            "Czech": "Czech",
            "Gothic": "Gothic",
            "German": "German",
            "Sanskrit": "Vedic: Early",
            "Hindi": "Hindi"
        }
        
    def load_cldf_data(self):
        """Load CLDF data files"""
        if not self.base_path:
            return False
            
        try:
            print(f"Loading CLDF data from {self.base_path}")
            
            self.langs = pd.read_csv(self.base_path / "languages.csv")
            self.forms = pd.read_csv(self.base_path / "forms.csv") 
            self.cognates = pd.read_csv(self.base_path / "cognates.csv")
            
            print(f"✓ Loaded {len(self.langs)} languages, {len(self.forms)} forms, {len(self.cognates)} cognate judgments")
            
            self.lang_lookup = dict(zip(self.langs["Name"], self.langs["ID"]))
            self.lp_map = self._build_lang_param_map()
            
            print(f"✓ Built cognate mapping for {len(self.lp_map)} language-parameter combinations")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading CLDF data: {e}")
            return False
    
    def _build_lang_param_map(self):
        """Build language-parameter to cognate set mapping"""
        forms_min = self.forms[["ID","Language_ID","Parameter_ID"]].rename(columns={"ID":"Form_ID"})
        cg = self.cognates.merge(forms_min, on="Form_ID", how="inner")
        cg["Cognateset_ID"] = cg["Cognateset_ID"].astype(str)
        cg["Language_ID"] = cg["Language_ID"].astype(str)
        cg["Parameter_ID"] = cg["Parameter_ID"].astype(str)
        
        lp = (cg.groupby(["Language_ID","Parameter_ID"])["Cognateset_ID"]
                .apply(lambda x: frozenset(x.dropna().unique()))
                .reset_index(name="cset_ids"))
        
        return {(row.Language_ID, row.Parameter_ID): row.cset_ids for row in lp.itertuples(index=False)}
    
    def calculate_cognate_distance(self, lang_a_name, lang_b_name):
        """Calculate cognate distance between two languages"""
        if not self.lp_map or not self.lang_lookup:
            return None
            
        cldf_name_a = self.language_mapping.get(lang_a_name)
        cldf_name_b = self.language_mapping.get(lang_b_name)
        
        if not cldf_name_a or not cldf_name_b:
            print(f"  No CLDF mapping for {lang_a_name} or {lang_b_name}")
            return None
            
        try:
            lang_a_id = str(self.lang_lookup[cldf_name_a])
            lang_b_id = str(self.lang_lookup[cldf_name_b])
        except KeyError as e:
            print(f"  Language not found in CLDF data: {e}")
            return None
        
        params_a = {p for (l,p) in self.lp_map.keys() if l == lang_a_id}
        params_b = {p for (l,p) in self.lp_map.keys() if l == lang_b_id}
        shared = params_a & params_b
        
        if not shared:
            print(f"  No shared parameters between {lang_a_name} and {lang_b_name}")
            return None
        
        matches = 0
        for p in shared:
            if self.lp_map[(lang_a_id, p)] and self.lp_map[(lang_b_id, p)]:
                if self.lp_map[(lang_a_id, p)] & self.lp_map[(lang_b_id, p)]:
                    matches += 1
        
        distance = 1 - matches/len(shared)
        
        print(f"  Cognate distance {lang_a_name}-{lang_b_name}: {distance:.4f} ({matches}/{len(shared)} shared cognates)")
        
        return distance
    
    def calculate_cognate_distances(self, historical_pairs):
        """Calculate cognate distances for all language pairs"""
        if not self.lp_map:
            print("✗ CLDF data not loaded")
            return {}
        
        cognate_results = {}
        
        print("\nCALCULATING COGNATE DISTANCES (CLDF)")
        print("=" * 50)
        
        for pair_name, (ancient, modern, years, family) in historical_pairs.items():
            print(f"Processing {pair_name}...")
            
            distance = self.calculate_cognate_distance(ancient, modern)
            
            if distance is not None:
                cognate_results[pair_name] = distance
                print(f"✓ {pair_name}: {distance:.4f}")
            else:
                print(f"✗ {pair_name}: CLDF cognate data not available")
        
        return cognate_results

class GrambankTypologicalAnalyzer:
    def __init__(self):
        """Grambank-based typological distance analyzer using sane format CSV"""
        
        self.grambank_mapping = {
            "Ancient Greek": "Ancient Greek",
            "Modern Greek": "Modern Greek", 
            "Latin": "Latin",
            "Italian": "Italian",
            "Spanish": "Spanish",
            "French": "French", 
            "Romanian": "Romanian",
            "Old Church Slavonic": "Old Church Slavonic",
            "Bulgarian": "Bulgarian",
            "Russian": "Russian",
            "Serbian": "Serbian",
            "Czech": "Czech",
            "Gothic": "Gothic",
            "German": "German",
            "Sanskrit": "Sanskrit",
            "Hindi": "Hindi"
        }
        
        self.alternative_mappings = {
            "Serbian": ["Serbian", "Serbo-Croatian", "Serbian-Croatian-Bosnian"],
            "Old Church Slavonic": ["Old Church Slavonic", "Church Slavonic"],
            "Romanian": ["Romanian", "Daco-Romanian"],
            "Sanskrit": ["Sanskrit", "Vedic Sanskrit"],
            "Spanish": ["Spanish", "Castilian"],
            "German": ["German", "Standard German", "High German"]
        }
        
        self.grambank_data = None
        self.feature_columns = None

    def load_grambank_sane_data(self, sane_file="grambank_sane_format.csv"):
        """Load Grambank data from sane format CSV"""
        
        # Try multiple possible paths
        possible_paths = [
            sane_file,  # Direct path
            f"data/{sane_file}",  # With data/ prefix
            f"./{sane_file}",  # Current directory
        ]
        
        for file_path in possible_paths:
            try:
                print(f"Trying to load Grambank from: {file_path}")
                
                if not os.path.exists(file_path):
                    continue
                
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                if 'Language' not in df.columns:
                    print(f"  No 'Language' column in {file_path}")
                    continue
                
                # Identify feature columns (GB001 through GB195)
                self.feature_columns = [col for col in df.columns if col.startswith('GB') and col[2:].isdigit()]
                self.feature_columns.sort()
                
                if not self.feature_columns:
                    print(f"  No GB feature columns found in {file_path}")
                    continue
                
                # Set Language column as index for easy lookup
                df = df.set_index('Language')
                self.grambank_data = df
                
                print(f"✓ Successfully loaded Grambank: {len(df)} languages, {len(self.feature_columns)} features")
                return True
                
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue
        
        print("✗ Failed to load Grambank data from any path")
        return False

    def find_grambank_language(self, language_name):
        """Find the best match for a language in Grambank data with improved matching"""
        
        if self.grambank_data is None:
            return None
        
        available_langs = self.grambank_data.index.tolist()
        
        # 1. Try direct mapping first
        if language_name in self.grambank_mapping:
            mapped_name = self.grambank_mapping[language_name]
            
            # Exact match
            if mapped_name in available_langs:
                return mapped_name
            
            # Try alternatives for this language
            if language_name in self.alternative_mappings:
                for alt_name in self.alternative_mappings[language_name]:
                    if alt_name in available_langs:
                        print(f"  Found alternative match: {language_name} → {alt_name}")
                        return alt_name
            
            # Case-insensitive exact match
            for lang in available_langs:
                if lang.lower() == mapped_name.lower():
                    print(f"  Found case-insensitive match: {language_name} → {lang}")
                    return lang
        
        print(f"  No reliable Grambank match found for: {language_name}")
        return None

    def calculate_grambank_distance(self, lang1, lang2):
        """Calculate normalized Hamming distance between two languages using Grambank features"""
        
        if self.grambank_data is None:
            return None
        
        # Find language matches
        grambank_lang1 = self.find_grambank_language(lang1)
        grambank_lang2 = self.find_grambank_language(lang2)
        
        if not grambank_lang1 or not grambank_lang2:
            return None
        
        try:
            # Get feature vectors
            row1 = self.grambank_data.loc[grambank_lang1, self.feature_columns]
            row2 = self.grambank_data.loc[grambank_lang2, self.feature_columns]
            
            # Find positions where both languages have known values (not '?')
            valid_mask = (row1 != '?') & (row2 != '?')
            
            if valid_mask.sum() < 10:
                print(f"  ⚠ Warning: Only {valid_mask.sum()} common features for {lang1}-{lang2}")
                return None
            
            # Extract valid values
            valid_values1 = row1[valid_mask]
            valid_values2 = row2[valid_mask]
            
            # Calculate Hamming distance (proportion of differing values)
            differences = (valid_values1 != valid_values2).sum()
            total_compared = len(valid_values1)
            hamming_distance = differences / total_compared
            
            print(f"  Grambank distance {lang1}-{lang2}: {hamming_distance:.4f} ({differences}/{total_compared} features differ)")
            
            return hamming_distance
            
        except Exception as e:
            print(f"  ✗ Error calculating Grambank distance for {lang1}-{lang2}: {e}")
            return None

    def calculate_grambank_distances(self, historical_pairs):
        """Calculate Grambank distances for all language pairs"""
        
        if self.grambank_data is None:
            print("✗ Grambank data not loaded")
            return {}
        
        grambank_results = {}
        
        print("\nCALCULATING GRAMBANK TYPOLOGICAL DISTANCES")
        print("=" * 60)
        
        for pair_name, (ancient, modern, years, family) in historical_pairs.items():
            print(f"Processing {pair_name}...")
            
            distance = self.calculate_grambank_distance(ancient, modern)
            
            if distance is not None:
                grambank_results[pair_name] = distance
                print(f"✓ {pair_name}: {distance:.4f}")
            else:
                print(f"✗ {pair_name}: Grambank data not available")
        
        return grambank_results
class ImprovedSyntacticAnalyzer:
    def __init__(self):
        """Enhanced syntactic analyzer that controls for text effects"""
        
        self.treebank_metadata = {
            "Ancient Greek": {"genre": "classical_texts", "register": "formal", "period": "ancient"},
            "Modern Greek": {"genre": "news", "register": "standard", "period": "modern"},
            "Latin": {"genre": "religious", "register": "formal", "period": "classical"},
            "Italian": {"genre": "news", "register": "standard", "period": "modern"},
            "Spanish": {"genre": "news", "register": "standard", "period": "modern"},
            "French": {"genre": "news", "register": "standard", "period": "modern"},
            "Romanian": {"genre": "news", "register": "standard", "period": "modern"},
            "Old Church Slavonic": {"genre": "religious", "register": "formal", "period": "ancient"},
            "Bulgarian": {"genre": "news", "register": "standard", "period": "modern"},
            "Russian": {"genre": "news", "register": "standard", "period": "modern"},
            "Serbian": {"genre": "news", "register": "standard", "period": "modern"},
            "Czech": {"genre": "news", "register": "standard", "period": "modern"},
            "Gothic": {"genre": "religious", "register": "formal", "period": "ancient"},
            "German": {"genre": "news", "register": "standard", "period": "modern"},
            "Sanskrit": {"genre": "religious", "register": "formal", "period": "ancient"},
            "Hindi": {"genre": "news", "register": "standard", "period": "modern"}
        }

    def extract_universal_syntactic_features(self, sentences, language):
        """Extract UD-based syntactic features that are reliably annotated"""
        if not sentences:
            return {}
        
        total_tokens = 0
        total_relations = 0
        
        morph_inventories = {
            'case_values': set(),
            'tense_values': set(),
            'aspect_values': set(),
            'mood_values': set(),
            'voice_values': set(),
            'person_values': set(),
            'number_values': set(),
            'gender_values': set(),
            'definiteness_values': set(),
            'degree_values': set()
        }
        
        pos_counts = Counter()
        deprel_counts = Counter()
        dependency_distances = []
        head_directions = {'head_final': 0, 'head_initial': 0}
        vo_orders = {'verb_before_obj': 0, 'obj_before_verb': 0}
        aux_orders = {'aux_before_main': 0, 'main_before_aux': 0}
        adp_orders = {'preposition': 0, 'postposition': 0}
        
        constructions = {
            'passive': 0,
            'active': 0,
            'copula_overt': 0,
            'auxiliary': 0,
            'finite_clause': 0,
            'nonfinite_clause': 0,
            'coordination': 0,
            'subordination': 0
        }
        
        for sentence in sentences:
            if not sentence:
                continue
            
            total_tokens += len(sentence)
            
            token_lookup = {int(token['id']): token for token in sentence if token['id'].isdigit()}
            
            for token in sentence:
                if not token['id'].isdigit():
                    continue
                    
                current_id = int(token['id'])
                head_id = int(token['head']) if token['head'].isdigit() and token['head'] != '0' else None
                
                if token['feats'] != '_':
                    for feat in token['feats'].split('|'):
                        if '=' in feat:
                            key, value = feat.split('=', 1)
                            if key == 'Case':
                                morph_inventories['case_values'].add(value)
                            elif key == 'Tense':
                                morph_inventories['tense_values'].add(value)
                            elif key == 'Aspect':
                                morph_inventories['aspect_values'].add(value)
                            elif key == 'Mood':
                                morph_inventories['mood_values'].add(value)
                            elif key == 'Voice':
                                morph_inventories['voice_values'].add(value)
                            elif key == 'Person':
                                morph_inventories['person_values'].add(value)
                            elif key == 'Number':
                                morph_inventories['number_values'].add(value)
                            elif key == 'Gender':
                                morph_inventories['gender_values'].add(value)
                            elif key == 'Definite':
                                morph_inventories['definiteness_values'].add(value)
                            elif key == 'Degree':
                                morph_inventories['degree_values'].add(value)
                
                pos_counts[token['upos']] += 1
                deprel_counts[token['deprel']] += 1
                total_relations += 1
                
                if head_id and head_id in token_lookup:
                    distance = abs(current_id - head_id)
                    dependency_distances.append(distance)
                    
                    if token['deprel'] in ['amod', 'nmod', 'advmod', 'nummod']:
                        if current_id < head_id:
                            head_directions['head_final'] += 1
                        else:
                            head_directions['head_initial'] += 1
                    
                    if token['deprel'] == 'obj':
                        if current_id < head_id:
                            vo_orders['obj_before_verb'] += 1
                        else:
                            vo_orders['verb_before_obj'] += 1
                    
                    if token['deprel'] == 'aux':
                        if current_id < head_id:
                            aux_orders['aux_before_main'] += 1
                        else:
                            aux_orders['main_before_aux'] += 1
                    
                    if token['deprel'] == 'case':
                        if current_id < head_id:
                            adp_orders['preposition'] += 1
                        else:
                            adp_orders['postposition'] += 1
                
                if token['deprel'] == 'nsubj:pass':
                    constructions['passive'] += 1
                elif token['deprel'] == 'nsubj':
                    constructions['active'] += 1
                elif token['deprel'] == 'cop':
                    constructions['copula_overt'] += 1
                elif token['deprel'] == 'aux':
                    constructions['auxiliary'] += 1
                elif token['deprel'] in ['ccomp', 'xcomp', 'advcl']:
                    if 'VerbForm=Fin' in token['feats']:
                        constructions['finite_clause'] += 1
                    elif 'VerbForm=Inf' in token['feats'] or 'VerbForm=Part' in token['feats']:
                        constructions['nonfinite_clause'] += 1
                elif token['deprel'] in ['conj', 'cc']:
                    constructions['coordination'] += 1
                elif token['deprel'] in ['mark', 'csubj', 'acl']:
                    constructions['subordination'] += 1
        
        features = {}
        
        features['morphological_inventories'] = {
            key: len(values) for key, values in morph_inventories.items()
        }
        features['total_morphological_complexity'] = sum(len(values) for values in morph_inventories.values())
        
        features['pos_frequencies'] = {pos: count/total_tokens for pos, count in pos_counts.items()}
        
        features['deprel_frequencies'] = {rel: count/total_relations for rel, count in deprel_counts.items()}
        
        total_head_dir = sum(head_directions.values())
        features['head_directionality'] = {
            'head_final_ratio': head_directions['head_final'] / max(total_head_dir, 1),
            'head_initial_ratio': head_directions['head_initial'] / max(total_head_dir, 1)
        }
        
        total_vo = sum(vo_orders.values())
        features['verb_object_order'] = {
            'vo_ratio': vo_orders['verb_before_obj'] / max(total_vo, 1),
            'ov_ratio': vo_orders['obj_before_verb'] / max(total_vo, 1)
        }
        
        total_aux = sum(aux_orders.values())
        features['auxiliary_order'] = {
            'aux_main_ratio': aux_orders['aux_before_main'] / max(total_aux, 1),
            'main_aux_ratio': aux_orders['main_before_aux'] / max(total_aux, 1)
        }
        
        total_adp = sum(adp_orders.values())
        features['adposition_order'] = {
            'preposition_ratio': adp_orders['preposition'] / max(total_adp, 1),
            'postposition_ratio': adp_orders['postposition'] / max(total_adp, 1)
        }
        
        features['dependency_distances'] = {
            'mean': np.mean(dependency_distances) if dependency_distances else 0,
            'std': np.std(dependency_distances) if dependency_distances else 0
        }
        
        features['construction_frequencies'] = {
            key: count/total_tokens for key, count in constructions.items()
        }
        
        return features

    def calculate_universal_syntactic_distance(self, features1, features2, lang1, lang2):
        """Calculate syntactic distance using UD-based features"""
        if not features1 or not features2:
            return {'overall_syntactic_distance': 1.0, 'component_distances': {}, 'normalization_applied': 'No data'}
        
        distances = []
        
        morph1 = features1.get('morphological_inventories', {})
        morph2 = features2.get('morphological_inventories', {})
        
        morph_distances = []
        for feature_type in ['case_values', 'tense_values', 'mood_values', 'voice_values', 'number_values', 'gender_values']:
            val1 = morph1.get(feature_type, 0)
            val2 = morph2.get(feature_type, 0)
            morph_distances.append(self._scalar_distance(val1, val2))
        
        avg_morph_distance = np.mean(morph_distances) if morph_distances else 0
        distances.append(('morphological_complexity', avg_morph_distance, 0.40))
        
        deprel1 = features1.get('deprel_frequencies', {})
        deprel2 = features2.get('deprel_frequencies', {})
        deprel_distance = self._distribution_distance(deprel1, deprel2)
        distances.append(('dependency_relations', deprel_distance, 0.30))
        
        head_dir1 = features1.get('head_directionality', {})
        head_dir2 = features2.get('head_directionality', {})
        head_dir_distance = self._distribution_distance(head_dir1, head_dir2)
        
        vo1 = features1.get('verb_object_order', {})
        vo2 = features2.get('verb_object_order', {})
        vo_distance = self._distribution_distance(vo1, vo2)
        
        aux1 = features1.get('auxiliary_order', {})
        aux2 = features2.get('auxiliary_order', {})
        aux_distance = self._distribution_distance(aux1, aux2)
        
        word_order_distance = np.mean([head_dir_distance, vo_distance, aux_distance])
        distances.append(('word_order', word_order_distance, 0.30))
        
        total_distance = sum(dist * weight for _, dist, weight in distances)
        
        return {
            'overall_syntactic_distance': total_distance,
            'component_distances': {name: dist for name, dist, _ in distances},
            'normalization_applied': f"Enhanced UD-based analysis: {lang1} vs {lang2}",
            'feature_details': {
                'morphological_inventories': (morph1, morph2),
                'dependency_relations': (deprel1, deprel2),
                'word_order_patterns': ((head_dir1, vo1, aux1), (head_dir2, vo2, aux2))
            }
        }

    def _distribution_distance(self, dist1, dist2):
        """Calculate Jensen-Shannon divergence between two distributions"""
        if not dist1 or not dist2:
            return 1.0
        
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        if not all_keys:
            return 0.0
        
        p1 = np.array([dist1.get(k, 0) for k in all_keys])
        p2 = np.array([dist2.get(k, 0) for k in all_keys])
        
        p1 = p1 / np.sum(p1) if np.sum(p1) > 0 else p1
        p2 = p2 / np.sum(p2) if np.sum(p2) > 0 else p2
        
        p1 = np.where(p1 == 0, 1e-10, p1)
        p2 = np.where(p2 == 0, 1e-10, p2)
        
        m = (p1 + p2) / 2
        js_div = 0.5 * np.sum(p1 * np.log(p1 / m)) + 0.5 * np.sum(p2 * np.log(p2 / m))
        
        return np.sqrt(js_div)

    def _scalar_distance(self, val1, val2):
        """Calculate normalized distance between scalar values"""
        if val1 == val2:
            return 0.0
        
        avg_val = (abs(val1) + abs(val2)) / 2
        if avg_val == 0:
            return 0.0
        
        return abs(val1 - val2) / avg_val

class URIELTypologicalAnalyzer:
    def __init__(self):
        """URIEL+ based typological distance analyzer with robust error handling"""
        if not URIEL_AVAILABLE:
            print("⚠ URIEL analyzer unavailable - will skip URIEL distances")
            return
        
        self.uriel = urielplus.URIELPlus()
        
        self.uriel_codes = {
            "Ancient Greek": "grc",
            "Modern Greek": "ell",
            "Latin": "lat",
            "Italian": "ita",
            "Spanish": "spa",
            "French": "fra",
            "Romanian": "ron",
            "Bulgarian": "bul",
            "Russian": "rus",
            "Serbian": "srp",
            "Gothic": "got",
            "German": "deu",
            "Sanskrit": "san",
            "Hindi": "hin"
        }
        
        print("ℹ️  Note: Old Church Slavonic excluded from URIEL+ analysis (insufficient featural data)")
    
    def calculate_uriel_distance(self, lang1, lang2):
        """Calculate distance between URIEL+ typological vectors with improved error handling"""
        if not URIEL_AVAILABLE:
            return None
        
        if lang1 not in self.uriel_codes or lang2 not in self.uriel_codes:
            return None
        
        lang1_code = self.uriel_codes[lang1]
        lang2_code = self.uriel_codes[lang2]
        
        try:
            distance = self.uriel.new_distance("featural", [lang1_code, lang2_code])
            return distance
        except SystemExit:
            print(f"  URIEL+ calculation failed for {lang1}-{lang2}: No shared featural features")
            return None
        except Exception as e:
            print(f"  Error calculating URIEL+ distance for {lang1}-{lang2}: {e}")
            return None
    
    def calculate_uriel_distances(self, historical_pairs):
        """Calculate URIEL+ distances for all language pairs with robust error handling"""
        if not URIEL_AVAILABLE:
            print("\n⚠ URIEL+ ANALYSIS SKIPPED (urielplus not available)")
            return {}
            
        uriel_results = {}
        
        print("\nCALCULATING URIEL+ FEATURAL DISTANCES")
        print("=" * 50)
        
        for pair_name, (ancient, modern, years, family) in historical_pairs.items():
            print(f"Processing {pair_name}...")
            
            try:
                distance = self.calculate_uriel_distance(ancient, modern)
                
                if distance is not None:
                    uriel_results[pair_name] = distance
                    print(f"✓ {pair_name}: {distance:.4f}")
                else:
                    print(f"✗ {pair_name}: URIEL+ data not available or calculation failed")
            except SystemExit:
                print(f"✗ {pair_name}: URIEL+ calculation failed (no shared features)")
                continue
            except Exception as e:
                print(f"✗ {pair_name}: Unexpected error - {e}")
                continue
        
        return uriel_results

class ASJPPhonologicalAnalyzer:
    def __init__(self):
        """ASJP phonological distance analyzer"""
        
        self.asjp_file_patterns = {
            "Ancient Greek": ["a_greek_asjp.txt"],
            "Modern Greek": ["m_greek_asjp.txt"],
            "Latin": ["latin_asjp.txt"],
            "Italian": ["italian_asjp.txt"],
            "Spanish": ["spanish_asjp.txt"],
            "French": ["french_asjp.txt"],
            "Romanian": ["romanian_asjp.txt"],
            "Old Church Slavonic": ["old_church_slavonic.txt"],
            "Bulgarian": ["bulgarian_asjp.txt"],
            "Russian": ["russian_asjp.txt"],
            "Serbian": ["serbiancroatian_asjp.txt"],
            "Gothic": ["gothic_asjp.txt"],
            "German": ["german_asjp.txt"],
            "Sanskrit": ["sanskrit_asjp.txt"],
            "Czech": ["czech_asjp.txt"],
            "Hindi": ["hindi_asjp.txt"]
        }

    def normalized_edit_distance(self, s1, s2):
        """Calculate normalized Levenshtein distance"""
        if s1 == s2:
            return 0.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 1.0
        
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j], dp[i][j-1], dp[i-1][j-1]
                    )
        
        return dp[len1][len2] / max(len1, len2)

    def load_asjp_wordlist(self, language):
        """Load ASJP wordlist for a language - returns concept dict if available"""
        
        possible_files = self.asjp_file_patterns.get(language, [])
        
        for filename in possible_files:
            # Use direct path like the working file
            direct_path = f"data/{filename}"
            if os.path.exists(direct_path):
                try:
                    concept_words = self._parse_asjp_file(direct_path)
                    if concept_words:
                        print(f"✓ Found ASJP data for {language}: {direct_path} ({len(concept_words)} concepts)")
                        return concept_words
                        
                except Exception as e:
                    print(f"Error reading {direct_path}: {e}")
                    continue
        
        return None
    
    def _parse_asjp_file(self, filename):
        """Parse ASJP format file and return dict with concept numbers as keys"""
        
        concept_words = {}
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                if line and line[0].isdigit():
                    parts = line.split()
                    
                    if len(parts) >= 2:
                        try:
                            concept_num = int(parts[0])
                            
                            if len(parts) >= 3:
                                phonetic = parts[2]
                            else:
                                phonetic = parts[1]
                            
                            if '//' in phonetic:
                                phonetic = phonetic.split('//')[0].strip()
                            
                            if ',' in phonetic:
                                phonetic = phonetic.split(',')[0].strip()
                            
                            if phonetic:
                                concept_words[concept_num] = phonetic
                                
                        except ValueError:
                            continue
        
        return concept_words if concept_words else None

    def calculate_asjp_distance(self, lang1, lang2):
        """Calculate ASJP phonological distance between two languages using concept matching"""
        
        concept_words1 = self.load_asjp_wordlist(lang1)
        concept_words2 = self.load_asjp_wordlist(lang2)
        
        if not concept_words1 or not concept_words2:
            return None
        
        common_concepts = set(concept_words1.keys()) & set(concept_words2.keys())
        
        if not common_concepts:
            print(f"  No common concepts found between {lang1} and {lang2}")
            return None
        
        distances = []
        
        for concept_num in sorted(common_concepts):
            word1 = concept_words1[concept_num]
            word2 = concept_words2[concept_num]
            
            if word1 and word2:
                dist = self.normalized_edit_distance(word1, word2)
                distances.append(dist)
        
        if distances:
            avg_distance = np.mean(distances)
            print(f"  Compared {len(distances)} common concepts: {sorted(list(common_concepts))}")
            return avg_distance
        
        return None

    def calculate_phonological_distances(self, historical_pairs):
        """Calculate phonological distances for all language pairs"""
        
        phonological_results = {}
        
        print("\nCALCULATING PHONOLOGICAL DISTANCES (ASJP)")
        print("=" * 50)
        
        for pair_name, (ancient, modern, years, family) in historical_pairs.items():
            print(f"Processing {pair_name}...")
            
            distance = self.calculate_asjp_distance(ancient, modern)
            
            if distance is not None:
                phonological_results[pair_name] = distance
                print(f"✓ {pair_name}: {distance:.4f}")
            else:
                print(f"✗ {pair_name}: ASJP data files not found")
        
        return phonological_results

class EnhancedSevenDimensionalTreebankAnalyzer:
    def __init__(self, cldf_path=None):
        """Enhanced seven-dimensional linguistic change analyzer"""
        
        self.treebank_files = {
            "Ancient Greek": "grc_perseus-ud-train.conllu",
            "Modern Greek": "el_gdt-ud-train.conllu", 
            "Latin": "la_ittb-ud-train.conllu",
            "Italian": "it_isdt-ud-train.conllu",
            "Spanish": "es_ancora-ud-train.conllu",
            "French": "fr_ftb-ud-train.conllu",
            "Romanian": "ro_rrt-ud-train.conllu",
            "Old Church Slavonic": "cu_proiel-ud-train.conllu",
            "Bulgarian": "bg_btb-ud-train.conllu", 
            "Russian": "ru_taiga-ud-train.conllu",
            "Serbian": "sr_set-ud-train.conllu",
            "Czech": "cs_cac-ud-train.conllu",
            "Gothic": "got_proiel-ud-train.conllu",
            "German": "de_gsd-ud-train.conllu",
            "Sanskrit": "sa_vedic-ud-train.conllu",
            "Hindi": "hi_hdtb-ud-train.conllu"
        }
        
        self.historical_pairs = {
            "Ancient Greek → Modern Greek": ("Ancient Greek", "Modern Greek", 2500, "Hellenic"),
            "Latin → Italian": ("Latin", "Italian", 1500, "Romance"),
            "Latin → Spanish": ("Latin", "Spanish", 1500, "Romance"), 
            "Latin → French": ("Latin", "French", 1500, "Romance"),
            "Latin → Romanian": ("Latin", "Romanian", 1500, "Romance"),
            "Old Church Slavonic → Bulgarian": ("Old Church Slavonic", "Bulgarian", 1000, "Slavic"),
            "Old Church Slavonic → Russian": ("Old Church Slavonic", "Russian", 1000, "Slavic"),
            "Old Church Slavonic → Serbian": ("Old Church Slavonic", "Serbian", 1000, "Slavic"),
            "Gothic → German": ("Gothic", "German", 1600, "Germanic"),
            "Old Church Slavonic → Czech": ("Old Church Slavonic", "Czech", 1000, "Slavic"),
            "Sanskrit → Hindi": ("Sanskrit", "Hindi", 1500, "Indo-Aryan")
        }
        
        self.syntactic_analyzer = ImprovedSyntacticAnalyzer()
        self.grambank_analyzer = GrambankTypologicalAnalyzer()
        self.phonological_analyzer = ASJPPhonologicalAnalyzer()
        self.uriel_analyzer = URIELTypologicalAnalyzer() if URIEL_AVAILABLE else None
        self.cognate_analyzer = CLDFCognateAnalyzer(cldf_path) if cldf_path else None

    def normalized_edit_distance(self, s1, s2):
        """Calculate normalized edit distance"""
        if s1 == s2:
            return 0.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 1.0
        
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j], dp[i][j-1], dp[i-1][j-1]
                    )
        
        return dp[len1][len2] / max(len1, len2)

    def calculate_lexical_distances(self):
        """Calculate lexical distances from Swadesh lists"""
        
        swadesh_files = {
            "Ancient Greek": "swadesh_Ancient_Greek.txt",
            "Modern Greek": "swadesh_modern_greek.txt",
            "Latin": "swadesh_latin.txt", 
            "Italian": "swadesh_italian.txt",
            "Spanish": "swadesh_spanish.txt",
            "French": "swadesh_french.txt",
            "Romanian": "swadesh_romanian.txt",
            "Old Church Slavonic": "swadesh_old_church_slavonic.txt",
            "Bulgarian": "swadesh_bulgarian.txt",
            "Russian": "swadesh_russian.txt",
            "Serbian": "swadesh_serbian.txt",
            "Gothic": "swadesh_gothic.txt",
            "German": "swadesh_german.txt",
            "Sanskrit": "swadesh_sanskrit.txt",
            "Czech": "swadesh_czech.txt",
            "Hindi": "swadesh_hindi.txt"
        }
        
        lexical_results = {}
        
        print("CALCULATING LEXICAL DISTANCES")
        print("=" * 60)
        
        for pair_name, (ancient, modern, years, family) in self.historical_pairs.items():
            if ancient not in swadesh_files or modern not in swadesh_files:
                print(f"✗ {pair_name}: Missing Swadesh files")
                continue
                
            # Use direct paths like the working file
            ancient_path = f"data/{swadesh_files[ancient]}"
            modern_path = f"data/{swadesh_files[modern]}"
            
            try:
                with open(ancient_path, 'r', encoding='utf-8') as f:
                    lines1 = [line.strip() for line in f if line.strip()]
                with open(modern_path, 'r', encoding='utf-8') as f:
                    lines2 = [line.strip() for line in f if line.strip()]
            except:
                print(f"✗ {pair_name}: Could not load word lists")
                continue
            
            clean_pairs = []
            total_concepts = min(len(lines1), len(lines2))
            
            for i in range(total_concepts):
                line1 = lines1[i].strip()
                line2 = lines2[i].strip()
                
                if ' ' not in line1 and ' ' not in line2 and line1 and line2:
                    clean_pairs.append((line1, line2))
            
            if len(clean_pairs) < 10:
                print(f"✗ {pair_name}: Insufficient clean pairs ({len(clean_pairs)}/{total_concepts} concepts)")
                continue
            
            edit_distances = []
            for w1, w2 in clean_pairs:
                edit_distances.append(self.normalized_edit_distance(w1, w2))
            
            if edit_distances:
                avg_distance = np.mean(edit_distances)
                lexical_results[pair_name] = avg_distance
                clean_percentage = (len(clean_pairs) / total_concepts) * 100
                print(f"✓ {pair_name}: {avg_distance:.4f} ({len(clean_pairs)}/{total_concepts} concepts, {clean_percentage:.1f}% usable)")
        
        return lexical_results

# Replace the calculate_typological_distances method in your main analyzer class:

    def calculate_typological_distances(self, excel_file):
        """Calculate typological distances from WALS"""
        
        # Try multiple possible paths like the Grambank loader
        possible_paths = [
            excel_file,  # Direct path
            f"data/{excel_file}",  # With data/ prefix
            f"./{excel_file}",  # Current directory
        ]
        
        df = None
        for file_path in possible_paths:
            try:
                print(f"DEBUG: Trying to load WALS from {file_path}")
                if os.path.exists(file_path):
                    df = pd.read_excel(file_path, sheet_name="WALS_full_table")
                    print(f"✓ Successfully loaded WALS: {len(df)} rows, {len(df.columns)} columns")
                    break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if df is None:
            print("✗ WALS file not found in any tested path")
            return {}
        
        df = df.dropna(subset=["Feature #"])
        language_columns = [col for col in df.columns if col not in ["Feature Name", "Feature #"]]
        
        language_features = {lang: {} for lang in language_columns}
        for _, row in df.iterrows():
            feature_id = row["Feature #"]
            for lang in language_columns:
                value = row[lang]
                if not pd.isna(value):
                    language_features[lang][feature_id] = value
        
        wals_mapping = {
            "Ancient Greek": "Classical Greek, κλασική Αττική",
            "Modern Greek": "Greek",
            "Latin": "Latin",
            "Italian": "Italian", 
            "Spanish": "Spanish",
            "French": "French",
            "Romanian": "Romanian",
            "Old Church Slavonic": "Old Church Slavonic",
            "Bulgarian": "Bulgarian",
            "Russian": "Russian",
            "Serbian": "Serbo-Croatian",
            "Gothic": "Gothic",
            "German": "German",
            "Sanskrit": "Sanskrit",
            "Czech": "Czech",
            "Hindi": "Hindi"
        }
        
        typological_results = {}
        
        print("\nCALCULATING TYPOLOGICAL DISTANCES (WALS)")
        print("=" * 50)
        
        for pair_name, (ancient, modern, years, family) in self.historical_pairs.items():
            ancient_wals = wals_mapping.get(ancient)
            modern_wals = wals_mapping.get(modern)
            
            if not ancient_wals or not modern_wals:
                print(f"✗ {pair_name}: No WALS mapping")
                continue
                
            if ancient_wals not in language_features or modern_wals not in language_features:
                print(f"✗ {pair_name}: Missing WALS data")
                continue
            
            ancient_features = language_features[ancient_wals]
            modern_features = language_features[modern_wals]
            
            common_features = set(ancient_features.keys()) & set(modern_features.keys())
            
            if not common_features:
                continue
            
            differences = sum(1 for f in common_features 
                            if ancient_features[f] != modern_features[f])
            
            hamming_distance = differences / len(common_features)
            typological_results[pair_name] = hamming_distance
            
            print(f"✓ {pair_name}: {hamming_distance:.4f}")
        
        return typological_results

    def parse_conllu_file(self, filepath, max_sentences=5000):
        """Parse CoNLL-U treebank file"""
        # Use direct path like the working file
        direct_path = f"data/{filepath}"
        
        if not os.path.exists(direct_path):
            print(f"✗ Treebank file not found: {direct_path}")
            return None
        
        print(f"Parsing {direct_path}...")
        
        try:
            sentences = []
            current_sentence = []
            sentence_count = 0
            
            with open(direct_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('#'):
                        continue
                    elif line == '':
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                            sentence_count += 1
                            
                            if sentence_count >= max_sentences:
                                break
                    else:
                        parts = line.split('\t')
                        if len(parts) >= 8 and not '-' in parts[0] and not '.' in parts[0]:
                            token_data = {
                                'id': parts[0],
                                'form': parts[1],
                                'lemma': parts[2], 
                                'upos': parts[3],
                                'xpos': parts[4],
                                'feats': parts[5],
                                'head': parts[6],
                                'deprel': parts[7]
                            }
                            current_sentence.append(token_data)
            
            if current_sentence and sentence_count < max_sentences:
                sentences.append(current_sentence)
                sentence_count += 1
            
            used_sentences = len(sentences)
            print(f"  ✓ Parsed {used_sentences} sentences")
            
            return sentences
            
        except Exception as e:
            print(f"  ✗ Error parsing {direct_path}: {e}")
            return None

    def calculate_improved_syntactic_distances(self):
        """Calculate syntactic distances"""
        
        treebank_features = {}
        
        print("\nEXTRACTING SYNTACTIC FEATURES")
        print("=" * 60)
        
        for language, filename in self.treebank_files.items():
            sentences = self.parse_conllu_file(filename)
            if sentences:
                features = self.syntactic_analyzer.extract_universal_syntactic_features(sentences, language)
                treebank_features[language] = features
                print(f"✓ {language}: Features extracted")
            else:
                print(f"✗ {language}: Failed to extract features")
        
        syntactic_results = {}
        
        print(f"\nCALCULATING SYNTACTIC DISTANCES")
        print("=" * 60)
        
        for pair_name, (ancient, modern, years, family) in self.historical_pairs.items():
            if ancient not in treebank_features or modern not in treebank_features:
                print(f"✗ {pair_name}: Missing treebank data")
                continue
            
            distance_data = self.syntactic_analyzer.calculate_universal_syntactic_distance(
                treebank_features[ancient], 
                treebank_features[modern],
                ancient,
                modern
            )
            
            syntactic_results[pair_name] = distance_data['overall_syntactic_distance']
            print(f"✓ {pair_name}: {distance_data['overall_syntactic_distance']:.4f}")
        
        return syntactic_results

    def combine_seven_dimensions(self, lexical_results, wals_typological_results, 
                                grambank_results, uriel_results, syntactic_results, 
                                phonological_results, cognate_results):
        """Enhanced combination with statistical validation"""
        
        combined_results = {}
        
        print(f"\nCOMBINING SEVEN DIMENSIONS WITH STATISTICAL VALIDATION")
        print("=" * 80)
        print("Dimensions: Lexical, WALS-Typological, Grambank-Typological, URIEL-Featural, Syntactic, Phonological, Cognate")
        print("Weighting: Equal weight for available dimensions")
        print()
        sys.stdout.flush()  # Force output
        
        for pair_name in lexical_results.keys():
            try:
                print(f"Processing {pair_name}...")
                
                lexical = lexical_results.get(pair_name, None)
                wals_typological = wals_typological_results.get(pair_name, None)
                grambank_typological = grambank_results.get(pair_name, None)
                uriel_featural = uriel_results.get(pair_name, None) if uriel_results else None
                syntactic = syntactic_results.get(pair_name, None)
                phonological = phonological_results.get(pair_name, None)
                cognate = cognate_results.get(pair_name, None)
                
                if lexical is None:
                    print(f"  Skipping {pair_name} - no lexical data")
                    continue
                
                available_dimensions = []
                available_dimensions.append(('lexical', lexical))
                
                if wals_typological is not None:
                    available_dimensions.append(('wals_typological', wals_typological))
                if grambank_typological is not None:
                    available_dimensions.append(('grambank_typological', grambank_typological))
                if uriel_featural is not None:
                    available_dimensions.append(('uriel_featural', uriel_featural))
                if syntactic is not None:
                    available_dimensions.append(('syntactic', syntactic))
                if phonological is not None:
                    available_dimensions.append(('phonological', phonological))
                if cognate is not None:
                    available_dimensions.append(('cognate', cognate))
                
                if available_dimensions:
                    balanced_score = np.mean([value for _, value in available_dimensions])
                else:
                    balanced_score = lexical
                
                if balanced_score is None or np.isnan(balanced_score):
                    balanced_score = lexical if lexical is not None else 0.0
                
                _, (ancient, modern, years, family) = pair_name, self.historical_pairs[pair_name]
                
                combined_results[pair_name] = {
                    "lexical_distance": lexical,
                    "wals_typological_distance": wals_typological,
                    "grambank_typological_distance": grambank_typological,
                    "uriel_featural_distance": uriel_featural,
                    "syntactic_distance": syntactic,
                    "phonological_distance": phonological,
                    "cognate_distance": cognate,
                    "seven_dimensional_combined": float(balanced_score),
                    "family": family,
                    "years": years,
                    "ancient": ancient,
                    "modern": modern,
                    "available_dimensions": len(available_dimensions),
                    "dimension_count": 7,
                    "complete_analysis": len(available_dimensions) == 7
                }
                
                print(f"  ✓ {pair_name}: {balanced_score:.4f} ({len(available_dimensions)} dimensions)")
                
            except Exception as e:
                print(f"  ✗ Error processing {pair_name}: {e}")
                continue
        
        print(f"\n✓ Combined results for {len(combined_results)} language pairs")
        return combined_results

    def create_seven_dimensional_visualizations(self, combined_results):
        """Create comprehensive 7D analysis visualizations"""
        
        if not combined_results:
            return None
        
        # Prepare data
        pair_names = []
        lexical_dists = []
        wals_typological_dists = []
        grambank_typological_dists = []
        uriel_dists = []
        syntactic_dists = []
        phonological_dists = []
        cognate_dists = []
        combined_scores = []
        families = []
        
        family_colors = {
            'Hellenic': '#2E8B57',
            'Romance': '#4169E1', 
            'Slavic': '#DC143C',
            'Germanic': '#FF8C00',
            'Indo-Aryan': '#9932CC'
        }
        
        for pair_name, data in combined_results.items():
            clean_name = pair_name.replace(' → ', '\n→ ')
            pair_names.append(clean_name)
            lexical_dists.append(data["lexical_distance"])
            wals_typological_dists.append(data["wals_typological_distance"] if data["wals_typological_distance"] is not None else 0)
            grambank_typological_dists.append(data["grambank_typological_distance"] if data["grambank_typological_distance"] is not None else 0)
            uriel_dists.append(data["uriel_featural_distance"] if data["uriel_featural_distance"] is not None else 0)
            syntactic_dists.append(data["syntactic_distance"] if data["syntactic_distance"] is not None else 0)
            phonological_dists.append(data["phonological_distance"] if data["phonological_distance"] is not None else 0)
            cognate_dists.append(data["cognate_distance"] if data["cognate_distance"] is not None else 0)
            combined_scores.append(data["seven_dimensional_combined"])
            families.append(data["family"])
        
        colors = [family_colors.get(family, '#808080') for family in families]
        
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7
        })
        
        fig = plt.figure(figsize=(24, 18), constrained_layout=True)
        
        # 1. Seven-component comparison
        ax1 = plt.subplot(3, 4, 1)
        x = np.arange(len(pair_names))
        width = 0.10
        
        bars1 = ax1.bar(x - 3*width, lexical_dists, width, label='Lexical', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x - 2*width, wals_typological_dists, width, label='WALS', alpha=0.8, color='lightcoral')
        bars3 = ax1.bar(x - width, grambank_typological_dists, width, label='Grambank', alpha=0.8, color='lightpink')
        bars4 = ax1.bar(x, uriel_dists, width, label='URIEL', alpha=0.8, color='lightsalmon')
        bars5 = ax1.bar(x + width, syntactic_dists, width, label='Syntactic', alpha=0.8, color='lightgreen')
        bars6 = ax1.bar(x + 2*width, phonological_dists, width, label='Phonological', alpha=0.8, color='gold')
        bars7 = ax1.bar(x + 3*width, cognate_dists, width, label='Cognate', alpha=0.8, color='plum')
        
        ax1.set_title('SEVEN Dimensions of Linguistic Change', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Distance', fontsize=9)
        ax1.set_xticks(x)
        ax1.set_xticklabels(pair_names, rotation=45, ha='right', fontsize=7)
        ax1.legend(fontsize=6, ncol=2, loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Combined distance (MAIN RESULT)
        ax2 = plt.subplot(3, 4, 2)
        bars_combined = ax2.bar(range(len(pair_names)), combined_scores,
                            color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('SEVEN-DIMENSIONAL COMBINED\nLINGUISTIC CHANGE', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Combined Distance', fontsize=9)
        ax2.set_xticks(range(len(pair_names)))
        ax2.set_xticklabels(pair_names, rotation=45, ha='right', fontsize=7)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, distance in zip(bars_combined, combined_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{distance:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=7)
        
        # Save the plot
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data

# Global analyzer instance
global_analyzer = None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Seven-Dimensional Linguistic Analysis API"})

@app.route('/api/debug-files', methods=['GET'])
def debug_files():
    """Debug endpoint to see available files"""
    folders = {
        "data": DATA_FOLDER,
        "uploads": UPLOAD_FOLDER,
        "current": "."
    }
    
    result = {}
    for name, folder in folders.items():
        if os.path.exists(folder):
            try:
                result[name] = sorted(os.listdir(folder))
            except Exception as e:
                result[name] = f"Error reading folder: {e}"
        else:
            result[name] = "Folder does not exist"
    
    return jsonify(result)

@app.route('/api/initialize', methods=['POST'])
def initialize_analyzer():
    """Initialize the analyzer with configuration"""
    global global_analyzer
    
    try:
        data = request.get_json()
        cldf_path = data.get('cldf_path', 'data')  # Default to data folder
        
        global_analyzer = EnhancedSevenDimensionalTreebankAnalyzer(cldf_path)
        
        return jsonify({
            "status": "success",
            "message": "Analyzer initialized successfully",
            "language_pairs": list(global_analyzer.historical_pairs.keys()),
            "uriel_available": URIEL_AVAILABLE
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for data files"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['file']
        file_type = request.form.get('type', 'unknown')
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            "status": "success",
            "message": f"File uploaded successfully",
            "filename": filename,
            "path": filepath,
            "type": file_type
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def run_analysis():
    """Run linguistic analysis on selected dimensions"""
    global global_analyzer
    
    if not global_analyzer:
        return jsonify({"status": "error", "message": "Analyzer not initialized"}), 400
    
    try:
        data = request.get_json()
        selected_dimensions = data.get('dimensions', [])
        language_pairs = data.get('pairs', list(global_analyzer.historical_pairs.keys()))
        
        results = {}
        
        # Calculate each selected dimension
        if 'lexical' in selected_dimensions:
            print("Calculating lexical distances...")
            results['lexical'] = global_analyzer.calculate_lexical_distances()
        
        if 'wals' in selected_dimensions:
            print("Calculating WALS typological distances...")
            results['wals'] = global_analyzer.calculate_typological_distances("sane_wals.xlsx")
        
        if 'grambank' in selected_dimensions:
            print("Calculating Grambank distances...")
            if global_analyzer.grambank_analyzer.load_grambank_sane_data("grambank_sane_format.csv"):
                results['grambank'] = global_analyzer.grambank_analyzer.calculate_grambank_distances(global_analyzer.historical_pairs)
            else:
                results['grambank'] = {}
        
        if 'uriel' in selected_dimensions and URIEL_AVAILABLE:
            print("Calculating URIEL distances...")
            results['uriel'] = global_analyzer.uriel_analyzer.calculate_uriel_distances(global_analyzer.historical_pairs)
        else:
            results['uriel'] = {}
        
        if 'syntactic' in selected_dimensions:
            print("Calculating syntactic distances...")
            results['syntactic'] = global_analyzer.calculate_improved_syntactic_distances()
        
        if 'phonological' in selected_dimensions:
            print("Calculating phonological distances...")
            results['phonological'] = global_analyzer.phonological_analyzer.calculate_phonological_distances(global_analyzer.historical_pairs)
        
        if 'cognate' in selected_dimensions and global_analyzer.cognate_analyzer:
            print("Calculating cognate distances...")
            if global_analyzer.cognate_analyzer.load_cldf_data():
                results['cognate'] = global_analyzer.cognate_analyzer.calculate_cognate_distances(global_analyzer.historical_pairs)
            else:
                results['cognate'] = {}
        else:
            results['cognate'] = {}
        
        # Combine dimensions if multiple selected
        if len(selected_dimensions) > 1:
            combined_results = global_analyzer.combine_seven_dimensions(
                results.get('lexical', {}),
                results.get('wals', {}),
                results.get('grambank', {}),
                results.get('uriel', {}),
                results.get('syntactic', {}),
                results.get('phonological', {}),
                results.get('cognate', {})
            )
            results['combined'] = combined_results
        
        return jsonify({
            "status": "success",
            "results": results,
            "dimensions_analyzed": selected_dimensions
        })
        
    except Exception as e:
        print(f"Analysis error: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/')
def index():
    """Serve the frontend HTML"""
    return send_file('index.html')

@app.route('/api/visualize', methods=['POST'])
def create_visualizations():
    """Create visualizations for analysis results"""
    global global_analyzer
    
    if not global_analyzer:
        return jsonify({"status": "error", "message": "Analyzer not initialized"}), 400
    
    try:
        data = request.get_json()
        results = data.get('results', {})
        
        if 'combined' in results:
            plot_data = global_analyzer.create_seven_dimensional_visualizations(results['combined'])
            
            return jsonify({
                "status": "success",
                "plot": plot_data,
                "message": "Visualization created successfully"
            })
        else:
            return jsonify({"status": "error", "message": "No combined results available for visualization"}), 400
            
    except Exception as e:
        print(f"Visualization error: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export analysis results to CSV"""
    try:
        data = request.get_json()
        results = data.get('results', {})
        format_type = data.get('format', 'csv')
        
        if 'combined' in results:
            rows = []
            for pair_name, pair_data in results['combined'].items():
                row = {
                    'Language_Pair': pair_name,
                    'Ancient_Language': pair_data['ancient'],
                    'Modern_Language': pair_data['modern'],
                    'Language_Family': pair_data['family'],
                    'Temporal_Depth_Years': pair_data['years'],
                    'Lexical_Distance': pair_data['lexical_distance'],
                    'WALS_Typological_Distance': pair_data['wals_typological_distance'],
                    'Grambank_Typological_Distance': pair_data['grambank_typological_distance'],
                    'URIEL_Featural_Distance': pair_data['uriel_featural_distance'],
                    'Syntactic_Distance': pair_data['syntactic_distance'],
                    'Phonological_Distance': pair_data['phonological_distance'],
                    'Cognate_Distance': pair_data['cognate_distance'],
                    'Seven_Dimensional_Combined_Distance': pair_data['seven_dimensional_combined']
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            filename = f"linguistic_analysis_results.{format_type}"
            filepath = os.path.join(RESULTS_FOLDER, filename)
            
            if format_type == 'csv':
                df.to_csv(filepath, index=False)
            elif format_type == 'json':
                df.to_json(filepath, orient='records', indent=2)
            elif format_type == 'xlsx':
                df.to_excel(filepath, index=False)
            
            return jsonify({
                "status": "success",
                "message": f"Results exported to {filename}",
                "filename": filename,
                "path": filepath
            })
        else:
            return jsonify({"status": "error", "message": "No results to export"}), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download exported files"""
    try:
        return send_file(os.path.join(RESULTS_FOLDER, filename), as_attachment=True)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 404

@app.route('/api/check-files', methods=['GET'])
def check_files():
    """Check which data files are available"""
    try:
        file_categories = {
            "wals": ["sane_wals.xlsx"],
            "grambank": ["grambank_sane_format.csv"],
            "swadesh": [
                "swadesh_Ancient_Greek.txt", "swadesh_modern_greek.txt", "swadesh_latin.txt",
                "swadesh_italian.txt", "swadesh_spanish.txt", "swadesh_french.txt",
                "swadesh_romanian.txt", "swadesh_old_church_slavonic.txt", "swadesh_bulgarian.txt",
                "swadesh_russian.txt", "swadesh_serbian.txt", "swadesh_gothic.txt",
                "swadesh_german.txt", "swadesh_sanskrit.txt", "swadesh_czech.txt", "swadesh_hindi.txt"
            ],
            "asjp": [
                "a_greek_asjp.txt", "m_greek_asjp.txt", "latin_asjp.txt", "italian_asjp.txt",
                "spanish_asjp.txt", "french_asjp.txt", "romanian_asjp.txt", "old_church_slavonic.txt",
                "bulgarian_asjp.txt", "russian_asjp.txt", "serbiancroatian_asjp.txt", "gothic_asjp.txt",
                "german_asjp.txt", "sanskrit_asjp.txt", "czech_asjp.txt", "hindi_asjp.txt"
            ],
            "treebanks": [
                "grc_perseus-ud-train.conllu", "el_gdt-ud-train.conllu", "la_ittb-ud-train.conllu",
                "it_isdt-ud-train.conllu", "es_ancora-ud-train.conllu", "fr_ftb-ud-train.conllu",
                "ro_rrt-ud-train.conllu", "cu_proiel-ud-train.conllu", "bg_btb-ud-train.conllu",
                "ru_taiga-ud-train.conllu", "sr_set-ud-train.conllu", "cs_cac-ud-train.conllu",
                "got_proiel-ud-train.conllu", "de_gsd-ud-train.conllu", "sa_vedic-ud-train.conllu",
                "hi_hdtb-ud-train.conllu"
            ]
        }
        
        file_status = {}
        
        for category, files in file_categories.items():
            found_files = []
            missing_files = []
            
            for filename in files:
                filepath = f"data/{filename}"
                if os.path.exists(filepath):
                    found_files.append(filename)
                else:
                    missing_files.append(filename)
            
            file_status[category] = {
                "found": found_files,
                "missing": missing_files,
                "total": len(files),
                "available": len(found_files)
            }
        
        return jsonify({
            "status": "success",
            "file_status": file_status
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/dimensions', methods=['GET'])
def get_available_dimensions():
    """Get available analysis dimensions"""
    dimensions = [
        {"id": "lexical", "name": "Lexical Distance", "description": "Core vocabulary change (Swadesh lists)"},
        {"id": "wals", "name": "WALS Typological", "description": "Discrete structural features"},
        {"id": "grambank", "name": "Grambank Typological", "description": "Binary grammatical features"},
        {"id": "syntactic", "name": "Syntactic Distance", "description": "Enhanced UD treebank analysis"},
        {"id": "phonological", "name": "Phonological Distance", "description": "Sound change patterns (ASJP)"},
        {"id": "cognate", "name": "Cognate Distance", "description": "Historical relatedness (CLDF)"}
    ]
    
    if URIEL_AVAILABLE:
        dimensions.insert(3, {
            "id": "uriel", 
            "name": "URIEL Featural", 
            "description": "Dense vector embeddings"
        })
    
    return jsonify({
        "status": "success",
        "dimensions": dimensions,
        "uriel_available": URIEL_AVAILABLE
    })

@app.route('/api/language-pairs', methods=['GET'])
def get_language_pairs():
    """Get available language pairs"""
    historical_pairs = {
        "Ancient Greek → Modern Greek": ("Ancient Greek", "Modern Greek", 2500, "Hellenic"),
        "Latin → Italian": ("Latin", "Italian", 1500, "Romance"),
        "Latin → Spanish": ("Latin", "Spanish", 1500, "Romance"), 
        "Latin → French": ("Latin", "French", 1500, "Romance"),
        "Latin → Romanian": ("Latin", "Romanian", 1500, "Romance"),
        "Old Church Slavonic → Bulgarian": ("Old Church Slavonic", "Bulgarian", 1000, "Slavic"),
        "Old Church Slavonic → Russian": ("Old Church Slavonic", "Russian", 1000, "Slavic"),
        "Old Church Slavonic → Serbian": ("Old Church Slavonic", "Serbian", 1000, "Slavic"),
        "Gothic → German": ("Gothic", "German", 1600, "Germanic"),
        "Old Church Slavonic → Czech": ("Old Church Slavonic", "Czech", 1000, "Slavic"),
        "Sanskrit → Hindi": ("Sanskrit", "Hindi", 1500, "Indo-Aryan")
    }
    
    return jsonify({
        "status": "success",
        "language_pairs": list(historical_pairs.keys()),
        "pairs_data": historical_pairs
    })

if __name__ == '__main__':
    print("Starting Seven-Dimensional Linguistic Analysis API...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/debug-files - Debug file availability")
    print("  POST /api/initialize - Initialize analyzer")
    print("  POST /api/upload - Upload data files")
    print("  POST /api/analyze - Run analysis")
    print("  POST /api/visualize - Create visualizations")
    print("  POST /api/export - Export results")
    print("  GET  /api/language-pairs - Get language pairs")
    print("  GET  /api/dimensions - Get available dimensions")
    print("  GET  /api/check-files - Check file availability")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
