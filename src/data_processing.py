"""
Advanced Data Processing Module for Drug Toxicity Prediction
Handles molecular descriptor calculation, feature engineering, and data preprocessing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import warnings

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from mordred import Calculator, descriptors as mordred_descriptors

# Machine learning utilities
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Custom utilities
from .config import DESCRIPTOR_CONFIG, DATASETS, ADVANCED_FEATURES

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularDescriptorCalculator:
    """
    Advanced molecular descriptor calculator with multiple descriptor types
    """
    
    def __init__(self):
        self.rdkit_descriptors = DESCRIPTOR_CONFIG["rdkit_descriptors"]
        self.use_mordred = DESCRIPTOR_CONFIG["mordred_descriptors"]
        self.custom_descriptors = DESCRIPTOR_CONFIG["custom_descriptors"]
        
        # Initialize Mordred calculator if needed
        if self.use_mordred:
            self.mordred_calc = Calculator(mordred_descriptors, ignore_3D=False)
        
        # Descriptor mapping
        self.descriptor_functions = {
            "MolWt": Descriptors.MolWt,
            "LogP": Descriptors.MolLogP,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "TPSA": Descriptors.TPSA,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
            "RingCount": rdMolDescriptors.CalcRingCount,
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
            "SaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
            "Heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
            "AmideCount": rdMolDescriptors.CalcNumAmideBonds,
            "EsterCount": rdMolDescriptors.CalcNumEsterBonds,
            "CarbonylCount": rdMolDescriptors.CalcNumCarbonylBonds,
            "MolecularFormula": self._get_molecular_formula,
            "ExactMass": self._get_exact_mass,
            "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
            "FormalCharge": Chem.rdmolops.GetFormalCharge
        }
        
        # Custom descriptor functions
        self.custom_functions = {
            "bertz_ct": self._calculate_bertz_ct,
            "balaban_j": self._calculate_balaban_j,
            "wiener_index": self._calculate_wiener_index,
            "electrotopological": self._calculate_electrotopological,
            "molecular_complexity": self._calculate_molecular_complexity,
            "drug_likeness": self._calculate_drug_likeness,
            "synthetic_accessibility": self._calculate_synthetic_accessibility,
            "lipophilicity_efficiency": self._calculate_lipophilicity_efficiency,
            "topological_polar_surface_area": self._calculate_tpsa_3d,
            "molecular_surface_area": self._calculate_molecular_surface_area
        }
    
    def calculate_descriptors(self, smiles_list: List[str], 
                           include_mordred: bool = True,
                           include_custom: bool = True) -> pd.DataFrame:
        """
        Calculate comprehensive molecular descriptors for a list of SMILES
        
        Args:
            smiles_list: List of SMILES strings
            include_mordred: Whether to include Mordred descriptors
            include_custom: Whether to include custom descriptors
            
        Returns:
            DataFrame with calculated descriptors
        """
        logger.info(f"Calculating descriptors for {len(smiles_list)} molecules")
        
        descriptors_list = []
        failed_smiles = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Calculating descriptors")):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_smiles.append((i, smiles))
                    continue
                
                # Add hydrogens for 3D descriptors
                mol = Chem.AddHs(mol)
                
                # Calculate RDKit descriptors
                rdkit_desc = self._calculate_rdkit_descriptors(mol)
                
                # Calculate Mordred descriptors
                mordred_desc = {}
                if include_mordred and self.use_mordred:
                    mordred_desc = self._calculate_mordred_descriptors(mol)
                
                # Calculate custom descriptors
                custom_desc = {}
                if include_custom:
                    custom_desc = self._calculate_custom_descriptors(mol)
                
                # Combine all descriptors
                all_descriptors = {**rdkit_desc, **mordred_desc, **custom_desc}
                all_descriptors["SMILES"] = smiles
                all_descriptors["Index"] = i
                
                descriptors_list.append(all_descriptors)
                
            except Exception as e:
                logger.warning(f"Failed to calculate descriptors for SMILES {smiles}: {str(e)}")
                failed_smiles.append((i, smiles))
        
        if failed_smiles:
            logger.warning(f"Failed to process {len(failed_smiles)} SMILES")
        
        df = pd.DataFrame(descriptors_list)
        df.set_index("Index", inplace=True)
        
        logger.info(f"Successfully calculated {len(df.columns)} descriptors for {len(df)} molecules")
        return df
    
    def _calculate_rdkit_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate RDKit descriptors"""
        descriptors = {}
        
        for desc_name in self.rdkit_descriptors:
            if desc_name in self.descriptor_functions:
                try:
                    value = self.descriptor_functions[desc_name](mol)
                    descriptors[desc_name] = float(value) if value is not None else np.nan
                except Exception as e:
                    logger.debug(f"Failed to calculate {desc_name}: {str(e)}")
                    descriptors[desc_name] = np.nan
        
        return descriptors
    
    def _calculate_mordred_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate Mordred descriptors"""
        try:
            mordred_desc = self.mordred_calc(mol)
            # Convert to dict and handle missing values
            desc_dict = {}
            for desc, value in zip(mordred_desc.columns, mordred_desc.values[0]):
                if isinstance(value, (int, float)):
                    desc_dict[f"Mordred_{desc}"] = float(value) if not np.isnan(value) else np.nan
                else:
                    desc_dict[f"Mordred_{desc}"] = np.nan
            return desc_dict
        except Exception as e:
            logger.debug(f"Failed to calculate Mordred descriptors: {str(e)}")
            return {}
    
    def _calculate_custom_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate custom advanced descriptors"""
        descriptors = {}
        
        for desc_name in self.custom_descriptors:
            if desc_name in self.custom_functions:
                try:
                    value = self.custom_functions[desc_name](mol)
                    descriptors[f"Custom_{desc_name}"] = float(value) if value is not None else np.nan
                except Exception as e:
                    logger.debug(f"Failed to calculate custom descriptor {desc_name}: {str(e)}")
                    descriptors[f"Custom_{desc_name}"] = np.nan
        
        return descriptors
    
    def _get_molecular_formula(self, mol: Chem.Mol) -> str:
        """Get molecular formula"""
        try:
            return rdMolDescriptors.CalcMolFormula(mol)
        except:
            return ""
    
    def _get_exact_mass(self, mol: Chem.Mol) -> float:
        """Calculate exact molecular mass"""
        try:
            return rdMolDescriptors.CalcExactMolWt(mol)
        except:
            return np.nan
    
    def _calculate_bertz_ct(self, mol: Chem.Mol) -> float:
        """Calculate Bertz CT complexity index"""
        try:
            return Chem.GraphDescriptors.BertzCT(mol)
        except:
            return np.nan
    
    def _calculate_balaban_j(self, mol: Chem.Mol) -> float:
        """Calculate Balaban J index"""
        try:
            return Chem.GraphDescriptors.BalabanJ(mol)
        except:
            return np.nan
    
    def _calculate_wiener_index(self, mol: Chem.Mol) -> float:
        """Calculate Wiener index"""
        try:
            return Chem.GraphDescriptors.WienerIndex(mol)
        except:
            return np.nan
    
    def _calculate_electrotopological(self, mol: Chem.Mol) -> float:
        """Calculate electrotopological state index"""
        try:
            from rdkit.Chem.EState import EState_VSA
            return sum(EState_VSA.EState_VSA_(mol))
        except:
            return np.nan
    
    def _calculate_molecular_complexity(self, mol: Chem.Mol) -> float:
        """Calculate molecular complexity score"""
        try:
            # Combine multiple complexity metrics
            complexity = 0
            complexity += Chem.GraphDescriptors.BertzCT(mol) * 0.3
            complexity += mol.GetNumAtoms() * 0.2
            complexity += mol.GetNumBonds() * 0.2
            complexity += rdMolDescriptors.CalcNumAromaticRings(mol) * 0.3
            return complexity
        except:
            return np.nan
    
    def _calculate_drug_likeness(self, mol: Chem.Mol) -> float:
        """Calculate drug-likeness score (Lipinski's Rule of Five)"""
        try:
            # Lipinski's parameters
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Calculate drug-likeness score
            score = 0
            if mw <= 500: score += 1
            if logp <= 5: score += 1
            if hbd <= 5: score += 1
            if hba <= 10: score += 1
            
            return score / 4.0  # Normalize to 0-1
        except:
            return np.nan
    
    def _calculate_synthetic_accessibility(self, mol: Chem.Mol) -> float:
        """Calculate synthetic accessibility score (simplified)"""
        try:
            # Simplified SA score based on molecular features
            complexity = self._calculate_molecular_complexity(mol)
            ring_count = rdMolDescriptors.CalcRingCount(mol)
            hetero_count = rdMolDescriptors.CalcNumHeteroatoms(mol)
            
            # Higher complexity and more rings/heteroatoms = lower SA
            sa_score = 10 - (complexity / 100) - (ring_count * 0.5) - (hetero_count * 0.2)
            return max(1, min(10, sa_score))  # Clamp between 1-10
        except:
            return np.nan
    
    def _calculate_lipophilicity_efficiency(self, mol: Chem.Mol) -> float:
        """Calculate lipophilicity efficiency (LE = pIC50 / LogP)"""
        try:
            logp = Descriptors.MolLogP(mol)
            if logp > 0:
                # Simplified LE calculation
                mw = Descriptors.MolWt(mol)
                le = (1 / mw) * 1000 / logp  # Simplified formula
                return le
            return np.nan
        except:
            return np.nan
    
    def _calculate_tpsa_3d(self, mol: Chem.Mol) -> float:
        """Calculate 3D topological polar surface area"""
        try:
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            
            # Calculate 3D TPSA
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.CalcTPSA(mol)
        except:
            return np.nan
    
    def _calculate_molecular_surface_area(self, mol: Chem.Mol) -> float:
        """Calculate molecular surface area"""
        try:
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            
            # Calculate surface area (simplified)
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.CalcLabuteASA(mol)
        except:
            return np.nan


class FeatureEngineer:
    """
    Advanced feature engineering for molecular data
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.pca = None
    
    def preprocess_features(self, X: pd.DataFrame, 
                          method: str = "robust",
                          handle_missing: str = "median") -> pd.DataFrame:
        """
        Preprocess features with scaling and missing value handling
        
        Args:
            X: Feature matrix
            method: Scaling method ('standard', 'robust', 'minmax')
            handle_missing: Method to handle missing values ('median', 'mean', 'drop')
            
        Returns:
            Preprocessed feature matrix
        """
        logger.info(f"Preprocessing {X.shape[1]} features")
        
        # Handle missing values
        if handle_missing == "median":
            X = X.fillna(X.median())
        elif handle_missing == "mean":
            X = X.fillna(X.mean())
        elif handle_missing == "drop":
            X = X.dropna(axis=1)
        
        # Remove constant features
        constant_selector = VarianceThreshold(threshold=0)
        X_var = constant_selector.fit_transform(X)
        feature_names = X.columns[constant_selector.get_support()]
        X = pd.DataFrame(X_var, columns=feature_names, index=X.index)
        
        # Scale features
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.info(f"After preprocessing: {X.shape[1]} features")
        return X
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        method: str = "univariate",
                        k: int = 100) -> pd.DataFrame:
        """
        Feature selection using various methods
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('univariate', 'variance', 'pca')
            k: Number of features to select
            
        Returns:
            Selected feature matrix
        """
        logger.info(f"Selecting top {k} features using {method}")
        
        if method == "univariate":
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            
        elif method == "variance":
            # Select features with highest variance
            variances = X.var()
            selected_features = variances.nlargest(k).index
            X_selected = X[selected_features].values
            
        elif method == "pca":
            # Use PCA for dimensionality reduction
            self.pca = PCA(n_components=min(k, X.shape[1]))
            X_selected = self.pca.fit_transform(X)
            selected_features = [f"PC_{i+1}" for i in range(X_selected.shape[1])]
        
        X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        logger.info(f"Selected {X_final.shape[1]} features")
        
        return X_final
    
    def create_interaction_features(self, X: pd.DataFrame, 
                                  max_features: int = 50) -> pd.DataFrame:
        """
        Create interaction features between top correlated features
        
        Args:
            X: Feature matrix
            max_features: Maximum number of interaction features to create
            
        Returns:
            Feature matrix with interaction terms
        """
        logger.info("Creating interaction features")
        
        # Select top features by variance
        top_features = X.var().nlargest(20).index
        
        interaction_features = []
        interaction_names = []
        
        # Create pairwise interactions
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                if len(interaction_features) >= max_features:
                    break
                
                # Multiplicative interaction
                interaction = X[feat1] * X[feat2]
                interaction_features.append(interaction)
                interaction_names.append(f"{feat1}_x_{feat2}")
        
        # Combine with original features
        interaction_df = pd.DataFrame(
            np.column_stack(interaction_features),
            columns=interaction_names,
            index=X.index
        )
        
        X_combined = pd.concat([X, interaction_df], axis=1)
        logger.info(f"Added {len(interaction_features)} interaction features")
        
        return X_combined
    
    def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series,
                             method: str = "smote") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced datasets using resampling techniques
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Resampling method ('smote', 'adasyn', 'undersample')
            
        Returns:
            Resampled feature matrix and target
        """
        logger.info(f"Handling imbalanced data using {method}")
        
        if method == "smote":
            sampler = SMOTE(random_state=42)
        elif method == "adasyn":
            sampler = ADASYN(random_state=42)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns, index=range(len(X_resampled)))
        y_resampled = pd.Series(y_resampled, index=range(len(y_resampled)))
        
        logger.info(f"Resampled data shape: {X_resampled.shape}")
        logger.info(f"Class distribution: {y_resampled.value_counts().to_dict()}")
        
        return X_resampled, y_resampled


class DataLoader:
    """
    Advanced data loader for various drug toxicity datasets
    """
    
    def __init__(self):
        self.datasets = DATASETS
    
    def load_tox21_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Tox21 dataset with preprocessing
        
        Args:
            file_path: Path to Tox21 dataset file
            
        Returns:
            Preprocessed Tox21 dataset
        """
        if file_path is None:
            file_path = self.datasets["tox21"]["local_path"]
        
        logger.info(f"Loading Tox21 dataset from {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # Handle other formats
                df = pd.read_excel(file_path)
            
            # Basic preprocessing
            logger.info(f"Original dataset shape: {df.shape}")
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values in toxicity columns
            toxicity_columns = [col for col in df.columns if any(x in col.lower() 
                               for x in ['toxicity', 'assay', 'nr-', 'sr-'])]
            
            for col in toxicity_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(-1)  # -1 for missing/unknown
            
            logger.info(f"Processed dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Tox21 dataset: {str(e)}")
            raise
    
    def load_zinc_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load ZINC250k dataset with preprocessing
        
        Args:
            file_path: Path to ZINC dataset file
            
        Returns:
            Preprocessed ZINC dataset
        """
        if file_path is None:
            file_path = self.datasets["zinc250k"]["local_path"]
        
        logger.info(f"Loading ZINC dataset from {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            logger.info(f"Original dataset shape: {df.shape}")
            
            # Basic preprocessing
            df = df.drop_duplicates()
            
            # Ensure SMILES column exists
            if 'smiles' not in df.columns and 'SMILES' not in df.columns:
                raise ValueError("No SMILES column found in dataset")
            
            # Standardize SMILES column name
            if 'smiles' in df.columns:
                df = df.rename(columns={'smiles': 'SMILES'})
            
            # Remove invalid SMILES
            valid_smiles = []
            for smiles in df['SMILES']:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
            
            df = df[df['SMILES'].isin(valid_smiles)]
            
            logger.info(f"Processed dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load ZINC dataset: {str(e)}")
            raise
    
    def create_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a sample dataset for testing
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Sample dataset with SMILES and random toxicity labels
        """
        logger.info(f"Creating sample dataset with {n_samples} molecules")
        
        # Sample SMILES strings (common drug-like molecules)
        sample_smiles = [
            "CC(C)OC(=O)C(C)(C)C",  # Ibuprofen-like
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN(C)C=O",  # Dimethylformamide
            "CCN(CC)CC",  # Triethylamine
            "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
            "CC(C)C1=CC=CC=C1",  # Cumene
            "C1CCCCC1",  # Cyclohexane
            "CC(=O)N",  # Acetamide
            "CCO",  # Ethanol
            "C1=CC=C(C=C1)N",  # Aniline
        ]
        
        # Generate random samples
        np.random.seed(42)
        random_smiles = np.random.choice(sample_smiles, n_samples)
        
        # Generate random toxicity labels (0: non-toxic, 1: toxic)
        toxicity_labels = np.random.binomial(1, 0.3, n_samples)  # 30% toxic
        
        # Create DataFrame
        df = pd.DataFrame({
            'SMILES': random_smiles,
            'Toxicity': toxicity_labels
        })
        
        logger.info(f"Created sample dataset: {df.shape}")
        return df


# Utility functions
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def standardize_smiles(smiles: str) -> Optional[str]:
    """Standardize SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None


def calculate_fingerprints(smiles_list: List[str], 
                          fp_type: str = "morgan",
                          radius: int = 2,
                          n_bits: int = 2048) -> np.ndarray:
    """
    Calculate molecular fingerprints
    
    Args:
        smiles_list: List of SMILES strings
        fp_type: Fingerprint type ('morgan', 'maccs', 'rdkit')
        radius: Morgan fingerprint radius
        n_bits: Number of fingerprint bits
        
    Returns:
        Fingerprint matrix
    """
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprints.append(np.zeros(n_bits))
            continue
        
        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "maccs":
            fp = Chem.GetMACCSKeysFingerprint(mol)
        elif fp_type == "rdkit":
            fp = Chem.RDKFingerprint(mol)
        
        fp_array = np.array(fp)
        fingerprints.append(fp_array)
    
    return np.array(fingerprints)
