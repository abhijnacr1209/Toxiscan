# Drug Toxicity Prediction Package
__version__ = "1.0.0"
__author__ = "Drug Toxicity Team"

# Import main classes for easy access
from .data_processing import MolecularDescriptorCalculator, FeatureEngineer, DataLoader
from .ml_models import EnsembleModel, DeepLearningTrainer, UncertaintyQuantifier
from .explainable_ai import SHAPExplainer, LIMEExplainer, MolecularHighlighter, ExplainabilityDashboard
from .molecular_viewer import MolecularViewer, MolecularAnimation, MolecularDashboard
from .molecule_builder import MoleculeBuilder, InteractiveMoleculeEditor
from .similarity_analyzer import ChemicalSimilarityAnalyzer, ChemicalSpaceVisualizer, ChemicalSpaceAnalyzer
from .chatbot import ToxicityChatbot
from .report_generator import ToxicityReportGenerator, BatchReportGenerator

__all__ = [
    # Data processing
    'MolecularDescriptorCalculator',
    'FeatureEngineer', 
    'DataLoader',
    
    # Machine learning
    'EnsembleModel',
    'DeepLearningTrainer',
    'UncertaintyQuantifier',
    
    # Explainable AI
    'SHAPExplainer',
    'LIMEExplainer',
    'MolecularHighlighter',
    'ExplainabilityDashboard',
    
    # Visualization
    'MolecularViewer',
    'MolecularAnimation',
    'MolecularDashboard',
    
    # Molecule builder
    'MoleculeBuilder',
    'InteractiveMoleculeEditor',
    
    # Similarity analysis
    'ChemicalSimilarityAnalyzer',
    'ChemicalSpaceVisualizer',
    'ChemicalSpaceAnalyzer',
    
    # Chatbot
    'ToxicityChatbot',
    
    # Report generation
    'ToxicityReportGenerator',
    'BatchReportGenerator'
]
