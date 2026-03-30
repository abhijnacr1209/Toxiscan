"""
Advanced AI Chatbot for Drug Toxicity Queries
Intelligent assistant with domain knowledge and contextual understanding
"""

import json
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import time

# NLP libraries (if available)
try:
    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("NLP libraries not available. Using rule-based responses.")

# Chemistry knowledge base
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToxicityChatbot:
    """
    Advanced AI chatbot specialized in drug toxicity and molecular chemistry
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize the toxicity chatbot
        
        Args:
            knowledge_base_path: Path to knowledge base JSON file
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.conversation_history = []
        self.context = {}
        
        # Initialize NLP models if available
        if NLP_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.similarity_threshold = 0.7
                logger.info("NLP models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load NLP models: {e}")
                NLP_AVAILABLE = False
        else:
            self.sentence_model = None
        
        # Initialize specialized response generators
        self.response_generators = {
            'molecular_properties': MolecularPropertyResponder(),
            'toxicity_mechanisms': ToxicityMechanismResponder(),
            'model_explanations': ModelExplanationResponder(),
            'chemical_concepts': ChemicalConceptResponder(),
            'prediction_interpretation': PredictionInterpreter(),
            'drug_development': DrugDevelopmentResponder()
        }
        
        logger.info("Toxicity chatbot initialized")
    
    def _load_knowledge_base(self, knowledge_base_path: Optional[str]) -> Dict:
        """Load knowledge base from file or create default"""
        if knowledge_base_path and Path(knowledge_base_path).exists():
            try:
                with open(knowledge_base_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
        
        # Default knowledge base
        return {
            "toxicity_concepts": {
                "logp": {
                    "definition": "LogP (octanol-water partition coefficient) measures lipophilicity",
                    "typical_range": "-1 to 5 for drug-like molecules",
                    "implications": {
                        "high": "High LogP (>3) indicates good membrane penetration but poor solubility",
                        "low": "Low LogP (<1) indicates good solubility but poor membrane penetration"
                    },
                    "toxicity_relevance": "Very high LogP (>5) can lead to bioaccumulation and increased toxicity"
                },
                "tpsa": {
                    "definition": "Topological Polar Surface Area predicts molecular polarity",
                    "typical_range": "0-200 Å² for drug-like molecules",
                    "implications": {
                        "high": "High TPSA (>140 Å²) may reduce oral bioavailability",
                        "low": "Low TPSA (<90 Å²) may enable CNS penetration"
                    },
                    "toxicity_relevance": "TPSA affects absorption, distribution, and metabolism"
                },
                "herg": {
                    "definition": "hERG (human Ether-à-go-go-Related Gene) potassium channel",
                    "function": "Critical for cardiac electrical activity",
                    "toxicity_relevance": "hERG inhibition causes QT prolongation and arrhythmias",
                    "clinical_importance": "Major cause of drug withdrawals and safety failures"
                },
                "pains": {
                    "definition": "PAINS (Pan-Assay Interference Compounds)",
                    "description": "Substructures that cause false positives in screening assays",
                    "examples": ["catechols", "rhodanines", "quinones"],
                    "toxicity_relevance": "Often flagged as toxic due to assay interference"
                },
                "lipinski": {
                    "definition": "Lipinski's Rule of Five for drug-likeness",
                    "rules": [
                        "Molecular weight ≤ 500 Da",
                        "LogP ≤ 5",
                        "H-bond donors ≤ 5",
                        "H-bond acceptors ≤ 10"
                    ],
                    "toxicity_relevance": "Violations often correlate with poor ADME and potential toxicity"
                }
            },
            "toxicity_mechanisms": {
                "reactive_metabolites": {
                    "description": "Metabolic activation produces reactive intermediates",
                    "examples": ["epoxides", "quinones", "imines"],
                    "consequences": "Protein adduct formation, DNA damage, oxidative stress"
                },
                "mitochondrial_toxicity": {
                    "description": "Disruption of mitochondrial function",
                    "mechanisms": ["Uncoupling oxidative phosphorylation", "Inhibiting electron transport"],
                    "consequences": "Energy depletion, cell death"
                },
                "dna_intercalation": {
                    "description": "Molecules insert between DNA base pairs",
                    "structural_features": ["Planar aromatic systems", "Intercalating functional groups"],
                    "consequences": "Mutagenesis, carcinogenesis"
                },
                "oxidative_stress": {
                    "description": "Generation of reactive oxygen species",
                    "mechanisms": ["Redox cycling", "Enzyme inhibition"],
                    "consequences": "Cellular damage, inflammation"
                }
            },
            "model_explanations": {
                "shap": {
                    "definition": "SHAP (SHapley Additive exPlanations)",
                    "purpose": "Explain individual predictions by allocating feature contributions",
                    "interpretation": {
                        "positive": "Feature increases toxicity risk",
                        "negative": "Feature decreases toxicity risk",
                        "magnitude": "Size of contribution to prediction"
                    }
                },
                "lime": {
                    "definition": "LIME (Local Interpretable Model-agnostic Explanations)",
                    "purpose": "Explain predictions using locally linear surrogate models",
                    "interpretation": "Shows which features most influence the specific prediction"
                },
                "feature_importance": {
                    "definition": "Global importance of features across the dataset",
                    "methods": ["Gini importance", "Permutation importance", "SHAP importance"],
                    "use": "Understanding which molecular properties drive toxicity predictions"
                }
            }
        }
    
    def process_query(self, user_message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user query and generate intelligent response
        
        Args:
            user_message: User's message
            context: Additional context (e.g., current prediction results)
            
        Returns:
            Response dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Update context
        if context:
            self.context.update(context)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': time.time()
        })
        
        # Analyze query intent
        intent = self._analyze_intent(user_message)
        
        # Generate response based on intent
        response = self._generate_response(user_message, intent, context)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'message': response['answer'],
            'timestamp': time.time(),
            'intent': intent,
            'confidence': response.get('confidence', 0.0)
        })
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        response['processing_time'] = time.time() - start_time
        return response
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent using keyword matching and NLP"""
        message_lower = message.lower()
        
        # Keyword-based intent classification
        intent_keywords = {
            'molecular_properties': [
                'logp', 'tpsa', 'molecular weight', 'lipinski', 'hbd', 'hba',
                'rotatable bonds', 'polar surface area', 'descriptors'
            ],
            'toxicity_mechanisms': [
                'toxicity', 'mechanism', 'reactive', 'metabolism', 'mitochondrial',
                'dna', 'oxidative', 'stress', 'herg', 'pains'
            ],
            'model_explanations': [
                'shap', 'lime', 'explain', 'feature importance', 'prediction',
                'model', 'accuracy', 'confidence', 'interpret'
            ],
            'chemical_concepts': [
                'smiles', 'fingerprint', 'scaffold', 'similarity', 'substructure',
                'functional group', 'aromatic', 'bond', 'atom'
            ],
            'prediction_interpretation': [
                'risk', 'score', 'probability', 'confidence', 'high risk',
                'low risk', 'moderate risk', 'toxic', 'non-toxic'
            ],
            'drug_development': [
                'drug', 'development', 'clinical', 'fda', 'approval', 'safety',
                'adme', 'pharmacokinetics', 'bioavailability'
            ]
        }
        
        # Count keyword matches
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            intent_scores[intent] = score
        
        # Return intent with highest score
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'general'
    
    def _generate_response(self, message: str, intent: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Generate response based on intent and context"""
        
        # Use specialized responder if available
        if intent in self.response_generators:
            return self.response_generators[intent].respond(message, context, self.knowledge_base)
        
        # Fallback to general response
        return self._generate_general_response(message, context)
    
    def _generate_general_response(self, message: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Generate general response using knowledge base"""
        message_lower = message.lower()
        
        # Search knowledge base for relevant information
        for category, concepts in self.knowledge_base.items():
            for concept, info in concepts.items():
                if concept in message_lower:
                    return {
                        'answer': self._format_concept_response(concept, info),
                        'confidence': 0.8,
                        'sources': [f"Knowledge base: {category}"]
                    }
        
        # Context-aware response
        if context and 'predictions' in context:
            return self._generate_contextual_response(message, context)
        
        # Fallback response
        return {
            'answer': self._generate_help_response(message),
            'confidence': 0.5,
            'sources': ['General knowledge']
        }
    
    def _format_concept_response(self, concept: str, info: Dict) -> str:
        """Format concept information into readable response"""
        response = f"**{concept.replace('_', ' ').title()}**\n\n"
        
        if 'definition' in info:
            response += f"**Definition:** {info['definition']}\n\n"
        
        if 'description' in info:
            response += f"**Description:** {info['description']}\n\n"
        
        if 'typical_range' in info:
            response += f"**Typical Range:** {info['typical_range']}\n\n"
        
        if 'implications' in info:
            response += "**Implications:**\n"
            if isinstance(info['implications'], dict):
                for key, value in info['implications'].items():
                    response += f"- {key.title()}: {value}\n"
            else:
                response += f"{info['implications']}\n"
            response += "\n"
        
        if 'toxicity_relevance' in info:
            response += f"**Toxicity Relevance:** {info['toxicity_relevance']}\n\n"
        
        if 'examples' in info:
            response += "**Examples:**\n"
            if isinstance(info['examples'], list):
                for example in info['examples']:
                    response += f"- {example}\n"
            else:
                response += f"{info['examples']}\n"
            response += "\n"
        
        if 'rules' in info:
            response += "**Rules:**\n"
            if isinstance(info['rules'], list):
                for rule in info['rules']:
                    response += f"- {rule}\n"
            else:
                response += f"{info['rules']}\n"
            response += "\n"
        
        return response
    
    def _generate_contextual_response(self, message: str, context: Dict) -> Dict[str, Any]:
        """Generate response based on current prediction context"""
        predictions = context.get('predictions', {})
        smiles = context.get('smiles', '')
        
        if not predictions:
            return {
                'answer': "I don't have prediction data to provide context-specific advice.",
                'confidence': 0.3
            }
        
        # Analyze prediction results
        max_prob = max(predictions.values()) if predictions else 0
        most_toxic = max(predictions, key=predictions.get) if predictions else 'Unknown'
        
        message_lower = message.lower()
        
        if 'risk' in message_lower or 'safe' in message_lower:
            risk_level = 'HIGH' if max_prob >= 0.7 else 'MODERATE' if max_prob >= 0.4 else 'LOW'
            
            response = f"""**Risk Assessment for Current Compound:**

**Overall Risk Level:** {risk_level}
**Highest Toxicity Probability:** {max_prob:.1%} for {most_toxic}

**Recommendations:**
"""
            
            if risk_level == 'HIGH':
                response += """- ⚠️ **HIGH RISK** - Avoid further development
- Consider significant structural modifications
- Conduct additional safety testing before proceeding
- Consult with toxicology experts"""
            elif risk_level == 'MODERATE':
                response += """- ⚡ **MODERATE RISK** - Proceed with caution
- Conduct additional in vitro assays
- Consider structural optimization
- Monitor safety parameters closely"""
            else:
                response += """- ✅ **LOW RISK** - Favorable safety profile
- Proceed to next development stage
- Continue with standard safety testing
- Maintain safety monitoring"""
            
            return {
                'answer': response,
                'confidence': 0.9,
                'sources': ['Current prediction analysis']
            }
        
        elif 'explain' in message_lower or 'why' in message_lower:
            # Generate explanation based on available feature importance
            response = f"""**Explanation for Current Prediction:**

The model predicts a {max_prob:.1%} probability of toxicity for the {most_toxic} endpoint.

**Key Factors:**
- Molecular properties contribute to this risk assessment
- The model has learned patterns from the Tox21 dataset
- Individual feature contributions can be viewed in the Explainable AI section

**For detailed explanations:**
- Use the SHAP analysis in the Explainable AI tab
- Review feature importance plots
- Check molecular highlighting for atom-level contributions"""
            
            return {
                'answer': response,
                'confidence': 0.8,
                'sources': ['Model interpretation']
            }
        
        else:
            return {
                'answer': f"""I can see you're asking about the current compound with SMILES: {smiles[:30]}...

The prediction shows {max_prob:.1%} maximum toxicity probability. 

Would you like me to:
- Explain the risk assessment?
- Discuss specific toxicity mechanisms?
- Analyze molecular properties?
- Suggest optimization strategies?

Please let me know what specific aspect you'd like to explore!""",
                'confidence': 0.7,
                'sources': ['Context analysis']
            }
    
    def _generate_help_response(self, message: str) -> str:
        """Generate helpful response for unrecognized queries"""
        return f"""I understand you're asking about: "{message}"

I'm here to help with drug toxicity and molecular chemistry questions. Here's what I can assist with:

**🧬 Molecular Properties:**
- LogP, TPSA, molecular weight, Lipinski's rules
- Descriptors and their implications for toxicity

**⚡ Toxicity Mechanisms:**
- hERG inhibition, reactive metabolites
- Mitochondrial toxicity, DNA intercalation
- Oxidative stress and PAINS compounds

**🤖 Model Explanations:**
- SHAP values and feature importance
- How to interpret prediction scores
- Model accuracy and limitations

**💊 Drug Development:**
- Safety assessment strategies
- ADME considerations
- Regulatory requirements

**Try asking about:**
- "What is LogP and how does it affect toxicity?"
- "Explain SHAP values"
- "What makes a molecule toxic?"
- "How accurate are the predictions?"

Could you rephrase your question or choose one of these topics?"""
    
    def get_conversation_summary(self) -> str:
        """Get summary of current conversation"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = "**Conversation Summary:**\n\n"
        
        # Extract key topics
        user_messages = [msg['message'] for msg in self.conversation_history if msg['role'] == 'user']
        intents = [msg.get('intent', 'general') for msg in self.conversation_history if msg['role'] == 'assistant']
        
        # Count intents
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        summary += f"**Messages exchanged:** {len(self.conversation_history)}\n"
        summary += f"**Topics discussed:** {', '.join(set(intents))}\n\n"
        
        # Recent questions
        recent_questions = user_messages[-3:] if len(user_messages) > 3 else user_messages
        if recent_questions:
            summary += "**Recent questions:**\n"
            for i, question in enumerate(recent_questions, 1):
                summary += f"{i}. {question}\n"
        
        return summary
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.context = {}
        logger.info("Conversation history cleared")


class SpecializedResponder:
    """Base class for specialized response generators"""
    
    def respond(self, message: str, context: Optional[Dict], knowledge_base: Dict) -> Dict[str, Any]:
        """Generate specialized response"""
        raise NotImplementedError


class MolecularPropertyResponder(SpecializedResponder):
    """Responder for molecular property questions"""
    
    def respond(self, message: str, context: Optional[Dict], knowledge_base: Dict) -> Dict[str, Any]:
        message_lower = message.lower()
        
        if 'logp' in message_lower:
            return {
                'answer': self._explain_logp(knowledge_base),
                'confidence': 0.9,
                'sources': ['Molecular chemistry knowledge']
            }
        
        elif 'tpsa' in message_lower:
            return {
                'answer': self._explain_tpsa(knowledge_base),
                'confidence': 0.9,
                'sources': ['Molecular chemistry knowledge']
            }
        
        elif 'lipinski' in message_lower:
            return {
                'answer': self._explain_lipinski(knowledge_base),
                'confidence': 0.9,
                'sources': ['Drug discovery knowledge']
            }
        
        elif context and 'smiles' in context:
            return self._analyze_molecule_properties(context['smiles'], knowledge_base)
        
        return {
            'answer': "I can help with molecular properties like LogP, TPSA, and Lipinski's rules. What specific property would you like to know about?",
            'confidence': 0.6
        }
    
    def _explain_logp(self, knowledge_base: Dict) -> str:
        info = knowledge_base['toxicity_concepts']['logp']
        return f"""**LogP (Partition Coefficient)**

{info['definition']}

**Key Points:**
- {info['typical_range']}
- {info['implications']['high']}
- {info['implications']['low']}

**Toxicity Relevance:**
{info['toxicity_relevance']}

**Optimal Range for Oral Drugs:** 1-3 (Lipinski: ≤5)"""
    
    def _explain_tpsa(self, knowledge_base: Dict) -> str:
        info = knowledge_base['toxicity_concepts']['tpsa']
        return f"""**TPSA (Topological Polar Surface Area)**

{info['definition']}

**Key Points:**
- {info['typical_range']}
- {info['implications']['high']}
- {info['implications']['low']}

**Toxicity Relevance:**
{info['toxicity_relevance']}

**Guidelines:**
- ≤140 Å²: Good oral bioavailability (Veber rule)
- ≤90 Å²: Potential CNS penetration"""
    
    def _explain_lipinski(self, knowledge_base: Dict) -> str:
        info = knowledge_base['toxicity_concepts']['lipinski']
        return f"""**Lipinski's Rule of Five**

{info['definition']}

**The Rules:**
{chr(10).join(f"- {rule}" for rule in info['rules'])}

**Toxicity Relevance:**
{info['toxicity_relevance']}

**Note:** More than 1 violation suggests potential ADME/toxicity issues"""
    
    def _analyze_molecule_properties(self, smiles: str, knowledge_base: Dict) -> Dict[str, Any]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    'answer': "Invalid SMILES string provided.",
                    'confidence': 0.1
                }
            
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Check Lipinski rules
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            
            response = f"""**Molecular Properties Analysis for: {smiles[:30]}...**

**Calculated Properties:**
- Molecular Weight: {mw:.1f} Da
- LogP: {logp:.2f}
- TPSA: {tpsa:.1f} Å²
- H-Bond Donors: {hbd}
- H-Bond Acceptors: {hba}
- Rotatable Bonds: {rotatable_bonds}

**Lipinski's Rule of Five:**
- Violations: {violations}/4
- Drug-likeness: {'Good' if violations <= 1 else 'Poor' if violations == 2 else 'Very Poor'}

**Toxicity Implications:**
"""
            
            if logp > 5:
                response += "- ⚠️ High LogP may increase bioaccumulation risk\n"
            if tpsa > 140:
                response += "- ⚠️ High TPSA may reduce oral bioavailability\n"
            if violations > 2:
                response += "- ⚠️ Multiple Lipinski violations suggest potential toxicity issues\n"
            
            if violations == 0:
                response += "- ✅ Favorable drug-like properties\n"
            
            return {
                'answer': response,
                'confidence': 0.95,
                'sources': ['RDKit calculations', 'Lipinski rules']
            }
            
        except Exception as e:
            return {
                'answer': f"Error analyzing molecular properties: {str(e)}",
                'confidence': 0.1
            }


class ToxicityMechanismResponder(SpecializedResponder):
    """Responder for toxicity mechanism questions"""
    
    def respond(self, message: str, context: Optional[Dict], knowledge_base: Dict) -> Dict[str, Any]:
        message_lower = message.lower()
        
        if 'herg' in message_lower:
            return {
                'answer': self._explain_herg(knowledge_base),
                'confidence': 0.9,
                'sources': ['Cardiac toxicology knowledge']
            }
        
        elif 'pains' in message_lower:
            return {
                'answer': self._explain_pains(knowledge_base),
                'confidence': 0.9,
                'sources': ['Assay interference knowledge']
            }
        
        elif 'reactive' in message_lower or 'metabolite' in message_lower:
            return {
                'answer': self._explain_reactive_metabolites(knowledge_base),
                'confidence': 0.9,
                'sources': ['Metabolic toxicology knowledge']
            }
        
        return {
            'answer': "I can explain various toxicity mechanisms including hERG inhibition, reactive metabolites, and PAINS compounds. What specific mechanism interests you?",
            'confidence': 0.6
        }
    
    def _explain_herg(self, knowledge_base: Dict) -> str:
        info = knowledge_base['toxicity_concepts']['herg']
        return f"""**hERG Channel Inhibition**

{info['definition']}

**Function:**
{info['function']}

**Toxicity Relevance:**
{info['toxicity_relevance']}

**Clinical Importance:**
{info['clinical_importance']}

**Risk Factors:**
- Basic, planar molecules
- Lipophilic compounds
- Certain aromatic heterocycles

**Testing:**
- In vitro hERG assays
- In vivo cardiac safety studies
- QT interval monitoring"""
    
    def _explain_pains(self, knowledge_base: Dict) -> str:
        info = knowledge_base['toxicity_concepts']['pains']
        return f"""**PAINS (Pan-Assay Interference Compounds)**

{info['definition']}

**Common PAINS Substructures:**
{', '.join(info['examples'])}

**Toxicity Relevance:**
{info['toxicity_relevance']}

**Why They're Problematic:**
- Give false positive results in screening
- Often appear toxic due to assay interference
- Waste resources in drug discovery

**Detection:**
- SMARTS pattern matching
- Computational filters
- Experimental validation"""
    
    def _explain_reactive_metabolites(self, knowledge_base: Dict) -> str:
        info = knowledge_base['toxicity_mechanisms']['reactive_metabolites']
        return f"""**Reactive Metabolites**

{info['description']}

**Common Reactive Species:**
{', '.join(info['examples'])}

**Formation Pathways:**
- Cytochrome P450 oxidation
- Phase I metabolism
- Environmental activation

**Consequences:**
{info['consequences']}

**Risk Mitigation:**
- Structural modification to remove metabolic hotspots
- In vitro metabolite identification
- Reactive metabolite trapping studies"""


class ModelExplanationResponder(SpecializedResponder):
    """Responder for model explanation questions"""
    
    def respond(self, message: str, context: Optional[Dict], knowledge_base: Dict) -> Dict[str, Any]:
        message_lower = message.lower()
        
        if 'shap' in message_lower:
            return {
                'answer': self._explain_shap(knowledge_base),
                'confidence': 0.9,
                'sources': ['Explainable AI knowledge']
            }
        
        elif 'lime' in message_lower:
            return {
                'answer': self._explain_lime(knowledge_base),
                'confidence': 0.9,
                'sources': ['Explainable AI knowledge']
            }
        
        elif 'feature importance' in message_lower:
            return {
                'answer': self._explain_feature_importance(knowledge_base),
                'confidence': 0.9,
                'sources': ['Machine learning knowledge']
            }
        
        return {
            'answer': "I can explain SHAP values, LIME explanations, and feature importance in toxicity models. What would you like to know?",
            'confidence': 0.6
        }
    
    def _explain_shap(self, knowledge_base: Dict) -> str:
        info = knowledge_base['model_explanations']['shap']
        return f"""**SHAP (SHapley Additive exPlanations)**

{info['purpose']}

**How It Works:**
{info['definition']}

**Interpretation:**
{chr(10).join(f"- {key}: {value}" for key, value in info['interpretation'].items())}

**Benefits:**
- Based on solid game theory
- Provides both local and global explanations
- Consistent and accurate feature attribution

**In Toxicity Prediction:**
- Shows which molecular features increase/decrease toxicity
- Helps understand model decision-making
- Guides molecular optimization"""
    
    def _explain_lime(self, knowledge_base: Dict) -> str:
        info = knowledge_base['model_explanations']['lime']
        return f"""**LIME (Local Interpretable Model-agnostic Explanations)**

{info['purpose']}

**How It Works:**
{info['definition']}

**Interpretation:**
{info['interpretation']}

**Benefits:**
- Model-agnostic (works with any ML model)
- Easy to understand explanations
- Focuses on individual predictions

**In Toxicity Prediction:**
- Explains why a specific compound is predicted as toxic
- Identifies key features for that prediction
- Helps validate model reasoning"""
    
    def _explain_feature_importance(self, knowledge_base: Dict) -> str:
        info = knowledge_base['model_explanations']['feature_importance']
        return f"""**Feature Importance**

{info['use']}

**Common Methods:**
{chr(10).join(f"- {method}" for method in info['methods'])}

**Interpretation:**
- Higher importance = more influence on predictions
- Global perspective across all compounds
- Helps identify key toxicity drivers

**In Toxicity Models:**
- Reveals which molecular properties matter most
- Guides feature selection and engineering
- Provides insights into toxicity mechanisms"""


class ChemicalConceptResponder(SpecializedResponder):
    """Responder for chemical concept questions"""
    
    def respond(self, message: str, context: Optional[Dict], knowledge_base: Dict) -> Dict[str, Any]:
        message_lower = message.lower()
        
        if 'smiles' in message_lower:
            return {
                'answer': self._explain_smiles(),
                'confidence': 0.9,
                'sources': ['Chemical informatics knowledge']
            }
        
        elif 'fingerprint' in message_lower:
            return {
                'answer': self._explain_fingerprints(),
                'confidence': 0.9,
                'sources': ['Chemical informatics knowledge']
            }
        
        elif 'scaffold' in message_lower:
            return {
                'answer': self._explain_scaffolds(),
                'confidence': 0.9,
                'sources': ['Medicinal chemistry knowledge']
            }
        
        return {
            'answer': "I can explain chemical concepts like SMILES notation, molecular fingerprints, and scaffolds. What would you like to know?",
            'confidence': 0.6
        }
    
    def _explain_smiles(self) -> str:
        return """**SMILES (Simplified Molecular Input Line Entry System)**

SMILES is a text-based notation for representing molecular structures.

**Basic Rules:**
- Atoms: Element symbols (C, N, O, etc.)
- Bonds: Single (=), Double (=), Triple (#), Aromatic (:)
- Branches: Parentheses ()
- Rings: Numbers to indicate connections
- Charges: + or - with numbers

**Examples:**
- Water: O
- Methane: C
- Ethanol: CCO
- Benzene: c1ccccc1
- Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O

**In Toxicity Prediction:**
- Input format for molecular structures
- Enables computational analysis
- Standardized representation"""
    
    def _explain_fingerprints(self) -> str:
        return """**Molecular Fingerprints**

Fingerprints are binary vectors representing molecular structure and features.

**Common Types:**
- **Morgan (ECFP):** Circular fingerprints around each atom
- **MACCS:** 166 predefined structural keys
- **RDKit:** Path-based fingerprints

**How They Work:**
- Encode presence/absence of substructures
- Enable rapid similarity calculations
- Used in machine learning models

**In Toxicity Prediction:**
- Capture structural features related to toxicity
- Enable similarity-based predictions
- Foundation for pattern recognition"""
    
    def _explain_scaffolds(self) -> str:
        return """**Molecular Scaffolds**

Scaffolds represent the core ring system of molecules.

**Murcko Scaffold:**
- Extracts ring systems and linkers
- Removes side chains
- Identifies structural framework

**Applications:**
- Scaffold hopping in drug design
- Chemical space analysis
- Series identification in medicinal chemistry

**In Toxicity Prediction:**
- Group compounds by structural class
- Identify toxicity trends across scaffolds
- Guide scaffold optimization strategies"""


class PredictionInterpreter(SpecializedResponder):
    """Responder for prediction interpretation questions"""
    
    def respond(self, message: str, context: Optional[Dict], knowledge_base: Dict) -> Dict[str, Any]:
        if not context or 'predictions' not in context:
            return {
                'answer': "I need prediction data to provide interpretation. Please make a prediction first.",
                'confidence': 0.3
            }
        
        return self._interpret_predictions(context)
    
    def _interpret_predictions(self, context: Dict) -> Dict[str, Any]:
        predictions = context['predictions']
        smiles = context.get('smiles', 'Unknown compound')
        
        # Analyze predictions
        max_prob = max(predictions.values()) if predictions else 0
        most_toxic = max(predictions, key=predictions.get) if predictions else 'Unknown'
        
        # Risk categorization
        risk_level = 'HIGH' if max_prob >= 0.7 else 'MODERATE' if max_prob >= 0.4 else 'LOW'
        
        response = f"""**Prediction Interpretation for: {smiles[:30]}...**

**Risk Assessment:**
- **Overall Risk Level:** {risk_level}
- **Highest Risk:** {most_toxic} ({max_prob:.1%})
- **Confidence:** {max_prob:.1%}

**Interpretation Guidelines:**
- **LOW Risk (0-40%):** Generally safe, proceed with standard testing
- **MODERATE Risk (40-70%):** Requires additional safety evaluation
- **HIGH Risk (70-100%):** Significant safety concerns, consider alternatives

**Next Steps:**
"""
        
        if risk_level == 'HIGH':
            response += """- ⚠️ Conduct thorough safety assessment
- Consider structural modification
- Evaluate alternative compounds"""
        elif risk_level == 'MODERATE':
            response += """- ⚡ Perform additional in vitro assays
- Consider optimization strategies
- Monitor safety parameters"""
        else:
            response += """- ✅ Proceed with development
- Continue standard safety testing
- Maintain safety monitoring"""
        
        return {
            'answer': response,
            'confidence': 0.9,
            'sources': ['Prediction analysis']
        }


class DrugDevelopmentResponder(SpecializedResponder):
    """Responder for drug development questions"""
    
    def respond(self, message: str, context: Optional[Dict], knowledge_base: Dict) -> Dict[str, Any]:
        message_lower = message.lower()
        
        if 'adme' in message_lower:
            return {
                'answer': self._explain_adme(),
                'confidence': 0.9,
                'sources': ['Pharmacology knowledge']
            }
        
        elif 'clinical' in message_lower or 'fda' in message_lower:
            return {
                'answer': self._explain_clinical_development(),
                'confidence': 0.9,
                'sources': ['Drug development knowledge']
            }
        
        elif 'safety' in message_lower:
            return {
                'answer': self._explain_safety_assessment(),
                'confidence': 0.9,
                'sources': ['Toxicology knowledge']
            }
        
        return {
            'answer': "I can help with drug development topics including ADME, clinical trials, and safety assessment. What would you like to know?",
            'confidence': 0.6
        }
    
    def _explain_adme(self) -> str:
        return """**ADME (Absorption, Distribution, Metabolism, Excretion)**

ADME properties determine how drugs behave in the body.

**Absorption:**
- How drug enters bloodstream
- Affected by solubility, permeability
- Key for oral bioavailability

**Distribution:**
- How drug spreads through body
- Affected by protein binding, lipophilicity
- Determines tissue concentrations

**Metabolism:**
- How body breaks down drug
- Mainly in liver (CYP enzymes)
- Can produce reactive metabolites

**Excretion:**
- How drug leaves body
- Mainly via kidneys, liver
- Affects dosing frequency

**Toxicity Relevance:**
- Poor ADME can lead to accumulation
- Metabolic activation can create toxic species
- Individual variations affect safety"""
    
    def _explain_clinical_development(self) -> str:
        return """**Clinical Development Process**

**Phase I:**
- Small group of healthy volunteers
- Assess safety, tolerability
- Determine dosing range

**Phase II:**
- Larger group of patients
- Evaluate efficacy, side effects
- Optimize dosing

**Phase III:**
- Large-scale patient studies
- Confirm efficacy, safety
- Compare to existing treatments

**Phase IV:**
- Post-marketing surveillance
- Long-term safety monitoring
- Rare adverse event detection

**Toxicity Testing:**
- Preclinical safety studies required
- GLP-compliant toxicology
- Regulatory safety packages"""
    
    def _explain_safety_assessment(self) -> str:
        return """**Drug Safety Assessment**

**In Vitro Testing:**
- Cytotoxicity assays
- hERG channel testing
- Metabolic stability
- Genotoxicity testing

**In Vivo Testing:**
- Acute toxicity studies
- Subchronic/chronic toxicity
- Organ-specific toxicity
- Carcinogenicity studies

**Safety Pharmacology:**
- Cardiovascular safety
- CNS safety
- Respiratory safety

**Regulatory Requirements:**
- ICH guidelines for safety testing
- GLP compliance
- Comprehensive safety packages
- Risk-benefit analysis"""
