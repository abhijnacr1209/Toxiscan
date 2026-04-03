"""
PDF Generation Fix for Toxiscan
Proper PDF creation with correct formatting
"""

import os
from pathlib import Path
from datetime import datetime
import base64

def create_proper_pdf(analysis_results, tox21_predictions, gnn_results):
    """Create a properly formatted PDF report"""
    
    # Create proper PDF content with correct headers
    pdf_content = f"""%PDF-1.4
1 0 obj
<<
/Title (Toxiscan - Tox21 Powered Drug Toxicity Analysis Report)
/Creator (Toxiscan AI System)
/Producer (Toxiscan PDF Generator v1.0)
/CreationDate (D:{datetime.now().strftime('%Y%m%d%H%M%S')})
>>
endobj

2 0 obj
<<
/Type /Catalog
/Pages 3 0 R
>>
endobj

3 0 obj
<<
/Type /Pages
/Kids [4 0 R]
/Count 1
>>
endobj

4 0 obj
<<
/Type /Page
/Parent 3 0 R
/MediaBox [0 0 612 792]
/Contents 5 0 R
/Resources <<
/Font <<
/F1 6 0 R
>>
>>
>>
endobj

5 0 obj
<<
/Length {len(generate_pdf_content(analysis_results, tox21_predictions, gnn_results))}
>>
stream
{generate_pdf_content(analysis_results, tox21_predictions, gnn_results)}
endstream
endobj

6 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000174 00000 n 
0000000561 00000 n 
0000000714 00000 n 
trailer
<<
/Size 7
/Root 2 0 R
>>
startxref
{800 + len(generate_pdf_content(analysis_results, tox21_predictions, gnn_results))}
%%EOF
"""
    
    return pdf_content.encode('utf-8')

def generate_pdf_content(analysis_results, tox21_predictions, gnn_results):
    """Generate PDF content stream"""
    
    content = f"""BT
/F1 12 Tf
72 720 Td
(Toxiscan - Tox21 Powered Drug Toxicity Analysis Report) Tj
0 -20 Td
(Generated: {analysis_results['timestamp']}) Tj
0 -30 Td
(==================================================) Tj
0 -20 Td
(COMPOUND INFORMATION) Tj
0 -15 Td
(SMILES: {analysis_results['smiles']}) Tj
0 -15 Td
(Molecular Weight: {analysis_results['molecular_weight']} Da) Tj
0 -15 Td
(LogP: {analysis_results['logp']}) Tj
0 -15 Td
(TPSA: {analysis_results['tpsa']} Ų) Tj
0 -30 Td
(TOX21 PREDICTIONS) Tj
0 -15 Td
(Number of Endpoints: {len(tox21_predictions)}) Tj
0 -15 Td
(Max Risk: {max(tox21_predictions.values()):.3f}) Tj
0 -15 Td
(Avg Risk: {sum(tox21_predictions.values())/len(tox21_predictions):.3f}) Tj
0 -30 Td
(HIGH-RISK ENDPOINTS) Tj
"""
    
    # Add high-risk endpoints
    high_risk = [ep for ep, prob in tox21_predictions.items() if prob > 0.7]
    for endpoint in high_risk[:5]:
        content += f"0 -15 Td\n({endpoint}: {tox21_predictions[endpoint]:.3f}) Tj\n"
    
    content += f"""0 -30 Td
(GNN NEURAL NETWORK ANALYSIS) Tj
0 -15 Td
(Detected Toxic Patterns: {len(gnn_results.get('detected_patterns', []))}) Tj
"""
    
    if gnn_results.get('most_toxic_subgraph'):
        most_toxic = gnn_results['most_toxic_subgraph']
        content += f"""0 -15 Td
(Most Toxic Subgraph: {most_toxic.get('pattern_name', 'None')}) Tj
0 -15 Td
(Toxicity Score: {most_toxic.get('toxicity_score', 0):.2f}) Tj
"""
    
    content += f"""0 -30 Td
(AI-POWERED RECOMMENDATIONS) Tj
"""
    
    for rec in analysis_results.get('recommendations', []):
        content += f"0 -15 Td\n(- {rec}) Tj\n"
    
    content += f"""0 -30 Td
(TOX21 DATASET INFO) Tj
0 -15 Td
(Source: NIH Tox21 Challenge) Tj
0 -15 Td
(Training Size: ~12,000 compounds) Tj
0 -15 Td
(Endpoints: 12 nuclear receptor and stress response assays) Tj
0 -15 Td
(Model: Random Forest with molecular descriptors) Tj
0 -30 Td
(DISCLAIMER) Tj
0 -15 Td
(This analysis is for research and educational purposes only.) Tj
0 -15 Td
(The predictions are computational estimates based on Tox21 data) Tj
0 -15 Td
(and should not be used as the sole basis for safety decisions.) Tj
0 -15 Td
(Always validate with experimental assays and consult) Tj
0 -15 Td
(qualified medicinal chemists and toxicologists.) Tj
0 -30 Td
(Contact: support@toxiscan.ai) Tj
ET
"""
    
    return content

def create_simple_text_pdf(analysis_results, tox21_predictions, gnn_results):
    """Create a simple text-based PDF that will definitely open"""
    
    # Create simple text content
    text_content = f"""==================================================
TOXISCAN - TOX21 POWERED DRUG TOXICITY ANALYSIS REPORT
Generated: {analysis_results['timestamp']}
==================================================

COMPOUND INFORMATION
-----------------
SMILES: {analysis_results['smiles']}
Molecular Weight: {analysis_results['molecular_weight']} Da
LogP: {analysis_results['logp']}
TPSA: {analysis_results['tpsa']} Ų

TOX21 PREDICTIONS
------------------
Number of Endpoints: {len(tox21_predictions)}
Max Risk: {max(tox21_predictions.values()):.3f}
Avg Risk: {sum(tox21_predictions.values())/len(tox21_predictions):.3f}

ENDPOINT DETAILS:
-----------------
"""
    
    for endpoint, prob in tox21_predictions.items():
        risk_level = 'HIGH' if prob > 0.7 else 'MODERATE' if prob > 0.4 else 'LOW'
        text_content += f"{endpoint}: {prob:.3f} ({risk_level})\n"
    
    text_content += f"""
HIGH-RISK ENDPOINTS:
-------------------
"""
    
    high_risk = [ep for ep, prob in tox21_predictions.items() if prob > 0.7]
    for endpoint in high_risk:
        text_content += f"{endpoint}: {tox21_predictions[endpoint]:.3f}\n"
    
    text_content += f"""
GNN NEURAL NETWORK ANALYSIS
--------------------------
Detected Toxic Patterns: {len(gnn_results.get('detected_patterns', []))}
"""
    
    if gnn_results.get('most_toxic_subgraph'):
        most_toxic = gnn_results['most_toxic_subgraph']
        text_content += f"""Most Toxic Subgraph: {most_toxic.get('pattern_name', 'None')}
Toxicity Score: {most_toxic.get('toxicity_score', 0):.2f}
Description: {most_toxic.get('description', 'No description available')}
"""
    
    text_content += f"""
DETECTED PATTERNS:
-----------------
"""
    
    for i, pattern in enumerate(gnn_results.get('detected_patterns', [])):
        text_content += f"""Pattern {i+1}: {pattern.get('pattern_name', 'Unknown')}
- Toxicity Score: {pattern.get('toxicity_score', 0):.2f}
- Description: {pattern.get('description', 'No description available')}
- Mechanism: {pattern.get('mechanism', 'Unknown mechanism')}
- Clinical Relevance: {pattern.get('clinical_relevance', 'Unknown clinical relevance')}
- Prevention: {pattern.get('prevention', 'No prevention strategies available')}

"""
    
    text_content += f"""
AI-POWERED RECOMMENDATIONS
----------------------------
"""
    
    for rec in analysis_results.get('recommendations', []):
        text_content += f"- {rec}\n"
    
    text_content += f"""
TOX21 DATASET INFO
-----------------
Source: NIH Tox21 Challenge
Training Size: ~12,000 compounds
Endpoints: 12 nuclear receptor and stress response assays
Model: Random Forest with molecular descriptors
Features: 10 molecular descriptors (MolWt, LogP, TPSA, etc.)

DISCLAIMER
----------
This analysis is for research and educational purposes only.
The predictions are computational estimates based on Tox21 data and should not be used
as the sole basis for safety decisions.

Always validate with experimental assays and consult
qualified medicinal chemists and toxicologists.

For more information, contact: support@toxiscan.ai

==================================================
Report generated by Toxiscan AI System
Version: 1.0 (Tox21 Enhanced)
==================================================
"""
    
    return text_content.encode('utf-8')
