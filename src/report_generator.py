"""
Advanced Report Generation Module
Creates comprehensive PDF/HTML reports with molecular visualizations and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings
from datetime import datetime
import base64
from io import BytesIO

# Report generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# HTML generation
import jinja2
from weasyprint import HTML, CSS

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDraw2D
from rdkit.Chem.Draw import rdMolDraw2D

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ToxicityReportGenerator:
    """
    Advanced report generator for drug toxicity analysis
    """
    
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Report templates
        self.template_env = jinja2.Environment(
            loader=jinja2.DictLoader({
                'html_report': self._get_html_template(),
                'executive_summary': self._get_executive_summary_template()
            })
        )
        
        logger.info("Report generator initialized")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for reports"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.black,
            leftIndent=20
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leading=14
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            backColor=colors.yellow,
            borderColor=colors.red,
            borderWidth=1,
            borderPadding=5
        ))
        
        # Success style
        self.styles.add(ParagraphStyle(
            name='Success',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.green,
            backColor=colors.lightgreen,
            borderColor=colors.green,
            borderWidth=1,
            borderPadding=5
        ))
    
    def generate_comprehensive_report(self,
                                    smiles: str,
                                    predictions: Dict[str, float],
                                    molecular_properties: Dict[str, float],
                                    explanations: Optional[Dict] = None,
                                    similar_compounds: Optional[List[Dict]] = None,
                                    structural_alerts: Optional[Dict] = None,
                                    output_format: str = "pdf") -> str:
        """
        Generate comprehensive toxicity report
        
        Args:
            smiles: SMILES string of the compound
            predictions: Toxicity predictions
            molecular_properties: Molecular properties
            explanations: Model explanations (SHAP/LIME)
            similar_compounds: List of similar compounds
            structural_alerts: Structural alerts
            output_format: Output format ('pdf' or 'html')
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"toxicity_report_{timestamp}.{output_format}"
        output_path = self.output_dir / filename
        
        logger.info(f"Generating comprehensive report: {filename}")
        
        # Prepare report data
        report_data = {
            'smiles': smiles,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'predictions': predictions,
            'properties': molecular_properties,
            'explanations': explanations or {},
            'similar_compounds': similar_compounds or [],
            'structural_alerts': structural_alerts or {},
            'risk_assessment': self._calculate_risk_assessment(predictions),
            'recommendations': self._generate_recommendations(predictions, molecular_properties)
        }
        
        # Generate report based on format
        if output_format == "pdf":
            self._generate_pdf_report(report_data, output_path)
        elif output_format == "html":
            self._generate_html_report(report_data, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Report generated successfully: {output_path}")
        return str(output_path)
    
    def _calculate_risk_assessment(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment"""
        if not predictions:
            return {'risk_level': 'UNKNOWN', 'confidence': 0.0}
        
        max_prob = max(predictions.values())
        avg_prob = np.mean(list(predictions.values()))
        
        # Determine risk level
        if max_prob >= 0.7:
            risk_level = 'HIGH'
            risk_color = 'red'
            description = 'High probability of toxicity - significant safety concerns'
        elif max_prob >= 0.4:
            risk_level = 'MODERATE'
            risk_color = 'orange'
            description = 'Moderate toxicity risk - requires further evaluation'
        else:
            risk_level = 'LOW'
            risk_color = 'green'
            description = 'Low toxicity risk - favorable safety profile'
        
        return {
            'risk_level': risk_level,
            'risk_color': risk_color,
            'max_probability': max_prob,
            'average_probability': avg_prob,
            'confidence': max_prob,  # Simplified confidence
            'description': description,
            'highest_risk_endpoint': max(predictions, key=predictions.get)
        }
    
    def _generate_recommendations(self, predictions: Dict[str, float], 
                                properties: Dict[str, float]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        max_prob = max(predictions.values()) if predictions else 0
        
        if max_prob >= 0.7:
            recommendations.extend([
                "⚠️ AVOID further development without significant structural modification",
                "Conduct comprehensive in vitro toxicity testing",
                "Consider alternative chemical scaffolds",
                "Consult with toxicology experts before proceeding"
            ])
        elif max_prob >= 0.4:
            recommendations.extend([
                "⚡ Proceed with CAUTION - additional safety testing required",
                "Conduct targeted toxicity assays for high-risk endpoints",
                "Consider structural optimization to reduce toxicity",
                "Monitor safety parameters closely in preclinical studies"
            ])
        else:
            recommendations.extend([
                "✅ LOW RISK - proceed with standard development pathway",
                "Continue with routine safety assessment",
                "Monitor for unexpected toxicities in later stages",
                "Maintain standard safety testing protocols"
            ])
        
        # Property-based recommendations
        if properties:
            if properties.get('LogP', 0) > 5:
                recommendations.append("Consider reducing LogP to improve safety profile")
            
            if properties.get('TPSA', 0) > 140:
                recommendations.append("High TPSA may affect bioavailability - consider optimization")
            
            if properties.get('MolWt', 0) > 500:
                recommendations.append("High molecular weight - consider reducing size")
        
        return recommendations
    
    def _generate_pdf_report(self, report_data: Dict, output_path: Path):
        """Generate PDF report"""
        doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=72, leftMargin=72)
        story = []
        
        # Title page
        story.extend(self._create_title_page(report_data))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(report_data))
        story.append(PageBreak())
        
        # Molecular information
        story.extend(self._create_molecular_section(report_data))
        story.append(PageBreak())
        
        # Toxicity predictions
        story.extend(self._create_predictions_section(report_data))
        story.append(PageBreak())
        
        # Risk assessment
        story.extend(self._create_risk_assessment_section(report_data))
        story.append(PageBreak())
        
        # Explanations (if available)
        if report_data['explanations']:
            story.extend(self._create_explanations_section(report_data))
            story.append(PageBreak())
        
        # Similar compounds (if available)
        if report_data['similar_compounds']:
            story.extend(self._create_similar_compounds_section(report_data))
            story.append(PageBreak())
        
        # Recommendations
        story.extend(self._create_recommendations_section(report_data))
        
        # Build PDF
        doc.build(story)
    
    def _create_title_page(self, report_data: Dict) -> List:
        """Create title page"""
        elements = []
        
        # Title
        elements.append(Paragraph("DRUG TOXICITY ASSESSMENT REPORT", self.styles['CustomTitle']))
        elements.append(Spacer(1, 50))
        
        # Compound information
        elements.append(Paragraph(f"Compound: {report_data['smiles']}", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 30))
        
        # Report metadata
        elements.append(Paragraph(f"Generated: {report_data['timestamp']}", self.styles['CustomBody']))
        elements.append(Spacer(1, 20))
        
        # Risk level indicator
        risk_assessment = report_data['risk_assessment']
        risk_style = self.styles['Warning'] if risk_assessment['risk_level'] == 'HIGH' else \
                    self.styles['Success'] if risk_assessment['risk_level'] == 'LOW' else \
                    self.styles['Normal']
        
        elements.append(Paragraph(f"RISK LEVEL: {risk_assessment['risk_level']}", risk_style))
        elements.append(Paragraph(risk_assessment['description'], self.styles['CustomBody']))
        
        return elements
    
    def _create_executive_summary(self, report_data: Dict) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 20))
        
        # Summary table
        summary_data = [
            ['Compound', report_data['smiles'][:50] + '...' if len(report_data['smiles']) > 50 else report_data['smiles']],
            ['Risk Level', report_data['risk_assessment']['risk_level']],
            ['Max Toxicity Probability', f"{report_data['risk_assessment']['max_probability']:.3f}"],
            ['Highest Risk Endpoint', report_data['risk_assessment']['highest_risk_endpoint']],
            ['Analysis Date', report_data['timestamp']]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Key findings
        elements.append(Paragraph("KEY FINDINGS", self.styles['SectionHeader']))
        
        findings = []
        risk_assessment = report_data['risk_assessment']
        
        findings.append(f"• Overall toxicity risk: {risk_assessment['risk_level']}")
        findings.append(f"• Maximum toxicity probability: {risk_assessment['max_probability']:.1%}")
        findings.append(f"• Primary concern: {risk_assessment['highest_risk_endpoint']}")
        
        if report_data['predictions']:
            high_risk_endpoints = [k for k, v in report_data['predictions'].items() if v > 0.5]
            if high_risk_endpoints:
                findings.append(f"• High-risk endpoints: {', '.join(high_risk_endpoints)}")
        
        for finding in findings:
            elements.append(Paragraph(finding, self.styles['CustomBody']))
        
        return elements
    
    def _create_molecular_section(self, report_data: Dict) -> List:
        """Create molecular information section"""
        elements = []
        
        elements.append(Paragraph("MOLECULAR INFORMATION", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 20))
        
        # SMILES and structure
        elements.append(Paragraph("Structure", self.styles['SectionHeader']))
        elements.append(Paragraph(f"SMILES: {report_data['smiles']}", self.styles['CustomBody']))
        
        # Add molecular structure image
        try:
            mol = Chem.MolFromSmiles(report_data['smiles'])
            if mol:
                # Generate 2D structure
                drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()
                
                # Save to temporary file and add to report
                temp_img_path = self.output_dir / "temp_molecule.png"
                drawer.WriteToFile(str(temp_img_path))
                
                img = Image(str(temp_img_path), width=4*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 20))
                
                # Clean up
                temp_img_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not generate molecular structure: {e}")
        
        # Molecular properties
        if report_data['properties']:
            elements.append(Paragraph("Molecular Properties", self.styles['SectionHeader']))
            
            prop_data = [['Property', 'Value']]
            for prop_name, prop_value in report_data['properties'].items():
                if isinstance(prop_value, float):
                    prop_data.append([prop_name, f"{prop_value:.3f}"])
                else:
                    prop_data.append([prop_name, str(prop_value)])
            
            prop_table = Table(prop_data, colWidths=[2*inch, 2*inch])
            prop_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(prop_table)
        
        return elements
    
    def _create_predictions_section(self, report_data: Dict) -> List:
        """Create toxicity predictions section"""
        elements = []
        
        elements.append(Paragraph("TOXICITY PREDICTIONS", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 20))
        
        if not report_data['predictions']:
            elements.append(Paragraph("No prediction data available.", self.styles['CustomBody']))
            return elements
        
        # Create predictions table
        pred_data = [['Toxicity Endpoint', 'Probability', 'Risk Level']]
        
        for endpoint, probability in report_data['predictions'].items():
            if probability >= 0.7:
                risk = 'HIGH'
                color = colors.red
            elif probability >= 0.4:
                risk = 'MODERATE'
                color = colors.orange
            else:
                risk = 'LOW'
                color = colors.green
            
            pred_data.append([endpoint, f"{probability:.3f}", risk])
        
        pred_table = Table(pred_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(pred_table)
        elements.append(Spacer(1, 20))
        
        # Create prediction chart
        try:
            chart_path = self._create_prediction_chart(report_data['predictions'])
            if chart_path and Path(chart_path).exists():
                img = Image(chart_path, width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 10))
                
                # Clean up
                Path(chart_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not create prediction chart: {e}")
        
        return elements
    
    def _create_risk_assessment_section(self, report_data: Dict) -> List:
        """Create risk assessment section"""
        elements = []
        
        elements.append(Paragraph("RISK ASSESSMENT", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 20))
        
        risk_assessment = report_data['risk_assessment']
        
        # Risk level with color coding
        risk_style = self.styles['Warning'] if risk_assessment['risk_level'] == 'HIGH' else \
                    self.styles['Success'] if risk_assessment['risk_level'] == 'LOW' else \
                    self.styles['Normal']
        
        elements.append(Paragraph(f"Overall Risk Level: {risk_assessment['risk_level']}", risk_style))
        elements.append(Spacer(1, 15))
        
        elements.append(Paragraph("Assessment Details", self.styles['SectionHeader']))
        
        details = [
            f"• Maximum toxicity probability: {risk_assessment['max_probability']:.1%}",
            f"• Average toxicity probability: {risk_assessment['average_probability']:.1%}",
            f"• Highest risk endpoint: {risk_assessment['highest_risk_endpoint']}",
            f"• Assessment confidence: {risk_assessment['confidence']:.1%}"
        ]
        
        for detail in details:
            elements.append(Paragraph(detail, self.styles['CustomBody']))
        
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Risk Interpretation", self.styles['SectionHeader']))
        elements.append(Paragraph(risk_assessment['description'], self.styles['CustomBody']))
        
        return elements
    
    def _create_explanations_section(self, report_data: Dict) -> List:
        """Create model explanations section"""
        elements = []
        
        elements.append(Paragraph("MODEL EXPLANATIONS", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 20))
        
        explanations = report_data['explanations']
        
        if not explanations:
            elements.append(Paragraph("No explanation data available.", self.styles['CustomBody']))
            return elements
        
        # SHAP explanations
        if 'shap' in explanations:
            elements.append(Paragraph("SHAP Analysis", self.styles['SectionHeader']))
            shap_data = explanations['shap']
            
            if 'feature_importance' in shap_data:
                elements.append(Paragraph("Top Contributing Features", self.styles['CustomBody']))
                
                # Create feature importance table
                feature_data = [['Feature', 'Importance', 'Direction']]
                
                for feature in shap_data['feature_importance'][:10]:  # Top 10
                    importance = feature.get('importance', feature.get('shap_value', 0))
                    direction = 'Increases Risk' if importance > 0 else 'Decreases Risk'
                    feature_data.append([feature.get('feature', 'Unknown'), f"{abs(importance):.3f}", direction])
                
                feature_table = Table(feature_data, colWidths=[2*inch, 1*inch, 1.5*inch])
                feature_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(feature_table)
                elements.append(Spacer(1, 15))
        
        # LIME explanations
        if 'lime' in explanations:
            elements.append(Paragraph("LIME Analysis", self.styles['SectionHeader']))
            lime_data = explanations['lime']
            
            if 'feature_contributions' in lime_data:
                elements.append(Paragraph("Local Feature Contributions", self.styles['CustomBody']))
                
                contributions = lime_data['feature_contributions'][:10]
                for feature, contribution in contributions:
                    direction = "increases" if contribution > 0 else "decreases"
                    elements.append(Paragraph(
                        f"• {feature}: {direction} toxicity risk ({contribution:.3f})",
                        self.styles['CustomBody']
                    ))
        
        return elements
    
    def _create_similar_compounds_section(self, report_data: Dict) -> List:
        """Create similar compounds section"""
        elements = []
        
        elements.append(Paragraph("SIMILAR COMPOUNDS ANALYSIS", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 20))
        
        similar_compounds = report_data['similar_compounds']
        
        if not similar_compounds:
            elements.append(Paragraph("No similar compounds data available.", self.styles['CustomBody']))
            return elements
        
        # Create similar compounds table
        similar_data = [['Compound', 'Similarity', 'Toxicity', 'Risk Level']]
        
        for compound in similar_compounds[:10]:  # Top 10
            similarity = compound.get('similarity', 0)
            toxicity = compound.get('toxicity', 0)
            
            if toxicity >= 0.7:
                risk = 'HIGH'
            elif toxicity >= 0.4:
                risk = 'MODERATE'
            else:
                risk = 'LOW'
            
            name = compound.get('name', compound.get('smiles', 'Unknown')[:30])
            similar_data.append([name, f"{similarity:.3f}", f"{toxicity:.3f}", risk])
        
        similar_table = Table(similar_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch])
        similar_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(similar_table)
        
        return elements
    
    def _create_recommendations_section(self, report_data: Dict) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("RECOMMENDATIONS", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 20))
        
        recommendations = report_data['recommendations']
        
        if not recommendations:
            elements.append(Paragraph("No specific recommendations available.", self.styles['CustomBody']))
            return elements
        
        for i, recommendation in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {recommendation}", self.styles['CustomBody']))
            elements.append(Spacer(1, 10))
        
        # Disclaimer
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("DISCLAIMER", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "This report is generated by computational models and should be used as a guide only. "
            "Experimental validation is required before making any decisions based on these results. "
            "The predictions are based on the Tox21 dataset and may not capture all toxicity mechanisms.",
            self.styles['CustomBody']
        ))
        
        return elements
    
    def _create_prediction_chart(self, predictions: Dict[str, float]) -> str:
        """Create prediction chart and return path"""
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            endpoints = list(predictions.keys())
            probabilities = list(predictions.values())
            
            # Create bar chart
            bars = ax.bar(endpoints, probabilities, color=['red' if p > 0.7 else 'orange' if p > 0.4 else 'green' for p in probabilities])
            
            ax.set_xlabel('Toxicity Endpoints')
            ax.set_ylabel('Probability')
            ax.set_title('Toxicity Prediction Results')
            ax.set_ylim(0, 1)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / "prediction_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error creating prediction chart: {e}")
            return None
    
    def _generate_html_report(self, report_data: Dict, output_path: Path):
        """Generate HTML report"""
        template = self.template_env.get_template('html_report')
        
        # Add molecular structure image
        try:
            mol = Chem.MolFromSmiles(report_data['smiles'])
            if mol:
                drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()
                
                # Convert to base64
                img_buffer = BytesIO()
                drawer.WriteToPng(img_buffer)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                report_data['molecule_image'] = f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.warning(f"Could not generate molecular structure for HTML: {e}")
            report_data['molecule_image'] = None
        
        # Generate HTML
        html_content = template.render(**report_data)
        
        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_html_template(self) -> str:
        """Get HTML report template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Toxicity Assessment Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 3px solid #1f77b4; padding-bottom: 20px; margin-bottom: 30px; }
        .title { color: #1f77b4; font-size: 28px; font-weight: bold; }
        .subtitle { color: #333; font-size: 18px; margin-top: 10px; }
        .section { margin: 30px 0; }
        .section-title { color: #1f77b4; font-size: 20px; border-bottom: 2px solid #1f77b4; padding-bottom: 5px; }
        .risk-high { background-color: #ffebee; border: 2px solid #f44336; padding: 15px; border-radius: 5px; }
        .risk-moderate { background-color: #fff3e0; border: 2px solid #ff9800; padding: 15px; border-radius: 5px; }
        .risk-low { background-color: #e8f5e8; border: 2px solid #4caf50; padding: 15px; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #1f77b4; color: white; }
        .molecule-structure { text-align: center; margin: 20px 0; }
        .recommendations { background-color: #f5f5f5; padding: 20px; border-radius: 5px; }
        .recommendations ul { margin: 0; padding-left: 20px; }
        .disclaimer { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">DRUG TOXICITY ASSESSMENT REPORT</div>
        <div class="subtitle">Compound: {{ smiles }}</div>
        <div class="subtitle">Generated: {{ timestamp }}</div>
    </div>

    <div class="section">
        <div class="section-title">Risk Assessment</div>
        <div class="risk-{{ risk_assessment.risk_color }}">
            <h3>RISK LEVEL: {{ risk_assessment.risk_level }}</h3>
            <p>{{ risk_assessment.description }}</p>
            <p><strong>Maximum Toxicity Probability:</strong> {{ "%.1f"|format(risk_assessment.max_probability * 100) }}%</p>
            <p><strong>Highest Risk Endpoint:</strong> {{ risk_assessment.highest_risk_endpoint }}</p>
        </div>
    </div>

    {% if molecule_image %}
    <div class="section">
        <div class="section-title">Molecular Structure</div>
        <div class="molecule-structure">
            <img src="{{ molecule_image }}" alt="Molecular Structure" style="max-width: 100%;">
        </div>
    </div>
    {% endif %}

    <div class="section">
        <div class="section-title">Toxicity Predictions</div>
        <table>
            <thead>
                <tr>
                    <th>Toxicity Endpoint</th>
                    <th>Probability</th>
                    <th>Risk Level</th>
                </tr>
            </thead>
            <tbody>
                {% for endpoint, probability in predictions.items() %}
                <tr>
                    <td>{{ endpoint }}</td>
                    <td>{{ "%.3f"|format(probability) }}</td>
                    <td>
                        {% if probability >= 0.7 %}HIGH{% elif probability >= 0.4 %}MODERATE{% else %}LOW{% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    {% if properties %}
    <div class="section">
        <div class="section-title">Molecular Properties</div>
        <table>
            <thead>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for prop_name, prop_value in properties.items() %}
                <tr>
                    <td>{{ prop_name }}</td>
                    <td>{{ "%.3f"|format(prop_value) if prop_value is number else prop_value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    <div class="section">
        <div class="section-title">Recommendations</div>
        <div class="recommendations">
            <ul>
                {% for recommendation in recommendations %}
                <li>{{ recommendation }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>

    <div class="disclaimer">
        <strong>Disclaimer:</strong> This report is generated by computational models and should be used as a guide only. 
        Experimental validation is required before making any decisions based on these results.
    </div>
</body>
</html>
        """
    
    def _get_executive_summary_template(self) -> str:
        """Get executive summary template"""
        return """
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p><strong>Compound:</strong> {{ smiles }}</p>
            <p><strong>Risk Level:</strong> <span class="risk-{{ risk_assessment.risk_color }}">{{ risk_assessment.risk_level }}</span></p>
            <p><strong>Maximum Toxicity Probability:</strong> {{ "%.1f"|format(risk_assessment.max_probability * 100) }}%</p>
            <p><strong>Key Finding:</strong> {{ risk_assessment.description }}</p>
        </div>
        """


class BatchReportGenerator:
    """
    Generate reports for multiple compounds
    """
    
    def __init__(self, output_dir: str = "outputs/batch_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.single_generator = ToxicityReportGenerator(str(self.output_dir))
    
    def generate_batch_report(self, compounds_data: List[Dict[str, Any]], 
                            output_format: str = "pdf") -> str:
        """
        Generate batch report for multiple compounds
        
        Args:
            compounds_data: List of compound data dictionaries
            output_format: Output format
            
        Returns:
            Path to generated batch report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_toxicity_report_{timestamp}.{output_format}"
        output_path = self.output_dir / filename
        
        logger.info(f"Generating batch report for {len(compounds_data)} compounds")
        
        if output_format == "pdf":
            self._generate_batch_pdf(compounds_data, output_path)
        elif output_format == "html":
            self._generate_batch_html(compounds_data, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Batch report generated: {output_path}")
        return str(output_path)
    
    def _generate_batch_pdf(self, compounds_data: List[Dict], output_path: Path):
        """Generate batch PDF report"""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        
        # Title
        styles = getSampleStyleSheet()
        story.append(Paragraph("BATCH TOXICITY ASSESSMENT REPORT", styles['Title']))
        story.append(Spacer(1, 20))
        
        # Summary table
        table_data = [['Compound', 'Risk Level', 'Max Probability', 'Highest Risk']]
        
        for compound in compounds_data:
            smiles = compound['smiles'][:30] + '...' if len(compound['smiles']) > 30 else compound['smiles']
            risk_level = compound.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
            max_prob = compound.get('risk_assessment', {}).get('max_probability', 0)
            highest_risk = compound.get('risk_assessment', {}).get('highest_risk_endpoint', 'UNKNOWN')
            
            table_data.append([smiles, risk_level, f"{max_prob:.3f}", highest_risk])
        
        batch_table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch, 1.5*inch])
        batch_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(batch_table)
        
        # Build PDF
        doc.build(story)
    
    def _generate_batch_html(self, compounds_data: List[Dict], output_path: Path):
        """Generate batch HTML report"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Toxicity Assessment Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; border-bottom: 3px solid #1f77b4; padding-bottom: 20px; }
        .title { color: #1f77b4; font-size: 24px; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #1f77b4; color: white; }
        .risk-high { color: #f44336; font-weight: bold; }
        .risk-moderate { color: #ff9800; font-weight: bold; }
        .risk-low { color: #4caf50; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">BATCH TOXICITY ASSESSMENT REPORT</div>
        <div>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Compound (SMILES)</th>
                <th>Risk Level</th>
                <th>Max Probability</th>
                <th>Highest Risk Endpoint</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for compound in compounds_data:
            smiles = compound['smiles'][:50] + '...' if len(compound['smiles']) > 50 else compound['smiles']
            risk_assessment = compound.get('risk_assessment', {})
            risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
            max_prob = risk_assessment.get('max_probability', 0)
            highest_risk = risk_assessment.get('highest_risk_endpoint', 'UNKNOWN')
            
            risk_class = f"risk-{risk_level.lower()}" if risk_level.lower() in ['high', 'moderate', 'low'] else ""
            
            html_content += f"""
            <tr>
                <td>{smiles}</td>
                <td class="{risk_class}">{risk_level}</td>
                <td>{max_prob:.3f}</td>
                <td>{highest_risk}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


# Utility functions
def create_quick_report(smiles: str, predictions: Dict[str, float]) -> str:
    """
    Create a quick summary report
    
    Args:
        smiles: SMILES string
        predictions: Toxicity predictions
        
    Returns:
        Report summary text
    """
    if not predictions:
        return "No prediction data available."
    
    max_prob = max(predictions.values())
    most_toxic = max(predictions, key=predictions.get)
    
    risk_level = 'HIGH' if max_prob >= 0.7 else 'MODERATE' if max_prob >= 0.4 else 'LOW'
    
    summary = f"""
TOXICITY ASSESSMENT SUMMARY
============================
Compound: {smiles}
Risk Level: {risk_level}
Maximum Toxicity Probability: {max_prob:.1%}
Highest Risk Endpoint: {most_toxic}

RECOMMENDATIONS:
"""
    
    if risk_level == 'HIGH':
        summary += "- Avoid further development without significant modification\n"
        summary += "- Conduct comprehensive toxicity testing\n"
        summary += "- Consider alternative scaffolds\n"
    elif risk_level == 'MODERATE':
        summary += "- Proceed with caution and additional testing\n"
        summary += "- Consider structural optimization\n"
        summary += "- Monitor safety parameters closely\n"
    else:
        summary += "- Low risk - proceed with standard development\n"
        summary += "- Continue routine safety assessment\n"
    
    return summary
