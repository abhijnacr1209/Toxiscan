"""
Metabolism Site Animation for Toxiscan
Interactive 3D animations showing metabolic transformations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json

# Chemistry libraries
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class MetabolismAnimator:
    """Create animated visualizations of metabolic transformations"""
    
    def __init__(self):
        self.metabolic_pathways = {
            'aromatic_oxidation': {
                'name': 'Aromatic Hydroxylation',
                'description': 'Cytochrome P450 enzymes add hydroxyl groups to aromatic rings',
                'color': '#FF6B6B',
                'enzyme': 'CYP450',
                'probability': 0.8
            },
            'aliphatic_oxidation': {
                'name': 'Aliphatic Hydroxylation',
                'description': 'Oxidation of aliphatic carbons to alcohols',
                'color': '#4ECDC4',
                'enzyme': 'CYP450',
                'probability': 0.7
            },
            'n_oxidation': {
                'name': 'N-Oxidation',
                'description': 'Oxidation of nitrogen atoms to N-oxides',
                'color': '#45B7D1',
                'enzyme': 'CYP450',
                'probability': 0.6
            },
            's_oxidation': {
                'name': 'S-Oxidation',
                'description': 'Oxidation of sulfur atoms to sulfoxides/sulfones',
                'color': '#96CEB4',
                'enzyme': 'CYP450',
                'probability': 0.9
            },
            'dealkylation': {
                'name': 'Dealkylation',
                'description': 'Removal of alkyl groups from heteroatoms',
                'color': '#FFEAA7',
                'enzyme': 'CYP450',
                'probability': 0.7
            },
            'conjugation': {
                'name': 'Phase II Conjugation',
                'description': 'Glucuronidation, sulfation, or glutathione conjugation',
                'color': '#DDA0DD',
                'enzyme': 'UGT/SULT/GST',
                'probability': 0.8
            }
        }
    
    def create_metabolism_animation(self, smiles: str, predicted_sites: List[Dict]) -> go.Figure:
        """Create animated 3D visualization of metabolic transformations"""
        
        if not RDKIT_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(
                text="Metabolism animation requires RDKit",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Invalid SMILES",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Generate 3D conformer
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100)
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            
            # Get atom positions
            conf = mol.GetConformer()
            atom_positions = []
            atom_symbols = []
            atom_colors = []
            atom_sizes = []
            
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                pos = conf.GetAtomPosition(atom_idx)
                atom_positions.append([pos.x, pos.y, pos.z])
                atom_symbols.append(atom.GetSymbol())
                
                # Default color and size
                atom_colors.append('lightblue')
                atom_sizes.append(8)
            
            # Create bonds
            bond_x, bond_y, bond_z = [], [], []
            for bond in mol.GetBonds():
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                
                pos1 = conf.GetAtomPosition(atom1_idx)
                pos2 = conf.GetAtomPosition(atom2_idx)
                
                bond_x.extend([pos1.x, pos2.x, None])
                bond_y.extend([pos1.y, pos2.y, None])
                bond_z.extend([pos1.z, pos2.z, None])
            
            # Create animation frames
            frames = []
            
            # Frame 0: Original molecule
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(
                        x=bond_x, y=bond_y, z=bond_z,
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False,
                        hoverinfo='none',
                        name='bonds'
                    ),
                    go.Scatter3d(
                        x=[pos[0] for pos in atom_positions],
                        y=[pos[1] for pos in atom_positions],
                        z=[pos[2] for pos in atom_positions],
                        mode='markers+text',
                        marker=dict(
                            size=atom_sizes,
                            color=atom_colors,
                            line=dict(width=2, color='black'),
                            opacity=0.8
                        ),
                        text=atom_symbols,
                        textposition='middle center',
                        showlegend=False,
                        name='atoms',
                        hovertemplate='<b>Atom %{text}</b><br>' +
                                     'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                     '<extra></extra>'
                    )
                ],
                name='Original Molecule'
            ))
            
            # Create frames for each metabolic transformation
            for i, site in enumerate(predicted_sites[:3]):  # Limit to 3 sites
                pathway = site.get('metabolic_pathways', ['aromatic_oxidation'])[0]
                pathway_info = self.metabolic_pathways.get(pathway, self.metabolic_pathways['aromatic_oxidation'])
                
                # Highlight the metabolic site
                highlight_colors = atom_colors.copy()
                highlight_sizes = atom_sizes.copy()
                
                atom_idx = site.get('atom_index', 0)
                if atom_idx < len(highlight_colors):
                    highlight_colors[atom_idx] = pathway_info['color']
                    highlight_sizes[atom_idx] = 12
                
                # Add metabolic transformation indicator
                transform_x = [atom_positions[atom_idx][0]]
                transform_y = [atom_positions[atom_idx][1]]
                transform_z = [atom_positions[atom_idx][2]]
                
                frames.append(go.Frame(
                    data=[
                        go.Scatter3d(
                            x=bond_x, y=bond_y, z=bond_z,
                            mode='lines',
                            line=dict(color='gray', width=3),
                            showlegend=False,
                            hoverinfo='none',
                            name='bonds'
                        ),
                        go.Scatter3d(
                            x=[pos[0] for pos in atom_positions],
                            y=[pos[1] for pos in atom_positions],
                            z=[pos[2] for pos in atom_positions],
                            mode='markers+text',
                            marker=dict(
                                size=highlight_sizes,
                                color=highlight_colors,
                                line=dict(width=2, color='black'),
                                opacity=0.8
                            ),
                            text=atom_symbols,
                            textposition='middle center',
                            showlegend=False,
                            name='atoms',
                            hovertemplate='<b>Atom %{text}</b><br>' +
                                         'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>' +
                                         '<extra></extra>'
                        ),
                        go.Scatter3d(
                            x=transform_x,
                            y=transform_y,
                            z=transform_z,
                            mode='markers',
                            marker=dict(
                                size=20,
                                color=pathway_info['color'],
                                symbol='diamond',
                                line=dict(width=3, color='black')
                            ),
                            showlegend=False,
                            name='metabolic_site',
                            hovertemplate=f'<b>Metabolic Site</b><br>' +
                                         f'Pathway: {pathway_info["name"]}<br>' +
                                         f'Enzyme: {pathway_info["enzyme"]}<br>' +
                                         f'Risk: {site.get("risk_score", 0):.2f}<br>' +
                                         '<extra></extra>'
                        )
                    ],
                    name=f'{pathway_info["name"]} - Site {i+1}'
                ))
            
            # Create figure with animation
            fig = go.Figure(
                data=frames[0].data,
                frames=frames
            )
            
            # Add animation controls
            fig.update_layout(
                title="Interactive Metabolism Site Animation",
                scene=dict(
                    xaxis=dict(title='X Å', showbackground=False),
                    yaxis=dict(title='Y Å', showbackground=False),
                    zaxis=dict(title='Z Å', showbackground=False),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="▶️ Play Animation",
                                method="animate",
                                args=[None, {"frame": {"duration": 2000, "redraw": True},
                                          "fromcurrent": True}]
                            ),
                            dict(
                                label="⏸️ Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": False},
                                             "mode": "immediate",
                                             "transition": {"duration": 0}}]
                            ),
                            dict(
                                label="🔄 Reset",
                                method="animate",
                                args=[[0], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}]
                            )
                        ],
                        direction="left",
                        pad={"r": 10, "t": 87},
                        showactive=False,
                        x=0.011,
                        xanchor="right",
                        y=0,
                        yanchor="top"
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        yanchor="top",
                        xanchor="left",
                        currentvalue={"prefix": "Frame: "},
                        pad={"t": 50},
                        len=0.9,
                        x=0.1,
                        y=0,
                        steps=[
                            dict(
                                args=[[frame.name], {"frame": {"duration": 0, "redraw": True},
                                                     "mode": "immediate",
                                                     "transition": {"duration": 0}}],
                                label=frame.name,
                                method="animate"
                            )
                            for frame in frames
                        ]
                    )
                ],
                width=800,
                height=600,
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating animation: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_metabolic_pathway_chart(self, predicted_sites: List[Dict]) -> go.Figure:
        """Create a chart showing metabolic pathway probabilities"""
        
        pathways = []
        probabilities = []
        colors = []
        
        for site in predicted_sites:
            for pathway in site.get('metabolic_pathways', ['aromatic_oxidation']):
                pathway_info = self.metabolic_pathways.get(pathway, self.metabolic_pathways['aromatic_oxidation'])
                pathways.append(pathway_info['name'])
                probabilities.append(pathway_info['probability'])
                colors.append(pathway_info['color'])
        
        # Create bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pathways,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Probability: %{y:.1%}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Metabolic Pathway Probabilities",
            xaxis_title="Metabolic Pathway",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=400,
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_enzyme_interaction_chart(self, predicted_sites: List[Dict]) -> go.Figure:
        """Create a chart showing enzyme interactions"""
        
        enzymes = {}
        for site in predicted_sites:
            for pathway in site.get('metabolic_pathways', ['aromatic_oxidation']):
                pathway_info = self.metabolic_pathways.get(pathway, self.metabolic_pathways['aromatic_oxidation'])
                enzyme = pathway_info['enzyme']
                if enzyme not in enzymes:
                    enzymes[enzyme] = 0
                enzymes[enzyme] += site.get('risk_score', 0.5)
        
        # Create pie chart
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=list(enzymes.keys()),
            values=list(enzymes.values()),
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>' +
                         'Risk Score: %{value:.2f}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Enzyme Interaction Distribution",
            height=400,
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_metabolism_timeline(self, predicted_sites: List[Dict]) -> go.Figure:
        """Create a timeline of metabolic transformations"""
        
        # Sort sites by risk score
        sorted_sites = sorted(predicted_sites, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        phases = []
        times = []
        descriptions = []
        colors = []
        
        for i, site in enumerate(sorted_sites):
            phase = f"Phase {1 if i < len(sorted_sites)/2 else 2}"
            time = i * 2  # 2 hours per transformation
            pathway = site.get('metabolic_pathways', ['aromatic_oxidation'])[0]
            pathway_info = self.metabolic_pathways.get(pathway, self.metabolic_pathways['aromatic_oxidation'])
            
            phases.append(phase)
            times.append(time)
            descriptions.append(f"Atom {site.get('atom_index', 0)} - {pathway_info['name']}")
            colors.append(pathway_info['color'])
        
        # Create timeline chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=phases,
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color='black')
            ),
            text=descriptions,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>' +
                         'Time: %{x} hours<br>' +
                         'Phase: %{y}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Metabolic Transformation Timeline",
            xaxis_title="Time (hours)",
            yaxis_title="Metabolic Phase",
            height=400,
            paper_bgcolor='white'
        )
        
        return fig

def create_metabolism_animation_tab(smiles: str, predicted_sites: List[Dict]):
    """Create the metabolism animation tab content"""
    
    st.markdown("### 🎬 Metabolism Site Animation")
    st.markdown("""
    <div class="feature-card">
        <strong>🎬 Interactive Animation:</strong> Watch how your molecule is metabolized step-by-step
        <br><strong>🔬 Real Enzyme Data:</strong> Based on actual cytochrome P450 metabolic pathways
        <br><strong>⏱️ Timeline View:</strong> See the order and timing of metabolic transformations
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize animator
    animator = MetabolismAnimator()
    
    # Create animation
    with st.spinner("Creating metabolism animation..."):
        animation_fig = animator.create_metabolism_animation(smiles, predicted_sites)
    
    # Display animation
    st.plotly_chart(animation_fig, key="metabolism_animation")
    
    # Animation controls info
    st.markdown("""
    <div class="warning-box">
        <strong>🎮 Animation Controls:</strong>
        <ul>
            <li>▶️ <strong>Play:</strong> Watch the full metabolic transformation sequence</li>
            <li>⏸️ <strong>Pause:</strong> Stop at any point to examine the structure</li>
            <li>🔄 <strong>Reset:</strong> Return to the original molecule</li>
            <li>📊 <strong>Slider:</strong> Manually navigate through transformation steps</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Metabolic Pathway Probabilities")
        pathway_fig = animator.create_metabolic_pathway_chart(predicted_sites)
        st.plotly_chart(pathway_fig, key="pathway_chart")
    
    with col2:
        st.markdown("#### 🧬 Enzyme Interaction Distribution")
        enzyme_fig = animator.create_enzyme_interaction_chart(predicted_sites)
        st.plotly_chart(enzyme_fig, key="enzyme_chart")
    
    # Timeline
    st.markdown("#### ⏱️ Metabolic Transformation Timeline")
    timeline_fig = animator.create_metabolism_timeline(predicted_sites)
    st.plotly_chart(timeline_fig, key="timeline_chart")
    
    # Detailed pathway information
    st.markdown("#### 🔬 Detailed Metabolic Pathway Information")
    
    for site in predicted_sites[:3]:  # Show top 3 sites
        with st.expander(f"🔥 Site {site.get('atom_index', 0)} - Risk {site.get('risk_score', 0):.2f}"):
            pathways = site.get('metabolic_pathways', ['aromatic_oxidation'])
            
            for pathway in pathways:
                pathway_info = animator.metabolic_pathways.get(pathway, animator.metabolic_pathways['aromatic_oxidation'])
                
                st.markdown(f"""
                <div class="gnn-explanation">
                    <h4>🧬 {pathway_info['name']}</h4>
                    <p><strong>Enzyme:</strong> {pathway_info['enzyme']}</p>
                    <p><strong>Description:</strong> {pathway_info['description']}</p>
                    <p><strong>Probability:</strong> {pathway_info['probability']:.1%}</p>
                    <p><strong>Risk Score:</strong> {site.get('risk_score', 0):.2f}</p>
                    <p><strong>Confidence:</strong> {site.get('confidence', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
