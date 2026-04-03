"""
Human Organ Metabolism Video Animation
Creates animated videos showing how human organs react to chemicals during metabolism
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
import json

class OrganMetabolismAnimator:
    """Create animated videos of human organ metabolism"""
    
    def __init__(self):
        self.organs = {
            'liver': {
                'name': 'Liver',
                'position': (0, 0),
                'size': 80,
                'color': '#8B4513',
                'function': 'Primary metabolism site - CYP450 enzymes',
                'enzymes': ['CYP3A4', 'CYP2D6', 'CYP2C9', 'CYP1A2'],
                'metabolism_rate': 0.9
            },
            'kidney': {
                'name': 'Kidneys',
                'position': (-150, -50),
                'size': 60,
                'color': '#FF6B6B',
                'function': 'Excretion and some metabolism',
                'enzymes': ['UGT', 'SULT'],
                'metabolism_rate': 0.4
            },
            'lung': {
                'name': 'Lungs',
                'position': (150, -50),
                'size': 70,
                'color': '#FF69B4',
                'function': 'Metabolism of inhaled compounds',
                'enzymes': ['CYP1A1', 'CYP2B6'],
                'metabolism_rate': 0.3
            },
            'heart': {
                'name': 'Heart',
                'position': (0, 100),
                'size': 50,
                'color': '#DC143C',
                'function': 'Limited metabolism',
                'enzymes': ['MAO', 'COMT'],
                'metabolism_rate': 0.2
            },
            'brain': {
                'name': 'Brain',
                'position': (0, -100),
                'size': 60,
                'color': '#4B0082',
                'function': 'Blood-brain barrier protection',
                'enzymes': ['MAO', 'COMT'],
                'metabolism_rate': 0.1
            },
            'intestine': {
                'name': 'Intestine',
                'position': (-100, 50),
                'size': 55,
                'color': '#FF8C00',
                'function': 'First-pass metabolism',
                'enzymes': ['CYP3A4', 'UGT'],
                'metabolism_rate': 0.6
            }
        }
        
        self.metabolism_steps = [
            'absorption',
            'distribution',
            'liver_metabolism',
            'kidney_excretion',
            'other_organ_metabolism',
            'elimination'
        ]
    
    def create_organ_animation_video(self, smiles: str, tox21_predictions: Dict) -> go.Figure:
        """Create an animated video showing organ metabolism"""
        
        frames = []
        
        # Calculate overall toxicity
        max_risk = max(tox21_predictions.values())
        toxicity_level = 'high' if max_risk > 0.7 else 'medium' if max_risk > 0.4 else 'low'
        
        # Step 0: Initial state - organs at rest
        frames.append(self._create_initial_frame())
        
        # Step 1: Chemical absorption
        frames.append(self._create_absorption_frame(smiles, toxicity_level))
        
        # Step 2: Distribution to organs
        frames.append(self._create_distribution_frame(smiles, toxicity_level))
        
        # Step 3: Liver metabolism (main event)
        frames.append(self._create_liver_metabolism_frame(smiles, tox21_predictions, toxicity_level))
        
        # Step 4: Kidney processing
        frames.append(self._create_kidney_frame(smiles, toxicity_level))
        
        # Step 5: Other organ metabolism
        frames.append(self._create_other_organs_frame(smiles, tox21_predictions, toxicity_level))
        
        # Step 6: Elimination
        frames.append(self._create_elimination_frame(smiles, toxicity_level))
        
        # Create figure with animation
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Update layout for video-like experience
        fig.update_layout(
            title="Human Organ Metabolism Animation",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            width=800,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="▶️ Play Video",
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
                            label="🔄 Restart",
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
            ]
        )
        
        return fig
    
    def _create_initial_frame(self) -> go.Frame:
        """Create initial frame with organs at rest"""
        traces = []
        
        for organ_id, organ in self.organs.items():
            # Draw organ
            traces.append(go.Scatter(
                x=[organ['position'][0]],
                y=[organ['position'][1]],
                mode='markers',
                marker=dict(
                    size=organ['size'],
                    color=organ['color'],
                    line=dict(width=2, color='black'),
                    opacity=0.7
                ),
                name=organ['name'],
                text=organ['name'],
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>' +
                             f'Function: {organ["function"]}<br>' +
                             'Status: At Rest<br>' +
                             '<extra></extra>'
            ))
        
        return go.Frame(data=traces, name="Initial State")
    
    def _create_absorption_frame(self, smiles: str, toxicity_level: str) -> go.Frame:
        """Create absorption frame"""
        traces = []
        
        for organ_id, organ in self.organs.items():
            # Highlight organs involved in absorption
            if organ_id in ['intestine', 'lung']:
                color = '#FFD700' if toxicity_level == 'high' else '#FFA500' if toxicity_level == 'medium' else '#90EE90'
                size = organ['size'] * 1.2
                opacity = 0.9
            else:
                color = organ['color']
                size = organ['size']
                opacity = 0.5
            
            traces.append(go.Scatter(
                x=[organ['position'][0]],
                y=[organ['position'][1]],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='black'),
                    opacity=opacity
                ),
                name=organ['name'],
                text=organ['name'],
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>' +
                             f'Function: {organ["function"]}<br>' +
                             'Status: Absorbing Chemical<br>' +
                             '<extra></extra>'
            ))
        
        # Add chemical particles
        for i in range(5):
            traces.append(go.Scatter(
                x=[-200 + i * 20],
                y=[0],
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    symbol='circle',
                    opacity=0.8
                ),
                name=f'Chemical_{i}',
                hovertemplate='Chemical Molecule<br>' +
                             'Moving to absorption sites<br>' +
                             '<extra></extra>'
            ))
        
        return go.Frame(data=traces, name="Chemical Absorption")
    
    def _create_distribution_frame(self, smiles: str, toxicity_level: str) -> go.Frame:
        """Create distribution frame"""
        traces = []
        
        for organ_id, organ in self.organs.items():
            # All organs receive blood flow
            color = '#FFD700' if toxicity_level == 'high' else '#FFA500' if toxicity_level == 'medium' else '#90EE90'
            size = organ['size'] * 1.1
            opacity = 0.8
            
            traces.append(go.Scatter(
                x=[organ['position'][0]],
                y=[organ['position'][1]],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='black'),
                    opacity=opacity
                ),
                name=organ['name'],
                text=organ['name'],
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>' +
                             f'Function: {organ["function"]}<br>' +
                             'Status: Receiving Chemical<br>' +
                             '<extra></extra>'
            ))
        
        # Add chemical particles distributed
        for organ_id, organ in self.organs.items():
            for i in range(2):
                traces.append(go.Scatter(
                    x=[organ['position'][0] + np.random.uniform(-20, 20)],
                    y=[organ['position'][1] + np.random.uniform(-20, 20)],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='blue',
                        symbol='circle',
                        opacity=0.7
                    ),
                    name=f'Chemical_{organ_id}_{i}',
                    hovertemplate='Chemical Molecule<br>' +
                                 'Distributed to organ<br>' +
                                 '<extra></extra>'
                ))
        
        return go.Frame(data=traces, name="Distribution")
    
    def _create_liver_metabolism_frame(self, smiles: str, tox21_predictions: Dict, toxicity_level: str) -> go.Frame:
        """Create liver metabolism frame"""
        traces = []
        
        for organ_id, organ in self.organs.items():
            if organ_id == 'liver':
                # Liver is highly active
                color = '#FF0000' if toxicity_level == 'high' else '#FF6347' if toxicity_level == 'medium' else '#32CD32'
                size = organ['size'] * 1.5
                opacity = 1.0
                status = 'Active Metabolism'
            else:
                color = organ['color']
                size = organ['size']
                opacity = 0.4
                status = 'Waiting'
            
            traces.append(go.Scatter(
                x=[organ['position'][0]],
                y=[organ['position'][1]],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=3, color='black'),
                    opacity=opacity
                ),
                name=organ['name'],
                text=organ['name'],
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>' +
                             f'Function: {organ["function"]}<br>' +
                             f'Status: {status}<br>' +
                             f'Enzymes: {", ".join(organ["enzymes"])}<br>' +
                             '<extra></extra>'
            ))
        
        # Add metabolism particles around liver
        for i in range(8):
            angle = i * 45 * np.pi / 180
            x = self.organs['liver']['position'][0] + 60 * np.cos(angle)
            y = self.organs['liver']['position'][1] + 60 * np.sin(angle)
            
            traces.append(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    opacity=0.9
                ),
                name=f'Metabolite_{i}',
                hovertemplate='Metabolite Product<br>' +
                             'Enzyme Processing<br>' +
                             '<extra></extra>'
            ))
        
        return go.Frame(data=traces, name="Liver Metabolism")
    
    def _create_kidney_frame(self, smiles: str, toxicity_level: str) -> go.Frame:
        """Create kidney processing frame"""
        traces = []
        
        for organ_id, organ in self.organs.items():
            if organ_id == 'kidney':
                # Kidneys are processing
                color = '#FF69B4' if toxicity_level == 'high' else '#FFB6C1' if toxicity_level == 'medium' else '#DDA0DD'
                size = organ['size'] * 1.3
                opacity = 0.9
                status = 'Filtering & Excreting'
            elif organ_id == 'liver':
                # Liver still active but less
                color = '#FFA500'
                size = organ['size'] * 1.2
                opacity = 0.7
                status = 'Continuing Metabolism'
            else:
                color = organ['color']
                size = organ['size']
                opacity = 0.3
                status = 'Minimal Activity'
            
            traces.append(go.Scatter(
                x=[organ['position'][0]],
                y=[organ['position'][1]],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='black'),
                    opacity=opacity
                ),
                name=organ['name'],
                text=organ['name'],
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>' +
                             f'Function: {organ["function"]}<br>' +
                             f'Status: {status}<br>' +
                             '<extra></extra>'
            ))
        
        # Add excretion particles
        for i in range(6):
            x = self.organs['kidney']['position'][0] + np.random.uniform(-30, 30)
            y = self.organs['kidney']['position'][1] + np.random.uniform(-30, 30)
            
            traces.append(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=8,
                    color='purple',
                    symbol='triangle-down',
                    opacity=0.8
                ),
                name=f'Excreted_{i}',
                hovertemplate='Excreted Metabolite<br>' +
                             'Kidney Filtration<br>' +
                             '<extra></extra>'
            ))
        
        return go.Frame(data=traces, name="Kidney Processing")
    
    def _create_other_organs_frame(self, smiles: str, tox21_predictions: Dict, toxicity_level: str) -> go.Frame:
        """Create other organs metabolism frame"""
        traces = []
        
        for organ_id, organ in self.organs.items():
            if organ_id in ['brain', 'heart']:
                # These organs have minimal metabolism
                color = '#DDA0DD' if toxicity_level == 'high' else '#E6E6FA' if toxicity_level == 'medium' else '#F0F8FF'
                size = organ['size'] * 1.1
                opacity = 0.6
                status = 'Minimal Processing'
            elif organ_id in ['lung', 'intestine']:
                # Secondary metabolism
                color = '#FFB6C1'
                size = organ['size'] * 1.2
                opacity = 0.7
                status = 'Secondary Metabolism'
            else:
                # Liver and kidney still working
                color = '#FFA500'
                size = organ['size'] * 1.1
                opacity = 0.6
                status = 'Continuing Work'
            
            traces.append(go.Scatter(
                x=[organ['position'][0]],
                y=[organ['position'][1]],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='black'),
                    opacity=opacity
                ),
                name=organ['name'],
                text=organ['name'],
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>' +
                             f'Function: {organ["function"]}<br>' +
                             f'Status: {status}<br>' +
                             f'Enzymes: {", ".join(organ["enzymes"])}<br>' +
                             '<extra></extra>'
            ))
        
        # Add scattered metabolite particles
        for organ_id, organ in self.organs.items():
            if organ['metabolism_rate'] > 0.2:
                for i in range(int(organ['metabolism_rate'] * 3)):
                    x = organ['position'][0] + np.random.uniform(-25, 25)
                    y = organ['position'][1] + np.random.uniform(-25, 25)
                    
                    traces.append(go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color='orange',
                            symbol='square',
                            opacity=0.6
                        ),
                        name=f'Metabolite_{organ_id}_{i}',
                        hovertemplate='Minor Metabolite<br>' +
                                     'Secondary Processing<br>' +
                                     '<extra></extra>'
                    ))
        
        return go.Frame(data=traces, name="Other Organ Metabolism")
    
    def _create_elimination_frame(self, smiles: str, toxicity_level: str) -> go.Frame:
        """Create elimination frame"""
        traces = []
        
        for organ_id, organ in self.organs.items():
            # All organs returning to normal
            color = organ['color']
            size = organ['size']
            opacity = 0.5
            status = 'Returning to Normal'
            
            traces.append(go.Scatter(
                x=[organ['position'][0]],
                y=[organ['position'][1]],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='black'),
                    opacity=opacity
                ),
                name=organ['name'],
                text=organ['name'],
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>' +
                             f'Function: {organ["function"]}<br>' +
                             f'Status: {status}<br>' +
                             '<extra></extra>'
            ))
        
        # Add eliminated particles moving away
        for i in range(10):
            x = np.random.uniform(-250, 250)
            y = np.random.uniform(-150, 150)
            
            traces.append(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=4,
                    color='gray',
                    symbol='circle',
                    opacity=0.4
                ),
                name=f'Eliminated_{i}',
                hovertemplate='Eliminated Compound<br>' +
                             'Leaving Body<br>' +
                             '<extra></extra>'
            ))
        
        return go.Frame(data=traces, name="Elimination")
    
    def create_organ_info_table(self) -> pd.DataFrame:
        """Create a table with organ information"""
        data = []
        for organ_id, organ in self.organs.items():
            data.append({
                'Organ': organ['name'],
                'Primary Function': organ['function'],
                'Key Enzymes': ', '.join(organ['enzymes']),
                'Metabolism Rate': f"{organ['metabolism_rate']:.0%}"
            })
        
        return pd.DataFrame(data)
    
    def create_metabolism_timeline(self, tox21_predictions: Dict) -> go.Figure:
        """Create a timeline of metabolism process"""
        
        max_risk = max(tox21_predictions.values())
        metabolism_speed = 'Fast' if max_risk > 0.7 else 'Medium' if max_risk > 0.4 else 'Slow'
        
        steps = [
            {'time': 0, 'process': 'Absorption', 'duration': 30},
            {'time': 30, 'process': 'Distribution', 'duration': 15},
            {'time': 45, 'process': 'Liver Metabolism', 'duration': 60},
            {'time': 105, 'process': 'Kidney Processing', 'duration': 45},
            {'time': 150, 'process': 'Other Organs', 'duration': 30},
            {'time': 180, 'process': 'Elimination', 'duration': 60}
        ]
        
        fig = go.Figure()
        
        for i, step in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[step['time'], step['time'] + step['duration']],
                y=[i, i],
                mode='lines+markers',
                line=dict(width=20, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][i]),
                marker=dict(size=10),
                name=step['process'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Time: %{x} min<br>' +
                             f'Duration: {step["duration"]} min<br>' +
                             '<extra></extra>',
                text=[step['process']]
            ))
        
        fig.update_layout(
            title=f'Metabolism Timeline (Speed: {metabolism_speed})',
            xaxis_title='Time (minutes)',
            yaxis_title='Process Steps',
            yaxis=dict(ticktext=[step['process'] for step in steps], tickval=list(range(len(steps)))),
            height=400,
            paper_bgcolor='white'
        )
        
        return fig

def create_organ_metabolism_tab(smiles: str, tox21_predictions: Dict):
    """Create the organ metabolism video tab"""
    
    st.markdown("### 🎬 Human Organ Metabolism Video")
    st.markdown("""
    <div class="feature-card">
        <strong>🎬 Organ Metabolism Animation:</strong> Watch how human organs process chemicals in real-time
        <br><strong>🔬 Realistic Physiology:</strong> Based on actual human organ functions and enzyme systems
        <br><strong>⏱️ Timeline View:</strong> See the complete metabolism process from absorption to elimination
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize animator
    animator = OrganMetabolismAnimator()
    
    # Create organ metabolism video
    with st.spinner("Creating organ metabolism animation..."):
        video_fig = animator.create_organ_animation_video(smiles, tox21_predictions)
    
    # Display the video
    st.plotly_chart(video_fig, key="organ_metabolism_video")
    
    # Video controls info
    st.markdown("""
    <div class="warning-box">
        <strong>🎮 Video Controls:</strong>
        <ul>
            <li>▶️ <strong>Play Video:</strong> Watch the complete metabolism process</li>
            <li>⏸️ <strong>Pause:</strong> Stop at any point to examine organ activity</li>
            <li>🔄 <strong>Restart:</strong> Replay the video from the beginning</li>
        </ul>
        <p>The video shows: Absorption → Distribution → Liver Metabolism → Kidney Processing → Other Organs → Elimination</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Organ information table
    st.markdown("#### 🏥 Organ Function Information")
    
    organ_info_df = animator.create_organ_info_table()
    st.dataframe(organ_info_df, use_container_width=True)
    
    # Metabolism timeline
    st.markdown("#### ⏱️ Metabolism Process Timeline")
    
    timeline_fig = animator.create_metabolism_timeline(tox21_predictions)
    st.plotly_chart(timeline_fig, key="metabolism_timeline")
    
    # Detailed organ information
    st.markdown("#### 🧬 Detailed Organ Analysis")
    
    max_risk = max(tox21_predictions.values())
    toxicity_level = 'High' if max_risk > 0.7 else 'Medium' if max_risk > 0.4 else 'Low'
    
    organs = animator.organs
    for organ_id, organ in organs.items():
        with st.expander(f"🏥 {organ['name']} - {organ['function']}"):
            st.markdown(f"""
            **Primary Role:** {organ['function']}
            
            **Key Enzymes:** {', '.join(organ['enzymes'])}
            
            **Metabolism Rate:** {organ['metabolism_rate']:.0%}
            
            **Toxicity Impact:** {'High activity expected' if toxicity_level == 'High' and organ['metabolism_rate'] > 0.5 else 'Normal processing' if toxicity_level == 'Medium' else 'Minimal impact'}
            
            **Clinical Relevance:** This organ plays a {'critical' if organ['metabolism_rate'] > 0.7 else 'important' if organ['metabolism_rate'] > 0.4 else 'minor'} role in drug metabolism.
            """)
            
            # Add organ-specific warnings
            if organ_id == 'liver':
                st.markdown("""
                <div class="warning-box">
                    🚨 **Liver Warning:** Primary metabolism site. High toxicity compounds may cause liver stress or damage.
                </div>
                """, unsafe_allow_html=True)
            elif organ_id == 'kidney':
                st.markdown("""
                <div class="warning-box">
                    ⚠️ **Kidney Warning:** Responsible for excretion. High toxicity may affect kidney function.
                </div>
                """, unsafe_allow_html=True)
            elif organ_id == 'brain':
                st.markdown("""
                <div class="warning-box">
                    🧠 **Brain Protection:** Blood-brain barrier limits chemical entry. High toxicity may still affect CNS.
                </div>
                """, unsafe_allow_html=True)

def create_simple_organ_demo():
    """Create a simple demonstration of organ metabolism"""
    
    st.markdown("### 🎬 Organ Metabolism Demo")
    st.markdown("""
    <div class="feature-card">
        <strong>🎬 Demo Video:</strong> Shows how aspirin is processed by human organs
        <br><strong>🔬 Real Physiology:</strong> Based on actual human metabolism
        <br><strong>⏱️ 6 Steps:</strong> Complete metabolism process visualization
    </div>
    """, unsafe_allow_html=True)
    
    # Create demo animation
    animator = OrganMetabolismAnimator()
    
    # Demo predictions for aspirin
    demo_predictions = {
        'NR-AR': 0.1,
        'NR-AhR': 0.2,
        'NR-AR-LBD': 0.1,
        'NR-ER': 0.1,
        'NR-ER-LBD': 0.1,
        'NR-PPAR-gamma': 0.1,
        'SR-ARE': 0.3,
        'SR-ATAD5': 0.1,
        'SR-HSE': 0.2,
        'SR-MMP': 0.1,
        'SR-p53': 0.1
    }
    
    # Create demo video
    demo_fig = animator.create_organ_animation_video("CC(=O)Oc1ccccc1C(=O)O", demo_predictions)
    
    st.plotly_chart(demo_fig, key="organ_demo")
    
    st.markdown("#### 📋 Demo Information")
    st.markdown("""
    **Compound:** Aspirin (Acetylsalicylic Acid)
    **Toxicity Level:** Low
    **Processing Time:** ~3 hours
    **Primary Organs:** Liver (metabolism), Kidneys (excretion)
    """)
