"""
Advanced Molecular Visualization Module
Interactive 2D/3D molecular structure viewer with highlighting capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import io
import base64
from pathlib import Path
import warnings

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDraw2D, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import rdGeometry as rdGeom
from rdkit.Chem import rdMolDescriptors, Descriptors

# 3D visualization (if available)
try:
    import py3Dmol
    P3Dmol_AVAILABLE = True
except ImportError:
    P3Dmol_AVAILABLE = False
    warnings.warn("py3Dmol not available. 3D visualization will be limited.")

# Plotly for interactive plots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


class MolecularViewer:
    """
    Advanced molecular visualization with 2D/3D rendering and highlighting
    """
    
    def __init__(self):
        self.default_size = (400, 400)
        self.default_bond_width = 2
        self.default_atom_radius = 0.5
        self.color_schemes = {
            'default': {
                'C': '#909090',
                'N': '#3050F8',
                'O': '#FF0D0D',
                'F': '#90E050',
                'P': '#FF8000',
                'S': '#FFFF30',
                'Cl': '#1FF01F',
                'Br': '#A62929',
                'I': '#940094',
                'H': '#FFFFFF'
            },
            'highlight': {
                'high_importance': '#FF0000',
                'medium_importance': '#FFA500',
                'low_importance': '#00FF00',
                'negative_importance': '#0000FF'
            }
        }
    
    def visualize_molecule_2d(self, smiles: str, 
                             size: Tuple[int, int] = (400, 400),
                             highlight_atoms: List[int] = None,
                             atom_colors: Dict[int, str] = None,
                             bond_colors: Dict[Tuple[int, int], str] = None,
                             show_atom_numbers: bool = False,
                             wedge_bonds: bool = True) -> plt.Figure:
        """
        Create 2D molecular structure visualization
        
        Args:
            smiles: SMILES string of the molecule
            size: Image size (width, height)
            highlight_atoms: List of atom indices to highlight
            atom_colors: Dictionary mapping atom indices to colors
            bond_colors: Dictionary mapping bond tuples to colors
            show_atom_numbers: Whether to show atom numbers
            wedge_bonds: Whether to use wedge bonds for stereochemistry
            
        Returns:
            Matplotlib figure
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Add hydrogens for better visualization
            mol = Chem.AddHs(mol)
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Get atom positions
            conf = mol.GetConformer()
            atom_positions = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_positions.append((pos.x, pos.y))
            
            # Draw bonds
            for bond in mol.GetBonds():
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                
                pos1 = atom_positions[atom1_idx]
                pos2 = atom_positions[atom2_idx]
                
                # Determine bond color
                bond_key = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
                bond_color = bond_colors.get(bond_key, '#000000') if bond_colors else '#000000'
                
                # Draw bond
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           color=bond_color, linewidth=self.default_bond_width)
                elif bond.GetBondType() == Chem.BondType.DOUBLE:
                    # Draw double bond
                    offset = 0.1
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx_norm = dx / length
                        dy_norm = dy / length
                        offset_x = -dy_norm * offset
                        offset_y = dx_norm * offset
                        
                        ax.plot([pos1[0] + offset_x, pos2[0] + offset_x], 
                               [pos1[1] + offset_y, pos2[1] + offset_y], 
                               color=bond_color, linewidth=self.default_bond_width)
                        ax.plot([pos1[0] - offset_x, pos2[0] - offset_x], 
                               [pos1[1] - offset_y, pos2[1] - offset_y], 
                               color=bond_color, linewidth=self.default_bond_width)
                elif bond.GetBondType() == Chem.BondType.TRIPLE:
                    # Draw triple bond
                    offset = 0.15
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx_norm = dx / length
                        dy_norm = dy / length
                        offset_x = -dy_norm * offset
                        offset_y = dx_norm * offset
                        
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               color=bond_color, linewidth=self.default_bond_width)
                        ax.plot([pos1[0] + offset_x, pos2[0] + offset_x], 
                               [pos1[1] + offset_y, pos2[1] + offset_y], 
                               color=bond_color, linewidth=self.default_bond_width)
                        ax.plot([pos1[0] - offset_x, pos2[0] - offset_x], 
                               [pos1[1] - offset_y, pos2[1] - offset_y], 
                               color=bond_color, linewidth=self.default_bond_width)
            
            # Draw atoms
            for i, atom in enumerate(mol.GetAtoms()):
                pos = atom_positions[i]
                atom_symbol = atom.GetSymbol()
                atomic_num = atom.GetAtomicNum()
                
                # Determine atom color
                if highlight_atoms and i in highlight_atoms:
                    atom_color = atom_colors.get(i, '#FF0000') if atom_colors else '#FF0000'
                else:
                    atom_color = self.color_schemes['default'].get(atom_symbol, '#808080')
                
                # Draw atom circle
                circle = plt.Circle(pos, self.default_atom_radius, 
                                  color=atom_color, ec='black', linewidth=1)
                ax.add_patch(circle)
                
                # Add atom symbol
                ax.text(pos[0], pos[1], atom_symbol, 
                       ha='center', va='center', fontsize=8, 
                       color='white' if atomic_num > 1 else 'black')
                
                # Add atom number if requested
                if show_atom_numbers:
                    ax.text(pos[0], pos[1] - 1.2, str(i), 
                           ha='center', va='center', fontsize=6, color='black')
            
            # Set plot limits
            if atom_positions:
                x_coords = [pos[0] for pos in atom_positions]
                y_coords = [pos[1] for pos in atom_positions]
                margin = 2.0
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            # Create error figure
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, f"Error rendering molecule:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig
    
    def visualize_molecule_3d(self, smiles: str,
                             width: int = 400,
                             height: int = 400,
                             style: str = 'stick',
                             background: str = 'white',
                             show_atom_labels: bool = True) -> str:
        """
        Create 3D molecular structure visualization using py3Dmol
        
        Args:
            smiles: SMILES string of the molecule
            width: Viewer width
            height: Viewer height
            style: Visualization style ('stick', 'sphere', 'line')
            background: Background color
            show_atom_labels: Whether to show atom labels
            
        Returns:
            HTML string for 3D viewer
        """
        if not P3Dmol_AVAILABLE:
            return "<p>3D visualization not available. Please install py3Dmol.</p>"
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "<p>Invalid SMILES string</p>"
            
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            
            # Convert to PDB format for py3Dmol
            pdb = Chem.rdmolfiles.MolToPDBBlock(mol)
            
            # Create py3Dmol viewer
            viewer = py3Dmol.view(width=width, height=height)
            viewer.addModel(pdb, 'pdb')
            
            # Set style
            if style == 'stick':
                viewer.setStyle({'stick': {}})
            elif style == 'sphere':
                viewer.setStyle({'sphere': {}})
            elif style == 'line':
                viewer.setStyle({'line': {}})
            
            # Set background
            viewer.setBackgroundColor(background)
            
            # Add atom labels if requested
            if show_atom_labels:
                viewer.addLabel('C', {'position': {'x': 0, 'y': 0, 'z': 0}})
            
            # Zoom to fit
            viewer.zoomTo()
            
            return viewer._make_html().data()
            
        except Exception as e:
            return f"<p>Error creating 3D visualization: {str(e)}</p>"
    
    def highlight_toxicity_features(self, smiles: str,
                                  feature_importance: Dict[str, float],
                                  atom_importance: Dict[int, float] = None,
                                  threshold: float = 0.5) -> plt.Figure:
        """
        Highlight molecular features based on toxicity importance
        
        Args:
            smiles: SMILES string
            feature_importance: Dictionary of feature importances
            atom_importance: Dictionary of atom-level importances
            threshold: Threshold for highlighting
            
        Returns:
            Matplotlib figure with highlighted molecule
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Get atom positions
            conf = mol.GetConformer()
            atom_positions = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_positions.append((pos.x, pos.y))
            
            # Determine atom colors based on importance
            atom_colors = {}
            if atom_importance:
                for atom_idx, importance in atom_importance.items():
                    if importance > threshold:
                        atom_colors[atom_idx] = self.color_schemes['highlight']['high_importance']
                    elif importance > threshold * 0.5:
                        atom_colors[atom_idx] = self.color_schemes['highlight']['medium_importance']
                    else:
                        atom_colors[atom_idx] = self.color_schemes['highlight']['low_importance']
            
            # Draw molecule with highlighting
            drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
            drawer.SetFontSize(0.8)
            
            # Prepare highlighting
            if atom_colors:
                highlight_atoms = list(atom_colors.keys())
                highlight_colors = list(atom_colors.values())
                
                rdMolDraw2D.PrepareAndDrawMolecule(
                    drawer, mol, 
                    highlightAtoms=highlight_atoms,
                    highlightAtomColors={i: c for i, c in zip(highlight_atoms, highlight_colors)}
                )
            else:
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            
            drawer.FinishDrawing()
            
            # Convert to matplotlib
            from PIL import Image
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                drawer.WriteToFile(tmp.name)
                img = Image.open(tmp.name)
                ax.imshow(img)
                Path(tmp.name).unlink()  # Clean up
            
            # Add importance legend
            legend_elements = [
                patches.Patch(color=self.color_schemes['highlight']['high_importance'], 
                            label='High Importance'),
                patches.Patch(color=self.color_schemes['highlight']['medium_importance'], 
                            label='Medium Importance'),
                patches.Patch(color=self.color_schemes['highlight']['low_importance'], 
                            label='Low Importance')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, f"Error highlighting features:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig
    
    def create_interactive_plotly_viewer(self, smiles: str,
                                       show_atom_labels: bool = True,
                                       show_bond_info: bool = True) -> go.Figure:
        """
        Create interactive Plotly molecular viewer
        
        Args:
            smiles: SMILES string
            show_atom_labels: Whether to show atom labels
            show_bond_info: Whether to show bond information on hover
            
        Returns:
            Plotly figure
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Add hydrogens and generate 2D coordinates
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            
            # Get atom positions and properties
            conf = mol.GetConformer()
            atom_data = []
            
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_data.append({
                    'x': pos.x,
                    'y': pos.y,
                    'z': pos.z,
                    'symbol': atom.GetSymbol(),
                    'atomic_num': atom.GetAtomicNum(),
                    'index': atom.GetIdx(),
                    'formal_charge': atom.GetFormalCharge(),
                    'hybridization': str(atom.GetHybridization())
                })
            
            # Get bond data
            bond_data = []
            for bond in mol.GetBonds():
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                
                pos1 = conf.GetAtomPosition(atom1_idx)
                pos2 = conf.GetAtomPosition(atom2_idx)
                
                bond_data.append({
                    'x': [pos1.x, pos2.x],
                    'y': [pos1.y, pos2.y],
                    'z': [pos1.z, pos2.z],
                    'bond_type': str(bond.GetBondType()),
                    'bond_order': bond.GetBondTypeAsDouble(),
                    'is_aromatic': bond.GetIsAromatic(),
                    'atom1': atom1_idx,
                    'atom2': atom2_idx
                })
            
            # Create figure
            fig = go.Figure()
            
            # Add bonds
            for bond in bond_data:
                fig.add_trace(go.Scatter(
                    x=bond['x'],
                    y=bond['y'],
                    mode='lines',
                    line=dict(width=3*bond['bond_order'], color='black'),
                    showlegend=False,
                    hoverinfo='text',
                    text=f"Bond: {bond['bond_type']}, Order: {bond['bond_order']}, Aromatic: {bond['is_aromatic']}"
                ))
            
            # Add atoms
            atom_df = pd.DataFrame(atom_data)
            
            for symbol in atom_df['symbol'].unique():
                subset = atom_df[atom_df['symbol'] == symbol]
                
                fig.add_trace(go.Scatter(
                    x=subset['x'],
                    y=subset['y'],
                    mode='markers+text' if show_atom_labels else 'markers',
                    marker=dict(
                        size=15,
                        color=self.color_schemes['default'].get(symbol, '#808080'),
                        line=dict(width=1, color='black')
                    ),
                    text=subset['symbol'] if show_atom_labels else None,
                    textposition='middle center',
                    textfont=dict(size=10, color='white' if symbol != 'H' else 'black'),
                    name=symbol,
                    hoverinfo='text',
                    text=[f"Atom: {row['symbol']} ({row['index']})<br>"
                          f"Atomic Number: {row['atomic_num']}<br>"
                          f"Charge: {row['formal_charge']}<br>"
                          f"Hybridization: {row['hybridization']}" 
                          for _, row in subset.iterrows()]
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Interactive Molecular Structure: {smiles}",
                showlegend=True,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                width=600,
                height=600
            )
            
            # Set equal aspect ratio
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating interactive viewer:<br>{str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
    
    def create_molecular_property_heatmap(self, smiles_list: List[str],
                                        properties: Dict[str, List[float]],
                                        title: str = "Molecular Property Heatmap") -> go.Figure:
        """
        Create heatmap of molecular properties for multiple compounds
        
        Args:
            smiles_list: List of SMILES strings
            properties: Dictionary of property names to values
            title: Heatmap title
            
        Returns:
            Plotly heatmap figure
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(properties)
            df.index = [f"Compound {i+1}" for i in range(len(smiles_list))]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale='RdYlBu_r',
                showscale=True,
                text=np.round(df.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Properties",
                yaxis_title="Compounds",
                width=800,
                height=max(400, len(smiles_list) * 30)
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating heatmap:<br>{str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
    
    def create_conformer_comparison(self, smiles: str, num_conformers: int = 5) -> go.Figure:
        """
        Create comparison of multiple molecular conformers
        
        Args:
            smiles: SMILES string
            num_conformers: Number of conformers to generate
            
        Returns:
            Plotly figure with conformer comparison
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate multiple conformers
            conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, 
                                                     randomSeed=42, pruneRmsThresh=0.5)
            
            # Optimize conformers
            for cid in conformer_ids:
                AllChem.UFFOptimizeMolecule(mol, confId=cid)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f"Conformer {i+1}" for i in range(min(num_conformers, 6))],
                specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Plot each conformer
            for i, cid in enumerate(conformer_ids[:6]):
                row = i // 3 + 1
                col = i % 3 + 1
                
                # Get coordinates for this conformer
                conf = mol.GetConformer(cid)
                atom_positions = []
                atom_symbols = []
                
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    atom_positions.append([pos.x, pos.y, pos.z])
                    atom_symbols.append(atom.GetSymbol())
                
                atom_positions = np.array(atom_positions)
                
                # Plot atoms
                fig.add_trace(
                    go.Scatter(
                        x=atom_positions[:, 0],
                        y=atom_positions[:, 1],
                        mode='markers+text',
                        marker=dict(size=8, color='blue'),
                        text=atom_symbols,
                        textposition='middle center',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Plot bonds (simplified 2D projection)
                for bond in mol.GetBonds():
                    atom1_idx = bond.GetBeginAtomIdx()
                    atom2_idx = bond.GetEndAtomIdx()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[atom_positions[atom1_idx, 0], atom_positions[atom2_idx, 0]],
                            y=[atom_positions[atom1_idx, 1], atom_positions[atom2_idx, 1]],
                            mode='lines',
                            line=dict(color='black', width=2),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=f"Molecular Conformers: {smiles}",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating conformers:<br>{str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
    
    def visualize_molecule(self, smiles: str, 
                         visualization_type: str = "2d",
                         **kwargs) -> Union[plt.Figure, str, go.Figure]:
        """
        Main visualization method with multiple options
        
        Args:
            smiles: SMILES string
            visualization_type: Type of visualization ('2d', '3d', 'interactive', 'heatmap')
            **kwargs: Additional arguments for specific visualization types
            
        Returns:
            Visualization object (matplotlib figure, HTML string, or Plotly figure)
        """
        if visualization_type == "2d":
            return self.visualize_molecule_2d(smiles, **kwargs)
        elif visualization_type == "3d":
            return self.visualize_molecule_3d(smiles, **kwargs)
        elif visualization_type == "interactive":
            return self.create_interactive_plotly_viewer(smiles, **kwargs)
        elif visualization_type == "heatmap":
            return self.create_molecular_property_heatmap(**kwargs)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")


class MolecularAnimation:
    """
    Create animated molecular visualizations
    """
    
    def __init__(self):
        self.viewer = MolecularViewer()
    
    def create_rotation_animation(self, smiles: str, 
                                frames: int = 36,
                                output_format: str = "gif") -> str:
        """
        Create 3D rotation animation
        
        Args:
            smiles: SMILES string
            frames: Number of rotation frames
            output_format: Output format ('gif', 'mp4', 'html')
            
        Returns:
            Path to animation file or HTML string
        """
        if not P3Dmol_AVAILABLE:
            return "<p>3D animation not available. Please install py3Dmol.</p>"
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "<p>Invalid SMILES string</p>"
            
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            
            # Convert to PDB format
            pdb = Chem.rdmolfiles.MolToPDBBlock(mol)
            
            # Create py3Dmol viewer with animation
            viewer = py3Dmol.view(width=400, height=400)
            viewer.addModel(pdb, 'pdb')
            viewer.setStyle({'stick': {}})
            viewer.setBackgroundColor('white')
            
            # Create rotation animation
            viewer.animate({
                'loop': 'backandforth',
                'interval': 50
            })
            
            # Add rotation frames
            for i in range(frames):
                angle = i * (360 / frames)
                viewer.rotate(angle, 'y')
            
            viewer.zoomTo()
            
            if output_format == "html":
                return viewer._make_html().data()
            else:
                # For GIF/MP4, would need additional libraries
                return "<p>Animation export requires additional setup</p>"
                
        except Exception as e:
            return f"<p>Error creating animation: {str(e)}</p>"
    
    def create_vibration_animation(self, smiles: str,
                                 vibration_mode: int = 0) -> str:
        """
        Create molecular vibration animation
        
        Args:
            smiles: SMILES string
            vibration_mode: Vibration mode to animate
            
        Returns:
            HTML string for animation
        """
        # This is a placeholder - real implementation would require
        # quantum chemistry calculations for vibrational modes
        return "<p>Vibration animation requires quantum chemistry calculations</p>"


class MolecularDashboard:
    """
    Comprehensive molecular visualization dashboard
    """
    
    def __init__(self):
        self.viewer = MolecularViewer()
        self.animation = MolecularAnimation()
    
    def create_comprehensive_dashboard(self, smiles: str) -> Dict[str, Any]:
        """
        Create comprehensive visualization dashboard
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with multiple visualizations
        """
        dashboard = {
            'smiles': smiles,
            'visualizations': {}
        }
        
        try:
            # 2D structure
            dashboard['visualizations']['2d_structure'] = self.viewer.visualize_molecule_2d(smiles)
            
            # 3D structure (if available)
            if P3Dmol_AVAILABLE:
                dashboard['visualizations']['3d_structure'] = self.viewer.visualize_molecule_3d(smiles)
            
            # Interactive viewer
            dashboard['visualizations']['interactive'] = self.viewer.create_interactive_plotly_viewer(smiles)
            
            # Conformer comparison
            dashboard['visualizations']['conformers'] = self.viewer.create_conformer_comparison(smiles)
            
            # Molecular properties
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties = {
                    'Molecular Weight': [Descriptors.MolWt(mol)],
                    'LogP': [Descriptors.MolLogP(mol)],
                    'TPSA': [Descriptors.TPSA(mol)],
                    'Num H Donors': [Descriptors.NumHDonors(mol)],
                    'Num H Acceptors': [Descriptors.NumHAcceptors(mol)],
                    'Rotatable Bonds': [Descriptors.NumRotatableBonds(mol)]
                }
                
                dashboard['visualizations']['properties'] = self.viewer.create_molecular_property_heatmap(
                    [smiles], properties, "Molecular Properties"
                )
            
        except Exception as e:
            dashboard['error'] = str(e)
        
        return dashboard


# Utility functions
def save_molecule_image(fig: plt.Figure, filename: str, dpi: int = 300):
    """Save molecular visualization to file"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def molecule_to_base64(fig: plt.Figure, format: str = 'png') -> str:
    """Convert molecular visualization to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format=format, bbox_inches='tight', pad_inches=0.1)
    buffer.seek(0)
    
    image_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/{format};base64,{image_base64}"


def create_molecular_gallery(smiles_list: List[str], 
                           titles: List[str] = None,
                           cols: int = 3) -> plt.Figure:
    """
    Create gallery of molecular structures
    
    Args:
        smiles_list: List of SMILES strings
        titles: List of titles for each molecule
        cols: Number of columns in gallery
        
    Returns:
        Matplotlib figure with molecular gallery
    """
    n_molecules = len(smiles_list)
    rows = (n_molecules + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    viewer = MolecularViewer()
    
    for i, smiles in enumerate(smiles_list):
        row = i // cols
        col = i % cols
        
        mol_fig = viewer.visualize_molecule_2d(smiles, size=(300, 300))
        
        # Copy the molecular figure to the subplot
        for ax in mol_fig.axes:
            for child in ax.get_children():
                axes[row, col].add_child(child)
        
        axes[row, col].set_title(titles[i] if titles and i < len(titles) else f"Molecule {i+1}")
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_molecules, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig
