"""
Advanced Real-Time Molecule Builder Interface
Interactive drag-and-drop molecule editor with toxicity prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import logging
from pathlib import Path
import warnings

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDraw2D, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import rdGeometry as rdGeom

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches

# Data structures
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AtomType(Enum):
    """Supported atom types for molecule builder"""
    CARBON = "C"
    NITROGEN = "N"
    OXYGEN = "O"
    FLUORINE = "F"
    CHLORINE = "Cl"
    BROMINE = "Br"
    IODINE = "I"
    SULFUR = "S"
    PHOSPHORUS = "P"
    HYDROGEN = "H"


class BondType(Enum):
    """Supported bond types"""
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4


@dataclass
class Atom:
    """Atom representation for molecule builder"""
    element: str
    x: float
    y: float
    z: float = 0.0
    index: int = 0
    charge: int = 0
    is_aromatic: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'element': self.element,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'index': self.index,
            'charge': self.charge,
            'is_aromatic': self.is_aromatic
        }


@dataclass
class Bond:
    """Bond representation for molecule builder"""
    atom1_idx: int
    atom2_idx: int
    bond_type: BondType
    is_aromatic: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'atom1_idx': self.atom1_idx,
            'atom2_idx': self.atom2_idx,
            'bond_type': self.bond_type.value,
            'is_aromatic': self.is_aromatic
        }


class MoleculeBuilder:
    """
    Advanced real-time molecule builder with interactive editing
    """
    
    def __init__(self):
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.selected_atom: Optional[int] = None
        self.selected_bond: Optional[int] = None
        self.current_tool: str = "add_atom"
        self.current_atom_type: AtomType = AtomType.CARBON
        self.current_bond_type: BondType = BondType.SINGLE
        self.grid_size = 20
        self.snap_to_grid = True
        self.auto_add_hydrogens = True
        
        # Visual settings
        self.atom_colors = {
            'C': '#909090',
            'N': '#3050F8',
            'O': '#FF0D0D',
            'F': '#90E050',
            'Cl': '#1FF01F',
            'Br': '#A62929',
            'I': '#940094',
            'S': '#FFFF30',
            'P': '#FF8000',
            'H': '#FFFFFF'
        }
        
        self.atom_sizes = {
            'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Cl': 1.75,
            'Br': 1.85, 'I': 1.98, 'S': 1.80, 'P': 1.80, 'H': 1.10
        }
        
        logger.info("Molecule builder initialized")
    
    def add_atom(self, x: float, y: float, element: str = None) -> int:
        """
        Add an atom to the molecule
        
        Args:
            x: X coordinate
            y: Y coordinate
            element: Element symbol (optional, uses current atom type)
            
        Returns:
            Index of added atom
        """
        if element is None:
            element = self.current_atom_type.value
        
        # Snap to grid if enabled
        if self.snap_to_grid:
            x = round(x / self.grid_size) * self.grid_size
            y = round(y / self.grid_size) * self.grid_size
        
        # Create new atom
        atom = Atom(
            element=element,
            x=x,
            y=y,
            index=len(self.atoms)
        )
        
        self.atoms.append(atom)
        logger.info(f"Added atom {element} at ({x}, {y})")
        
        return atom.index
    
    def add_bond(self, atom1_idx: int, atom2_idx: int, bond_type: BondType = None) -> bool:
        """
        Add a bond between two atoms
        
        Args:
            atom1_idx: Index of first atom
            atom2_idx: Index of second atom
            bond_type: Type of bond (optional, uses current bond type)
            
        Returns:
            True if bond added successfully
        """
        if bond_type is None:
            bond_type = self.current_bond_type
        
        # Validate atom indices
        if atom1_idx >= len(self.atoms) or atom2_idx >= len(self.atoms):
            return False
        
        # Check if bond already exists
        for bond in self.bonds:
            if (bond.atom1_idx == atom1_idx and bond.atom2_idx == atom2_idx) or \
               (bond.atom1_idx == atom2_idx and bond.atom2_idx == atom1_idx):
                return False
        
        # Create new bond
        bond = Bond(
            atom1_idx=atom1_idx,
            atom2_idx=atom2_idx,
            bond_type=bond_type
        )
        
        self.bonds.append(bond)
        logger.info(f"Added bond between atoms {atom1_idx} and {atom2_idx}")
        
        return True
    
    def delete_atom(self, atom_idx: int) -> bool:
        """
        Delete an atom and its connected bonds
        
        Args:
            atom_idx: Index of atom to delete
            
        Returns:
            True if atom deleted successfully
        """
        if atom_idx >= len(self.atoms):
            return False
        
        # Remove connected bonds
        self.bonds = [bond for bond in self.bonds 
                     if bond.atom1_idx != atom_idx and bond.atom2_idx != atom_idx]
        
        # Remove atom
        del self.atoms[atom_idx]
        
        # Update atom indices
        for atom in self.atoms:
            if atom.index > atom_idx:
                atom.index -= 1
        
        # Update bond indices
        for bond in self.bonds:
            if bond.atom1_idx > atom_idx:
                bond.atom1_idx -= 1
            if bond.atom2_idx > atom_idx:
                bond.atom2_idx -= 1
        
        logger.info(f"Deleted atom {atom_idx}")
        return True
    
    def delete_bond(self, bond_idx: int) -> bool:
        """
        Delete a bond
        
        Args:
            bond_idx: Index of bond to delete
            
        Returns:
            True if bond deleted successfully
        """
        if bond_idx >= len(self.bonds):
            return False
        
        del self.bonds[bond_idx]
        logger.info(f"Deleted bond {bond_idx}")
        return True
    
    def find_nearest_atom(self, x: float, y: float, max_distance: float = 20) -> Optional[int]:
        """
        Find the nearest atom to the given coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            max_distance: Maximum search distance
            
        Returns:
            Index of nearest atom or None
        """
        min_distance = float('inf')
        nearest_atom = None
        
        for atom in self.atoms:
            distance = np.sqrt((atom.x - x)**2 + (atom.y - y)**2)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_atom = atom.index
        
        return nearest_atom
    
    def find_nearest_bond(self, x: float, y: float, max_distance: float = 15) -> Optional[int]:
        """
        Find the nearest bond to the given coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            max_distance: Maximum search distance
            
        Returns:
            Index of nearest bond or None
        """
        min_distance = float('inf')
        nearest_bond = None
        
        for i, bond in enumerate(self.bonds):
            atom1 = self.atoms[bond.atom1_idx]
            atom2 = self.atoms[bond.atom2_idx]
            
            # Calculate distance from point to line segment
            distance = self._point_to_line_distance(x, y, atom1.x, atom1.y, atom2.x, atom2.y)
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_bond = i
        
        return nearest_bond
    
    def _point_to_line_distance(self, px: float, py: float, 
                                x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance from point to line segment"""
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length**2))
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        return np.sqrt((px - projection_x)**2 + (py - projection_y)**2)
    
    def to_rdkit_mol(self) -> Optional[Chem.Mol]:
        """
        Convert builder molecule to RDKit molecule
        
        Returns:
            RDKit molecule or None if conversion fails
        """
        if not self.atoms:
            return None
        
        # Create RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        atom_map = {}
        for i, atom in enumerate(self.atoms):
            rd_atom = Chem.Atom(atom.element)
            rd_atom.SetFormalCharge(atom.charge)
            rd_atom.SetIsAromatic(atom.is_aromatic)
            
            atom_idx = mol.AddAtom(rd_atom)
            atom_map[i] = atom_idx
        
        # Add bonds
        for bond in self.bonds:
            atom1_idx = atom_map[bond.atom1_idx]
            atom2_idx = atom_map[bond.atom2_idx]
            
            if bond.bond_type == BondType.SINGLE:
                mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
            elif bond.bond_type == BondType.DOUBLE:
                mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.DOUBLE)
            elif bond.bond_type == BondType.TRIPLE:
                mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.TRIPLE)
            elif bond.bond_type == BondType.AROMATIC:
                mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.AROMATIC)
        
        # Convert to regular molecule
        mol = mol.GetMol()
        
        # Add hydrogens if enabled
        if self.auto_add_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Generate 2D coordinates
        try:
            AllChem.Compute2DCoords(mol)
        except:
            logger.warning("Failed to generate 2D coordinates")
        
        return mol
    
    def from_rdkit_mol(self, mol: Chem.Mol) -> bool:
        """
        Load molecule from RDKit molecule
        
        Args:
            mol: RDKit molecule to load
            
        Returns:
            True if loaded successfully
        """
        try:
            # Clear current molecule
            self.atoms = []
            self.bonds = []
            
            # Get conformer coordinates
            conf = mol.GetConformer()
            
            # Add atoms
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                new_atom = Atom(
                    element=atom.GetSymbol(),
                    x=pos.x,
                    y=pos.y,
                    z=pos.z,
                    index=len(self.atoms),
                    charge=atom.GetFormalCharge(),
                    is_aromatic=atom.GetIsAromatic()
                )
                self.atoms.append(new_atom)
            
            # Add bonds
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.SINGLE:
                    bt = BondType.SINGLE
                elif bond_type == Chem.BondType.DOUBLE:
                    bt = BondType.DOUBLE
                elif bond_type == Chem.BondType.TRIPLE:
                    bt = BondType.TRIPLE
                elif bond_type == Chem.BondType.AROMATIC:
                    bt = BondType.AROMATIC
                else:
                    bt = BondType.SINGLE
                
                new_bond = Bond(
                    atom1_idx=bond.GetBeginAtomIdx(),
                    atom2_idx=bond.GetEndAtomIdx(),
                    bond_type=bt,
                    is_aromatic=bond.GetIsAromatic()
                )
                self.bonds.append(new_bond)
            
            logger.info(f"Loaded molecule with {len(self.atoms)} atoms and {len(self.bonds)} bonds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RDKit molecule: {e}")
            return False
    
    def to_smiles(self) -> str:
        """Convert molecule to SMILES string"""
        mol = self.to_rdkit_mol()
        if mol is None:
            return ""
        
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return ""
    
    def from_smiles(self, smiles: str) -> bool:
        """
        Load molecule from SMILES string
        
        Args:
            smiles: SMILES string to load
            
        Returns:
            True if loaded successfully
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            return self.from_rdkit_mol(mol)
            
        except Exception as e:
            logger.error(f"Failed to load SMILES: {e}")
            return False
    
    def clear(self):
        """Clear the molecule builder"""
        self.atoms = []
        self.bonds = []
        self.selected_atom = None
        self.selected_bond = None
        logger.info("Molecule builder cleared")
    
    def get_molecular_formula(self) -> str:
        """Get molecular formula"""
        mol = self.to_rdkit_mol()
        if mol is None:
            return ""
        
        try:
            return rdMolDescriptors.CalcMolFormula(mol)
        except:
            return ""
    
    def get_molecular_weight(self) -> float:
        """Get molecular weight"""
        mol = self.to_rdkit_mol()
        if mol is None:
            return 0.0
        
        try:
            return rdMolDescriptors.CalcExactMolWt(mol)
        except:
            return 0.0
    
    def validate_molecule(self) -> Dict[str, Any]:
        """
        Validate the molecule and return diagnostic information
        
        Returns:
            Dictionary with validation results
        """
        mol = self.to_rdkit_mol()
        if mol is None:
            return {
                'valid': False,
                'errors': ['Cannot create RDKit molecule'],
                'warnings': []
            }
        
        errors = []
        warnings_list = []
        
        # Check for valence issues
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            errors.append(f"Valence error: {str(e)}")
        
        # Check for unusual charges
        total_charge = Chem.rdmolops.GetFormalCharge(mol)
        if abs(total_charge) > 3:
            warnings_list.append(f"High formal charge: {total_charge}")
        
        # Check for unusual atom patterns
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                errors.append("Invalid atom found")
            
            # Check for unusual valence states
            valence = atom.GetTotalValence()
            if atom.GetSymbol() == 'C' and valence > 4:
                warnings_list.append(f"Unusual carbon valence: {valence}")
            elif atom.GetSymbol() == 'N' and valence > 5:
                warnings_list.append(f"Unusual nitrogen valence: {valence}")
            elif atom.GetSymbol() == 'O' and valence > 2:
                warnings_list.append(f"Unusual oxygen valence: {valence}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings_list,
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'formula': self.get_molecular_formula(),
            'molecular_weight': self.get_molecular_weight()
        }
    
    def to_dict(self) -> Dict:
        """Convert molecule builder to dictionary"""
        return {
            'atoms': [atom.to_dict() for atom in self.atoms],
            'bonds': [bond.to_dict() for bond in self.bonds],
            'settings': {
                'current_atom_type': self.current_atom_type.value,
                'current_bond_type': self.current_bond_type.value,
                'snap_to_grid': self.snap_to_grid,
                'auto_add_hydrogens': self.auto_add_hydrogens,
                'grid_size': self.grid_size
            }
        }
    
    def from_dict(self, data: Dict) -> bool:
        """
        Load molecule builder from dictionary
        
        Args:
            data: Dictionary with molecule data
            
        Returns:
            True if loaded successfully
        """
        try:
            # Clear current molecule
            self.clear()
            
            # Load settings
            settings = data.get('settings', {})
            self.current_atom_type = AtomType(settings.get('current_atom_type', 'C'))
            self.current_bond_type = BondType(settings.get('current_bond_type', 1))
            self.snap_to_grid = settings.get('snap_to_grid', True)
            self.auto_add_hydrogens = settings.get('auto_add_hydrogens', True)
            self.grid_size = settings.get('grid_size', 20)
            
            # Load atoms
            for atom_data in data.get('atoms', []):
                atom = Atom(
                    element=atom_data['element'],
                    x=atom_data['x'],
                    y=atom_data['y'],
                    z=atom_data.get('z', 0.0),
                    index=atom_data['index'],
                    charge=atom_data.get('charge', 0),
                    is_aromatic=atom_data.get('is_aromatic', False)
                )
                self.atoms.append(atom)
            
            # Load bonds
            for bond_data in data.get('bonds', []):
                bond = Bond(
                    atom1_idx=bond_data['atom1_idx'],
                    atom2_idx=bond_data['atom2_idx'],
                    bond_type=BondType(bond_data['bond_type']),
                    is_aromatic=bond_data.get('is_aromatic', False)
                )
                self.bonds.append(bond)
            
            logger.info(f"Loaded molecule from dictionary: {len(self.atoms)} atoms, {len(self.bonds)} bonds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from dictionary: {e}")
            return False


class InteractiveMoleculeEditor:
    """
    Interactive molecule editor with real-time visualization
    """
    
    def __init__(self):
        self.builder = MoleculeBuilder()
        self.fig_size = (800, 600)
        self.show_grid = True
        self.show_atom_labels = True
        self.show_bond_labels = False
        
    def create_interactive_plot(self) -> go.Figure:
        """
        Create interactive plot for molecule editing
        
        Returns:
            Plotly figure for interactive editing
        """
        fig = go.Figure()
        
        # Add grid if enabled
        if self.show_grid and self.builder.snap_to_grid:
            grid_size = self.builder.grid_size
            grid_range = 400
            
            # Vertical grid lines
            for x in range(-grid_range, grid_range + 1, grid_size):
                fig.add_shape(
                    type="line",
                    x0=x, y0=-grid_range, x1=x, y1=grid_range,
                    line=dict(color="lightgray", width=1, dash="dot")
                )
            
            # Horizontal grid lines
            for y in range(-grid_range, grid_range + 1, grid_size):
                fig.add_shape(
                    type="line",
                    x0=-grid_range, y0=y, x1=grid_range, y1=y,
                    line=dict(color="lightgray", width=1, dash="dot")
                )
        
        # Draw bonds
        for bond in self.builder.bonds:
            atom1 = self.builder.atoms[bond.atom1_idx]
            atom2 = self.builder.atoms[bond.atom2_idx]
            
            # Determine bond appearance
            bond_color = 'black'
            bond_width = 3
            
            if bond.bond_type == BondType.DOUBLE:
                bond_width = 5
            elif bond.bond_type == BondType.TRIPLE:
                bond_width = 7
            elif bond.bond_type == BondType.AROMATIC:
                bond_color = 'purple'
                bond_width = 4
            
            # Draw bond line
            fig.add_trace(go.Scatter(
                x=[atom1.x, atom2.x],
                y=[atom1.y, atom2.y],
                mode='lines',
                line=dict(color=bond_color, width=bond_width),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Draw atoms
        if self.builder.atoms:
            atom_x = [atom.x for atom in self.builder.atoms]
            atom_y = [atom.y for atom in self.builder.atoms]
            atom_symbols = [atom.element for atom in self.builder.atoms]
            atom_colors = [self.builder.atom_colors.get(atom.element, '#808080') 
                          for atom in self.builder.atoms]
            atom_sizes = [self.builder.atom_sizes.get(atom.element, 1.5) * 10 
                         for atom in self.builder.atoms]
            
            # Add atom markers
            fig.add_trace(go.Scatter(
                x=atom_x,
                y=atom_y,
                mode='markers',
                marker=dict(
                    size=atom_sizes,
                    color=atom_colors,
                    line=dict(width=2, color='black')
                ),
                text=atom_symbols if self.show_atom_labels else None,
                textposition='middle center',
                textfont=dict(size=12, color='white'),
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>' +
                             'Position: (%{x:.1f}, %{y:.1f})<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Molecule Builder",
            xaxis=dict(
                title="X",
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                title="Y",
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='white',
            width=self.fig_size[0],
            height=self.fig_size[1],
            dragmode='drawrect',
            hovermode='closest'
        )
        
        # Set axis limits
        axis_limit = 300
        fig.update_xaxes(range=[-axis_limit, axis_limit])
        fig.update_yaxes(range=[-axis_limit, axis_limit])
        
        return fig
    
    def create_toolbar_controls(self) -> Dict[str, Any]:
        """
        Create toolbar control definitions
        
        Returns:
            Dictionary with control definitions
        """
        return {
            'atom_types': [
                {'symbol': 'C', 'name': 'Carbon', 'color': '#909090'},
                {'symbol': 'N', 'name': 'Nitrogen', 'color': '#3050F8'},
                {'symbol': 'O', 'name': 'Oxygen', 'color': '#FF0D0D'},
                {'symbol': 'F', 'name': 'Fluorine', 'color': '#90E050'},
                {'symbol': 'Cl', 'name': 'Chlorine', 'color': '#1FF01F'},
                {'symbol': 'Br', 'name': 'Bromine', 'color': '#A62929'},
                {'symbol': 'I', 'name': 'Iodine', 'color': '#940094'},
                {'symbol': 'S', 'name': 'Sulfur', 'color': '#FFFF30'},
                {'symbol': 'P', 'name': 'Phosphorus', 'color': '#FF8000'},
                {'symbol': 'H', 'name': 'Hydrogen', 'color': '#FFFFFF'}
            ],
            'bond_types': [
                {'type': 'SINGLE', 'name': 'Single Bond', 'symbol': '—'},
                {'type': 'DOUBLE', 'name': 'Double Bond', 'symbol': '='},
                {'type': 'TRIPLE', 'name': 'Triple Bond', 'symbol': '≡'},
                {'type': 'AROMATIC', 'name': 'Aromatic Bond', 'symbol': '⎯'}
            ],
            'tools': [
                {'id': 'add_atom', 'name': 'Add Atom', 'icon': '⚛️'},
                {'id': 'add_bond', 'name': 'Add Bond', 'icon': '🔗'},
                {'id': 'delete', 'name': 'Delete', 'icon': '🗑️'},
                {'id': 'move', 'name': 'Move', 'icon': '✋'},
                {'id': 'select', 'name': 'Select', 'icon': '👆'}
            ],
            'settings': {
                'snap_to_grid': True,
                'show_grid': True,
                'auto_add_hydrogens': True,
                'grid_size': 20
            }
        }
    
    def handle_click_event(self, x: float, y: float, tool: str) -> Dict[str, Any]:
        """
        Handle mouse click events in the editor
        
        Args:
            x: Click X coordinate
            y: Click Y coordinate
            tool: Current tool
            
        Returns:
            Result of the action
        """
        result = {'action': tool, 'success': False, 'message': ''}
        
        if tool == 'add_atom':
            atom_idx = self.builder.add_atom(x, y)
            result['success'] = True
            result['message'] = f'Added {self.builder.current_atom_type.value} atom'
            result['atom_idx'] = atom_idx
            
        elif tool == 'add_bond':
            # Find nearest atom
            nearest_atom = self.builder.find_nearest_atom(x, y)
            if nearest_atom is not None:
                if self.builder.selected_atom is None:
                    self.builder.selected_atom = nearest_atom
                    result['success'] = True
                    result['message'] = f'Selected atom {nearest_atom}'
                else:
                    # Add bond between selected atom and nearest atom
                    if self.builder.add_bond(self.builder.selected_atom, nearest_atom):
                        result['success'] = True
                        result['message'] = f'Added bond between atoms {self.builder.selected_atom} and {nearest_atom}'
                    else:
                        result['message'] = 'Bond already exists or invalid'
                    self.builder.selected_atom = None
            else:
                result['message'] = 'No atom found nearby'
                
        elif tool == 'delete':
            # Try to delete nearest atom or bond
            nearest_atom = self.builder.find_nearest_atom(x, y)
            if nearest_atom is not None:
                if self.builder.delete_atom(nearest_atom):
                    result['success'] = True
                    result['message'] = f'Deleted atom {nearest_atom}'
            else:
                nearest_bond = self.builder.find_nearest_bond(x, y)
                if nearest_bond is not None:
                    if self.builder.delete_bond(nearest_bond):
                        result['success'] = True
                        result['message'] = f'Deleted bond {nearest_bond}'
                else:
                    result['message'] = 'No atom or bond found nearby'
                    
        elif tool == 'select':
            nearest_atom = self.builder.find_nearest_atom(x, y)
            if nearest_atom is not None:
                self.builder.selected_atom = nearest_atom
                result['success'] = True
                result['message'] = f'Selected atom {nearest_atom}'
            else:
                result['message'] = 'No atom found nearby'
                
        elif tool == 'move':
            # Implementation would depend on drag events
            result['message'] = 'Move tool requires drag events'
        
        return result
    
    def get_molecule_info(self) -> Dict[str, Any]:
        """
        Get current molecule information
        
        Returns:
            Dictionary with molecule details
        """
        validation = self.builder.validate_molecule()
        
        return {
            'smiles': self.builder.to_smiles(),
            'formula': validation.get('formula', ''),
            'molecular_weight': validation.get('molecular_weight', 0.0),
            'num_atoms': len(self.builder.atoms),
            'num_bonds': len(self.builder.bonds),
            'validation': validation,
            'atom_types': list(set(atom.element for atom in self.builder.atoms)),
            'bond_types': [bond.bond_type.name for bond in self.builder.bonds]
        }
    
    def export_molecule(self, format: str = 'smiles') -> str:
        """
        Export molecule in specified format
        
        Args:
            format: Export format ('smiles', 'mol', 'sdf', 'json')
            
        Returns:
            Exported molecule string
        """
        if format == 'smiles':
            return self.builder.to_smiles()
        elif format == 'json':
            return json.dumps(self.builder.to_dict(), indent=2)
        elif format == 'mol':
            mol = self.builder.to_rdkit_mol()
            if mol:
                return Chem.MolToMolBlock(mol)
        elif format == 'sdf':
            mol = self.builder.to_rdkit_mol()
            if mol:
                return Chem.MolToMolBlock(mol)
        
        return ""
    
    def import_molecule(self, data: str, format: str = 'smiles') -> bool:
        """
        Import molecule from string
        
        Args:
            data: Molecule data
            format: Data format ('smiles', 'mol', 'json')
            
        Returns:
            True if imported successfully
        """
        if format == 'smiles':
            return self.builder.from_smiles(data)
        elif format == 'json':
            try:
                molecule_data = json.loads(data)
                return self.builder.from_dict(molecule_data)
            except:
                return False
        elif format == 'mol':
            try:
                mol = Chem.MolFromMolBlock(data)
                if mol:
                    return self.builder.from_rdkit_mol(mol)
            except:
                return False
        
        return False


# Utility functions
def create_molecule_template_library() -> Dict[str, str]:
    """
    Create a library of common molecular templates
    
    Returns:
        Dictionary of template names to SMILES
    """
    return {
        'Benzene': 'c1ccccc1',
        'Pyridine': 'c1ccncc1',
        'Pyrimidine': 'c1cncnc1',
        'Imidazole': 'c1ncc[nH]1',
        'Thiophene': 'c1ccsc1',
        'Furan': 'c1ccoc1',
        'Pyrrole': 'c1cc[nH]c1',
        'Cyclohexane': 'C1CCCCC1',
        'Cyclopentane': 'C1CCCC1',
        'Methane': 'C',
        'Ethane': 'CC',
        'Propane': 'CCC',
        'Butane': 'CCCC',
        'Water': 'O',
        'Ammonia': 'N',
        'Hydrogen Sulfide': 'S',
        'Methanol': 'CO',
        'Ethanol': 'CCO',
        'Acetone': 'CC(=O)C',
        'Formaldehyde': 'C=O',
        'Acetaldehyde': 'CC=O',
        'Acetic Acid': 'CC(=O)O',
        'Methylamine': 'CN',
        'Ethylamine': 'CCN',
        'Aniline': 'c1ccccc1N',
        'Phenol': 'c1ccccc1O',
        'Benzoic Acid': 'c1ccccc1C(=O)O',
        'Toluene': 'Cc1ccccc1',
        'Xylene': 'Cc1ccc(C)cc1',
        'Naphthalene': 'c1ccc2ccccc2c1',
        'Anthracene': 'c1ccc2c(c1)ccc3c2cccc3',
        'Pyrene': 'c1ccc2c(c1)ccc3c4cccc5cccc2c3c45',
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Nicotine': 'c1ccc2c(c1)[nH]cnc2',
        'Dopamine': 'C1=CC(=C(C=C1CCN)O)O',
        'Serotonin': 'CC1=CN(C2=C1C=CC(=C2)O)CO',
        'Glucose': 'C(C1C(C(C(C(O1)O)O)O)O)O',
        'Adenine': 'C1=NC2=C(N1)C(=NC=N2)N',
        'Guanine': 'C1=NC2=C(N1)C(=NC=N2)N',
        'Cytosine': 'C1=NC(=O)NC(=C1)N',
        'Thymine': 'CC1=CN(C(=O)NC1=O)C',
        'Uracil': 'C1=CN(C(=O)NC1=O)O'
    }


def validate_smiles_interactive(smiles: str) -> Dict[str, Any]:
    """
    Validate SMILES string with detailed feedback
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        Validation result with detailed information
    """
    result = {
        'valid': False,
        'smiles': smiles,
        'errors': [],
        'warnings': [],
        'molecule_info': {}
    }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['errors'].append('Invalid SMILES syntax')
            return result
        
        result['valid'] = True
        
        # Basic molecule information
        result['molecule_info'] = {
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'formula': rdMolDescriptors.CalcMolFormula(mol),
            'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol)
        }
        
        # Check for unusual features
        total_charge = Chem.rdmolops.GetFormalCharge(mol)
        if total_charge != 0:
            result['warnings'].append(f'Charged molecule: {total_charge:+d}')
        
        # Check for functional groups
        functional_groups = {
            'carboxylic_acid': Chem.MolFromSmarts('C(=O)O'),
            'aldehyde': Chem.MolFromSmarts('C=O'),
            'ketone': Chem.MolFromSmarts('C(=O)C'),
            'alcohol': Chem.MolFromSmarts('[OH]'),
            'amine': Chem.MolFromSmarts('[NH2]'),
            'amide': Chem.MolFromSmarts('C(=O)N'),
            'ester': Chem.MolFromSmarts('C(=O)OC'),
            'ether': Chem.MolFromSmarts('COC'),
            'phenol': Chem.MolFromSmarts('c1ccc(cc1)O'),
            'nitro': Chem.MolFromSmarts('[N+](=O)[O-]'),
            'halogen': Chem.MolFromSmarts('[F,Cl,Br,I]')
        }
        
        found_groups = []
        for group_name, group_pattern in functional_groups.items():
            if group_pattern and mol.HasSubstructMatch(group_pattern):
                found_groups.append(group_name)
        
        if found_groups:
            result['molecule_info']['functional_groups'] = found_groups
        
    except Exception as e:
        result['errors'].append(f'Validation error: {str(e)}')
    
    return result
