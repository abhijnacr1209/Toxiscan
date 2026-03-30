"""
Advanced Chemical Similarity Analysis and Chemical Space Visualization
Implements multiple similarity metrics, clustering, and dimensionality reduction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.Fingerprints import Fingerprinter
from rdkit.ML.Cluster import Butina

# Machine learning and dimensionality reduction
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import umap

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Chemical similarity (if available)
try:
    import chemfp
    CHEMFP_AVAILABLE = True
except ImportError:
    CHEMFP_AVAILABLE = False
    warnings.warn("chemfp not available. Using RDKit for similarity calculations.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ChemicalSimilarityAnalyzer:
    """
    Advanced chemical similarity analysis with multiple metrics and methods
    """
    
    def __init__(self):
        self.fingerprint_cache = {}
        self.similarity_cache = {}
        self.molecule_cache = {}
        
        # Fingerprint types
        self.fingerprint_types = {
            'morgan': self._calculate_morgan_fingerprint,
            'maccs': self._calculate_maccs_fingerprint,
            'rdkit': self._calculate_rdkit_fingerprint,
            'topological': self._calculate_topological_fingerprint,
            'atom_pair': self._calculate_atom_pair_fingerprint
        }
        
        # Similarity metrics
        self.similarity_metrics = {
            'tanimoto': DataStructs.TanimotoSimilarity,
            'dice': DataStructs.DiceSimilarity,
            'cosine': DataStructs.CosineSimilarity,
            'sokal': DataStructs.SokalSimilarity,
            'russell': DataStructs.RussellSimilarity,
            'rogot_goldberg': DataStructs.RogotGoldbergSimilarity
        }
        
        logger.info("Chemical similarity analyzer initialized")
    
    def calculate_fingerprints(self, smiles_list: List[str], 
                            fp_type: str = 'morgan',
                            **kwargs) -> np.ndarray:
        """
        Calculate molecular fingerprints for a list of SMILES
        
        Args:
            smiles_list: List of SMILES strings
            fp_type: Type of fingerprint to calculate
            **kwargs: Additional parameters for fingerprint calculation
            
        Returns:
            Array of fingerprints
        """
        if fp_type not in self.fingerprint_types:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        logger.info(f"Calculating {fp_type} fingerprints for {len(smiles_list)} molecules")
        
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            # Check cache first
            cache_key = f"{smiles}_{fp_type}_{hash(tuple(sorted(kwargs.items())))}"
            if cache_key in self.fingerprint_cache:
                fp = self.fingerprint_cache[cache_key]
                fingerprints.append(fp)
                valid_smiles.append(smiles)
                continue
            
            # Calculate fingerprint
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES: {smiles}")
                    continue
                
                fp = self.fingerprint_types[fp_type](mol, **kwargs)
                fingerprints.append(fp)
                valid_smiles.append(smiles)
                
                # Cache result
                self.fingerprint_cache[cache_key] = fp
                
            except Exception as e:
                logger.warning(f"Failed to calculate fingerprint for {smiles}: {e}")
        
        if not fingerprints:
            raise ValueError("No valid molecules found")
        
        # Convert to numpy array
        if fp_type in ['maccs']:  # MACCS returns bit vectors
            fp_array = np.zeros((len(fingerprints), len(fingerprints[0])), dtype=np.int8)
            for i, fp in enumerate(fingerprints):
                DataStructs.ConvertToNumpyArray(fp, fp_array[i])
        else:
            fp_array = np.array(fingerprints)
        
        logger.info(f"Successfully calculated {len(fingerprints)} fingerprints")
        return fp_array, valid_smiles
    
    def _calculate_morgan_fingerprint(self, mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """Calculate Morgan (ECFP) fingerprint"""
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    
    def _calculate_maccs_fingerprint(self, mol: Chem.Mol) -> Chem.rdMolDescriptors.MACCSkeys:
        """Calculate MACCS keys fingerprint"""
        return Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    
    def _calculate_rdkit_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate RDKit fingerprint"""
        fp = Chem.RDKFingerprint(mol)
        return np.array(fp)
    
    def _calculate_topological_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate topological fingerprint"""
        fp = Chem.TopologicalFingerprint(mol)
        return np.array(fp)
    
    def _calculate_atom_pair_fingerprint(self, mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
        """Calculate atom pair fingerprint"""
        fp = Chem.GetAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        return np.array(fp)
    
    def calculate_similarity_matrix(self, fingerprints: np.ndarray,
                                 metric: str = 'tanimoto') -> np.ndarray:
        """
        Calculate pairwise similarity matrix
        
        Args:
            fingerprints: Array of fingerprints
            metric: Similarity metric to use
            
        Returns:
            Similarity matrix
        """
        logger.info(f"Calculating similarity matrix using {metric} metric")
        
        n_molecules = len(fingerprints)
        similarity_matrix = np.zeros((n_molecules, n_molecules))
        
        if metric not in self.similarity_metrics:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        similarity_func = self.similarity_metrics[metric]
        
        # Calculate upper triangle (symmetric matrix)
        for i in range(n_molecules):
            for j in range(i, n_molecules):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Convert back to RDKit fingerprint for similarity calculation
                    fp1 = DataStructs.ExplicitBitVect(fingerprints[i])
                    fp2 = DataStructs.ExplicitBitVect(fingerprints[j])
                    similarity = similarity_func(fp1, fp2)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def find_similar_compounds(self, query_smiles: str, 
                             target_smiles_list: List[str],
                             fp_type: str = 'morgan',
                             metric: str = 'tanimoto',
                             top_k: int = 10,
                             threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find compounds similar to a query compound
        
        Args:
            query_smiles: Query SMILES string
            target_smiles_list: List of target SMILES strings
            fp_type: Fingerprint type
            metric: Similarity metric
            top_k: Number of similar compounds to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar compounds with similarity scores
        """
        logger.info(f"Finding similar compounds for {query_smiles}")
        
        # Calculate query fingerprint
        query_fp, _ = self.calculate_fingerprints([query_smiles], fp_type=fp_type)
        if len(query_fp) == 0:
            raise ValueError(f"Invalid query SMILES: {query_smiles}")
        
        query_fp = query_fp[0]
        
        # Calculate target fingerprints
        target_fps, valid_targets = self.calculate_fingerprints(target_smiles_list, fp_type=fp_type)
        
        # Calculate similarities
        similarities = []
        similarity_func = self.similarity_metrics[metric]
        
        for i, target_fp in enumerate(target_fps):
            # Convert to RDKit fingerprints
            fp1 = DataStructs.ExplicitBitVect(query_fp)
            fp2 = DataStructs.ExplicitBitVect(target_fp)
            
            similarity = similarity_func(fp1, fp2)
            
            if similarity >= threshold:
                similarities.append({
                    'smiles': valid_targets[i],
                    'similarity': similarity,
                    'index': i
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def cluster_molecules(self, smiles_list: List[str],
                        fp_type: str = 'morgan',
                        method: str = 'dbscan',
                        **kwargs) -> Dict[str, Any]:
        """
        Cluster molecules based on structural similarity
        
        Args:
            smiles_list: List of SMILES strings
            fp_type: Fingerprint type
            method: Clustering method ('dbscan', 'kmeans', 'hierarchical', 'butina')
            **kwargs: Additional parameters for clustering
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Clustering {len(smiles_list)} molecules using {method}")
        
        # Calculate fingerprints
        fingerprints, valid_smiles = self.calculate_fingerprints(smiles_list, fp_type=fp_type)
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(fingerprints)
        
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Perform clustering
        if method == 'dbscan':
            eps = kwargs.get('eps', 0.3)
            min_samples = kwargs.get('min_samples', 2)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
        elif method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 5)
            
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(fingerprints)
            
        elif method == 'hierarchical':
            n_clusters = kwargs.get('n_clusters', 5)
            linkage = kwargs.get('linkage', 'average')
            
            clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                              linkage=linkage, 
                                              affinity='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
        elif method == 'butina':
            # Butina clustering (RDKit specific)
            cutoff = kwargs.get('cutoff', 0.3)
            
            # Convert to distance matrix for Butina
            dists = []
            for i in range(len(fingerprints)):
                row = []
                for j in range(len(fingerprints)):
                    if i == j:
                        row.append(0.0)
                    else:
                        fp1 = DataStructs.ExplicitBitVect(fingerprints[i])
                        fp2 = DataStructs.ExplicitBitVect(fingerprints[j])
                        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                        distance = 1 - similarity
                        row.append(distance)
                dists.append(row)
            
            clusters = Butina.ClusterData(dists, len(fingerprints), cutoff, isDistData=True)
            cluster_labels = np.full(len(fingerprints), -1)
            
            for cluster_id, cluster in enumerate(clusters):
                for mol_idx in cluster:
                    cluster_labels[mol_idx] = cluster_id
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Organize results
        results = {
            'method': method,
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': list(cluster_labels).count(-1),
            'clusters': {}
        }
        
        # Group molecules by cluster
        for i, label in enumerate(cluster_labels):
            if label not in results['clusters']:
                results['clusters'][label] = []
            results['clusters'][label].append({
                'smiles': valid_smiles[i],
                'index': i
            })
        
        logger.info(f"Clustering completed: {results['n_clusters']} clusters, {results['n_noise']} noise points")
        return results
    
    def calculate_chemical_space_coordinates(self, smiles_list: List[str],
                                          method: str = 'tsne',
                                          fp_type: str = 'morgan',
                                          **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate 2D coordinates for chemical space visualization
        
        Args:
            smiles_list: List of SMILES strings
            method: Dimensionality reduction method ('tsne', 'pca', 'umap', 'mds')
            fp_type: Fingerprint type
            **kwargs: Additional parameters for the method
            
        Returns:
            Tuple of (coordinates, valid_smiles)
        """
        logger.info(f"Calculating chemical space coordinates using {method}")
        
        # Calculate fingerprints
        fingerprints, valid_smiles = self.calculate_fingerprints(smiles_list, fp_type=fp_type)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            perplexity = kwargs.get('perplexity', min(30, len(fingerprints) - 1))
            n_iter = kwargs.get('n_iter', 1000)
            
            tsne = TSNE(n_components=2, perplexity=perplexity, 
                       n_iter=n_iter, random_state=42)
            coordinates = tsne.fit_transform(fingerprints)
            
        elif method == 'pca':
            n_components = kwargs.get('n_components', 2)
            
            pca = PCA(n_components=n_components, random_state=42)
            coordinates = pca.fit_transform(fingerprints)
            
        elif method == 'umap':
            n_neighbors = kwargs.get('n_neighbors', 15)
            min_dist = kwargs.get('min_dist', 0.1)
            
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            coordinates = reducer.fit_transform(fingerprints)
            
        elif method == 'mds':
            n_components = kwargs.get('n_components', 2)
            
            # Calculate distance matrix
            similarity_matrix = self.calculate_similarity_matrix(fingerprints)
            distance_matrix = 1 - similarity_matrix
            
            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
            coordinates = mds.fit_transform(distance_matrix)
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        logger.info(f"Chemical space coordinates calculated for {len(valid_smiles)} molecules")
        return coordinates, valid_smiles
    
    def analyze_scaffold_diversity(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Analyze scaffold diversity in a set of molecules
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with scaffold analysis results
        """
        logger.info("Analyzing scaffold diversity")
        
        scaffolds = {}
        molecules = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Get Murcko scaffold
                scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                
                if scaffold_smiles not in scaffolds:
                    scaffolds[scaffold_smiles] = []
                
                scaffolds[scaffold_smiles].append(smiles)
                molecules.append(smiles)
                
            except Exception as e:
                logger.warning(f"Failed to process {smiles}: {e}")
        
        # Calculate diversity metrics
        n_scaffolds = len(scaffolds)
        n_molecules = len(molecules)
        
        # Scaffold frequency distribution
        scaffold_counts = [len(compounds) for compounds in scaffolds.values()]
        
        # Diversity metrics
        scaffold_diversity_index = n_scaffolds / n_molecules if n_molecules > 0 else 0
        shannon_entropy = -sum((count/n_molecules) * np.log(count/n_molecules) 
                             for count in scaffold_counts if count > 0)
        
        results = {
            'n_molecules': n_molecules,
            'n_scaffolds': n_scaffolds,
            'scaffold_diversity_index': scaffold_diversity_index,
            'shannon_entropy': shannon_entropy,
            'scaffolds': {},
            'most_common_scaffolds': []
        }
        
        # Organize scaffold information
        for scaffold_smiles, compounds in scaffolds.items():
            results['scaffolds'][scaffold_smiles] = {
                'count': len(compounds),
                'compounds': compounds
            }
        
        # Get most common scaffolds
        sorted_scaffolds = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
        results['most_common_scaffolds'] = [
            {'smiles': scaffold, 'count': len(compounds), 'compounds': compounds}
            for scaffold, compounds in sorted_scaffolds[:10]
        ]
        
        logger.info(f"Scaffold analysis completed: {n_scaffolds} scaffolds from {n_molecules} molecules")
        return results
    
    def calculate_property_correlations(self, smiles_list: List[str],
                                      properties: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate correlations between molecular similarity and properties
        
        Args:
            smiles_list: List of SMILES strings
            properties: Dictionary of property names to values
            
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info("Calculating property-similarity correlations")
        
        # Calculate fingerprints and similarity matrix
        fingerprints, valid_smiles = self.calculate_fingerprints(smiles_list)
        similarity_matrix = self.calculate_similarity_matrix(fingerprints)
        
        # Filter properties to match valid molecules
        valid_properties = {}
        for prop_name, prop_values in properties.items():
            if len(prop_values) == len(smiles_list):
                # Filter to match valid molecules
                valid_indices = [i for i, smiles in enumerate(smiles_list) if smiles in valid_smiles]
                valid_properties[prop_name] = [prop_values[i] for i in valid_indices]
        
        # Calculate correlations
        correlations = {}
        for prop_name, prop_values in valid_properties.items():
            # For each molecule, calculate average similarity to all others
            avg_similarities = []
            prop_correlations = []
            
            for i in range(len(valid_smiles)):
                # Exclude self-similarity
                similarities = [similarity_matrix[i, j] for j in range(len(valid_smiles)) if i != j]
                avg_sim = np.mean(similarities)
                avg_similarities.append(avg_sim)
                prop_correlations.append(prop_values[i])
            
            # Calculate correlation
            if len(avg_similarities) > 1:
                correlation = np.corrcoef(avg_similarities, prop_correlations)[0, 1]
                correlations[prop_name] = {
                    'correlation': correlation if not np.isnan(correlation) else 0.0,
                    'avg_similarity': avg_similarities,
                    'property_values': prop_correlations
                }
        
        return correlations


class ChemicalSpaceVisualizer:
    """
    Advanced visualization of chemical space and similarity relationships
    """
    
    def __init__(self):
        self.color_palettes = {
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'magma': px.colors.sequential.Magma,
            'blues': px.colors.sequential.Blues,
            'reds': px.colors.sequential.Reds
        }
    
    def plot_chemical_space(self, coordinates: np.ndarray,
                          smiles_list: List[str],
                          properties: Optional[Dict[str, List[float]]] = None,
                          color_by: str = 'index',
                          title: str = "Chemical Space Visualization") -> go.Figure:
        """
        Create interactive chemical space plot
        
        Args:
            coordinates: 2D coordinates for molecules
            smiles_list: List of SMILES strings
            properties: Dictionary of properties for coloring
            color_by: Property to color points by
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Determine color values
        if color_by == 'index':
            color_values = list(range(len(smiles_list)))
            color_scale = 'Viridis'
        elif properties and color_by in properties:
            color_values = properties[color_by]
            color_scale = 'Viridis'
        else:
            color_values = [1] * len(smiles_list)
            color_scale = 'Blues'
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=color_values,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(title=color_by),
                line=dict(width=1, color='black')
            ),
            text=smiles_list,
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         f'{color_by}: %{{marker.color:.3f}}<br>' +
                         '<extra></extra>',
            name='Molecules'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=800,
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray,
                             smiles_list: List[str],
                             title: str = "Molecular Similarity Heatmap") -> go.Figure:
        """
        Create similarity heatmap
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            smiles_list: List of SMILES strings
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Truncate SMILES for display
        display_labels = [s[:15] + '...' if len(s) > 15 else s for s in smiles_list]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=display_labels,
            y=display_labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Similarity"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            width=800,
            height=600,
            xaxis_title="Compounds",
            yaxis_title="Compounds"
        )
        
        return fig
    
    def plot_clustering_results(self, coordinates: np.ndarray,
                             cluster_labels: List[int],
                             smiles_list: List[str],
                             title: str = "Molecular Clustering Results") -> go.Figure:
        """
        Plot clustering results in chemical space
        
        Args:
            coordinates: 2D coordinates
            cluster_labels: Cluster assignment for each molecule
            smiles_list: List of SMILES strings
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Get unique clusters
        unique_clusters = list(set(cluster_labels))
        colors = px.colors.qualitative.Set3
        
        fig = go.Figure()
        
        for i, cluster_id in enumerate(unique_clusters):
            # Get points in this cluster
            mask = [label == cluster_id for label in cluster_labels]
            cluster_coords = coordinates[mask]
            cluster_smiles = [smiles for smiles, m in zip(smiles_list, mask) if m]
            
            # Handle noise points
            if cluster_id == -1:
                color = 'black'
                name = 'Noise'
            else:
                color = colors[i % len(colors)]
                name = f'Cluster {cluster_id}'
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    line=dict(width=1, color='black')
                ),
                text=cluster_smiles,
                hovertemplate='<b>%{text}</b><br>' +
                             'Cluster: ' + name + '<br>' +
                             'X: %{x:.2f}<br>' +
                             'Y: %{y:.2f}<br>' +
                             '<extra></extra>',
                name=name
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=800,
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def plot_scaffold_network(self, scaffolds: Dict[str, List[str]],
                           title: str = "Scaffold Network") -> go.Figure:
        """
        Create network plot of scaffold relationships
        
        Args:
            scaffolds: Dictionary of scaffold SMILES to compound lists
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Create nodes for scaffolds
        scaffold_ids = list(range(len(scaffolds)))
        scaffold_smiles = list(scaffolds.keys())
        scaffold_sizes = [len(compounds) for compounds in scaffolds.values()]
        
        # Layout nodes in a circle
        n_scaffolds = len(scaffolds)
        angles = np.linspace(0, 2*np.pi, n_scaffolds, endpoint=False)
        x_coords = np.cos(angles)
        y_coords = np.sin(angles)
        
        # Add scaffold nodes
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=[s*5 + 10 for s in scaffold_sizes],  # Size based on number of compounds
                color=scaffold_sizes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Number of Compounds"),
                line=dict(width=2, color='black')
            ),
            text=[f"Scaffold {i+1}" for i in range(n_scaffolds)],
            textposition='middle center',
            hovertemplate='<b>Scaffold %{text}</b><br>' +
                         'SMILES: %{customdata}<br>' +
                         'Compounds: %{marker.color}<br>' +
                         '<extra></extra>',
            customdata=scaffold_smiles,
            name='Scaffolds'
        ))
        
        # Add edges based on scaffold similarity (simplified)
        for i in range(n_scaffolds):
            for j in range(i+1, n_scaffolds):
                # Calculate scaffold similarity (simplified)
                try:
                    mol1 = Chem.MolFromSmiles(scaffold_smiles[i])
                    mol2 = Chem.MolFromSmiles(scaffold_smiles[j])
                    if mol1 and mol2:
                        fp1 = Chem.RDKFingerprint(mol1)
                        fp2 = Chem.RDKFingerprint(mol2)
                        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                        
                        if similarity > 0.3:  # Only show significant similarities
                            fig.add_shape(
                                type="line",
                                x0=x_coords[i], y0=y_coords[i],
                                x1=x_coords[j], y1=y_coords[j],
                                line=dict(color=f"rgba(100,100,100,{similarity})", width=similarity*5)
                            )
                except:
                    pass
        
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_property_similarity_correlation(self, correlations: Dict[str, Any],
                                          title: str = "Property-Similarity Correlations") -> go.Figure:
        """
        Plot correlation between molecular similarity and properties
        
        Args:
            correlations: Dictionary from similarity analysis
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(correlations.keys())[:4],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        for i, (prop_name, data) in enumerate(correlations.items()):
            if i >= 4:  # Limit to 4 properties
                break
            
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=data['avg_similarity'],
                    y=data['property_values'],
                    mode='markers',
                    name=prop_name,
                    text=[f"Similarity: {s:.3f}<br>Property: {p:.3f}" 
                          for s, p in zip(data['avg_similarity'], data['property_values'])],
                    hovertemplate='<b>%{fullData.name}</b><br>%{text}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add trend line
            z = np.polyfit(data['avg_similarity'], data['property_values'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(data['avg_similarity']), max(data['avg_similarity']), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name=f'{prop_name} trend',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Update subplot axes
            fig.update_xaxes(title_text="Average Similarity", row=row, col=col)
            fig.update_yaxes(title_text=prop_name, row=row, col=col)
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        return fig


class ChemicalSpaceAnalyzer:
    """
    Main class for comprehensive chemical space analysis
    """
    
    def __init__(self):
        self.similarity_analyzer = ChemicalSimilarityAnalyzer()
        self.visualizer = ChemicalSpaceVisualizer()
    
    def comprehensive_analysis(self, smiles_list: List[str],
                            properties: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive chemical space analysis
        
        Args:
            smiles_list: List of SMILES strings
            properties: Optional molecular properties
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Starting comprehensive chemical space analysis for {len(smiles_list)} molecules")
        
        results = {
            'summary': {
                'n_molecules': len(smiles_list),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # 1. Chemical space coordinates
        try:
            coords_tsne, valid_smiles = self.similarity_analyzer.calculate_chemical_space_coordinates(
                smiles_list, method='tsne'
            )
            coords_pca, _ = self.similarity_analyzer.calculate_chemical_space_coordinates(
                smiles_list, method='pca'
            )
            coords_umap, _ = self.similarity_analyzer.calculate_chemical_space_coordinates(
                smiles_list, method='umap'
            )
            
            results['coordinates'] = {
                'tsne': coords_tsne.tolist(),
                'pca': coords_pca.tolist(),
                'umap': coords_umap.tolist(),
                'valid_smiles': valid_smiles
            }
            
        except Exception as e:
            logger.error(f"Error calculating coordinates: {e}")
            results['coordinates'] = None
        
        # 2. Similarity analysis
        try:
            fingerprints, _ = self.similarity_analyzer.calculate_fingerprints(smiles_list)
            similarity_matrix = self.similarity_analyzer.calculate_similarity_matrix(fingerprints)
            
            results['similarity'] = {
                'matrix': similarity_matrix.tolist(),
                'avg_similarity': float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                'max_similarity': float(np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                'min_similarity': float(np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
            }
            
        except Exception as e:
            logger.error(f"Error in similarity analysis: {e}")
            results['similarity'] = None
        
        # 3. Clustering analysis
        try:
            clustering_results = self.similarity_analyzer.cluster_molecules(
                smiles_list, method='dbscan', eps=0.3
            )
            results['clustering'] = clustering_results
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            results['clustering'] = None
        
        # 4. Scaffold analysis
        try:
            scaffold_results = self.similarity_analyzer.analyze_scaffold_diversity(smiles_list)
            results['scaffolds'] = scaffold_results
            
        except Exception as e:
            logger.error(f"Error in scaffold analysis: {e}")
            results['scaffolds'] = None
        
        # 5. Property correlations (if properties provided)
        if properties:
            try:
                correlation_results = self.similarity_analyzer.calculate_property_correlations(
                    smiles_list, properties
                )
                results['property_correlations'] = correlation_results
                
            except Exception as e:
                logger.error(f"Error in property correlation analysis: {e}")
                results['property_correlations'] = None
        
        logger.info("Comprehensive chemical space analysis completed")
        return results
    
    def create_analysis_dashboard(self, analysis_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create dashboard with multiple visualization plots
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        if analysis_results['coordinates'] and analysis_results['coordinates']['valid_smiles']:
            valid_smiles = analysis_results['coordinates']['valid_smiles']
            coords_tsne = np.array(analysis_results['coordinates']['tsne'])
            
            # Chemical space plot
            figures['chemical_space'] = self.visualizer.plot_chemical_space(
                coords_tsne, valid_smiles, title="Chemical Space (t-SNE)"
            )
        
        if analysis_results['similarity']:
            similarity_matrix = np.array(analysis_results['similarity']['matrix'])
            valid_smiles = analysis_results['coordinates']['valid_smiles'] if analysis_results['coordinates'] else list(range(len(similarity_matrix)))
            
            # Similarity heatmap
            figures['similarity_heatmap'] = self.visualizer.plot_similarity_heatmap(
                similarity_matrix, valid_smiles
            )
        
        if analysis_results['clustering'] and analysis_results['coordinates']:
            coords_tsne = np.array(analysis_results['coordinates']['tsne'])
            cluster_labels = analysis_results['clustering']['cluster_labels']
            valid_smiles = analysis_results['coordinates']['valid_smiles']
            
            # Clustering plot
            figures['clustering'] = self.visualizer.plot_clustering_results(
                coords_tsne, cluster_labels, valid_smiles
            )
        
        if analysis_results['scaffolds']:
            # Scaffold network
            figures['scaffold_network'] = self.visualizer.plot_scaffold_network(
                analysis_results['scaffolds']['scaffolds']
            )
        
        if analysis_results['property_correlations']:
            # Property correlations
            figures['property_correlations'] = self.visualizer.plot_property_similarity_correlation(
                analysis_results['property_correlations']
            )
        
        return figures


# Utility functions
def calculate_molecular_diversity_index(smiles_list: List[str]) -> float:
    """
    Calculate molecular diversity index for a set of compounds
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Diversity index (0-1, higher = more diverse)
    """
    analyzer = ChemicalSimilarityAnalyzer()
    
    try:
        fingerprints, valid_smiles = analyzer.calculate_fingerprints(smiles_list)
        similarity_matrix = analyzer.calculate_similarity_matrix(fingerprints)
        
        # Calculate mean pairwise similarity (excluding diagonal)
        n = len(similarity_matrix)
        if n <= 1:
            return 0.0
        
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        mean_similarity = np.mean(upper_triangle)
        
        # Diversity index = 1 - mean similarity
        diversity = 1.0 - mean_similarity
        return max(0.0, min(1.0, diversity))
        
    except Exception as e:
        logger.error(f"Error calculating diversity index: {e}")
        return 0.0


def find_diverse_compound_set(smiles_list: List[str], 
                            n_select: int,
                            method: str = 'maxmin') -> List[str]:
    """
    Select a diverse subset of compounds
    
    Args:
        smiles_list: List of SMILES strings
        n_select: Number of compounds to select
        method: Selection method ('maxmin', 'random')
        
    Returns:
        List of selected SMILES
    """
    if n_select >= len(smiles_list):
        return smiles_list
    
    if method == 'random':
        return np.random.choice(smiles_list, n_select, replace=False).tolist()
    
    elif method == 'maxmin':
        analyzer = ChemicalSimilarityAnalyzer()
        
        # Calculate fingerprints
        fingerprints, valid_smiles = analyzer.calculate_fingerprints(smiles_list)
        similarity_matrix = analyzer.calculate_similarity_matrix(fingerprints)
        
        # MaxMin algorithm
        selected = [0]  # Start with first compound
        remaining = list(range(1, len(valid_smiles)))
        
        while len(selected) < n_select and remaining:
            # Find compound with minimum similarity to selected set
            min_similarities = []
            
            for candidate in remaining:
                # Calculate minimum similarity to selected compounds
                similarities = [similarity_matrix[candidate, sel] for sel in selected]
                min_similarities.append(min(similarities))
            
            # Select compound with maximum minimum similarity
            best_idx = np.argmax(min_similarities)
            best_candidate = remaining[best_idx]
            
            selected.append(best_candidate)
            remaining.pop(best_idx)
        
        return [valid_smiles[i] for i in selected]
    
    else:
        raise ValueError(f"Unknown selection method: {method}")
