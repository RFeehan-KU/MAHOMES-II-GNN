#    MAHOMES II - GNN
#    Copyright (C) 2023 University of Kansas
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# @file  data.py
# @brief Transforms protein metal binding sites into graph objects for use with GNNs
# @author Ryan Feehan <RFeehan93@gmail.com>


import os
import numpy as np
import pandas as pd

from Bio.PDB.PDBParser import PDBParser
import scipy.spatial as ss

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected

## Since this isn't intended for production usage,
## It is only set-up to work for sites in the MAHOMES-II dataset and test-set sites
labels_df = pd.read_csv("input_data/metal_sites_for_dl.csv")
labels_df.set_index(['id'], inplace=True, drop=True)

## for one hot encoding, where all metal ions are represented as the same 
metal_codes = ['ZN', 'FE', 'MN','MG', 'NI', 'CO', 'CU', 'MO']
elem_codes = ['C', 'H', 'O', 'N', 'S', 'M', 'X']

## For creating a set of site graphs 
# nodes are all atoms within 15 Å of the gifven metal ion
# edges for atoms within 4.5 Ā
class SiteDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SiteDataset, self).__init__(root, transform=None,
                 pre_transform=None)

    @property
    def raw_file_names(self):
        return [filename for filename in os.scandir(self.root+"/raw")]

    @property
    def processed_file_names(self):
        return [os.path.splitext(os.path.basename(file))[0]+'.pt' for file in self.raw_paths]

    def download(self):
        pass

    def process(self):
        data_list =[]
        for tmp_file in self.raw_paths:
           id =  os.path.splitext(os.path.basename(tmp_file))[0]
           raw_file="%s/raw/%s.pdb"%(self.root, id)
           if os.path.isfile(raw_file):
            atoms=self._get_structure(id, raw_file)
            site =  self._get_site(atoms, id)
            if len(atoms)==0:
                continue
            try:
                node_feats =  torch.FloatTensor([self._one_hot_elem(e) for e in site['element']])
            except:
                print(id)
                os.system("rm %s"%(raw_file))
                
                continue
            node_pos = torch.FloatTensor(site[['x', 'y', 'z']].to_numpy())
            kd_tree = ss.KDTree(node_pos)
            edge_tuples = list(kd_tree.query_pairs(4.5))
            edges = torch.LongTensor(edge_tuples).t().contiguous()
            edges = to_undirected(edges)

            label = self._get_label(id)
            if label==-1:
                os.system("rm %s"%(raw_file))
                continue
            label=torch.tensor(label)
            if label>-1:
                new_entry = Data(node_feats, edges, y=label, pos=node_pos)
                data_list.append(new_entry)
                torch.save(new_entry, self.root + "/processed/%s.pt"%id)
            else:
                os.system("rm %s"%(raw_file))
          
        self.data = data_list 


    def len(self):
        return len(self.raw_file_names)
    
    def get(self, idx):
        
        data = torch.load("%s/processed/%s"%(self.root, self.processed_file_names[idx]))
        return data
    
    # get structure from a pdb file with biopython
    def _get_structure(self, id, file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(id, file)
        atom_data = []
        for atom in structure.get_atoms():
           residue = atom.get_parent()
           chain = residue.get_parent()
           new_atom= [chain.id, residue.id[1], residue.resname, atom.coord[0], atom.coord[1], atom.coord[2], atom.name, atom.element, atom.serial_number]
           atom_data.append(new_atom)

        atom_df = pd.DataFrame(atom_data, columns=['chainID', 'seq_num', 'resname', 'x', 'y', 'z', 'name', 'element', 'serial'])
        return atom_df
    
    def _get_site(self, struc, site_id, site_dist_cutoff=15):
        struc.reset_index(inplace=True)
        seq_num=int(site_id.split('_')[-1])
        lig_atom=struc.loc[struc['seq_num']==seq_num]
        if len(lig_atom)>0:
           ## uses mean to deal with metal res ids with multiple metal atoms
           lig_coords =  np.array([lig_atom['x'].mean(), lig_atom['y'].mean(), lig_atom['z'].mean()])
           kd_tree = ss.KDTree(struc[['x','y','z']].values)
           site_atoms_idx = kd_tree.query_ball_point(lig_coords, r=site_dist_cutoff, p=2.0)
           site_atoms_df = struc.iloc[site_atoms_idx].reset_index(drop=True)
           return(site_atoms_df)
           
        else: return([])

    def _get_label(self, site_id):
        if site_id in labels_df.index:
            try:
                label = int(labels_df.loc[site_id].Enzyme)
                return(label)
            except:
                print(site_id)
                return(-1)
        else: return(-1)

    def _one_hot_elem(self, x):
        if x in metal_codes:
            x = elem_codes[-2]
        elif x not in elem_codes:
           x = elem_codes[-1]
        return list(map(lambda s: x == s, elem_codes))
