import argparse
import csv
import glob
import math
import os
import random
import sys
import numpy as np
from collections import defaultdict

import mendeleev
from functools import partial,reduce
from tqdm import tqdm
import multiprocessing as mp
from schrodinger.application.jaguar.packages.shared import \
    uniquify_with_comparison
from schrodinger.application.jaguar.utils import LewisModes, mmjag_update_lewis, mmjag_reset_connectivity
from schrodinger.application.jaguar.autots_bonding import clean_st
from schrodinger.application.matsci import clusterstruct
from schrodinger.application.jaguar.utils import group_with_comparison, get_total_charge
#from schrodinger.application.jaguar.utils import group_items, get_total_charge
from schrodinger.application.matsci.nano.xtal import (connect_atoms, get_cell,
                                                      is_infinite)
from schrodinger.comparison import are_conformers
from schrodinger.structure import StructureReader, create_new_structure
from schrodinger.structutils.analyze import (evaluate_asl, evaluate_smarts,
                                             has_valid_lewis_structure,
                                             hydrogens_present)
from schrodinger.structutils.build import remove_alternate_positions
from schrodinger.structutils.measure import get_close_atoms

MAX_VALENCIES = {'H': 4, 'He': 4, 'Li': 8, 'Be': 8, 'B': 8, 'C': 8, 'N': 5, 'O': 5, 'F': 5, 'Ne': 8, 'Na': 8, 'Mg': 8, 'Al': 8, 'Si': 8, 'P': 8, 'S': 8, 'Cl': 8, 'Ar': 8, 'K': 8, 'Ca': 8, 'Sc': 9, 'Ti': 9, 'V': 9, 'Cr': 9, 'Mn': 9, 'Fe': 9, 'Co': 9, 'Ni': 9, 'Cu': 9, 'Zn': 9, 'Ga': 9, 'Ge': 9, 'As': 8, 'Se': 8, 'Br': 8, 'Kr': 8, 'Rb': 8, 'Sr': 8, 'Y': 9, 'Zr': 9, 'Nb': 9, 'Mo': 9, 'Tc': 9, 'Ru': 9, 'Rh': 9, 'Pd': 9, 'Ag': 9, 'Cd': 9, 'In': 9, 'Sn': 9, 'Sb': 9, 'Te': 8, 'I': 8, 'Xe': 8, 'Cs': 8, 'Ba': 8, 'La': 9, 'Ce': 9, 'Pr': 10, 'Nd': 9, 'Pm': 9, 'Sm': 9, 'Eu': 9, 'Gd': 9, 'Tb': 9, 'Dy': 9, 'Ho': 9, 'Er': 9, 'Tm': 9, 'Yb': 9, 'Lu': 9, 'Hf': 9, 'Ta': 9, 'W': 9, 'Re': 9, 'Os': 9, 'Ir': 9, 'Pt': 9, 'Au': 9, 'Hg': 9, 'Tl': 9, 'Pb': 9, 'Bi': 9, 'Po': 9, 'At': 8, 'Rn': 8, 'Fr': 9, 'Ra': 9, 'Ac': 9, 'Th': 9, 'Pa': 9, 'U': 9, 'Np': 9, 'Pu': 9, 'Am': 9, 'Cm': 9, 'Bk': 9, 'Cf': 9, 'Es': 9, 'Fm': 9, 'Md': 9, 'No': 9, 'Lr': 9, 'Rf': 9, 'Db': 9, 'Sg': 9, 'Bh': 9, 'Hs': 9, 'Mt': 9, 'Ds': 9, 'Rg': 9, 'Cn': 9, 'Nh': 9, 'Fl': 9, 'Mc': 9, 'Lv': 9, 'Ts': 1, 'Og': 1, 'DU': 15, 'Lp': 15, '': 15}


def reduce_to_minimal(st):
    mol_list = [mol.extractStructure() for mol in st.molecule]
    mol_list.sort(key=lambda x: x.atom_total)
    # group those molecules by conformers
    grouped_mol_list = group_with_comparison(mol_list, are_conformers)
    #grouped_mol_list = group_items(mol_list, are_conformers)
    # represent the structure as the counts of each type of molecule
    # and a representative structure
    st_mols = frozenset((len(grp), tuple(grp)) for grp in grouped_mol_list)
    counts = list(zip(*st_mols))[0]
    divisor = reduce(lambda x, y: math.gcd(x, y), counts, counts[0])
    reduced_st = create_new_structure()
    #st_mols = sorted(st_mols, key=lambda x: -len(evaluate_asl(x[1], 'metals')))
    for count, mol_sts in st_mols:
        for idx in range(count//divisor):
            reduced_st.extend(mol_sts[idx])
    remove_free_oxygen(reduced_st)
    #remove_nitrate(reduced_st)
    #remove_ammoniums(reduced_st)
    return reduced_st


def is_too_large(st):
    return all(len(mol.atom) > 250 for mol in st.molecule)

def mg_mg_bond(st):
    mg_list = ('#5', '#15', '#33')
    mg_smarts = f'[{",".join(mg_list)}][{",".join(mg_list)}]'
    s_smarts = '[#16][#16][#16]'
    return bool(evaluate_smarts(st, mg_smarts) + evaluate_smarts(st, s_smarts))


def remove_common_solvents(st):
    # DCM/chloroform/Ctet/+missingH, toluene, benzene/F/Clbenzene/o-diF/o-diClbenzene,THF,ether, water, pentane, hexane, NO2Me, acetone, MeCN, DMF
    for solv in ('[Cl,BrX1][CX4]([Cl,BrX1])([Cl,Br,#1X1])[Cl,#1X1]','[Cl,BrX1][CD2][Cl,BrX1]',
                 '[#1X1][CX3]1[CX3]([#1X1])[CX3]([#1X1])[CX3]([CX4]([#1X1])([#1X1])[#1X1])[CX3]([#1X1])[CX3]1[#1X1]',
                 '[#1X1][CX3]1[CX3]([#1X1])[CX3]([#1X1])[CX3]([F,Cl,#1X1])[CX3]([F,Cl,#1X1])[CX3]1[#1X1]',
                 '[#1X1][CX4]1([#1X1])[OX2][CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]1([#1X1])[#1X1]',
                 '[#1X1][CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[OX2][CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[#1X1]',
                 '[#1X1][OX2][#1X1]', '[#1X1][CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[#1X1]', '[#1X1][CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[#1X1]', '[#1X1][CX4]([#1X1])([#1X1])[#7X3]([#8X1])[#8X1]', '[#1X1][CX4]([#1X1])([#1X1])[CX3]([OX1])[CX4]([#1X1])([#1X1])[#1X1]',
                 '[#1X1][CX4]([#1X1])([#1X1])[#6D2][#7D1]',
                 '[#1X1][CX4]([#1X1])([#1X1])[NX3]([CX4]([#1X1])([#1X1])[#1X1])[CX3]([OX1])[#1X1]'):
        solvs = {frozenset(i) for i in evaluate_smarts(st, solv)}
        solv_atoms = set()
        for ats in solvs:
            solv_atoms.update(ats)
        st.deleteAtoms(solv_atoms)

def extract_molecules(cod_code, cod_path, output_path):
    cif_name = get_cif_location(cod_path, cod_code)
    print(cod_code, 'start')
    try:
        st = StructureReader.read(cif_name)
    except:
        open(cod_code, 'w').close()
        return cod_code
    # If there are no main groups, we aren't going to use it anyway
    if not mg_mg_bond(st):
        open(cod_code, 'w').close()
        return cod_code
    st = remove_alternate_positions(st)
    resolve_disorder(st)
    connect_atoms(st, max_valencies=MAX_VALENCIES, cov_factor=1.2)
    try:
        st = get_cell(st)
    except ValueError:
        open(cod_code, 'w').close()
        return cod_code
    clusterstruct.contract_structure(st)
    connect_atoms(st, max_valencies=MAX_VALENCIES, cov_factor=1.2)
    clusterstruct.contract_structure(st)
    connect_atoms(st, max_valencies=MAX_VALENCIES)
    if is_infinite(st) or get_close_atoms(st, 0.6):
        open(cod_code, 'w').close()
        return cod_code
    try:
        # One last way of setting the sigma skeleton
        st = clean_st(st)
    except (RuntimeError, AssertionError): 
        connect_atoms(st, max_valencies=MAX_VALENCIES)
    for bond in st.bond:
        bond.order = 1
    for at in st.atom:
        at.formal_charge = 0
    remove_common_solvents(st)
    #pb_correction(st)
    remove_F2_bonds(st)
    remove_metal_metal_bonds(st)
    # Skip things that consist solely of very large molecules that
    # we won't be taking anyway
    if is_too_large(st):
        open(cod_code, 'w').close()
        return cod_code
    st = reduce_to_minimal(st)
    if not mg_mg_bond(st):
        open(cod_code, 'w').close()
        return cod_code
     
    if not get_close_atoms(st, 0.2):
        try:
            spin = guess_spin_state(st)
        except ValueError:
            open(cod_code, 'w').close()
            return cod_code
        charge = get_total_charge(st)
        st.write(os.path.join(output_path, f"{cod_code}_{charge}_{spin}.mae"))
#    mol_sts = [mol.extractStructure() for mol in st.molecule]
#    mol_sts = uniquify_with_comparison(
#        mol_sts, are_conformers, use_lewis_structure=False
#    )
#    if len(mol_sts) == 1:
#        mol_sts[0].property['i_m_Molecular_charge'] = get_total_charge(st) // st.mol_total
#    elif len(mol_sts) > 1 and not get_close_atoms(st, 0.2):
#        mol_sts = [st.copy()]
#        
#    for idx, mol_st in enumerate(mol_sts):
#        if not mg_mg_bond(mol_st) or len(evaluate_asl(mol_st, "metals")) > 1:
#            continue
#        try:
#            spin = guess_spin_state(mol_st)
#        except ValueError:
#            open(cod_code, 'w').close()
#            return cod_code
#        charge = get_total_charge(mol_st)
#        mol_st.write(os.path.join(output_path, f"{cod_code}_molecule_{idx}_{charge}_{spin}.mae"))
    print(cod_code, 'end')

def main(cif_csv, cod_path, output_path, n_cores, n_chunks, chunk_idx):
    with open(cif_csv, 'r') as fh:
        csv_reader = csv.reader(fh, delimiter='\t')
        cif_list = [line[0] for line in csv_reader]
    cif_list = cif_list[1:] # Remove header line
    done_codes = set()
    done_codes = {os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(output_path,'*.mae'))}
    print(len(done_codes))
    if os.path.exists('skip_list.txt'):
        with open('skip_list.txt', 'r') as fh:
            done_codes.update([f.strip() for f in fh.readlines()])
    cif_list = [cif for cif in cif_list if cif not in done_codes]
    print(len(cif_list))
    chunks_to_process = np.array_split(cif_list, n_chunks)
    chunk = chunks_to_process[chunk_idx]
    print(len(chunk))
    fxn = partial(extract_molecules, cod_path=cod_path, output_path=output_path)
    with mp.Pool(n_cores) as pool:
        skip_list = list(tqdm(pool.imap(fxn, chunk), total=len(chunk)))


if __name__ == "__main__":
    args = parse_args()
    main(args.cif_csv, args.cod_path, args.output_path, args.num_workers, args.total_chunks, args.chunk_idx)
