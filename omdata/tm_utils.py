import itertools
import os 
import argparse
import mendeleev
from schrodinger.structutils.analyze import (evaluate_asl, evaluate_smarts,
                                             has_valid_lewis_structure,
                                             hydrogens_present)

def has_actinide(st):
    return any(at.atomic_number > 83 for at in st.atom)

def guess_spin_state(st):
    metals = evaluate_asl(st, "metals")
    # We will assume antiferromagnetic coupling for multimetallic systems
    # to ensure we don't put the spin state outside our acceptable range
    total_spin = 0
    local_spins = []
    for metal_idx in metals:
        metal_at = st.atom[metal_idx]
        local_spin = mendeleev.element(metal_at.element).ec.ionize(metal_at.formal_charge).unpaired_electrons()
        # Assume 2nd, 3rd row TMs are low spin, Ln are high spin
        if metal_at.atomic_number > 36 and not metal_at.atomic_number in range(59,70): 
            local_spin = local_spin % 2 
        if local_spin > 0:
            local_spins.append(local_spin)
    for idx, local_spin in enumerate(local_spins):
        total_spin += (-1) ** idx * local_spin
    return abs(total_spin) + 1

def get_cif_location(cod_path, cif_entry):
    dir1 = cif_entry[0]
    dir2 = cif_entry[1:3]
    dir3 = cif_entry[3:5]
    return os.path.join(cod_path, dir1, dir2, dir3, cif_entry + ".cif")

def has_collisions(st):
    def get_expected_length(at1, at2):
        total = 0
        for at in (at1, at2):
            radius = mendeleev.element(at.element).covalent_radius_pyykko
            total += radius
        return total / 100.0
    
    return any(bond.length < 0.55 * get_expected_length(*bond.atom) for bond in st.bond)

def fix_carbonyls(st):
    def revise_carbonyl(st, carb):
        O = st.atom[carb[0]]
        C = st.atom[carb[1]]
        M = st.atom[carb[2]]
        st.getBond(O,C).order = 3
        st.getBond(C,M).order = 1
        O.formal_charge = 1
        C.formal_charge = -1
        M.formal_charge -= 2

    for term_pattern in ('[OX1+0]=[CX2-2]=[!#6,!#7,!#8,!#16,!#1]', '[NX2+0]=[CX2-2]=[!#6,!#7,!#8,!#16,!#1]'):
        terminal_carbonyls = evaluate_smarts(st, term_pattern)
        metals = evaluate_asl(st, "metals")
        for carb in terminal_carbonyls:
            if carb[2] not in metals:
                continue
            revise_carbonyl(st, carb)
    bridging_carbonyls = evaluate_smarts(st, '[OX1+0]=[CX2-2]([!#6,!#7,!#8,!#16,!#1])[!#6,!#7,!#8,!#16,!#1]')
    for carb in bridging_carbonyls:
        if carb[2] not in metals or carb[3] not in metals or st.atom[carb[2]].formal_charge < st.atom[carb[3]].formal_charge:
            continue
        revise_carbonyl(st, carb)

def remove_metal_metal_bonds(st):
    metals = evaluate_asl(st, "metals")
    for at1, at2 in itertools.combinations(metals, 2):
        bond = st.getBond(at1, at2)
        if bond is not None:
            n_mol = st.mol_total
            st.deleteBond(*bond.atom)
            if st.mol_total != n_mol:
                st.addBond(*bond.atom, 1)


def resolve_disorder(st):
    disordered_atoms = [at for at in st.atom if int(at.property.get('s_cif_disorder_group',0)) > 1]
    st.deleteAtoms(disordered_atoms)
    low_occupancy = [at for at in st.atom if at.property.get('r_m_pdb_occupancy', 1) < 0.5]
    st.deleteAtoms(low_occupancy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_csv", required=True, type=str)
    parser.add_argument("--cod_path", required=True, type=str)
    parser.add_argument("--output_path", default='.', type=str)
    parser.add_argument("--total_chunks", default=1, type=int)
    parser.add_argument("--chunk_idx", default=0, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    return parser.parse_args()



def remove_ammoniums(st):
    patterns =('[C,#1][NX4]([C,N,#1])([C,#1])[C,#1]','[C,#1][NX3]1[CX3][CX3][CX3][CX3][CX3]1')
    charges = (-1,-1)
    for smarts, charge in zip(patterns, charges):
        ammoniums = {frozenset(i) for i in evaluate_smarts(st,smarts)}
        ammonium_ats = set()
        for nit in ammoniums:
            ammonium_ats.update(nit)
        ammonium_mols = {st.atom[at].molecule_number for at in ammonium_ats}
        ats_to_delete = set()
        charge_adjust = 0
        for mol in ammonium_mols:
            if not {at.element for at in st.molecule[mol].atom}.intersection(['B', 'P', 'As']):
                mol_st = st.molecule[mol].extractStructure()
                local_count = {frozenset(i) for i in evaluate_smarts(mol_st,smarts)}
                ats_to_delete.update((at.index for at in st.molecule[mol].atom))
                charge_adjust += charge * len(local_count)
        st.deleteAtoms(ats_to_delete)
        st.property['i_m_Molecular_charge'] = st.property.get('i_m_Molecular_charge', 0) + charge_adjust



def remove_nitrate(st):
    patterns =('[OX1][NX3]([OX1])([OX1])','[OX1][CX3]([OX1])([OX1])', '[OX1][ClX4]([OX1])([OX1])([OX1])', '[OX1][IX3]([OX1])([OX1])','[OX1][BrX3]([OX1])([OX1])','[OX1][SX4]([OX1])([OX1])[OX1]','[OX1][PX4]([OX1])([OX1])[OX1]') 
    charges = (1,2,1,1,1,2,3)
    for smarts, charge in zip(patterns, charges):
        nitrates = {frozenset(i) for i in evaluate_smarts(st,smarts)}
        nitrate_atoms = set()
        for nit in nitrates:
            nitrate_atoms.update(nit)
        st.deleteAtoms(nitrate_atoms)
        st.property['i_m_Molecular_charge'] = st.property.get('i_m_Molecular_charge', 0) + len(nitrates)*charge

def remove_free_oxygen(st):
    free_O = evaluate_smarts(st, '[OX0]')
    st.deleteAtoms([at for at_list in free_O for at in at_list])

def remove_F2_bonds(st):
    f2 = {frozenset(i) for i in evaluate_smarts(st,'[FX2][FX2]')}
    for at1, at2 in f2:
        st.deleteBond(at1, at2)

def pb_correction(st):
    pb_atoms = evaluate_asl(st, "at.ele Pb")
    if not pb_atoms:
        return
    st_copy = st.copy()
    connect_atoms(st_copy, max_valencies=MAX_VALENCIES, cov_factor=1.1)
    for pb in pb_atoms:
        for other_atom in st_copy.atom[pb].bonded_atoms:
            if not st.areBound(pb, other_atom.index):
                st.addBond(pb, other_atom.index, 1)


def equilibrate_metals(st):
    metals = evaluate_asl(st, "metals")
    #charge_dict = defaultdict(list)
    #for metal_idx in metals:
    #    at = st.atom[metal_idx]
    #    charge_dict[at.element].append((at.formal_charge, metal_idx))
    #for charges in charge_dict.values():
    charges = []
    for metal_idx in metals:
        at = st.atom[metal_idx]
        if at.element in {'Li','Na','K','Rb','Cs'} and at.formal_charge == 1:
            continue
        elif at.element in {'Be','Mg','Ca','Sr','Ba','Zn','Cd'} and at.formal_charge == 2:
            continue
        elif at.element in {'Al','La','Lu','Pr','Nd', 'Pm', 'Eu', 'Gd', 'Tb', 'Dy','Ho', 'Er', 'Tm','Sc', 'Y'} and at.formal_charge == 3:
            continue
        charges.append((at.formal_charge, metal_idx, at.element))
    if charges:
        min_chg = min(charges)
        max_chg = max(charges)
        chg_diff = max_chg[0] - min_chg[0]
        while chg_diff > 1: 
            shift = chg_diff // 2
            charges.remove(min_chg)
            charges.remove(max_chg)
            st.atom[min_chg[1]].formal_charge += shift
            st.atom[max_chg[1]].formal_charge -= shift
            charges.append((min_chg[0] + shift, min_chg[1]))
            charges.append((max_chg[0] - shift, max_chg[1]))
            min_chg = min(charges)
            max_chg = max(charges)
            chg_diff = max_chg[0] - min_chg[0]