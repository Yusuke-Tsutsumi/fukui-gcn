#!/usr/bin/env python
# coding: utf-8

# In[4]:


import functools
import pandas as pd
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from rdkit.Chem.rdchem import HybridizationType

from os.path import join
from rdkit.Chem.AtomPairs.Utils import NumPiElectrons


# In[5]:


filepath="/home/tsutsumi/fukui-gcn-test/"
PERIODIC_TABLE = pd.read_csv(
    join(filepath, "data", "atom_data.csv"), index_col=0)
RD_PT = Chem.rdchem.GetPeriodicTable()


# In[7]:


def element(a):
    """ Return the element """

    return a.GetSymbol()


def is_element(a, symbol="C"):
    """ Is the atom of a given element """
    return element(a) == symbol


element_features = {"is_{}".format(e): functools.partial(is_element, symbol=e)
                    for e in ("B", "S", "C", "P", "O", "N", "I", "Cl", "F", "Br")}


def total_num_Hs(a):
    
    return a.GetTotalNumHs()

def atomic_number(a):
    """ Atomic number of atom """

    return a.GetAtomicNum()


def atomic_mass(a):
    """ Atomic mass of atom """

    return a.GetMass()


def explicit_valence(a):
    """ Explicit valence of atom """
    return a.GetExplicitValence()


def implicit_valence(a):
    """ Implicit valence of atom """

    return a.GetImplicitValence()


def valence(a):
    """ returns the valence of the atom """

    return explicit_valence(a) + implicit_valence(a)


def degree(a):
    """ returns the degree of the atom """

    return a.GetDegree()

def degree_with_Hs(a):
    
    return a.GetTotalDegree()

def n_valence_electrons(a):
    """ return the number of valance electrons an atom has """

    return RD_PT.GetNOuterElecs(a.GetAtomicNum())


def n_pi_electrons(a):
    """ returns number of pi electrons """

    return NumPiElectrons(a)


def n_lone_pairs(a):
    """ returns the number of lone pairs assicitaed with the atom """

    return int(0.5 * (n_valence_electrons(a) - degree(a) - n_hydrogens(a) -
                      formal_charge(a) - n_pi_electrons(a)))


def van_der_waals_radius(a):
    """ returns van der waals radius of the atom """
    return PERIODIC_TABLE.van_der_waals_radius[a.GetAtomicNum()]


def formal_charge(a):
    """ Formal charge of atom """

    return a.GetFormalCharge()


def num_implicit_hydrogens(a):
    """ Number of implicit hydrogens """

    return a.GetNumImplicitHs()


def num_explicit_hydrogens(a):
    """ Number of explicit hydrodgens """

    return a.GetNumExplicitHs()


def n_hydrogens(a):
    """ Number of hydrogens """

    return num_implicit_hydrogens(a) + num_explicit_hydrogens(a)


def n_radical_elec(a):
    
    return a.GetNumRadicalElectrons()

def crippen_log_p_contrib(a):
    """ Hacky way of getting logP contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][0]


def crippen_molar_refractivity_contrib(a):
    """ Hacky way of getting molar refractivity contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][1]


def tpsa_contrib(a):
    """ Hacky way of getting total polar surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcTPSAContribs(m)[idx]


def labute_asa_contrib(a):
    """ Hacky way of getting accessible surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcLabuteASAContribs(m)[0][idx]

def inertial_shape_factor(a):
    
    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors.CalcInertialShapeFactor(m)


def Gyration_radius(a):
    
    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors.CalcRadiusOfGyration(m)

def gasteiger_charge(a, force_calc=False):
    """ Hacky way of getting gasteiger charge """

    res = a.GetPropsAsDict().get("_GasteigerCharge", None)
    if res and not force_calc:
        return float(res)
    else:
        m = a.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(m)
        return float(a.GetProp("_GasteigerCharge"))


def pauling_electronegativity(a):
    return PERIODIC_TABLE.pauling_electronegativity[
        a.GetAtomicNum()]


def first_ionization(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "first_ionisation_energy"]


def group(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "group"]



def period(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "period"]


def covalent_radius(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "covalent_radius"]

def electron_affinity(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "electron_affinity"]

def atomic_polarisability(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "atomic_polarisability"]

def van_der_waals_volume(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "van_der_waals_volume"]



# In[8]:


ATOM_FEATURES = {
    "atomic_number": atomic_number,
    "atomic_mass": atomic_mass,
    "formal_charge": formal_charge,
    "gasteiger_charge": gasteiger_charge,
    "pauling_electronegativity": pauling_electronegativity,
    "first_ionisation": first_ionization,
    "group": group,
    "period": period,
    "valence": valence,
    "degree": degree,
    "n_valence_electrons": n_valence_electrons,
    "n_pi_electrons": n_pi_electrons,
    "n_lone_pairs": n_lone_pairs,
    "van_der_waals_radius": van_der_waals_radius,
    "n_hydrogens": n_hydrogens,
    "log_p_contrib": crippen_log_p_contrib,
    "molar_refractivity_contrib": crippen_molar_refractivity_contrib,
    "total_polar_surface_area_contrib": tpsa_contrib,
    "total_labute_accessible_surface_area": labute_asa_contrib,
    'degree_with_Hs':degree_with_Hs,
    'total_num_Hs':total_num_Hs,
    'inertial_shape_factor':inertial_shape_factor,
    'Gyration_radius':Gyration_radius,
    'covalent_radius':covalent_radius,
    'electron_affinity':electron_affinity,
    'atomic_polarisability':atomic_polarisability,
    'van_der_waals_volume':van_der_waals_volume,
}
ATOM_FEATURES.update(element_features)

# In[ ]:




