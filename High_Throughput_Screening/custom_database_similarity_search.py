"""
**Author**: Ajay Khanna
**Date**: Dec.10.2023
**Place**: UC Merced

### 📧 Contact Information
- **GitHub**: [Ajaykhanna](https://github.com/Ajaykhanna) 🐱<200d>💻
- **Twitter**: [@samdig](https://twitter.com/samdig) 🐦
- **LinkedIn**: [ajay-khanna](https://www.linkedin.com/in/ajay-khanna) 💼
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

def banner(title, char="=", width=20):
    """
    This function prints a banner with the specified title.
    """
    print(char * width, f"{title}", char * width)

def load_database(database_file: str) -> list[Chem.rdchem.Mol]:
    """Load a database of molecules from a CSV file.

    Args:
        database_file (str): The path to the CSV file containing the molecular database.

    Returns:
        list[rdkit.Chem.rdchem.Mol]: A list of RDKit Mol objects representing the molecules in the database.
    """
    df = pd.read_csv(database_file)
    return [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]


def similarity_search(
    query_smi: str, database: list, threshold: float = 0.7, num_results: int = 5
):
    """
    Performs a similarity search on a custom database of molecules using the provided query SMILES string.

    Args:
        query_smi (str): The SMILES string of the query molecule.
        database (list): A list of RDKit Mol objects representing the molecules in the custom database.
        threshold (float, optional): The minimum similarity threshold for a molecule to be considered a match. Defaults to 0.7.
        num_results (int, optional): The maximum number of similar molecules to return. Defaults to 5.

    Returns:
        list: A list of tuples, where each tuple contains a RDKit Mol object and its similarity score to the query molecule.
    """
    query_mol = Chem.MolFromSmiles(query_smi)
    query_fp = FingerprintMols.FingerprintMol(query_mol)

    similar_molecules = []
    for mol in database:
        similarity = DataStructs.FingerprintSimilarity(
            query_fp, FingerprintMols.FingerprintMol(mol)
        )
        if similarity >= threshold:
            similar_molecules.append((mol, similarity))

    similar_molecules.sort(key=lambda x: x[1], reverse=True)
    return similar_molecules[:num_results]


# Example usage
if __name__ == "__main__":
    # Load your custom database
    database_file = "./database.csv"
    database = load_database(database_file)

    # Print Banner
    banner(title="Similarity Search with a Custom Database")
    # Define a query molecule (SMILES string)
    query_smi = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin

    # Perform similarity search
    similar_results = similarity_search(query_smi, database)

    # Print results
    print(f"Similar molecules to {query_smi}:")
    for mol, similarity in similar_results:
        smi = Chem.MolToSmiles(mol)
        print(f"SMILES: {smi}, Similarity: {similarity:.3f}")
