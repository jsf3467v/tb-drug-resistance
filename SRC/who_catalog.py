"""WHO mutation catalog reader. Keeps the grading groups associated with
resistance, maps WHO gene and drug names onto the system's spelling, and yields
deduplicated mutations for the graph loader.
"""

import warnings
from pathlib import Path

import pandas as pd
from config import DRUG_ALIASES

GENE_LOCUS = {
    'Rv0667': 'rpoB', 'Rv1908c': 'katG', 'Rv1484': 'inhA', 'Rv3795': 'embB',
    'Rv2043c': 'pncA', 'Rv0006': 'gyrA', 'Rv0005': 'gyrB', 'Rv0668': 'rpoC',
    'MTB000019': 'rrs', 'MTB000020': 'rrl', 'Rv2416c': 'eis', 'Rv0678': 'Rv0678',
    'Rv0701': 'rplC', 'Rv1694': 'tlyA', 'Rv3854c': 'ethA', 'Rv3919c': 'gid',
    'Rv1772': 'pepQ', 'Rv1258c': 'tap', 'Rv1267c': 'clpC1', 'Rv0676c': 'mmpR5',
    'Rv1129c': 'mshA', 'Rv0565c': 'ddn', 'Rv3547': 'fbiA', 'Rv3261': 'fbiB',
    'Rv1173': 'fbiC', 'Rv0407': 'fgd1', 'Rv2983': 'fbiD', 'Rv1905c': 'fprA',
    'Rv3806c': 'ubiA', 'Rv2535c': 'pepQ', 'Rv0340': 'iniA', 'Rv0341': 'iniB',
    'Rv0342': 'iniC', 'Rv1630': 'rpsA', 'Rv0682': 'rpsL', 'Rv3423c': 'alr',
    'Rv0486': 'ald', 'Rv3790': 'dprE2', 'Rv1979c': 'lprG', 'Rv0849': 'glpK',
    'Rv2752c': 'Rv2752c', 'Rv2477c': 'Rv2477c', 'Rv1438': 'thyA',
    'Rv2447c': 'folC', 'Rv3002c': 'thyX', 'Rv1626': 'ndh', 'Rv0885': 'embC',
    'Rv3804c': 'embA', 'Rv3265c': 'aftA', 'Rv2220': 'glf', 'Rv0193': 'pykA',
    'Rv3232c': 'alr', 'Rv3266c': 'dprE1', 'Rv0450c': 'mmpL5', 'Rv1854c': 'ndh',
    'Rv1483': 'fabG1', 'Rv2459': 'ribD', 'Rv1592c': 'ndhA'
}

# Three symbols are each reached from two loci: ndh (Rv1626, Rv1854c), alr
# (Rv3423c, Rv3232c), and pepQ (Rv1772, Rv2535c). mutation_id is built from the
# symbol, so the same variant token under either locus collapses to one node.
# A test pins these, so the set cannot grow unnoticed.

# Grading groups 1 and 2 are associated with resistance. Groups 3 to 5 are dropped.
GRADING_CONFIDENCE = {1: 'high', 2: 'moderate'}

# Case-insensitive lookup from either a locus id or a gene symbol to the standard
# symbol, so normalization is one dict hit rather than a per-row scan.
GENE_LOOKUP = {locus.lower(): name for locus, name in GENE_LOCUS.items()}
GENE_LOOKUP.update({name.lower(): name for name in GENE_LOCUS.values()})

# Resolved against this module rather than the working directory.
DATA_DIR = Path(__file__).resolve().parent.parent / "Datasets"
WHO_CATALOG_FILE = DATA_DIR / "WHO-UCN-TB-2023.7-eng.xlsx"


class WHOCatalog:
    """The WHO catalog workbook as rows the graph loader can merge."""

    def __init__(self, filepath=None):
        self.filepath = filepath or WHO_CATALOG_FILE
        self.data = None

    def read(self):
        # openpyxl warns about a conditional formatting extension this reader
        # never looks at. Scoped, so importing this module stays silent about
        # every other warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Conditional Formatting extension")
            df = pd.read_excel(self.filepath, sheet_name=0, header=2)
        self.data = self.graded_rows(df)
        return self.data

    @staticmethod
    def graded_rows(df):
        """Rows in grading groups 1 and 2, carrying confidence from the grading."""
        gradings = [c for c in df.columns if 'grading' in str(c).lower()]
        if not gradings:
            raise ValueError(f"no grading column in {list(df.columns)}")

        grading = next((c for c in gradings if 'final' in str(c).lower()), gradings[0])
        cols = ['drug', 'gene', 'mutation', 'variant', 'tier']
        df = df[cols + [grading]].dropna(subset=['drug', 'gene', 'tier', grading]).copy()
        group = df[grading].str.extract(r'^\s*(\d)')[0].astype('Int64')
        df['confidence'] = group.map(GRADING_CONFIDENCE)
        return df.dropna(subset=['confidence'])

    @staticmethod
    def normalize_gene(who_gene_name):
        """Map a WHO gene identifier, locus id or symbol, to the standard symbol."""
        if pd.isna(who_gene_name):
            return None
        gene = str(who_gene_name).strip()
        return GENE_LOOKUP.get(gene.lower(), gene)

    @staticmethod
    def normalize_drug(drug_name):
        """Map a WHO drug name to the system's canonical spelling."""
        if pd.isna(drug_name):
            return None
        drug = str(drug_name).lower().strip()
        return DRUG_ALIASES.get(drug, drug)

    def stats(self):
        if self.data is None:
            self.read()

        # Tiers are counted as they appear rather than against a fixed 1 and 2.
        # On the 2023 catalog every resistance-associated row is tier 1, so a
        # standing tier 2 line reported zero on every run. Confidence is the
        # more useful axis, since it becomes the level on the resistance edge.
        return {
            'total_mutations': len(self.data),
            'unique_drugs': self.data['drug'].nunique(),
            'unique_genes': self.data['gene'].nunique(),
            'by_tier': self.data['tier'].astype(int).value_counts().sort_index().to_dict(),
            'by_confidence': {level: int((self.data['confidence'] == level).sum())
                              for level in GRADING_CONFIDENCE.values()},
        }

    def unique_mutations(self, df):
        """Canonical ids per isolate, deduped on mutation_id and drug."""
        gene = df['gene'].map(self.normalize_gene)
        token = df['variant'].fillna(df['mutation']).astype(str).str.split('_', n=1).str[-1]

        out = pd.DataFrame({
            'mutation_id': (gene.astype(str) + '_' + token).to_numpy(),
            'gene': gene.to_numpy(),
            'drug': df['drug'].map(self.normalize_drug).to_numpy(),
            'tier': df['tier'].astype(int).to_numpy(),
            'confidence': df['confidence'].to_numpy(),
        })
        out = out.drop_duplicates(subset=['mutation_id', 'drug'])
        return out.to_dict('records')

    def batch_mutations(self, batch_size=1000):
        """Yield deduplicated WHO mutations in batches."""
        mutations = self.read()
        unique = self.unique_mutations(mutations)
        print(f"WHO catalog: {len(mutations)} rows -> {len(unique)} unique mutations")

        for i in range(0, len(unique), batch_size):
            yield unique[i:i + batch_size]


def main():
    stats = WHOCatalog().stats()
    print("WHO Data")
    print(f"Total mutations: {stats['total_mutations']:,}")
    print(f"Drugs: {stats['unique_drugs']}")
    print(f"Genes: {stats['unique_genes']}")
    print("Tier: " + ", ".join(f"{t} {n:,}" for t, n in stats['by_tier'].items()))
    print("Confidence: " + ", ".join(f"{c} {n:,}" for c, n in stats['by_confidence'].items()))


if __name__ == '__main__':
    main()