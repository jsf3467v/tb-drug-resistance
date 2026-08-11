"""WHO mutation catalog reader. Keeps the grading groups associated with resistance,
maps gene and drug names onto the system spelling, and yields deduplicated
mutations for the graph loader."""

import re
import warnings
from pathlib import Path

import pandas as pd

from config import DRUG_ALIASES

# One locus per symbol, since a shared symbol would merge distinct variants onto
# one node.
GENE_LOCUS = {
    'Rv0667': 'rpoB', 'Rv1908c': 'katG', 'Rv1484': 'inhA', 'Rv3795': 'embB',
    'Rv2043c': 'pncA', 'Rv0006': 'gyrA', 'Rv0005': 'gyrB', 'Rv0668': 'rpoC',
    'MTB000019': 'rrs', 'MTB000020': 'rrl', 'Rv2416c': 'eis',
    'Rv0701': 'rplC', 'Rv1694': 'tlyA', 'Rv3854c': 'ethA', 'Rv3919c': 'gid',
    'Rv1258c': 'tap', 'Rv1267c': 'clpC1', 'Rv0676c': 'mmpL5',
    'Rv1129c': 'mshA', 'Rv0565c': 'ddn', 'Rv3547': 'fbiA', 'Rv3261': 'fbiB',
    'Rv1173': 'fbiC', 'Rv0407': 'fgd1', 'Rv2983': 'fbiD', 'Rv1905c': 'fprA',
    'Rv3806c': 'ubiA', 'Rv2535c': 'pepQ', 'Rv0340': 'iniA', 'Rv0341': 'iniB',
    'Rv0342': 'iniC', 'Rv1630': 'rpsA', 'Rv0682': 'rpsL', 'Rv3423c': 'alr',
    'Rv0486': 'ald', 'Rv3790': 'dprE2', 'Rv1979c': 'lprG', 'Rv0849': 'glpK',
    'Rv1438': 'thyA', 'Rv2447c': 'folC', 'Rv3002c': 'thyX', 'Rv0885': 'embC',
    'Rv3804c': 'embA', 'Rv3265c': 'aftA', 'Rv2220': 'glf', 'Rv0193': 'pykA',
    'Rv3266c': 'dprE1', 'Rv0450c': 'mmpL4', 'Rv1854c': 'ndh',
    'Rv1483': 'fabG1', 'Rv2459': 'ribD', 'Rv1592c': 'ndhA',
    'Rv0678': 'mmpR5'
}

# Loci with no accepted gene symbol. The locus passes through as the node name, so
# unmapped_genes leaves them out rather than reporting a gap with no fix.
LOCUS_PASSTHROUGH = frozenset({'Rv2477c', 'Rv2752c'})

LOCUS_PATTERN = re.compile(r'Rv\d+[A-Za-z]?|MTB\d+')

# Groups 1 and 2 are associated with resistance. Groups 3 to 5 are dropped.
GRADING_CONFIDENCE = {1: 'high', 2: 'moderate'}

# A lower group is the stronger call, so the number ranks the level.
CONFIDENCE_RANK = {level: group for group, level in GRADING_CONFIDENCE.items()}

GENE_LOOKUP = {locus.lower(): name for locus, name in GENE_LOCUS.items()}
GENE_LOOKUP.update({name.lower(): name for name in GENE_LOCUS.values()})

DATA_DIR = Path(__file__).resolve().parent.parent / "Datasets"

# The version separator ships as either a dot or an underscore, so the workbook is
# matched rather than named.
CATALOG_PATTERN = "WHO-UCN-TB-2023?7-eng.xlsx"


def catalog_file(data_dir=DATA_DIR):
    """The workbook, or None when absent. The graph loader skips the merge rather
    than failing, so absence is not an error."""
    found = sorted(data_dir.glob(CATALOG_PATTERN))
    return found[0] if found else None


class WHOCatalog:
    """The WHO catalog workbook as rows the graph loader can merge."""

    def __init__(self, filepath=None):
        self.filepath = filepath or catalog_file()
        self.data = None

    def read(self):
        if self.filepath is None:
            raise FileNotFoundError(f"no workbook matching {CATALOG_PATTERN} in {DATA_DIR}")

        # openpyxl warns about a conditional formatting extension never read here.
        # Scoped, so every other warning still surfaces.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Conditional Formatting extension")
            df = pd.read_excel(self.filepath, sheet_name=0, header=2)
        self.data = self.graded_rows(df)
        return self.data

    @staticmethod
    def graded_rows(df):
        """Rows in groups 1 and 2 carrying a variant token, since a row with neither
        would key a mutation node on a null id."""
        gradings = [c for c in df.columns if 'grading' in str(c).lower()]
        if not gradings:
            raise ValueError(f"no grading column in {list(df.columns)}")

        grading = next((c for c in gradings if 'final' in str(c).lower()), gradings[0])
        cols = ['drug', 'gene', 'mutation', 'variant', 'tier']
        df = df[cols + [grading]].dropna(subset=['drug', 'gene', 'tier', grading]).copy()
        group = df[grading].astype(str).str.extract(r'^\s*(\d)')[0].astype('Int64')
        df['confidence'] = group.map(GRADING_CONFIDENCE)
        df = df[df['variant'].notna() | df['mutation'].notna()]
        return df.dropna(subset=['confidence'])

    @staticmethod
    def normalize_gene(who_gene_name):
        """Map a locus id or symbol to the standard gene symbol."""
        if pd.isna(who_gene_name):
            return None
        gene = str(who_gene_name).strip()
        return GENE_LOOKUP.get(gene.lower(), gene)

    @staticmethod
    def normalize_drug(drug_name):
        """Map a drug name to the canonical spelling."""
        if pd.isna(drug_name):
            return None
        drug = str(drug_name).lower().strip()
        return DRUG_ALIASES.get(drug, drug)

    def rows(self):
        """Graded rows, parsed once and held."""
        if self.data is None:
            self.read()
        return self.data

    def stats(self):
        # Tiers are counted as they appear, since every resistance-associated row in
        # the 2023 catalog is tier 1 and a fixed tier 2 line read zero.
        data = self.rows()
        return {
            'total_mutations': len(data),
            'unique_drugs': data['drug'].nunique(),
            'unique_genes': data['gene'].nunique(),
            'by_tier': data['tier'].astype(int).value_counts().sort_index().to_dict(),
            'by_confidence': {level: int((data['confidence'] == level).sum())
                              for level in GRADING_CONFIDENCE.values()},
        }

    def unmapped_genes(self):
        """Unresolved locus identifiers with their row counts. Symbols are left out,
        since they pass through and the graph merges on the symbol."""
        genes = self.rows()['gene'].astype(str).str.strip()
        gaps = (genes.str.fullmatch(LOCUS_PATTERN)
                & ~genes.str.lower().isin(GENE_LOOKUP)
                & ~genes.isin(LOCUS_PASSTHROUGH))
        return {gene: int(n) for gene, n in genes[gaps].value_counts().items()}

    def unique_mutations(self, df):
        """Canonical ids deduped on mutation_id and drug. The surviving confidence
        becomes the edge level, so the stronger grading is kept."""
        gene = df['gene'].map(self.normalize_gene)
        token = df['variant'].fillna(df['mutation']).astype(str).str.split('_', n=1).str[-1]

        out = pd.DataFrame({
            'mutation_id': (gene.astype(str) + '_' + token).to_numpy(),
            'gene': gene.to_numpy(),
            'drug': df['drug'].map(self.normalize_drug).to_numpy(),
            'tier': df['tier'].astype(int).to_numpy(),
            'confidence': df['confidence'].to_numpy(),
        })
        rank = out['confidence'].map(CONFIDENCE_RANK).to_numpy()
        out = out.iloc[rank.argsort(kind='stable')]
        return out.drop_duplicates(subset=['mutation_id', 'drug']).to_dict('records')

    def batch_mutations(self, batch_size=1000):
        """Yield deduplicated WHO mutations in batches."""
        mutations = self.rows()
        unique = self.unique_mutations(mutations)
        print(f"WHO catalog: {len(mutations):,} graded rows -> {len(unique):,} mutation-drug "
              f"pairs across {len({u['mutation_id'] for u in unique}):,} mutations")

        for i in range(0, len(unique), batch_size):
            yield unique[i:i + batch_size]


def main():
    catalog = WHOCatalog()
    stats = catalog.stats()
    tiers = ", ".join(f"tier {t} {n:,}" for t, n in stats['by_tier'].items())
    confidence = ", ".join(f"{c} {n:,}" for c, n in stats['by_confidence'].items())
    print(f"WHO catalog: {stats['total_mutations']:,} graded rows, "
          f"{stats['unique_drugs']} drugs, {stats['unique_genes']} genes")
    print(f"  {tiers}   {confidence}")

    gaps = catalog.unmapped_genes()
    if gaps:
        print("  unresolved loci " + ", ".join(f"{g} {n:,}" for g, n in gaps.items()))


if __name__ == '__main__':
    main()