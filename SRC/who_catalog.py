import re
from pathlib import Path

import pandas as pd
import numpy as np


GENE_LOCUS = {
    'Rv0667': 'rpoB', 'Rv1908c': 'katG', 'Rv1484': 'inhA', 'Rv3795': 'embB',
    'Rv2043c': 'pncA', 'Rv0006': 'gyrA', 'Rv0005': 'gyrB', 'Rv0668': 'rpoC',
    'MTB000019': 'rrs', 'MTB000020': 'rrl', 'Rv2416c': 'eis', 'Rv0678': 'Rv0678',
    'Rv0701': 'rplC', 'Rv1694': 'tlyA', 'Rv3854c': 'ethA', 'Rv3919c': 'gid',
    'Rv1772': 'pepQ', 'Rv1258c': 'tap', 'Rv1267c': 'clpC1', 'Rv0676c': 'mmpR5',
    'Rv1129c': 'mshA', 'Rv0565c': 'ddn', 'Rv3547': 'fbiA', 'Rv3261': 'fbiB',
    'Rv1173': 'fbiC', 'Rv0407': 'fgd1', 'Rv2983': 'fbiD', 'Rv1905c': 'fprA',
    'Rv3806c': 'ubiA', 'Rv2535c': 'pepQ', 'Rv0340': 'iniA', 'Rv0341': 'iniB',
    'Rv0342': 'iniC', 'Rv1630': 'rpsL', 'Rv0682': 'rpsA', 'Rv3423c': 'alr',
    'Rv0486': 'ald', 'Rv3790': 'dprE2', 'Rv1979c': 'lprG', 'Rv0849': 'glpK',
    'Rv2752c': 'Rv2752c', 'Rv2477c': 'Rv2477c', 'Rv1634': 'rpsA', 'Rv1438': 'thyA',
    'Rv2447c': 'folC', 'Rv3002c': 'thyX', 'Rv1626': 'ndh', 'Rv0885': 'embC',
    'Rv3804c': 'embA', 'Rv3265c': 'aftA', 'Rv2220': 'glf', 'Rv0193': 'pykA',
    'Rv3232c': 'alr', 'Rv3266c': 'dprE1', 'Rv0450c': 'mmpL5', 'Rv1854c': 'ndh',
    'Rv1483': 'fabG1', 'Rv2459': 'ribD', 'Rv1592c': 'ndhA'
}

# WHO drug spellings that differ from the system's canonical names. Any drug not
# listed passes through unchanged, so only genuine remappings belong here.
DRUG_ALIASES = {
    'rifampicin': 'rifampin'
}

# Case-insensitive lookup from either a locus id or a gene symbol to the standard
# gene symbol, derived once from GENE_LOCUS so normalization is a single dict hit
# rather than a per-row scan of the table.
GENE_LOOKUP = {locus.lower(): name for locus, name in GENE_LOCUS.items()}
GENE_LOOKUP.update({name.lower(): name for name in GENE_LOCUS.values()})


# The WHO catalog ships in the project's Datasets folder. Resolve it relative to
# this module (SRC/who_catalog.py -> ../Datasets/) so loading no longer depends on
# the working directory after the project was reorganized into subfolders.
DATA_DIR = Path(__file__).resolve().parent.parent / "Datasets"
WHO_CATALOG_FILE = DATA_DIR / "WHO-UCN-TB-2023.7-eng.xlsx"


class WHOCatalog:
    def __init__(self, filepath=None):
        self.filepath = filepath or WHO_CATALOG_FILE
        self.data = None

    def read(self):
        df = pd.read_excel(self.filepath, sheet_name=0, header=2)
        self.data = self._clean(df)
        return self.data

    def _clean(self, df):
        core_cols = ['drug', 'gene', 'mutation', 'variant', 'tier']
        df = df[core_cols].copy()
        df = df.dropna(subset=['drug', 'gene', 'tier'])
        return df

    def _format_mutation_id(self, gene, variant):
        """Format a WHO variant to match the system's mutation IDs. A lowercase
        nucleotide change like '1349a>g' is reordered to 'a1349g'; an already
        gene-prefixed variant is kept; anything else is prefixed with the gene."""
        if not gene or not variant:
            return variant

        variant = str(variant).strip()

        if variant.startswith(f"{gene}_"):
            return variant

        nucl_match = re.match(r'^(\d+)([a-z])>([a-z])$', variant)
        if nucl_match:
            pos, ref, alt = nucl_match.groups()
            return f"{gene}_{ref}{pos}{alt}"

        return f"{gene}_{variant}"

    @staticmethod
    def _normalize_gene(who_gene_name):
        """Map a WHO gene identifier (locus id or symbol) to the standard symbol."""
        if pd.isna(who_gene_name):
            return None
        gene = str(who_gene_name).strip()
        return GENE_LOOKUP.get(gene.lower(), gene)

    @staticmethod
    def _normalize_drug(drug_name):
        """Normalize WHO drug names to the system's canonical spelling"""
        if pd.isna(drug_name):
            return None
        drug = str(drug_name).lower().strip()
        return DRUG_ALIASES.get(drug, drug)

    def stats(self):
        if self.data is None:
            self.read()

        return {
            'total_mutations': len(self.data),
            'unique_drugs': self.data['drug'].nunique(),
            'unique_genes': self.data['gene'].nunique(),
            'tier_1_count': (self.data['tier'] == 1).sum(),
            'tier_2_count': (self.data['tier'] == 2).sum()
        }

    def _unique_mutations(self, df):
        """Normalize the catalog column-wise and drop duplicate (mutation_id, drug)
        pairs. Only the gene-prefix formatting stays per-value, since its regex
        reordering has no clean column-wise form."""
        gene = df['gene'].map(self._normalize_gene)
        variant = df['variant'].fillna(df['mutation']).astype(str)
        tier = df['tier'].astype(int)

        out = pd.DataFrame({
            'mutation_id': [self._format_mutation_id(g, v) for g, v in zip(gene, variant)],
            'gene': gene.to_numpy(),
            'drug': df['drug'].map(self._normalize_drug).to_numpy(),
            'tier': tier.to_numpy(),
            'confidence': np.where(tier.to_numpy() == 1, 'high', 'moderate'),
        })
        out = out.drop_duplicates(subset=['mutation_id', 'drug'])
        return out.to_dict('records')

    def batch_mutations(self, batch_size=1000):
        """Yield deduplicated WHO mutations in batches"""
        mutations = self.read()
        unique = self._unique_mutations(mutations)
        print(f"WHO catalog: {len(mutations)} rows -> {len(unique)} unique mutations")

        for i in range(0, len(unique), batch_size):
            yield unique[i:i + batch_size]


def test():
    catalog = WHOCatalog()
    catalog.read()

    stats = catalog.stats()
    print("WHO Data")
    print(f"Total mutations: {stats['total_mutations']:,}")
    print(f"Drugs: {stats['unique_drugs']}")
    print(f"Genes: {stats['unique_genes']}")
    print(f"Tier 1: {stats['tier_1_count']:,}")
    print(f"Tier 2: {stats['tier_2_count']:,}")


if __name__ == '__main__':
    test()