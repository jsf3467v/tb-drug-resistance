"""CRyPTIC feature engineering: turn the release tables into one model-ready
table per isolate. Each row carries the catalog-graded resistance mutations,
the measured resistance profile (DST as the label, UKMYC concordance flagged),
the catalog genotypic profile, and a seeded stratified train/test split.

The rule-engine validation and the classifier both read this table, so the
label, the features, and the split are defined a single time. Reads only
parquet, imports nothing from the system, and resolves paths relative to the
project layout.
"""

import random
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "Datasets"
TABLE = DATA / "cryptic_features.parquet"

SEED = 42
TEST_FRACTION = 0.2

DRUG_ALIASES = {"rifampicin": "rifampin"}
FIRST_LINE = {"rifampin", "isoniazid", "ethambutol", "pyrazinamide"}
FLUOROQUINOLONES = {"levofloxacin", "moxifloxacin", "ofloxacin",
                    "ciprofloxacin", "gatifloxacin", "sitafloxacin", "fluoroquinolone"}
INJECTABLES = {"amikacin", "kanamycin", "capreomycin"}
SEVERITY = ["Susceptible", "MonoResistant", "PolyResistant", "MDR", "PreXDR", "XDR"]


def flat(df, key):
    """Move parquet index levels into columns when the key is not already one."""
    return df if key in df.columns else df.reset_index()


def drug_map(path):
    """Each 3-letter drug code mapped to its system drug name."""
    codes = pd.read_csv(path)
    names = codes["DRUG_NAME"].str.lower().map(lambda n: DRUG_ALIASES.get(n, n))
    return dict(zip(codes["DRUG_3_LETTER_CODE"], names))


def profile(drugs):
    """Resistance profile from a set of resistant drug names."""
    rif, inh = "rifampin" in drugs, "isoniazid" in drugs
    fq, inj = bool(drugs & FLUOROQUINOLONES), bool(drugs & INJECTABLES)
    if rif and inh and fq and inj:
        return "XDR"
    if rif and inh and (fq or inj):
        return "PreXDR"
    if rif and inh:
        return "MDR"
    first = len(drugs & FIRST_LINE)
    return "PolyResistant" if first > 1 else ("MonoResistant" if first else "Susceptible")


def resistant_profile(path, pheno_col, drugs):
    """Per-isolate profile from a phenotype table; only R/S calls are used."""
    df = flat(pd.read_parquet(path, columns=["UNIQUEID", "DRUG", pheno_col]), "UNIQUEID")
    calls = df[df[pheno_col].astype(str).isin(["R", "S"])]
    resistant = calls[calls[pheno_col].astype(str) == "R"].copy()
    resistant["drug"] = resistant["DRUG"].astype(str).map(drugs)
    sets = resistant.dropna(subset=["drug"]).groupby("UNIQUEID")["drug"].agg(set)
    out = pd.Series("Susceptible", index=calls["UNIQUEID"].unique())
    out.loc[sets.index] = sets.map(profile)
    return out


def resistance_features(path):
    """Catalog R-graded mutations per isolate, deduped, as a joined string."""
    eff = flat(pd.read_parquet(path, columns=["UNIQUEID", "GENE", "MUTATION", "PREDICTION"]), "GENE")
    r = eff[eff["PREDICTION"].astype(str) == "R"]
    gene = r["GENE"].astype("string").fillna("NA")
    mut = gene.str.cat(r["MUTATION"].astype("string").fillna("NA"), sep="_")
    pairs = r.assign(mut=mut).drop_duplicates(["UNIQUEID", "mut"])
    return pairs.groupby("UNIQUEID")["mut"].agg(lambda s: ";".join(sorted(s)))


def stratified_split(labels, seed=SEED, test_fraction=TEST_FRACTION):
    """A train/test label per isolate, holding out the same fraction of each class."""
    rng = random.Random(seed)
    split = pd.Series("train", index=labels.index)
    for value in labels.unique():
        members = list(labels.index[labels == value])
        rng.shuffle(members)
        cut = round(len(members) * test_fraction)
        split.loc[members[:cut]] = "test"
    return split


def dataset(rebuild=False):
    """The model-ready table, built from the release tables or read from cache."""
    if TABLE.exists() and not rebuild:
        return pd.read_parquet(TABLE)

    drugs = drug_map(DATA / "DRUG_CODES.csv")
    dst = resistant_profile(DATA / "DST_MEASUREMENTS.parquet", "PHENOTYPE", drugs)
    ukmyc = resistant_profile(DATA / "UKMYC_PHENOTYPES.parquet", "BINARY_PHENOTYPE", drugs)
    catalog = resistant_profile(DATA / "PREDICTIONS.parquet", "PREDICTION", drugs)
    features = resistance_features(DATA / "EFFECTS.parquet")

    table = pd.DataFrame({"label": dst, "ukmyc": ukmyc}).dropna(subset=["label"])
    table["concordant"] = (table["ukmyc"] == table["label"]).where(table["ukmyc"].notna())
    table["catalog"] = catalog.reindex(table.index)
    table["resistance"] = features.reindex(table.index).fillna("")
    table["split"] = stratified_split(table["label"])

    table = table.drop(columns="ukmyc").reset_index(names="uniqueid")
    table.to_parquet(TABLE, index=False)
    return table


def main():
    table = dataset(rebuild=True)
    measured_by_both = table["concordant"].notna()

    print(f"isolates: {len(table):,}")
    print("\nlabel balance:")
    print(table["label"].value_counts().reindex(SEVERITY, fill_value=0).to_string())
    print("\nsplit:")
    print(table["split"].value_counts().to_string())
    if measured_by_both.any():
        rate = table.loc[measured_by_both, "concordant"].mean()
        print(f"\nsecond opinion: {int(measured_by_both.sum()):,} isolates, concordance {rate:.1%}")
    print(f"\nsaved: {TABLE.name}")


if __name__ == "__main__":
    main()