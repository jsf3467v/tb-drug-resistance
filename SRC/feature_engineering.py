"""CRyPTIC release tables reduced to one row per isolate. Each row carries the
measured resistance profile as the label, whether the second assay agreed, and
the catalog's genotypic profile. The classification validation reads this table,
and drug classes come from config, so the label and the rule engine grade
against one definition.

The symbolic system has no training phase, so there is no held-out split and
every labeled isolate is scored. See README, Validation.
"""

from pathlib import Path

import pandas as pd
from config import DRUG_ALIASES, FLUOROQUINOLONES, INJECTABLES

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "Datasets"
TABLE = DATA / "cryptic_features.parquet"

# Release tables the cached table is derived from. The module file joins them so
# an edit to the class definitions below invalidates the cache as well.
SOURCES = ("DRUG_CODES.csv", "DST_MEASUREMENTS.parquet",
           "UKMYC_PHENOTYPES.parquet", "PREDICTIONS.parquet")

SEVERITY = ["Susceptible", "MonoResistant", "PolyResistant", "MDR", "PreXDR", "XDR"]


def flat(df, key):
    """Move parquet index levels into columns when the key is not already one.
    Pass the column the caller goes on to read, not a neighboring one."""
    return df if key in df.columns else df.reset_index()


def drug_map(path):
    """Each 3-letter drug code mapped to its system drug name."""
    codes = pd.read_csv(path)
    names = codes["DRUG_NAME"].str.lower().map(lambda n: DRUG_ALIASES.get(n, n))
    return dict(zip(codes["DRUG_3_LETTER_CODE"], names, strict=True))


def profile(drugs):
    """Resistance profile from a set of resistant drug names. The MDR tiers use
    the pre-2021 injectable-based WHO definitions, matching the rule engine,
    which states why in full.

    Mono and poly count every resistant drug rather than the first-line set
    alone. That follows the CDC wording, which counts any TB drug, and the
    extension to second-line agents that WHO flagged as likely needed for
    surveillance once reliable testing existed. WHO's own definition still reads
    first-line only, so this is a stated deviation from it and not an oversight.
    Isolates resistant to both isoniazid and rifampin resolve above this branch,
    so no tier at MDR or higher depends on the choice."""
    rif, inh = "rifampin" in drugs, "isoniazid" in drugs
    fq, inj = bool(drugs & FLUOROQUINOLONES), bool(drugs & INJECTABLES)
    if rif and inh and fq and inj:
        return "XDR"
    if rif and inh and (fq or inj):
        return "PreXDR"
    if rif and inh:
        return "MDR"
    return "PolyResistant" if len(drugs) > 1 else ("MonoResistant" if drugs else "Susceptible")


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


def cache_is_current():
    """True when the cached table is newer than every input it was built from.
    This module and config are inputs too, so editing the class definitions or
    the drug classes forces a rebuild. A missing source means not current."""
    if not TABLE.exists():
        return False
    built = TABLE.stat().st_mtime
    inputs = [DATA / name for name in SOURCES] + [Path(__file__).resolve(), BASE / "config.py"]
    return all(path.exists() and path.stat().st_mtime < built for path in inputs)


def dataset(rebuild=False):
    """The labeled table, built from the release tables or read from cache."""
    if cache_is_current() and not rebuild:
        return pd.read_parquet(TABLE)

    drugs = drug_map(DATA / "DRUG_CODES.csv")
    dst = resistant_profile(DATA / "DST_MEASUREMENTS.parquet", "PHENOTYPE", drugs)
    ukmyc = resistant_profile(DATA / "UKMYC_PHENOTYPES.parquet", "BINARY_PHENOTYPE", drugs)
    catalog = resistant_profile(DATA / "PREDICTIONS.parquet", "PREDICTION", drugs)

    table = pd.DataFrame({"label": dst, "ukmyc": ukmyc}).dropna(subset=["label"])
    table["concordant"] = (table["ukmyc"] == table["label"]) \
        .where(table["ukmyc"].notna()).astype("boolean")
    table["catalog"] = catalog.reindex(table.index)

    table = table.drop(columns="ukmyc").reset_index(names="uniqueid")
    table.to_parquet(TABLE, index=False)
    return table


def main():
    table = dataset(rebuild=True)
    measured_by_both = table["concordant"].notna()

    print(f"isolates: {len(table):,}")
    print("\nlabel balance:")
    print(table["label"].value_counts().reindex(SEVERITY, fill_value=0).to_string())
    if measured_by_both.any():
        rate = table.loc[measured_by_both, "concordant"].mean()
        print(f"\nsecond opinion: {int(measured_by_both.sum()):,} isolates, "
              f"concordance {rate:.1%}")
    print(f"\nsaved: {TABLE.name}")


if __name__ == "__main__":
    main()