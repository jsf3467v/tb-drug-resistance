"""CRyPTIC tables reduced to one row per isolate, holding the measured profile as
the label, second-assay agreement, the catalog profile, and second-line coverage.
Drug classes come from config, so the label and the rule engine grade alike."""

from pathlib import Path

import pandas as pd

from config import DRUG_ALIASES, FLUOROQUINOLONES, INJECTABLES, SEVERITY

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "Datasets"
TABLE = DATA / "cryptic_features.parquet"

# Release tables the cached table is derived from. The module file joins them so
# an edit to the class definitions below invalidates the cache as well.
SOURCES = ("DRUG_CODES.csv", "DST_MEASUREMENTS.parquet",
           "UKMYC_PHENOTYPES.parquet", "PREDICTIONS.parquet")

def flat(df, key):
    """Move parquet index levels into columns when the key is not one already."""
    return df if key in df.columns else df.reset_index()


def drug_map(path):
    """Each 3-letter drug code mapped to its system drug name."""
    codes = pd.read_csv(path)
    names = codes["DRUG_NAME"].str.lower().map(lambda n: DRUG_ALIASES.get(n, n))
    return dict(zip(codes["DRUG_3_LETTER_CODE"], names, strict=True))


def profile(drugs):
    """Profile from a set of resistant drug names, on the pre-2021 injectable-based
    definitions the rule engine uses. Mono and poly count every resistant drug
    rather than first-line only, which moves no tier at MDR or above."""
    rif, inh = "rifampin" in drugs, "isoniazid" in drugs
    fq, inj = bool(drugs & FLUOROQUINOLONES), bool(drugs & INJECTABLES)
    if rif and inh and fq and inj:
        return "XDR"
    if rif and inh and (fq or inj):
        return "PreXDR"
    if rif and inh:
        return "MDR"
    return "PolyResistant" if len(drugs) > 1 else ("MonoResistant" if drugs else "Susceptible")


def named_calls(path, pheno_col, drugs):
    """R and S rows with the drug resolved, plus the isolate index, which keeps
    isolates whose only calls name unmapped drugs."""
    df = flat(pd.read_parquet(path, columns=["UNIQUEID", "DRUG", pheno_col]), "UNIQUEID")
    calls = df[df[pheno_col].astype(str).isin(["R", "S"])]
    named = calls.assign(drug=calls["DRUG"].astype(str).map(drugs),
                         resistant=calls[pheno_col].astype(str).eq("R"))
    return calls["UNIQUEID"].unique(), named.dropna(subset=["drug"])


def isolate_profile(index, named):
    """Per-isolate profile. An isolate with no resistant call is susceptible."""
    sets = named[named["resistant"]].groupby("UNIQUEID")["drug"].agg(set)
    out = pd.Series("Susceptible", index=index)
    out.loc[sets.index] = sets.map(profile)
    return out


def second_line_coverage(index, named):
    """True where both a fluoroquinolone and an injectable were measured. Missing
    either caps the label at MDR whatever the genotype carries."""
    isolate = named["UNIQUEID"]
    fq = named["drug"].isin(FLUOROQUINOLONES).groupby(isolate).any()
    injectable = named["drug"].isin(INJECTABLES).groupby(isolate).any()
    return (fq & injectable).reindex(index, fill_value=False)


def resistant_profile(path, pheno_col, drugs):
    """Per-isolate profile from a phenotype table; only R/S calls are used."""
    return isolate_profile(*named_calls(path, pheno_col, drugs))


def cache_is_current():
    """True when the cache is newer than every input. This module and config count
    as inputs, so editing the drug classes forces a rebuild."""
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
    dst_index, dst_calls = named_calls(DATA / "DST_MEASUREMENTS.parquet", "PHENOTYPE", drugs)
    dst = isolate_profile(dst_index, dst_calls)
    ukmyc = resistant_profile(DATA / "UKMYC_PHENOTYPES.parquet", "BINARY_PHENOTYPE", drugs)
    catalog = resistant_profile(DATA / "PREDICTIONS.parquet", "PREDICTION", drugs)

    table = pd.DataFrame({"label": dst, "ukmyc": ukmyc}).dropna(subset=["label"])
    table["concordant"] = (table["ukmyc"] == table["label"]) \
        .where(table["ukmyc"].notna()).astype("boolean")
    table["catalog"] = catalog.reindex(table.index)
    table["second_line_tested"] = second_line_coverage(dst_index, dst_calls) \
        .reindex(table.index, fill_value=False)

    table = table.drop(columns="ukmyc").reset_index(names="uniqueid")
    table.to_parquet(TABLE, index=False)
    return table


def main():
    table = dataset(rebuild=True)
    both = table["concordant"].notna()
    counts = table["label"].value_counts().reindex(list(SEVERITY), fill_value=0)

    print(f"{len(table):,} isolates -> {TABLE.name}")
    print("  " + "  ".join(f"{tier} {n:,}" for tier, n in counts.items()))
    if both.any():
        print(f"  second opinion {int(both.sum()):,} isolates, "
              f"concordance {table.loc[both, 'concordant'].mean():.1%}")


if __name__ == "__main__":
    main()