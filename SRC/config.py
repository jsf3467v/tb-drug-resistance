SCHEMA = """
TB Drug Resistance Knowledge Graph Schema

NODE TYPES

Gene:
  Properties:
    - name: string (REQUIRED, UNIQUE) - e.g., "rpoB", "katG"
    - locus: string - e.g., "Rv0667"
    - function: string - e.g., "RNA polymerase beta subunit"
    - drug_target: string - e.g., "rifampin"

Drug:
  Properties:
    - name: string (REQUIRED, UNIQUE) - e.g., "rifampin"
    - class: string - "first-line", "second-line", "new-drug", "repurposed"
    - abbreviation: string - e.g., "RIF"
    - mechanism: string - e.g., "RNA polymerase inhibitor"

Mutation:
  Properties:
    - mutation_id: string (REQUIRED, UNIQUE) - e.g., "rpoB_p.Ser450Leu"
    - position: integer - e.g., 450
    - ref_amino_acid: string - e.g., "S"
    - alt_amino_acid: string - e.g., "L"

Strain:
  Properties:
    - strain_id: string (REQUIRED, UNIQUE) - e.g., "TB001"
    - lineage: string - e.g., "Beijing", "Euro-American"
    - country: string - e.g., "China", "USA"
    - year: integer - e.g., 2023, 2024

Patient:
  Properties:
    - patient_id: string (REQUIRED, UNIQUE) - e.g., "P001"
    - age: integer
    - sex: string - "M" or "F"
    - hiv_status: string - "positive" or "negative"
    - country: string
    - region: string - WHO region, e.g., "African", "SE_Asia", "European"
    - diabetes: boolean
    - previous_treatment: boolean

ResistanceProfile:
  Properties:
    - type: string (REQUIRED, UNIQUE) - "Susceptible", "MonoResistant", "PolyResistant", "MDR", "PreXDR", "XDR"
    - abbreviation: string - "S", "MR", "PR", "MDR", "PreXDR", "XDR"
    - description: string

RELATIONSHIP TYPES

(Mutation)-[:IN_GENE]->(Gene)
  Properties: none

(Mutation)-[:CONFERS_RESISTANCE]->(Drug)
  Properties:
    - level: string - "high", "moderate", or "low"

(Strain)-[:HAS_MUTATION]->(Mutation)
  Properties: none

(Strain)-[:HAS_PROFILE]->(ResistanceProfile)
  Properties: none

(Patient)-[:INFECTED_WITH]->(Strain)
  Properties:
    - date: date (optional)

(Strain)-[:TRANSMITTED_TO]->(Strain)
  Properties:
    - location: string (optional)
    - date: date (optional)

IMPORTANT NOTES
- Drug names are lowercase: "rifampin", "isoniazid", "ethambutol"
- Drug names use American spelling. Query rifampicin as rifampin.
- Gene names are exact: "rpoB", "katG", "inhA", "embB", "pncA", "gyrA", "gyrB"
- Use MATCH before WHERE
- For a question with several conditions, AND every condition in one WHERE so all must hold.
- Inline literal values directly in the query; the read path does not bind $parameters
"""

EXAMPLES = """
Example Query Patterns

Example 1: Simple drug resistance lookup
Question: What mutations cause rifampin resistance?
Cypher: MATCH (m:Mutation)-[r:CONFERS_RESISTANCE]->(d:Drug {name: 'rifampin'})
        MATCH (m)-[:IN_GENE]->(g:Gene)
        RETURN g.name as gene, m.mutation_id as mutation, r.level as resistance_level
        ORDER BY g.name, m.position

Example 2: Treatment options for strain
Question: What drugs can treat strain TB001?
Cypher: MATCH (s:Strain {strain_id: 'TB001'})-[:HAS_MUTATION]->(m:Mutation)-[:CONFERS_RESISTANCE]->(resistant_drug:Drug)
        WITH s, collect(DISTINCT resistant_drug.name) as resistant_drugs
        MATCH (d:Drug)
        WHERE NOT d.name IN resistant_drugs
        RETURN d.name as drug, d.class as drug_class
        ORDER BY d.class, d.name

Example 3: Strain profile
Question: What is the resistance profile of strain TB003?
Cypher: MATCH (s:Strain {strain_id: 'TB003'})
        OPTIONAL MATCH (s)-[:HAS_PROFILE]->(p:ResistanceProfile)
        OPTIONAL MATCH (s)-[:HAS_MUTATION]->(m:Mutation)
        OPTIONAL MATCH (m)-[:CONFERS_RESISTANCE]->(d:Drug)
        RETURN s.lineage as lineage, s.country as country, s.year as year,
               p.type as profile,
               collect(DISTINCT m.mutation_id) as mutations,
               collect(DISTINCT d.name) as resistant_to

Example 4: Gene mutation count
Question: Which genes have the most resistance mutations?
Cypher: MATCH (m:Mutation)-[:IN_GENE]->(g:Gene)
        MATCH (m)-[:CONFERS_RESISTANCE]->(d:Drug)
        RETURN g.name as gene, g.function as function,
               count(DISTINCT m) as mutation_count
        ORDER BY mutation_count DESC
        LIMIT 10

Example 5: Strains by resistance profile
Question: Show all MDR strains
Cypher: MATCH (s:Strain)-[:HAS_PROFILE]->(r:ResistanceProfile {type: 'MDR'})
        RETURN s.strain_id as strain, s.lineage as lineage,
               s.country as country, s.year as year
        ORDER BY s.year DESC, s.strain_id

Example 6: Patient treatment options
Question: What drugs can treat patient P003?
Cypher: MATCH (p:Patient {patient_id: 'P003'})-[:INFECTED_WITH]->(s:Strain)
        MATCH (s)-[:HAS_MUTATION]->(m:Mutation)-[:CONFERS_RESISTANCE]->(resistant_drug:Drug)
        WITH p, collect(DISTINCT resistant_drug.name) as resistant_drugs
        MATCH (d:Drug)
        WHERE NOT d.name IN resistant_drugs
        RETURN d.name as drug, d.class as drug_class, d.mechanism as mechanism
        ORDER BY d.class, d.name

Example 7: Mutations in specific gene
Question: Show all mutations in the rpoB gene
Cypher: MATCH (m:Mutation)-[:IN_GENE]->(g:Gene {name: 'rpoB'})
        OPTIONAL MATCH (m)-[:CONFERS_RESISTANCE]->(d:Drug)
        RETURN m.mutation_id as mutation, m.position as position,
               collect(DISTINCT d.name) as affects_drugs
        ORDER BY position

Example 8: Lineage comparison
Question: Compare resistance profiles of Beijing vs Euro-American lineages
Cypher: MATCH (s:Strain)-[:HAS_PROFILE]->(p:ResistanceProfile)
        WHERE s.lineage = 'Beijing' OR s.lineage = 'Euro-American'
        RETURN s.lineage as lineage, p.type as profile, count(s) as strain_count
        ORDER BY s.lineage, strain_count DESC

Example 9: High-risk patients
Question: Which patients have strains resistant to first-line drugs?
Cypher: MATCH (p:Patient)-[:INFECTED_WITH]->(s:Strain)-[:HAS_MUTATION]->(m:Mutation)
        MATCH (m)-[:CONFERS_RESISTANCE]->(d:Drug)
        WHERE d.class = 'first-line'
        RETURN p.patient_id as patient, p.age as age, p.hiv_status as hiv_status,
               s.strain_id as strain, collect(DISTINCT d.name) as resistant_drugs
        ORDER BY size(resistant_drugs) DESC

Example 10: Geographic distribution
Question: Show resistance profiles by country
Cypher: MATCH (s:Strain)-[:HAS_PROFILE]->(p:ResistanceProfile)
        RETURN s.country as country, p.type as profile, count(s) as count
        ORDER BY s.country, count DESC

Example 11: Comparative analysis with aggregation
Question: Compare Beijing lineage to Euro-American lineage
Cypher: MATCH (s:Strain)
        WHERE s.lineage = 'Beijing' OR s.lineage = 'Euro-American'
        OPTIONAL MATCH (s)-[:HAS_PROFILE]->(p:ResistanceProfile)
        OPTIONAL MATCH (s)-[:HAS_MUTATION]->(m:Mutation)
        RETURN s.lineage as lineage,
               count(DISTINCT s) as strain_count,
               collect(DISTINCT p.type) as profiles
        ORDER BY lineage
"""

# Drug-name variants. The WHO spelling rifampicin maps to the catalog's American
# spelling rifampin, so a query on either name resolves to the same drug node. The
# catalog loader and the NL layer import this map. The feature builder keeps its
# own copy to stay standalone.
DRUG_ALIASES = {'rifampicin': 'rifampin'}
