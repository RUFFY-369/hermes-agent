---
name: scientific-skills
description: Gateway to 134+ specialized scientific skills from scientific-agent-skills. Covers genomics, cheminformatics, quantum computing, research writing, medical analysis, and more. Fetches domain-specific reference material on demand.
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [science, biology, chemistry, physics, engineering, research, bioinformatics, cheminformatics, medical, data-science, analytics]
    category: research
---

# Scientific Agent Skills Gateway

Use when asked about molecular analysis, drug discovery, genomic/transcriptomic data processing, physics simulations, astronomical calculations, professional scientific writing, grant preparation, clinical reports, or any high-level scientific research task.

This skill is a gateway to the **scientific-agent-skills** library. Instead of bundling 134+ domain-specific skills, it indexes them and fetches what you need on demand.

## Sources

◆ **scientific-agent-skills** — 134+ specialized scientific and research skills
  Repo: https://github.com/K-Dense-AI/scientific-agent-skills
  Format: SKILL.md per domain with expert patterns, references, and executable scripts.

## How to fetch and use a skill

1. Identify the domain and skill name from the index below.
2. Clone the repo (shallow clone to save time):
   ```bash
   git clone --depth 1 https://github.com/K-Dense-AI/scientific-agent-skills.git /tmp/science-library
   ```
3. Read the specific skill:
   ```bash
   # Each skill is at: scientific-skills/<domain-name>/SKILL.md
   cat /tmp/science-library/scientific-skills/rdkit/SKILL.md
   ```
4. Follow the fetched skill as reference material. These are NOT Hermes-format skills — treat them as expert domain guides. They contain correct parameters, proper tool flags, and validated workflows.

## Skill Index by Domain

### Computational Biology & Genomics
scientific-agent-skills:
  anndata/ arboreto/ biopython/ bioservices/ cellxgene-census/ cobrapy/ deepchem/ deeptools/
  depmap/ dnanexus-integration/ esm/ etetoolkit/ flowio/ gget/ ginkgo-cloud-lab/ glycoengineering/
  gtars/ histolab/ imaging-data-commons/ labarchive-integration/ lamindb/ latchbio-integration/ omero-integration/ opentrons-integration/
  pathml/ phylogenetics/ polars-bio/ primekg/ protocolsio-integration/ pydeseq2/ pydicom/ pyhealth/
  pylabrobot/ pyopenms/ pysam/ pytdc/ scanpy/ scikit-bio/ scvelo/ scvi-tools/
  tiledbvcf/

### Cheminformatics & Drug Discovery
scientific-agent-skills:
  adaptyv/ datamol/ diffdock/ medchem/ molecular-dynamics/ molfeat/ rdkit/ rowan/
  torchdrug/

### Physics & Quantum Computing
scientific-agent-skills:
  astropy/ cirq/ geomaster/ pennylane/ pymatgen/ qiskit/ qutip/

### Data Science & Machine Learning
scientific-agent-skills:
  aeon/ dask/ deepchem/ geniml/ geopandas/ optimize-for-gpu/ polars/ pytorch-lightning/
  scikit-learn/ scikit-survival/ seaborn/ shap/ stable-baselines3/ statistical-analysis/ statsmodels/ timesfm-forecasting/
  torch-geometric/ transformers/ umap-learn/ vaex/ zarr-python/

### Research Management & Writing
scientific-agent-skills:
  citation-management/ consciousness-council/ docx/ exploratory-data-analysis/ hypogenic/ hypothesis-generation/ infographics/ iso-13485-certification/
  latex-posters/ literature-review/ markdown-mermaid-writing/ market-research-reports/ markitdown/ paper-lookup/ paperzilla/ parallel-web/
  pdf/ peer-review/ pptx-posters/ pptx/ research-grants/ research-lookup/ scholar-evaluation/ scientific-brainstorming/
  scientific-critical-thinking/ scientific-schematics/ scientific-slides/ scientific-visualization/ scientific-writing/ venue-templates/ what-if-oracle/ xlsx/
  zotero/

### Medical & Clinical Research
scientific-agent-skills:
  clinical-decision-support/ clinical-reports/ neurokit2/ neuropixels-analysis/ treatment-plans/

## Environment Setup

These skills assume a scientific research workstation. Common dependencies:

```bash
# Core Scientific Python
pip install numpy pandas scipy matplotlib seaborn scikit-learn

# Domain Specific
pip install biopython pysam rdkit scanpy anndata astropy qiskit pennylane

# System requirements (Ubuntu/Debian)
sudo apt install git build-essential python3-dev

# This library highly recommends 'uv' for dependency management
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Pitfalls

- The fetched skills are NOT in Hermes SKILL.md format. They use their own structure designed for the Agent Skills standard. Read them as expert reference material.
- Some skills require API keys (e.g., Benchling, Ginkgo, Zotero). Ensure environment variables are set before execution.
- This library uses `uv` for high-speed dependency isolation. If a skill has a `scripts/` or `examples/` folder with Python files, check for a `pyproject.toml` or `requirements.txt`.
- Computational biology tasks often involve large datasets. Be mindful of disk space in `/tmp/` and the local environment.
- Always check the `SECURITY.md` or scan results in the library for any safety considerations before running untrusted research scripts.
