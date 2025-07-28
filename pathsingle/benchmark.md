# Pathway Activity Using Single Cells Benchmark
Single-cell RNA-seq provides unprecedented insights into variations in cell types between conditions, tissue types, species and individuals. Differential gene expression analysis of the single-cell data is almost always followed by *gene set enrichment analysis*, where the aim is to identify gene programs, such as biological processes, gene ontologies or regulatory pathways that are over-represented in an experimental condition compared to control or other conditions, on the basis of differentially expressed (DE) genes.

### Pathway activity tests
Gene set tests test whether a pathway is enriched, in other words over-represented, in one condition compared to others, say, in healthy donors compared to severe COVID-19 patients. An alternative approach is to simply score the activity of a pathway or gene signature, in absolute sense, in individual cells, rather than testing for a differential activity between conditions. Some of the widely used tools for inference of gene set activity in general (including pathway activity) in individual cells include *VISION* {cite}`detomaso2019functional`, *AUCell* <cite>`aibar2017scenic`</cite>, pathway overdispersion analysis using *Pagoda2* <cite>`fan2016characterizing, lake2018integrative`</cite> and simple combined z-score <cite>`lee2008inferring`</cite>. 

*DoRothEA* <cite>`garcia2019benchmark`</cite> and *PROGENy* <cite>`schubert2018perturbation`</cite> are among functional analysis tools developed to infer transcription factor (TF) - target activities originally in Bulk RNA data. Holland et al. <cite>`holland2020robustness`</cite> found that Bulk RNA-seq methods *DoRothEA* and *PROGENy* have optimal performance in simulated scRNA-seq data, and even partially outperform tools specifically designed for scRNA-seq analysis despite the drop-out events and low library sizes in single cell data. Holland et al. also concluded that pathway and TF activity inference is more sensitive to the choice of gene sets rather than the statistical methods. This observation though can be specific to functional enrichment analyses and be explained by the fact that TF-target relations are context-specific; that is TF-target associations in one cell type may actually differ from another cell type or tissue.  

In contrast to Holand et al., Zhang et al. <cite>`zhang2020benchmarking`</cite> found that single-cell-based tools, specifically Pagoda2, outperform bulk-base methods from three different aspects of accuracy, stability and scalability. It should be noted that pathway and gene set activity inference tools inherently do not account for batch effects or biological variations other than the biological variation of interest. Therefore, it is up to the data analyst to ensure that the differential gene expression analysis step has worked properly.

### PROGENy
PROGENy is a comprehensive resource containing a curated collection of pathways and their target genes, with weights for each interaction. Here is a brief description of each pathway:
- Androgen: involved in the growth and development of the male reproductive organs.
- EGFR: regulates growth, survival, migration, apoptosis, proliferation, and differentiation in mammalian cells
- Estrogen: promotes the growth and development of the female reproductive organs.
- Hypoxia: promotes angiogenesis and metabolic reprogramming when O2 levels are low.
- JAK-STAT: involved in immunity, cell division, cell death, and tumor formation.
- MAPK: integrates external signals and promotes cell growth and proliferation.
- NFkB: regulates immune response, cytokine production and cell survival.
- p53: regulates cell cycle, apoptosis, DNA repair and tumor suppression.
- PI3K: promotes growth and proliferation.
- TGFb: involved in development, homeostasis, and repair of most tissues.
- TNFa: mediates haematopoiesis, immune surveillance, tumour regression and protection from infection.
- Trail: induces apoptosis.
- VEGF: mediates angiogenesis, vascular permeability, and cell migration.
- WNT: regulates organ morphogenesis during development and tissue repair.

### AUCell
Unlike the previous approach where we assessed gene set *enrichment* per *cluster* (or rather cell type), one can *score* the activity level of pathways and gene sets in each individual cell, that is based on absolute gene expression in the cell, regardless of expression of genes in the other cells. This we can achieve by activity scoring tools such as `AUCell`.

Similar to `GSEA`, we will be using the `decoupler` implementation of `AUCell`. Make sure to run the previous cell for downloading the REACTOME gene sets.

### Cluster Metrics
Following is a brief overview of the metrics we use to evaluate clustering results.

 - Silhouette Score: This score measures how similar a point is to its own cluster compared to other clusters. The silhouette_score function from sklearn.metrics is used. The Silhouette score ranges from -1 to 1, where a higher value indicates better-defined clusters. A score close to 0 suggests that clusters are overlapping or poorly defined. A negative value, as in this case, typically indicates that data points might have been assigned to the wrong clusters. Given that your score is slightly negative, it suggests potential issues with cluster separability or consistency.

 - Calinski-Harabasz Index: This index evaluates the ratio of the sum of between-cluster dispersion and of within-cluster dispersion. The calinski_harabasz_score function from sklearn.metrics is used. The Calinski-Harabasz index, also known as the Variance Ratio Criterion, is used to assess cluster separation. Higher values indicate better-defined clusters.
 
 - Dunn Index: This index is calculated by finding the minimum inter-cluster distance and dividing it by the maximum intra-cluster distance. The custom function dunn_index is used to compute this.
 The Dunn index evaluates cluster compactness and separation. Higher values are better, indicating well-separated and compact clusters.

 ### Datasets
 #### PBMC Benchmark
 The original dataset is from a single donor, from 10xgenomix. Fresh 68k PBMCs (Donor A).

- ~68,000 cells detected
- Sequenced on Illumina NextSeq 500 High Output with ~20,000 reads per cell
- 98bp read1 (transcript), 8bp I5 sample barcode, 14bp I7 GemCode barcode and 5bp read2 (UMI)
- Analysis run with --cells=24000
- It seems that preprocessing steps such as normalization and scaling were already performed on the dataset.

#### T-cells Challenge Dataset
- The data contained information from approximately 30,000 cells and over 15,000 genes.
- 30,000 T cells from a mouse tumor were collected after undergoing one of 73 different CRISPR-mediated single gene knockouts. 
- The challenge aimed to predict the proportions of cell states from five distinct categories: Progenitor, Effector, Terminal exhausted, Cycling, and Other.
- The organizers focused on evaluating seven target genes: AQR, BACH2, BHLHE40, ETS1, FOSB, MAFK, and STAT3.
- Data download: The challenge data is available on AWS cloud servers. You need to register for the challenge to access the original data:
    - https://www.topcoder.com/challenges/25f60820-2e69-444b-bc03-490686af2c87?tab=details
    - Download `sc_training.h5ad` from the forum link:
    - https://discussions.topcoder.com/discussion/25381/challenge-specifications-and-data-for-cancer-immunotherapy-data-science-grand-

### Running The Benchmark
- Select the dataset for running from the two options, PBMC or T-cells, by commenting out the relevant code in benchmark.py.
- Run the following:

`python benchmark.py`
- The challenge runs each method 30 times. It can be configured in the script itself.

### References
https://www.sc-best-practices.org/conditions/gsea_pathway.html

https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index

https://en.wikipedia.org/wiki/Dunn_index

https://en.wikipedia.org/wiki/Silhouette_(clustering)#:~:text=The%20silhouette%20score%20is%20specialized,distance%20or%20the%20Manhattan%20distance.

https://nbisweden.github.io/workshop-archive/workshop-scRNAseq/2020-01-27/labs/compiled/scanpy/scanpy_04_clustering.html

https://decoupler-py.readthedocs.io/en/latest/notebooks/progeny.

https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc68k_reduced.html#norm

https://scanpy-tutorials.readthedocs.io/en/latest/paga-paul15.html

https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0