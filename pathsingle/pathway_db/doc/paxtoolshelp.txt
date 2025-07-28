C:\Users\user\Documents\GitHub\PathSingle\pathsingle\pathway_db>java -jar paxtools.jar
#logback.classic pattern: %d %-5level [%thread] %logger{25} - %msg%n
(PaxtoolsMain Console) Available Operations:

merge <file1> <file2> <output>
        - merges file2 into file1 and writes it into output
toSIF <input> <output> [-extended] [-andSif] ["include=SIFType,.."] ["exclude=SIFType,.."] ["seqDb=db,.."] ["chemDb=db,.."] [-dontMergeInteractions] [-useNameIfNoId] [<property_accessor> ...]
        - converts a BioPAX model to SIF (default) or custom SIF-like text format;
          will use blacklist.txt (recommended) file in the current directory, if present.
        - Include or exclude to/from the analysis one or more relationship types by
          using 'include=' and/or 'exclude=', respectively, e.g., exclude=NEIGHBOR_OF,INTERACTS_WITH
          (mind using underscore instead of minus sign in the SIF type names; the default is to use all types).
        - With 'seqDb=' and 'chemDb=', you can specify standard sequence/gene/chemical ID type(s)
          (can be just a unique prefix) to match actual xref.db values in the BioPAX model,
          e.g., "seqDb=uniprot,hgnc,refseq", and in that order, means: if a UniProt entity ID is found,
          other ID types ain't used; otherwise, if an 'hgnc' ID/Symbol is found... and so on;
          when not specified, then 'hgnc' (in fact, 'HGNC Symbol') for bio-polymers -
          and ChEBI IDs or name (if '-useNameIfNoId' is set) for chemicals - are selected.
        - With '-extended' flag, the output will be the Pathway Commons TXT (Extended SIF) format:
          two sections separated with one blank line - first come inferred SIF interactions -
          'A    relationship-type       B' plus RESOURCE, PUBMED, PATHWAY, MEDIATOR extra columns,
          followed by interaction participants description section).
        - If '-andSif' flag is present (only makes sense together with '-extended'), then the
          classic SIF output file is also created (will have '.sif' extension).
        - Finally, <property_accessor>... list is to specify 4th, 5th etc. custom output columns;
          use pre-defined column names (accessors):
                MEDIATOR,
                PUBMED,
                PMC,
                COMMENTS,
                PATHWAY,
                PATHWAY_URI,
                RESOURCE,
                SOURCE_LOC,
                TARGET_LOC
          or custom biopax property path accessors (XPath-like expressions to apply to each mediator entity;
          see https://github.com/BioPAX/Paxtools/wiki/PatternBinaryInteractionFramework)
toSBGN <biopax.owl> <output.sbgn> [-nolayout]
        - converts model to the SBGN format and applies COSE layout unless optional -nolayout flag is set.
validate <path> <out> [xml|html|biopax] [auto-fix] [only-errors] [maxerrors=n] [notstrict]
        - validate BioPAX file/directory (up to ~25MB in total size, -
        otherwise download and run the stand-alone validator)
        in the directory using the online validator service
        (generates html or xml report, or gets the processed biopax
        (cannot be perfect though) see http://www.biopax.org/validator)
integrate <file1> <file2> <output>
        - integrates file2 into file1 and writes it into output (experimental)
toLevel3 <input> <output> [-psimiToComplexes]
        - converts BioPAX level 1 or 2, PSI-MI 2.5 and PSI-MITAB to the level 3 file;
        -psimiToComplexes forces PSI-MI Interactions become BioPAX Complexes instead MolecularInteractions.
toGSEA <input> <output> <db> [-crossSpecies] [-subPathways] [-notPathway] [organisms=9606,human,rat,..]
        - converts BioPAX data to the GSEA software format (GMT); options/flags:
        <db> - gene/protein ID type; values: uniprot, hgnc, refseq, etc. (a name or prefix to match
          ProteinReference/xref/db property values in the input BioPAX model).
        -crossSpecies - allows printing on the same line gene/protein IDs from different species;
        -subPathways - traverse into sub-pathways to collect all protein IDs for a pathway.
        -notPathway - also list those protein/gene IDs that cannot be reached from pathways.
        organisms - optional filter; a comma-separated list of taxonomy IDs and/or names

fetch <input> <output> [uris=URI1,..] [-absolute]
        - extracts a self-integral BioPAX sub-model from file1 and writes to the output; options:
        uri=... - an optional list of existing in the model BioPAX elements' full URIs;
        -absolute - set this flag to write full/absolute URIs to the output (i.e., 'rdf:about' instead 'rdf:ID').
getNeighbors <input> <id1,id2,..> <output>
        - nearest neighborhood graph query (id1,id2 - of Entity sub-class only)
summarize <input> <output> [--model] [--pathways] [--hgnc-ids] [--uniprot-ids] [--chebi-ids]
        - (experimental) summary of the input BioPAX model;
        runs one or several analyses and writes to the output file;
        '--model' - (default) BioPAX classes, properties and values summary;
        '--pathways' - pathways and sub-pathways hierarchy;
        '--hgnc-ids' - HGNC IDs/Symbols that occur in sequence entity references;
        '--uniprot-ids' - UniProt IDs in protein references;
        '--chebi-ids' - ChEBI IDs in small molecule references;
        '--uri-ids' - URI,type,name(s) and standard identifiers (in JSON format) for each physical entity;
        the options' order defines the results output order.
blacklist <input> <output>
        - creates a blacklist of ubiquitous small molecules, like ATP,
        from the BioPAX model and writes it to the output file. The blacklist can be used with
        paxtools graph queries or when converting from the SAME BioPAX data to the SIF formats.
pattern
        - BioPAX pattern search tool (opens a new dialog window)
help
        - prints this screen and exits

Commands can also use compressed input files (only '.gz').

java -jar paxtools.jar summarize owl\PathwayCommons12.humancyc.BIOPAX.owl --pathways
java -jar paxtools.jar toSIF owl/PathwayCommons12.kegg.BIOPAX.owl owl/PathwayCommons12.kegg.BIOPAX.sif -extended -useNameIfNoId "seqDb=hgnc,refseq,uniprot" "chemDb=chebi"
References:
https://download.baderlab.org/PathwayCommons/PC2/v12/
https://pmc.ncbi.nlm.nih.gov/articles/PMC3001121/

# Summarize to see pathways.
java -Xmx80g -XX:+UseG1GC -XX:+UseStringDeduplication -XX:MaxGCPauseMillis=500 -XX:G1HeapRegionSize=16M -jar paxtools.jar summarize owl/pc-biopax.owl kegg_paths.txt --pathways

# Getting reactome data.
wget https://reactome.org/download/current/biopax.zip
unzip biopax.zip

# Download UniProt ID to Gene Name mapping file for Human.
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz

# Decompress.
gunzip HUMAN_9606_idmapping.dat.gz

# Filter only Gene Name mappings (using awk).
awk -F'\t' '$2=="Gene_Name" {print $1"\t"$3}' HUMAN_9606_idmapping.dat > uniprot_to_gene.tsv