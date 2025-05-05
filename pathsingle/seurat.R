library(Seurat)
library(Matrix)

# Define the data directory
data_dir <- "c:/Users/user/Documents/GitHub/PathSingle/code/data/hg19b/"

# Read the matrix file
matrix_file <- file.path(data_dir, "matrix.mtx")
matrix_data <- readMM(matrix_file)

# Read the barcodes and genes files
barcodes_file <- file.path(data_dir, "barcodes.tsv")
genes_file <- file.path(data_dir, "genes.tsv")

barcodes <- readLines(barcodes_file)
genes <- readLines(genes_file)

# Assign gene names to columns and cell names (barcodes) to rows
colnames(matrix_data) <- barcodes
rownames(matrix_data) <- genes

# Create a Seurat object
pbmc <- CreateSeuratObject(counts = matrix_data, project = "PBMC")

# Normalize the data
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 1e4)

# Identify the 2000 most variable features using the "dispersion" method
pbmc <- FindVariableFeatures(pbmc, selection.method = 'vst', nfeatures = 2000)

# Scale the data
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

# Run PCA
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# Print the PCA results
print(pbmc[["pca"]])

# Extract PCA embeddings
pca_embeddings <- Embeddings(pbmc, "pca")

# Run K-means clustering with 10 clusters
set.seed(42)  # For reproducibility
kmeans_result <- kmeans(pca_embeddings, centers = 10)

# Assign K-means cluster labels to the Seurat object
pbmc$kmeans_clusters <- as.factor(kmeans_result$cluster)

# Set the active identity class to the K-means clusters
Idents(pbmc) <- "kmeans_clusters"

# Look at cluster IDs of the first 5 cells
head(Idents(pbmc), 5)

# Extract cluster identities
cluster_ids <- Idents(pbmc)

# Convert to a data frame
cluster_df <- data.frame(Cell = names(cluster_ids), Cluster = as.vector(cluster_ids))

# Save to a CSV file
write.csv(cluster_df, file = "c:/Users/user/Documents/GitHub/PathSingle/code/data/seurat_clusters.csv", row.names = FALSE)
write.csv(pca_embeddings, file = "c:/Users/user/Documents/GitHub/PathSingle/code/data/pca_embedings.csv", row.names = TRUE)
