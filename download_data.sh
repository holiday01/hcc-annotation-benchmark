#!/usr/bin/env bash
#
# Download all 6 benchmark datasets from NCBI GEO into ./data/.
#
# Usage:
#     bash download_data.sh                # download all
#     bash download_data.sh GSE125449      # download only one
#
# Requirements:
#     curl, gunzip, tar
#
# Each dataset goes into data/<GSEXXX>/suppl/ matching the directory layout
# the scripts in scripts/ expect (override with BENCHMARK_DATA_DIR env var,
# see scripts for details).

set -euo pipefail

DATA_DIR="${BENCHMARK_DATA_DIR:-$(pwd)/data}"
mkdir -p "${DATA_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build NCBI FTP URL for a GSE accession
#   GSE149614 → https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/
# ─────────────────────────────────────────────────────────────────────────────
gse_suppl_url() {
    local acc=$1
    local prefix="${acc%???}"          # GSE149614 → GSE149
    printf "https://ftp.ncbi.nlm.nih.gov/geo/series/%snnn/%s/suppl/" \
           "${prefix}" "${acc}"
}

download_gse() {
    local acc=$1
    local out="${DATA_DIR}/${acc}/suppl"
    mkdir -p "${out}"
    local url
    url=$(gse_suppl_url "${acc}")
    echo "─── ${acc} ───────────────────────────────────────────────"
    echo "  URL : ${url}"
    echo "  Out : ${out}"
    # Recursive mirror of the suppl/ directory (skip index.html, parent dirs,
    # robots.txt). Re-runs are safe — curl --remote-time keeps timestamps and
    # skips unchanged files when used with --time-cond, but for portability we
    # just always pull; GEO files are static.
    curl -fSL --retry 3 --retry-delay 5 \
         --create-dirs --output-dir "${out}" \
         -O -O -O -O -O -O -O -O -O -O \
         "${url}*" 2>/dev/null || true
    # Fallback: parse directory listing and download each file (more reliable
    # than glob expansion against curl).
    local listing
    listing=$(curl -fsSL "${url}" 2>/dev/null || true)
    if [[ -z "${listing}" ]]; then
        echo "  WARNING: could not list ${url}; skipping ${acc}"
        return
    fi
    # Pull href-style links (NCBI listing is HTML)
    echo "${listing}" | grep -oE 'href="[^"]+"' \
        | sed -E 's/href="([^"]+)"/\1/' \
        | grep -vE '^(\.\.|/)' \
        | while read -r fname; do
              [[ "${fname}" == */ ]] && continue   # skip subdirs
              [[ -f "${out}/${fname}" ]] && {
                  echo "  exists: ${fname}"
                  continue
              }
              echo "  fetch : ${fname}"
              curl -fSL --retry 3 --retry-delay 5 \
                   -o "${out}/${fname}" "${url}${fname}"
          done
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset post-processing (decompression for files the scripts expect
# uncompressed; e.g. GSE149614's 3.5 GB count.txt).
# ─────────────────────────────────────────────────────────────────────────────
post_process() {
    local acc=$1
    local out="${DATA_DIR}/${acc}/suppl"
    case "${acc}" in
        GSE149614)
            # 01_preprocess_benchmark_datasets.py expects the count.txt
            # uncompressed (it streams via pandas).
            if [[ -f "${out}/GSE149614_HCC.scRNAseq.S71915.count.txt.gz" \
                  && ! -f "${out}/GSE149614_HCC.scRNAseq.S71915.count.txt" ]]; then
                echo "  decompressing count.txt.gz (~3.5 GB)..."
                gunzip -k "${out}/GSE149614_HCC.scRNAseq.S71915.count.txt.gz"
            fi
            ;;
        GSE125449)
            # Decompress 10x triplets for Set1/Set2.
            for setname in Set1 Set2; do
                for ext in barcodes.tsv genes.tsv matrix.mtx; do
                    local f="${out}/GSE125449_${setname}_${ext}"
                    if [[ -f "${f}.gz" && ! -f "${f}" ]]; then
                        echo "  decompressing ${setname}/${ext}.gz"
                        gunzip -k "${f}.gz"
                    fi
                done
            done
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
DATASETS=(GSE125449 GSE149614 GSE156625 GSE162616 GSE202642 GSE223204)

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
fi

for acc in "${DATASETS[@]}"; do
    download_gse "${acc}"
    post_process "${acc}"
done

echo ""
echo "Done. Data lives under: ${DATA_DIR}"
echo ""
echo "NOTE: the scripts in scripts/ currently reference absolute paths under"
echo "      /mnt/10t/assi_result/GEO_DATA/ and /mnt/10t/assi_result/HCC/."
echo "      Either symlink:"
echo "          ln -s \"${DATA_DIR}\" /mnt/10t/assi_result/GEO_DATA"
echo "      or set BENCHMARK_DATA_DIR=${DATA_DIR} and edit the GEO_DATA"
echo "      reference at the top of 01_preprocess_benchmark_datasets.py and"
echo "      16_prepare_gse156625_external.py to point at \$BENCHMARK_DATA_DIR."
