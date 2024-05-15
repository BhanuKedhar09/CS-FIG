#!/bin/bash

PDF_DIR="/lstr/sahara/datalab-ml/z1974769/cs_pap1_2015/"
STAT_FILE="/lstr/sahara/datalab-ml/z1974769/outputs_pdffigures/real_outputs/stat_file.json"
FIGURE_DATA_PREFIX="/lstr/sahara/datalab-ml/z1974769/outputs_pdffigures/real_outputs/figure_data"
FULL_TEXT_PREFIX="/lstr/sahara/datalab-ml/z1974769/outputs_pdffigures/real_outputs/full_text"
FIGURE_IMAGE_PREFIX="/lstr/sahara/datalab-ml/z1974769/outputs_pdffigures/real_outputs/figure_images"
FIGURE_FORMAT="png"
DESIRED_DIR="/lstr/sahara/datalab-ml/z1974769/pdffigures2/"

export JAVA_HOME=/usr/bin/javac
export PATH=$JAVA_HOME:$PATH
export SBT_OPTS="-Xmx4G"
# export PATH=/usr/bin/javac:$PATH

cd $DESIRED_DIR

# Count and log the number of PDF files to be processed
total_files=$(find "$PDF_DIR" -type f -name "*.pdf" | wc -l)
echo "Total PDF files to process: $total_files"
echo "Starting to process PDF files in $PDF_DIR"
# echo $JAVA

# Execute the sbt command
if sbt "runMain org.allenai.pdffigures2.FigureExtractorBatchCli $PDF_DIR \
    -s $STAT_FILE \
    -m $FIGURE_IMAGE_PREFIX \
    -d $FIGURE_DATA_PREFIX \
    -f $FIGURE_FORMAT"; then
    echo "All files processed successfully."
else
    echo "Error occurred during processing. Check logs for details."
fi

echo "Figure extraction completed."
