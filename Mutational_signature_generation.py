# -*- coding: utf-8 -*-
"""sigprofiler.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JKQYfrgW_OxkcgTGF-5-RRff_5sz2KQ5
"""

import pandas as pd

def transpose_csv_with_headers(input_file_path, output_file_path):
    # Load the input CSV file
    df = pd.read_csv(input_file_path)

    # Transpose the dataframe. Reset index to keep the column headers as a row in the transposed DataFrame
    df_transposed = df.T.reset_index()

    # Rename columns to reflect original row numbers or customize as needed
    df_transposed.columns = ['OriginalColumnNames'] + list(range(1, len(df_transposed.columns)))

    # Save the transposed dataframe to a new CSV file, with headers
    df_transposed.to_csv(output_file_path, header=True, index=False)

# Example usage
input_file_path = '/content/sbs_96_matrix.csv'  # Update this to your input file path
output_file_path = '/content/sbs_96_/output_transposed.csv'  # Update this to your desired output file path

transpose_csv_with_headers(input_file_path, output_file_path)

print(f"Transposed CSV, including original column names, has been saved to: {output_file_path}")

pip install SigProfilerExtractor

from SigProfilerExtractor import sigpro as sig

sig.sigProfilerExtractor("matrix", "results", "/content/output_transposed.txt", reference_genome="GRCh37", minimum_signatures=1, maximum_signatures=10, nmf_replicates=100, cpu=-1)