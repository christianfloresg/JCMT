import pandas as pd
import re


def normalize_ra_dec(value):
    """
    Normalize RA or DEC by:
    - Handling mixed delimiters (colon, space).
    - Ensuring the format is consistent: HH:MM:SS or DD:MM:SS.

    Args:
        value (str): Input RA or DEC value.
    Returns:
        str: Normalized value in the format HH:MM:SS or DD:MM:SS.
    """
    if pd.isnull(value):  # Handle missing values
        return "00:00:00"

    value = str(value).replace(" ", ":").strip()  # Replace spaces with colons and remove extra spaces
    parts = re.split(r"[:]", value)
    if len(parts) == 2:  # If missing seconds
        parts.append("00")
    elif len(parts) == 1:  # If only hours or degrees provided
        parts.extend(["00", "00"])
    return ":".join(parts[:3])  # Return only HH:MM:SS or DD:MM:SS


def ra_to_seconds(ra):
    """Convert RA (HH:MM:SS) to seconds for sorting."""
    hh, mm, ss = map(float, ra.split(":"))
    return hh * 3600 + mm * 60 + ss


def format_ra_dec(value):
    """
    Format RA or DEC values in HH:MM:SS or DD:MM:SS format.

    Args:
        value (str): RA or DEC value in normalized format.
    Returns:
        str: Formatted RA or DEC value in HH:MM:SS format.
    """
    return value  # Since normalize_ra_dec already formats them with colons


def create_latex_table(file_path, output_file):
    """
    Creates a LaTeX deluxetable* from an Excel or Google Sheets file, sorted by RA.

    Args:
        file_path (str): Path to the Excel or CSV file.
        output_file (str): Path to save the generated LaTeX file.
    """
    # Read the Excel or CSV file into a pandas DataFrame
    df = pd.read_excel(file_path)  # Use read_csv() for CSV files

    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # Ensure the column names match the table structure
    expected_columns = [
        "Source Name", "RA", "DEC", "Cloud", "alpha", "Class", "HCO+ Obs.", "C18O Obs."
    ]

    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the file. Please check the column names.")

    # Normalize RA and DEC
    df['RA'] = df['RA'].apply(normalize_ra_dec)
    df['DEC'] = df['DEC'].apply(normalize_ra_dec)

    # Sort by RA
    df['RA_seconds'] = df['RA'].apply(ra_to_seconds)
    df = df.sort_values(by='RA_seconds').drop(columns=['RA_seconds'])  # Drop the temporary sorting column

    # Format RA and DEC for LaTeX
    df['RA'] = df['RA'].apply(format_ra_dec)
    df['DEC'] = df['DEC'].apply(format_ra_dec)

    # Begin LaTeX table
    latex_table = r"""
\begin{deluxetable*}{lccccccc}
\tablecaption{Summary of protostellar properties \label{tab:observation_info}}
\tablecolumns{8}
%\tablenum{1}
%\tablewidth{30pt}
\tabletypesize{\footnotesize}
\tablehead{
\colhead{Source Name} & \colhead{RA} & \colhead{DEC} & \colhead{Cloud} & \colhead{$\alpha$} & \colhead{Class} & \colhead{HCO+ Obs.} & \colhead{C18O Obs.}
}
%\colnumbers
\startdata
"""

    # Add rows from the DataFrame
    for _, row in df.iterrows():
        latex_table += f"    {row['Source Name']} & {row['RA']} & {row['DEC']} & {row['Cloud']} & {row['alpha']} & {row['Class']} & {row['HCO+ Obs.']} & {row['C18O Obs.']} \\\\ \n"

    # End LaTeX table
    latex_table += r"""
\enddata
\tablecomments{and Serpens from Dzib et al. (2011).}
\end{deluxetable*}
"""

    # Write the LaTeX table to the output file
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_file}.")


# Example Usage
file_path = "protostellar_data.xlsx"  # Replace with the path to your Excel or CSV file
output_file = "protostellar_table.tex"  # Replace with the desired LaTeX output file name
create_latex_table(file_path, output_file)
