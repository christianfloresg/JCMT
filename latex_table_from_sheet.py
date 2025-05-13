import pandas as pd
import re
from datetime import date,datetime
today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)

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
    print(ra)
    return hh * 3600 + mm * 60 + round(ss,2)


def format_ra_dec(value):
    """
    Format RA or DEC values in HH:MM:SS or DD:MM:SS format.

    Args:
        value (str): RA or DEC value in normalized format.
    Returns:
        str: Formatted RA or DEC value in HH:MM:SS format.
    """
    return value  # Since normalize_ra_dec already formats them with colons

def create_latex_table_concentration_factor(file_path, output_file):
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
        "Source Name", "HCO+_concentration", "C18O_concentration", "Integ.Beam.HCO+", "Unce.Beam.HCO+",
        "Integ.Beam.C18O", "Unce.Beam.C18O","Interpretation", "Note"
    ]

    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the file. Please check the column names.")

    # Begin LaTeX table
    latex_table = r"""
\begin{deluxetable*}{lcccccc}
\tablecaption{Concentration factors and envelope interpretation\label{tab:concentration_table}}
\tablecolumns{8}
%\tablenum{1}
%\tablewidth{30pt}
\tabletypesize{\footnotesize}
\tablehead{
\colhead{Source Name} & \colhead{HCO$^+$} & \colhead{C$^{18}$O} & \colhead{Integrated} &
 \colhead{Integrated} & \colhead{Interpretation} & \colhead{Note} \\
\colhead{ } & \colhead{concentration} & \colhead{concentration} & \colhead{intensity HCO$^+$} &
 \colhead{intensity C$^{18}$O} & \colhead{} & \colhead{}\\
\colhead{} & \colhead{ factor } & \colhead{factor} & \colhead{(K km/s)} &
\colhead{(K km/s)} & \colhead{ } & \colhead{ }
}
%\colnumbers
\startdata
"""

    # Add rows from the DataFrame
    for _, row in df.iterrows():
        latex_table += f"    {row['Source Name']} & {row['HCO+_concentration']} & {row['C18O_concentration']} " \
                       f"& {row['Integ.Beam.HCO+']} \pm {row['Unce.Beam.HCO+']} & " \
                       f"{row['Integ.Beam.C18O']} \pm {row['Unce.Beam.C18O']} & {row['Interpretation']} & {row['Note']} \\\\ \n"

    # End LaTeX table
    latex_table += r"""
\enddata
\tablecomments{Columns 2 and 5 are concentration factors. Columns 3 and 4 are the integratined intensity within the central beam.}
\end{deluxetable*}
"""

    # Write the LaTeX table to the output file
    with open(output_file+ '_' + today + '_' + hour_now+'.tex', "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_file+ '_' + today + '_' + hour_now}.")


def create_latex_table_spectral_parameters(file_path, output_file):
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
        "Source Name", "Peak_C18O_central", "C18O_RMS", "C18O_Peak_SNR", "C18O_Velocity", "Peak_HCO+_central", "HCO+_RMS","HCO+_Peak_SNR", "HCO+_Velocity"
    ]

    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the file. Please check the column names.")

    # Begin LaTeX table
    latex_table = r"""
\begin{deluxetable*}{lcccccccc}
\tablecaption{Summary of protostellar properties \label{tab:observation_info}}
\tablecolumns{9}
%\tablenum{1}
%\tablewidth{30pt}
\tabletypesize{\footnotesize}
\tablehead{
\colhead{Source Name} & \colhead{Peak C18O} & \colhead{C18O RMS} & \colhead{C18O peak RMS} & \colhead{C18O velocity} & 
 \colhead{Peak HCO+} & \colhead{HCO+ RMS} & \colhead{HCO+ peak RMS} &\colhead{HCO+ velocity} \\
\colhead{} & \colhead{central (K)} & \colhead{ (K)} & \colhead{} & \colhead{(km/s)} & 
& \colhead{central (K)} & \colhead{ (K)} & \colhead{} & \colhead{(km/s)}
}
%\colnumbers
\startdata
"""

    # Add rows from the DataFrame
    for _, row in df.iterrows():
        latex_table += f"    {row['Source Name']} & {row['Peak_C18O_central']} & {row['C18O_RMS']} & {row['C18O_Peak_SNR']} " \
                       f"& {row['C18O_Velocity']} & {row['Peak_HCO+_central']} & {row['HCO+_RMS']} & {row['HCO+_Peak_SNR']}" \
                       f" & {row['HCO+_Velocity']} \\\\ \n"

    # End LaTeX table
    latex_table += r"""
\enddata
\tablecomments{References for the A$_v$ value are given in the last column. }
\end{deluxetable*}
"""

    # Write the LaTeX table to the output file
    with open(output_file+ '_' + today + '_' + hour_now+'.tex', "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_file+ '_' + today + '_' + hour_now}.")

def create_latex_table_obs_parameters(file_path, output_file):
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
        "Source Name", "RA", "DEC", "Cloud", "alpha", "Class", "Av", "Reference"
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
\colhead{Source Name} & \colhead{RA} & \colhead{DEC} & \colhead{Cloud} & \colhead{$\alpha$} & \colhead{Class} & \colhead{A$_v$} & \colhead{Reference}
}
%\colnumbers
\startdata
"""

    # Add rows from the DataFrame
    for _, row in df.iterrows():
        latex_table += f"    {row['Source Name']} & {row['RA']} & {row['DEC']} & {row['Cloud']} & {row['alpha']} & {row['Class']} & {row['Av']} & {row['Reference']} \\\\ \n"

    # End LaTeX table
    latex_table += r"""
\enddata
\tablecomments{References for the A$_v$ value are given in the last column. }
\end{deluxetable*}
"""

    # Write the LaTeX table to the output file
    with open(output_file+ '_' + today + '_' + hour_now+'.tex', "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_file+ '_' + today + '_' + hour_now}.")



if __name__ == "__main__":

    # Example Usage
    file_path = "sheets_and_latex/May-2025/JCMT_concentration_table_for_paper.xlsx"  # Replace with the path to your Excel or CSV file
    output_file = "sheets_and_latex/May-2025/protostellar_concentration_table"  # Replace with the desired LaTeX output file name
    # create_latex_table_obs_parameters(file_path, output_file)
    # create_latex_table_spectral_parameters(file_path, output_file)
    create_latex_table_concentration_factor(file_path, output_file)