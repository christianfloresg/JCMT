import pandas as pd
import numpy as np
import matplotlib as mpl
import os
# df = pd.DataFrame({
#     "strings": ["Adam", "Mike"],
#     "ints": [1, 3],
#     "floats": [1.123, 1000.23]
# })

def read_file(folder,filename):
    # Read the file with three columns: strings, float, and float
    data = np.genfromtxt(os.path.join(folder,filename), dtype=[('col1', 'U20'), ('col2', 'f8'), ('col3', 'f8')],
                         delimiter=None, encoding='utf-8', comments='#')

    # Accessing data
    star_name = data['col1']  # First column (strings)
    one_sigma_c_factor = data['col2']  # Second column (numerical)
    three_sigma_c_factor = data['col3']  # Third column (numerical)


    return star_name,one_sigma_c_factor,three_sigma_c_factor


def merge_tables():

    star_name,one_sigma_c_factor,three_sigma_c_factor = read_file(folder='text_files',filename="concentrations_HCO+.txt")
    star_name_C18O,one_sigma_c_factor_C18O,three_sigma_c_factor_C18O = \
        read_file(folder='text_files',filename="concentrations_C18O.txt")

    df = pd.DataFrame({
        "Names": star_name,
        r"1sigma_HCO+": one_sigma_c_factor,
        "3sigma_HCO+": three_sigma_c_factor
    })

    df_C18O = pd.DataFrame({
        "Names": star_name_C18O,
        "1sigma_C18O": one_sigma_c_factor_C18O,
        "3sigma_C18O": three_sigma_c_factor_C18O
    })


    result = pd.merge(df, df_C18O, on="Names",how="outer")

    return result

def sorted_table():

    merged_table = merge_tables()
    sorted_df = merged_table.sort_values(by="1sigma_HCO+",ascending=False)

    sorted_df.to_html("molecular_tracers2.html", index=False)
    print("Table saved to molecular_tracers.html")

if __name__ == "__main__":
    sorted_table()
