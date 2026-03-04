from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
import matplotlib.pyplot as plt

def find_simbad_source_in_file(file_name, search_word):
    """
    Reads a text file, ignoring lines starting with '#', and searches for a word in the first column.
    If found, returns the rest of the line. If not found, prints a message and exits.

    :param file_name: The name of the text file
    :param search_word: The word to search for in the first column of the file
    :return: The rest of the line if the word is found, otherwise prints a message
    """
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Skip lines starting with '#' or that are empty
                if line.startswith("#") or not line.strip():
                    continue

                # Split the line into first word and the rest, without unnecessary stripping
                parts = line.split(maxsplit=1)

                if parts[0] == search_word:
                    # Return the rest of the line as soon as the word is found
                    print("You are using the coordinates of " + parts[1])
                    return parts[1] if len(parts) > 1 else ''

        # If no match is found
        print("This source is not in the list.")

    except FileNotFoundError:
        print(f"Error: The file {file_name} does not exist.")

    return None


def find_word_in_file(file_name, search_word, position):
    """
    Reads a text file, ignoring lines starting with '#', and searches for a word in the first column.
    If found, returns the word at the requested position in the same line. If not found, prints a message and exits.

    :param file_name: The name of the text file
    :param search_word: The word to search for in the first column of the file
    :param position: The index of the word in the line to return (0-based)
    :return: The word at the specified position if the word is found, otherwise prints a message
    """

    header = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Skip lines starting with '#' or that are empty
                if line.startswith("#") or not line.strip():
                    header.append(line.split())
                    # print(header)
                    continue

                # Split the line into words
                parts = line.split()

                if parts[0] == search_word:
                    # Check if the requested position is within bounds
                    if position < len(parts):
                        print(f"For source {line.split()[0]}, the value of "
                              f"{header[0][position+1]} is {parts[position]} {header[1][position+1]}")
                        return parts[position]
                    else:
                        print(f"Error: The requested position {position} is out of bounds.")
                        return None

        # If no match is found
        print("This source is not in the list.")

    except FileNotFoundError:
        print(f"Error: The file {file_name} does not exist.")

    return None


def closest_idx(lst, val):
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()
    return idx

def get_icrs_coordinates(object_name):
    '''
    get coordinates of astronomical objects by querying to SIMBAD names
    :param object_name:
    :return:
    '''
    # Initialize Simbad object and customize output to include coordinates
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('coordinates')

    # Query Simbad for the object
    result_table = custom_simbad.query_object(object_name)

    # Check if the query returned any results
    if result_table is None:
        print(f"Object '{object_name}' not found in SIMBAD.")
        return None

    # Extract the RA and DEC columns from the result
    ra = result_table['RA'][0]  # Right Ascension in HMS (hours, minutes, seconds)
    dec = result_table['DEC'][0]  # Declination in DMS (degrees, arcminutes, arcseconds)


    # Convert RA and DEC to a SkyCoord object in the ICRS frame
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='icrs')
    # coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='fk5')
    # Return the ICRS coordinates in degrees
    return coord
