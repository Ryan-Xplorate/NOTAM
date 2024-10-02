import streamlit as st
import re
import math
from math import radians, sin, cos, atan2, sqrt, degrees, asin, floor
from xml.etree import ElementTree as ET
import base64
import csv
from geopy.distance import geodesic
from pyproj import Geod

# =========================
# Helper Functions
# =========================

def dms_to_dd(degrees, minutes, seconds):
    """Convert Degrees, Minutes, Seconds to Decimal Degrees."""
    return degrees + (minutes / 60.0) + (seconds / 3600.0)

def dd_to_dms(dd, is_lat=True):
    """Convert Decimal Degrees to Degrees, Minutes, Seconds."""
    direction = ''
    if is_lat:
        direction = 'N' if dd >= 0 else 'S'
    else:
        direction = 'E' if dd >= 0 else 'W'
    dd = abs(dd)
    degrees_val = int(dd)
    minutes_val = int((dd - degrees_val) * 60)
    seconds_val = round((dd - degrees_val - minutes_val / 60) * 3600)
    return f"{degrees_val:02d}{minutes_val:02d}{seconds_val:02d}{direction}"

def parse_coordinates(coord_str):
    """Parse a coordinate string in DMS format and return decimal degrees."""
    pattern = r'(\d+)(\d{2})(\d{2})([NS])\s+(\d+)(\d{2})(\d{2})([EW])'
    match = re.search(pattern, coord_str)
    if match:
        lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = match.groups()
        lat = dms_to_dd(int(lat_d), int(lat_m), int(lat_s))
        lon = dms_to_dd(int(lon_d), int(lon_m), int(lon_s))
        if lat_dir == 'S':
            lat = -lat
        if lon_dir == 'W':
            lon = -lon
        return lat, lon
    return None

def create_buffer(lat1, lon1, lat2, lon2, width_nm):
    """Create buffer coordinates around a line defined by two points."""
    R = 6371  # Earth's radius in km
    width_km = width_nm * 1.852  # Convert nautical miles to km

    # Convert coordinates to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])

    # Calculate bearing
    bearing = atan2(sin(lon2_rad - lon1_rad) * cos(lat2_rad), 
                    cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(lon2_rad - lon1_rad))

    # Calculate perpendicular bearings
    bearing_left = bearing - math.pi / 2
    bearing_right = bearing + math.pi / 2

    # Calculate buffer points
    def get_buffer_point(lat, lon, bearing):
        lat_new = asin(sin(lat) * cos(width_km / R) + cos(lat) * sin(width_km / R) * cos(bearing))
        lon_new = lon + atan2(sin(bearing) * sin(width_km / R) * cos(lat),
                              cos(width_km / R) - sin(lat) * sin(lat_new))
        return degrees(lat_new), degrees(lon_new)

    left1_lat, left1_lon = get_buffer_point(lat1_rad, lon1_rad, bearing_left)
    right1_lat, right1_lon = get_buffer_point(lat1_rad, lon1_rad, bearing_right)
    left2_lat, left2_lon = get_buffer_point(lat2_rad, lon2_rad, bearing_left)
    right2_lat, right2_lon = get_buffer_point(lat2_rad, lon2_rad, bearing_right)

    return [(left1_lat, left1_lon), (left2_lat, left2_lon), 
            (right2_lat, right2_lon), (right1_lat, right1_lon)]

def create_kml(coords, buffer_coords, width_nm):
    """Create a KML string with a line and its buffer zone."""
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>NOTAM Line and Buffer</name>
    <description>Line with {width_nm}NM buffer on each side</description>
    <Style id="lineStyle">
      <LineStyle>
        <color>ff0000ff</color>
        <width>2</width>
      </LineStyle>
    </Style>
    <Style id="polyStyle">
      <LineStyle>
        <color>7f0000ff</color>
        <width>2</width>
      </LineStyle>
      <PolyStyle>
        <color>3f0000ff</color>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>NOTAM Line</name>
      <styleUrl>#lineStyle</styleUrl>
      <LineString>
        <coordinates>
          {coords[1]},{coords[0]},0
          {coords[3]},{coords[2]},0
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>Buffer Zone</name>
      <styleUrl>#polyStyle</styleUrl>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {buffer_coords[0][1]},{buffer_coords[0][0]},0
              {buffer_coords[1][1]},{buffer_coords[1][0]},0
              {buffer_coords[2][1]},{buffer_coords[2][0]},0
              {buffer_coords[3][1]},{buffer_coords[3][0]},0
              {buffer_coords[0][1]},{buffer_coords[0][0]},0
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>'''
    return kml

def process_notam(notam_text):
    """Process NOTAM text to extract coordinates and generate KML."""
    coord_pattern = r'PSN\s+(\d{6}[NS]\s+\d{7}[EW])'

    # Find all matches
    coords = re.findall(coord_pattern, notam_text)
    st.write(f"Extracted Coordinates: {coords}")  # Debug statement

    if len(coords) < 2:
        return "Error: Could not find two coordinate pairs in the NOTAM text."

    # Parse the first two coordinate pairs
    start_coord = parse_coordinates(coords[0])
    end_coord = parse_coordinates(coords[1])
    st.write(f"Parsed Start Coord: {start_coord}, End Coord: {end_coord}")  # Debug

    if not start_coord or not end_coord:
        return "Error: Could not parse coordinates."

    # Extract buffer width
    width_pattern = r'(\d+)\s*NM\s+EITHER\s+SIDE'
    width_match = re.search(width_pattern, notam_text)
    st.write(f"Buffer Width Match: {width_match}")  # Debug

    if width_match:
        width_nm = int(width_match.group(1))
        st.write(f"Buffer Width (NM): {width_nm}")  # Debug
    else:
        return "Error: Could not find buffer width in the NOTAM text."

    # Create buffer coordinates
    buffer_coords = create_buffer(start_coord[0], start_coord[1], end_coord[0], end_coord[1], width_nm)
    st.write(f"Buffer Coordinates: {buffer_coords}")  # Debug

    # Create KML
    kml_content = create_kml(start_coord + end_coord, buffer_coords, width_nm)

    return kml_content

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate HTML for downloading a binary file."""
    bin_str = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">Download {file_label}</a>'
    return href

def extract_kml_coordinates(kml_content):
    """Extract coordinates from an uploaded KML file."""
    try:
        root = ET.fromstring(kml_content)
    except ET.ParseError:
        st.error("Error parsing KML file. Please ensure it's a valid KML file.")
        return []

    namespace = {"ns": "http://www.opengis.net/kml/2.2"}
    coords_list = []

    # Find all <coordinates> tags in the KML
    for placemark in root.findall(".//ns:Placemark", namespaces=namespace):
        for coords in placemark.findall(".//ns:coordinates", namespaces=namespace):
            # Extract and split coordinates
            raw_coords = coords.text.strip().split()
            for coord in raw_coords:
                try:
                    lon, lat, _ = coord.split(",")  # Extract longitude, latitude
                    coords_list.append((float(lat), float(lon)))
                except ValueError:
                    st.warning(f"Invalid coordinate format: {coord}")
    st.write(f"Extracted Coordinates from KML: {coords_list}")  # Debug
    return coords_list

# =========================
# Find closest aerodrome
# =========================

def load_aerodromes(file_path):
    """Load aerodromes from a CSV file."""
    aerodromes = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            aerodromes.append({
                'name': row['name'],
                'code': row['code'],
                'lat': float(row['lat']),
                'lon': float(row['lon'])
            })
    return aerodromes

def find_closest_aerodrome(lat, lon, aerodromes):
    """Find the closest aerodrome to the given coordinates."""
    closest = min(aerodromes, key=lambda x: geodesic((lat, lon), (x['lat'], x['lon'])).nautical)
    distance = geodesic((lat, lon), (closest['lat'], closest['lon'])).nautical
    return closest, distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the initial bearing from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dL = lon2 - lon1
    X = cos(lat2) * sin(dL)
    Y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)
    initial_bearing = atan2(X, Y)
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def magnetic_bearing(true_bearing, magnetic_variation):
    """Convert true bearing to magnetic bearing."""
    return (true_bearing - magnetic_variation + 360) % 360

# ... (existing code) ...

def format_coordinates_dd_to_dms(coords):
    """Format decimal degrees coordinates to DMS (Degrees, Minutes, Seconds) string."""
    formatted_coords = []
    for lat, lon in coords[:2]:  # Assuming only two coordinates are needed
        psn = f"{dd_to_dms(lat, is_lat=True)} {dd_to_dms(lon, is_lat=False)}"
        formatted_coords.append(psn)
    return formatted_coords

# Add this new function
def get_aerodrome_info(kml_coords, aerodromes):
    """Get aerodrome information for the start and end points of the KML file."""
    start_point = kml_coords[0]
    end_point = kml_coords[-1]
    
    start_aerodrome, start_distance = find_closest_aerodrome(start_point[0], start_point[1], aerodromes)
    end_aerodrome, end_distance = find_closest_aerodrome(end_point[0], end_point[1], aerodromes)
    
    # Calculate bearings FROM the aerodromes TO the points
    start_bearing = calculate_bearing(start_aerodrome['lat'], start_aerodrome['lon'], start_point[0], start_point[1])
    end_bearing = calculate_bearing(end_aerodrome['lat'], end_aerodrome['lon'], end_point[0], end_point[1])
    
    # Convert to magnetic bearing (assuming 10 degrees east variation)
    magnetic_variation = 10
    start_mag_bearing = magnetic_bearing(start_bearing, magnetic_variation)
    end_mag_bearing = magnetic_bearing(end_bearing, magnetic_variation)
    
    return {
        'start': {
            'name': start_aerodrome['name'],
            'code': start_aerodrome['code'],
            'bearing': round(start_mag_bearing),
            'distance': round(start_distance, 1)
        },
        'end': {
            'name': end_aerodrome['name'],
            'code': end_aerodrome['code'],
            'bearing': round(end_mag_bearing),
            'distance': round(end_distance, 1)
        }
    }

def create_notam_template(distance1, psn1, br1, mag1, name1, ad1,
                         psn2, br2, mag2, name2, ad2, freq1, freq2, telephone):
    """Create a formatted NOTAM text based on the provided template."""
    template = f"""WI {distance1}NM EITHER SIDE OF A LINE BTN  
PSN {psn1} BRG {br1} MAG {mag1}NM FM {name1} AD {ad1} AND 
PSN {psn2} BRG {br2} MAG {mag2}NM FM {name2} AD {ad2}
OPR WILL BCST ON CTAF {freq1} AND 
MNT BRISBANE CENTRE FREQ {freq2} 15MIN PRIOR LAUNCH AND 
AT 15MIN INTERVALS WHILST AIRBORNE
OPR CTC TEL: {telephone}
UA EQUIPPED WITH ADS-B IN/OUT"""
    return template


def format_coordinates_dd_to_dms(coords):
    """Format decimal degrees coordinates to DMS (Degrees, Minutes, Seconds) string."""
    formatted_coords = []
    for lat, lon in coords[:2]:  # Assuming only two coordinates are needed
        psn = f"{dd_to_dms(lat, is_lat=True)} {dd_to_dms(lon, is_lat=False)}"
        formatted_coords.append(psn)
    return formatted_coords


# =========================
# Streamlit App
# =========================

def render_streamlit_app():
    st.title("NOTAM Manager")
    # Load aerodromes
    aerodromes = load_aerodromes('aerodromes.csv')

    # Sidebar for navigation
    menu_option = st.sidebar.radio("Menu", ["Read NOTAM", "Create NOTAM"])

    if menu_option == "Read NOTAM":
        st.header("Read NOTAM - either side of a line")
        st.write("Paste your NOTAM text below to extract coordinates and generate a KML file.")
        notam_text = st.text_area("NOTAM Text", height=200)

        if st.button("Generate KML", use_container_width=True, type="primary"):
            if notam_text:
                result = process_notam(notam_text)
                if isinstance(result, str) and result.startswith("Error"):
                    st.error(result)
                else:
                    st.success("KML file generated successfully!")
                    st.markdown(get_binary_file_downloader_html(result.encode(), 'notam_polygon.kml'), unsafe_allow_html=True)
            else:
                st.warning("Please enter NOTAM text.")

    elif menu_option == "Create NOTAM":
        st.header("Create NOTAM - either side of a line")
        st.write("Upload a KML file to extract coordinates and fill in the NOTAM template.")

        # Upload KML file
        uploaded_file = st.file_uploader("Upload KML file", type=["kml"])
        kml_coords = []

        if uploaded_file:
            try:
                kml_content = uploaded_file.read().decode("utf-8")
                kml_coords = extract_kml_coordinates(kml_content)
                
                if len(kml_coords) < 2:
                    st.error("The uploaded KML file does not contain at least two valid coordinate points.")
                else:
                    # Sort the coordinates by latitude, northernmost first
                    kml_coords.sort(key=lambda x: x[0], reverse=True)
                    st.success("Coordinates extracted successfully!")
                    
                    # Get aerodrome information
                    aerodrome_info = get_aerodrome_info(kml_coords, aerodromes)
            except Exception as e:
                st.error(f"An error occurred while processing the KML file: {e}")

        # Input fields for NOTAM details
        st.subheader("Enter NOTAM Details")

        distance1 = st.text_input("Distance either side of the line (NM)", value="2")

        # Layout: Use columns to organize input fields
        st.write("### Northern Point")
        col1, col2 = st.columns(2)

        with col1:
            name1 = st.text_input("Name of first aviation facility", value=aerodrome_info['start']['name'] if 'aerodrome_info' in locals() else "")
            ad1 = st.text_input("First aviation facility code", value=aerodrome_info['start']['code'] if 'aerodrome_info' in locals() else "")
        with col2:
            br1 = st.text_input("Bearing from first aviation facility (MAG)", value=str(aerodrome_info['start']['bearing']) if 'aerodrome_info' in locals() else "")
            mag1 = st.text_input("Distance from first aviation facility (NM)", value=str(aerodrome_info['start']['distance']) if 'aerodrome_info' in locals() else "")

        st.write("### Southern Point")
        col1, col2 = st.columns(2)
        with col1:
            name2 = st.text_input("Name of second aviation facility", value=aerodrome_info['end']['name'] if 'aerodrome_info' in locals() else "")
            ad2 = st.text_input("Second aviation facility code", value=aerodrome_info['end']['code'] if 'aerodrome_info' in locals() else "")
        with col2:
            br2 = st.text_input("Bearing from second aviation facility (MAG)", value=str(aerodrome_info['end']['bearing']) if 'aerodrome_info' in locals() else "")
            mag2 = st.text_input("Distance from second aviation facility (NM)", value=str(aerodrome_info['end']['distance']) if 'aerodrome_info' in locals() else "")

        st.write("### Communications")
        col1, col2, col3 = st.columns(3)
        with col1:
            freq1 = st.text_input("CTAF Frequency", value="126.7")
        with col2:
            freq2 = st.text_input("Centre Frequency", value="119.55")
        with col3:
            telephone = st.text_input("Telephone", value="xxxx xxx xxx")

        if st.button("Create NOTAM", use_container_width=True, type="primary"):
            if not kml_coords:
                st.error("Please upload a valid KML file with at least two coordinate points.")
            else:
                # Format coordinates to DMS and ensure northernmost is first
                formatted_psn = format_coordinates_dd_to_dms(kml_coords)
                psn1, psn2 = formatted_psn

                # Create NOTAM text with northern point first
                notam_text_generated = create_notam_template(
                    distance1, psn1, br1, mag1, name1, ad1,
                    psn2, br2, mag2, name2, ad2,
                    freq1, freq2, telephone
                )

                # Display generated NOTAM
                st.subheader("Generated NOTAM:")
                st.text_area("NOTAM Text", notam_text_generated, height=300)

                # Allow downloading the NOTAM text
                notam_bytes = notam_text_generated.encode()
                st.markdown(get_binary_file_downloader_html(notam_bytes, 'generated_notam.txt'), unsafe_allow_html=True)

if __name__ == "__main__":
    render_streamlit_app()
