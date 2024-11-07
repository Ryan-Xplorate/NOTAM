import streamlit as st
import re
import math
from math import radians, sin, cos, atan2, sqrt, degrees, asin, floor
from xml.etree import ElementTree as ET
import base64
import csv
from geopy.distance import geodesic
from pyproj import Geod
import textwrap
import pandas as pd
import numpy as np
from footer import footer

st.set_page_config(
    page_title="SDO 50 Performance and Weather", 
    page_icon="🚁", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

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
    st.write(f"Extracted Coordinates: {coords}")

    if len(coords) < 2:
        return "Error: Could not find two coordinate pairs in the NOTAM text."

    # Parse the first two coordinate pairs
    start_coord = parse_coordinates(coords[0])
    end_coord = parse_coordinates(coords[1])
    st.write(f"Parsed Start Coord: {start_coord}, End Coord: {end_coord}")

    if not start_coord or not end_coord:
        return "Error: Could not parse coordinates."

    # Extract buffer width
    width_pattern = r'(\d*\.?\d+)\s*NM\s+EITHER\s+SIDE'
    width_match = re.search(width_pattern, notam_text)
    st.write(f"Buffer Width Match: {width_match}")

    if width_match:
        width_nm = float(width_match.group(1))  # Changed from int() to float()
        st.write(f"Buffer Width (NM): {width_nm}")
    else:
        return "Error: Could not find buffer width in the NOTAM text."

    # Create buffer coordinates
    buffer_coords = create_buffer(start_coord[0], start_coord[1], end_coord[0], end_coord[1], width_nm)
    st.write(f"Buffer Coordinates: {buffer_coords}")

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

def format_coordinates_dd_to_dms(coords):
    """Format decimal degrees coordinates to DMS (Degrees, Minutes, Seconds) string."""
    formatted_coords = []
    for lat, lon in coords[:2]:  # Assuming only two coordinates are needed
        psn = f"{dd_to_dms(lat, is_lat=True)} {dd_to_dms(lon, is_lat=False)}"
        formatted_coords.append(psn)
    return formatted_coords

def get_aerodrome_info(kml_coords, aerodromes):
    """Get aerodrome information for the start and end points of the KML file."""
    start_point = kml_coords[0]
    end_point = kml_coords[-1]
    
    start_aerodrome, start_distance = find_closest_aerodrome(start_point[0], start_point[1], aerodromes)
    end_aerodrome, end_distance = find_closest_aerodrome(end_point[0], end_point[1], aerodromes)
    
    # Calculate bearings FROM the aerodromes TO the points
    start_bearing = calculate_bearing(start_aerodrome['lat'], start_aerodrome['lon'], start_point[0], start_point[1])
    end_bearing = calculate_bearing(end_aerodrome['lat'], end_aerodrome['lon'], end_point[0], end_point[1])
    
    # Convert to magnetic bearing (assuming 10 degrees east variation - Queensland/Chinchilla region only!)
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

def format_number(num):
    """Format number to remove .0 if whole number"""
    return str(int(num)) if num.is_integer() else str(num)

def create_summary_template(br1, br2, dist1, dist2, ad1, ad2):
    """Create summary with properly formatted distances and aerodrome handling"""
    dist1_fmt = format_number(float(dist1))
    dist2_fmt = format_number(float(dist2))
    
    # If both aerodromes are the same, only show one
    ad_part = f"FM {ad1}" if ad1 == ad2 else f"FM {ad1} / {ad2}"
    
    summary = f"UA OPS BTN BRG {br1}-{br2}MAG {dist1_fmt}NM-{dist2_fmt}NM {ad_part}"
    return summary

def create_notam_template(distance1, psn1, br1, mag1, name1, ad1,
                          psn2, br2, mag2, name2, ad2, freq1, freq2, telephone):  
    # Validate input fields
    if any(not field.replace(" ", "").replace(".", "").isdigit() for field in [freq1, freq2]) or not telephone.replace(" ", "").replace(".", "").isdigit():
        st.error("One or more fields contain invalid characters. Have you filled out communications properly?")
        return None
    
    # Create the NOTAM template
    template = textwrap.dedent(f"""
    WI {distance1} NM EITHER SIDE OF A LINE BTN  
    PSN {psn1} BRG {br1} MAG {mag1} NM FM {name1.upper()} AD ({ad1}) AND 
    PSN {psn2} BRG {br2} MAG {mag2} NM FM {name2.upper()} AD ({ad2})
    OPR WILL BCST ON CTAF {freq1} AND 
    MNT BRISBANE CENTRE FREQ {freq2} 15 MIN PRIOR LAUNCH AND 
    AT 15 MIN INTERVALS WHILST AIRBORNE
    OPR CTC TEL: {telephone}
    UA EQUIPPED WITH ADS-B IN/OUT
    """).strip()

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
   
    # Load aerodromes
    aerodromes = load_aerodromes('aerodromes.csv')

    # Sidebar for navigation
    st.sidebar.image("static/logo.png", use_column_width=True)
    menu_option = st.sidebar.radio("Menu", ["Read NOTAM", "Create NOTAM", "Check Aircraft Performance"])


    if menu_option == "Read NOTAM":
        st.title("NOTAM Manager 🚁")
        st.header("Read NOTAM - either side of a line")
        st.write("Paste your NOTAM text below to extract coordinates and generate a KML file.")
        notam_text = st.text_area("NOTAM Text", height=200)

        st.divider()

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
        st.title("NOTAM Manager 🚁")
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

        st.divider()

        distance1 = st.text_input("Distance either side of the line (NM)", value="2")

        st.divider()

        st.write("### Northern Point - <span style='color: yellow; font-size: 0.6em;'>automatically generated from KML file</span>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            name1 = st.text_input("Name of first aviation facility", value=aerodrome_info['start']['name'] if 'aerodrome_info' in locals() else "")
            ad1 = st.text_input("First aviation facility code", value=aerodrome_info['start']['code'] if 'aerodrome_info' in locals() else "")
        with col2:
            br1 = st.text_input("Bearing from first aviation facility (MAG)", value=str(aerodrome_info['start']['bearing']) if 'aerodrome_info' in locals() else "")
            mag1 = st.text_input("Distance from first aviation facility (NM)", value=str(aerodrome_info['start']['distance']) if 'aerodrome_info' in locals() else "")
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        st.write("### Southern Point - <span style='color: yellow; font-size: 0.6em;'>automatically generated from KML file</span>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            name2 = st.text_input("Name of second aviation facility", value=aerodrome_info['end']['name'] if 'aerodrome_info' in locals() else "")
            ad2 = st.text_input("Second aviation facility code", value=aerodrome_info['end']['code'] if 'aerodrome_info' in locals() else "")
        with col2:
            br2 = st.text_input("Bearing from second aviation facility (MAG)", value=str(aerodrome_info['end']['bearing']) if 'aerodrome_info' in locals() else "")
            mag2 = st.text_input("Distance from second aviation facility (NM)", value=str(aerodrome_info['end']['distance']) if 'aerodrome_info' in locals() else "")
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        st.write("### Communications")
        col1, col2, col3 = st.columns(3)
        with col1:
            freq1 = st.text_input("CTAF Frequency", value="xxx.xx")
        with col2:
            freq2 = st.text_input("Centre Frequency", value="xxx.xx")
        with col3:
            telephone = st.text_input("Telephone", value="xxxx xxx xxx")

        st.divider()

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

                summary_text_generated = create_summary_template(br1, br2, mag1, mag2, ad1, ad2)

                # Ensure that notam_text_generated is not None
                if notam_text_generated is not None:
                    st.subheader("Generated NOTAM:")
                    st.text_area("Summary Text", summary_text_generated, height=50)
                    st.text_area("NOTAM Text", notam_text_generated, height=300)

                    # Allow downloading the NOTAM text
                    notam_bytes = notam_text_generated.encode()
                    st.markdown(get_binary_file_downloader_html(notam_bytes, 'generated_notam.txt'), unsafe_allow_html=True)
        
    elif menu_option == "Check Aircraft Performance":
        # Load performance data at the start
        df = pd.read_csv('performance.csv', index_col='TOW (kg)')
        
        st.title("Performance and Atmospheric Analysis 🚁")
        
        # Input Collection Section
        temperature = st.number_input("Enter temperature (°C)", value=25)
        pressure = st.number_input("Enter pressure (hPa)", value=1013)
        dew_point = st.number_input("Enter dew point (°C)", value=10)
    
        use_feet = st.toggle("Use feet instead of meters for the deployment elevation", value=False)
        if use_feet:
            elevation_ft = st.number_input("Deployment elevation (ft)", value=1000)
        else:
            elevation_m = st.number_input("Deployment elevation (m)", value=300)
            elevation_ft = elevation_m * 3.28084  # Precise conversion
        
        takeoff_weight = st.number_input("Enter takeoff weight (kg)", value=80, min_value=46, max_value=87)
        if takeoff_weight == 87:
            st.warning("⚠️ Warning: You are at MTOW")
        
        def calculate_pressure_altitude(elevation_ft, pressure_hpa):
            """Calculate pressure altitude in feet"""
            standard_pressure = 1013.25  # hPa
            pressure_correction = (standard_pressure - pressure_hpa) * 30
            return elevation_ft + pressure_correction

        def calculate_density_altitude(pressure_altitude_ft, temperature_c, dew_point_c, pressure_hpa):
            """Calculate density altitude in feet"""
            standard_temp = 15.0  # °C
            temp_lapse_rate = -0.001981  # °C/ft
            
            # Vapor pressure calculation
            e = 6.11 * 10.0**((7.5 * dew_point_c) / (237.7 + dew_point_c))
            
            # Virtual temperature calculation
            temp_k = temperature_c + 273.15
            virtual_temp_k = temp_k / (1 - (e / pressure_hpa) * (1 - 0.622))
            virtual_temp_c = virtual_temp_k - 273.15
            
            # ISA temperature at pressure altitude
            isa_temp = standard_temp + (temp_lapse_rate * pressure_altitude_ft)
            
            # Final density altitude calculation
            return pressure_altitude_ft + (120 * (virtual_temp_c - isa_temp))
        
        def get_max_perf_for_weight(tow):
            """Get maximum allowable density altitude for a given weight"""
            if tow <= 75:
                return 7498
            elif tow <= 79:
                return 8131
            elif tow <= 83:
                return 8665
            elif tow <= 87:
                return 8952
            return 0  # Above MTOW

        # Calculate atmospheric conditions
        pressure_altitude = calculate_pressure_altitude(elevation_ft, pressure)
        density_altitude = calculate_density_altitude(pressure_altitude, temperature, dew_point, pressure)
        
        # Display atmospheric results
        st.divider()
        st.subheader("Atmospheric Conditions")
        cols = st.columns(2)
        with cols[0]:
            pa_difference = pressure_altitude - elevation_ft
            st.metric(
                "Pressure Altitude", 
                f"{pressure_altitude:.0f} ft / {(pressure_altitude * 0.3048):.0f} m", 
                f"{pa_difference:+.0f} ft from deployment elevation",
                delta_color="inverse"
            )
        with cols[1]:
            da_difference = density_altitude - elevation_ft
            st.metric(
                "Density Altitude", 
                f"{density_altitude:.0f} ft / {(density_altitude * 0.3048):.0f} m", 
                f"{da_difference:+.0f} ft from deployment elevation",
                delta_color="inverse"
            )
        
        def check_performance_limits(tow, da):
            """Check if current conditions exceed performance limits"""
            if tow <= 75:
                if da * 0.3048 <= 2400:
                    return 'ok'
                return 'caution'
            elif tow <= 79:
                if da * 0.3048  <= 2000:
                    return 'ok'
                elif da * 0.3048  <= 2400:
                    return 'caution'
                return 'prohibited'
            elif tow <= 83:
                if da * 0.3048  <= 1600:
                    return 'ok'
                elif da * 0.3048  <= 2200:
                    return 'caution'
                return 'prohibited'
            elif tow <= 87:
                if da * 0.3048  < 0:
                    return 'ok'
                elif da * 0.3048  <= 1600:
                    return 'caution'
                return 'prohibited'
            return 'prohibited'  # Above MTOW

        def get_interpolated_limit(df, tow, da):
            """Get interpolated performance limit for exact TOW and DA"""
            # Find the closest TOW rows
            tow_values = pd.to_numeric(df.index)
            lower_tow_idx = np.where(tow_values <= tow)[0][-1] if any(tow_values <= tow) else 0
            upper_tow_idx = np.where(tow_values >= tow)[0][0] if any(tow_values >= tow) else len(tow_values)-1
            
            # Find the closest DA columns
            da_values = pd.to_numeric(df.columns)
            lower_da_idx = np.where(da_values <= da)[0][-1] if any(da_values <= da) else 0
            upper_da_idx = np.where(da_values >= da)[0][0] if any(da_values >= da) else len(da_values)-1
            
            # Get the four corner values
            q11 = float(df.iloc[lower_tow_idx, lower_da_idx])
            q12 = float(df.iloc[lower_tow_idx, upper_da_idx])
            q21 = float(df.iloc[upper_tow_idx, lower_da_idx])
            q22 = float(df.iloc[upper_tow_idx, upper_da_idx])
            
            # Handle edge cases where we're at or beyond the table limits
            if lower_tow_idx == upper_tow_idx and lower_da_idx == upper_da_idx:
                return q11
            
            # Perform bilinear interpolation
            x = (da - da_values[lower_da_idx]) / (da_values[upper_da_idx] - da_values[lower_da_idx]) if lower_da_idx != upper_da_idx else 0
            y = (tow - tow_values[lower_tow_idx]) / (tow_values[upper_tow_idx] - tow_values[lower_tow_idx]) if lower_tow_idx != upper_tow_idx else 0
            
            interpolated_value = (1-x)*(1-y)*q11 + x*(1-y)*q12 + (1-x)*y*q21 + x*y*q22
            return interpolated_value

        def style_dataframe(df):
            """Apply styling to the dataframe with highlighted current conditions"""
            def highlight_performance(row):
                """Color code cells and highlight current conditions"""
                tow = float(row.name)
                styles = pd.Series('', index=row.index)
                values = row.astype(float)
                
                # Apply coloring based on TOW ranges
                if tow <= 75:
                    styles[values <= 7498] = 'background-color: #00ff0050'
                    styles[values > 7498] = 'background-color: #ffa50050'
                    
                elif tow <= 79:
                    styles[values <= 7916] = 'background-color: #00ff0050'
                    styles[(values > 7916) & (values <= 8131)] = 'background-color: #ffa50050'
                    styles[values > 8131] = 'background-color: #ff000050'
                    
                elif tow <= 83:
                    styles[values <= 8320] = 'background-color: #00ff0050'
                    styles[(values > 8320) & (values <= 8665)] = 'background-color: #ffa50050'
                    styles[values > 8665] = 'background-color: #ff000050'
                    
                elif tow <= 87:
                    styles[values <= 8000] = 'background-color: #00ff0050'
                    styles[(values > 8000) & (values <= 8952)] = 'background-color: #ffa50050'
                    styles[values > 8952] = 'background-color: #ff000050'
                    
                else:  # Above MTOW (> 87)
                    styles[:] = 'background-color: #ff000050'
                
                return styles
            
            # Set current conditions as attributes of the function
            highlight_performance.current_tow = takeoff_weight
            highlight_performance.current_da = density_altitude
            
            # Ensure index is numeric
            df.index = pd.to_numeric(df.index)
            
            # Apply the styling
            styled = df.style\
                .set_caption("Column far left: Takeoff Weight (kg) | Top Row: Density Altitude (m), colored cells, engine performance")\
                .format("{:.0f}")\
                .apply(highlight_performance, axis=1)
            
            return styled

        # Performance Analysis Section
        st.divider()
        st.subheader("Performance Analysis")
        
        # Get status and performance value
        status = check_performance_limits(takeoff_weight, density_altitude)
        performance_value = get_interpolated_limit(df, takeoff_weight, density_altitude)
        
        # Display status with context
        cols = st.columns([2, 3])
        with cols[0]:
            st.metric("Current Weight", f"{takeoff_weight:.1f} kg")
        with cols[1]:
            if status == 'ok':
                st.success("✅ NORMAL: Aircraft can operate safely")
            elif status == 'caution':
                st.warning(f"""⚠️ CAUTION: Degraded Performance
                - Max performance : {get_max_perf_for_weight(takeoff_weight):.0f} W
                - Current DA: {0.3048 * density_altitude:.0f} m""")
            else:  # prohibited
                st.error(f"""🚫 PROHIBITED: Operation Not Permitted
                - Max performance : {get_max_perf_for_weight(takeoff_weight):.0f} W
                - Current DA: {0.3048 * density_altitude:.0f} ft""")
        
        # Display the performance table
        st.divider()
        st.subheader("Performance Table")
        styled_df = style_dataframe(df)
        st.table(styled_df)
        
        # Add legend
        st.markdown("""
        ### Legend
        * 🟩 Normal Operations - Aircraft can operate safely
        * 🟨 Caution - Reduced performance, additional planning required
        * 🟥 Prohibited - Operations not permitted
        
        Note: Table shows the engine performance limits vs for various takeoff weights and density altitudes
        """)

footer()

if __name__ == "__main__":
    render_streamlit_app()