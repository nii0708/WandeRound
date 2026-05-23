import requests
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union
from concurrent.futures import ThreadPoolExecutor
import io
import sys


def get_exec_printed_result(code_string):
    """
    Executes a string of Python code and captures its printed output.

    Args:
        code_string (str): The Python code to execute.

    Returns:
        str: The captured printed output from the executed code.
    """
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code_string)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"Error during exec: {e}", file=sys.stderr)
        return ""
    finally:
        sys.stdout = old_stdout

    return redirected_output.getvalue()


def get_json(query):
    url = "https://overpass-api.de/api/interpreter"
    headers = {"User-Agent": "WandeRound/1.0"}
    response = requests.post(url, data={"data": query}, headers=headers, timeout=45)
    response.raise_for_status()
    return response.json()


def process_overpass(response):
    cleaned_resp = "\n".join(response.splitlines()[1:-1])
    return cleaned_resp


def get_osm_id_from_nominatim(place_name):
    """
    Get OSM ID from Nominatim for a specified place name
    """
    # URL encode the place name
    encoded_place = requests.utils.quote(place_name)

    # Nominatim API URL
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_place}&format=json&limit=1"

    # Add a user agent as per Nominatim usage policy
    headers = {
        "User-Agent": "GeocodingApp/1.0"  # Replace with your actual app name
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()

        if not data:
            return None, "No results found"

        result = data[0]
        lat = float(result.get("lat", 0))
        lon = float(result.get("lon", 0))

        return {"lat": lat, "lon": lon}, None

    except requests.exceptions.RequestException as e:
        return None, f"Error fetching data: {str(e)}"


def reverse_geocode(lat, lon, language="en"):
    """
    Return a short area name (suburb/neighbourhood/city_district/village/...)
    for the given coordinates using Nominatim. Returns None if nothing useful
    is found or the request fails.
    """
    url = (
        "https://nominatim.openstreetmap.org/reverse"
        f"?lat={lat}&lon={lon}&format=json&zoom=14&addressdetails=1"
        f"&accept-language={language}"
    )
    headers = {"User-Agent": "WandeRound/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException:
        return None

    address = data.get("address", {}) if isinstance(data, dict) else {}
    # Prefer narrowest meaningful area first
    for key in (
        "neighbourhood",
        "suburb",
        "quarter",
        "city_district",
        "village",
        "town",
        "hamlet",
        "city",
        "county",
    ):
        value = address.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def process_osm_data(data):
    """
    Process OpenStreetMap data from an Overpass API JSON file and create GeoDataFrames.

    Parameters:
    file_path (str): Path to the JSON file

    Returns:
    tuple: (nodes_gdf, ways_gdf, relations_gdf) - GeoDataFrames for nodes, ways, and relations
    """
    # Load the JSON data
    # with open(file_path, 'r') as f:
    #     data = json.load(f)

    # Step 1: Extract node coordinates into a lookup dictionary
    node_coords = {}
    for element in data["elements"]:
        if element["type"] == "node":
            node_id = element["id"]
            lat = element["lat"]
            lon = element["lon"]
            node_coords[node_id] = (lon, lat)  # GIS typically uses (lon, lat) order

    # Step 2: Process nodes to create a GeoDataFrame
    node_data = []
    for element in data["elements"]:
        # print('element : ',element)
        if (
            element["type"] == "node" and "tags" in element
        ):  # Only include nodes with tags
            node_id = element["id"]
            geometry = Point(node_coords[node_id])

            # Extract tags if they exist
            tags = element.get("tags", {})

            # Create a dictionary with id, geometry, and all tags
            node_dict = {"id": node_id, "geometry": geometry, "type": "node"}
            node_dict.update(tags)  # Add all tags as columns
            node_data.append(node_dict)

    # Create nodes GeoDataFrame
    if node_data:
        nodes_gdf = gpd.GeoDataFrame(node_data, crs="EPSG:4326")
    else:
        nodes_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        # print("No node features found.")

    # Step 3: Process ways and create a lookup dictionary for way geometries
    way_data = []
    way_geometry = {}  # Dictionary to store way geometries for relation processing

    for element in data["elements"]:
        if element["type"] == "way":
            way_id = element["id"]

            # Get coordinates for each node in the way
            way_coords = []
            for node_id in element["nodes"]:
                if node_id in node_coords:
                    way_coords.append(node_coords[node_id])
                else:
                    # print(f"Warning: Node {node_id} for way {way_id} not found in data.")
                    pass
            # If we have enough coordinates to form a valid geometry
            if len(way_coords) >= 2:
                # Check if it's a closed way (first and last points are the same)
                is_closed = len(way_coords) > 2 and way_coords[0] == way_coords[-1]

                # Create appropriate geometry based on whether the way is closed or not
                if is_closed:
                    # It's a closed way, likely representing a polygon (e.g., building)
                    try:
                        geometry = Polygon(way_coords)
                        way_geometry[way_id] = geometry
                    except Exception as e:
                        # print(f"Warning: Could not create Polygon for way {way_id}: {e}")
                        continue
                else:
                    # It's an open way, likely representing a line (e.g., road)
                    try:
                        geometry = LineString(way_coords)
                        way_geometry[way_id] = geometry
                    except Exception as e:
                        # print(f"Warning: Could not create LineString for way {way_id}: {e}")
                        continue

                # Extract tags if they exist
                tags = element.get("tags", {})

                # Only process as independent way if it has tags (otherwise it might just be part of a relation)
                if tags:
                    # Create a dictionary with id, geometry, and all tags
                    way_dict = {"id": way_id, "geometry": geometry, "type": "way"}
                    way_dict.update(tags)  # Add all tags as columns
                    way_data.append(way_dict)

    # Create ways GeoDataFrame
    if way_data:
        ways_gdf = gpd.GeoDataFrame(way_data, crs="EPSG:4326")
    else:
        ways_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        # print("No way features found.")

    # Step 4: Process relations
    relation_data = []
    for element in data["elements"]:
        if element["type"] == "relation":
            relation_id = element["id"]
            relation_type = element.get("tags", {}).get("type", "unknown")

            # Process multipolygon relations
            if relation_type == "multipolygon":
                outer_ways = []
                inner_ways = []

                # Sort members into outer and inner ways
                for member in element["members"]:
                    if member["type"] == "way":
                        way_id = member["ref"]
                        role = member["role"]

                        if way_id in way_geometry:
                            if role == "outer":
                                outer_ways.append(way_geometry[way_id])
                            elif role == "inner":
                                inner_ways.append(way_geometry[way_id])
                        else:
                            # print(f"Warning: Way {way_id} for relation {relation_id} not found in data.")
                            continue
                # Try to create a valid multipolygon
                try:
                    # For outer rings that may not form closed polygons
                    if any(isinstance(geom, LineString) for geom in outer_ways):
                        # Try to form polygons from the linestrings
                        polygons = list(polygonize(unary_union(outer_ways)))
                        if polygons:
                            outer_polygon = unary_union(polygons)
                        else:
                            # print(f"Warning: Could not create outer polygon for relation {relation_id}")
                            continue
                    else:
                        # We already have polygons
                        outer_polygon = unary_union(outer_ways)

                    # Process inner rings (holes)
                    if inner_ways:
                        inner_polygon = unary_union(inner_ways)

                        # Create multipolygon with holes
                        if isinstance(outer_polygon, Polygon):
                            # Single polygon with holes
                            if isinstance(inner_polygon, Polygon):
                                geometry = Polygon(
                                    outer_polygon.exterior, [inner_polygon.exterior]
                                )
                            else:  # Multiple holes
                                holes = [
                                    hole.exterior
                                    for hole in inner_polygon.geoms
                                    if isinstance(hole, Polygon)
                                ]
                                geometry = Polygon(outer_polygon.exterior, holes)
                        else:  # Multiple outer polygons
                            # This is more complex - would need to match holes to correct outer polygons
                            # For now, just create a multipolygon from the outer rings
                            geometry = outer_polygon
                            # print(f"Warning: Complex multipolygon for relation {relation_id}, ignoring holes")
                    else:
                        # No inner rings, just use the outer polygon
                        geometry = outer_polygon

                    # Extract tags
                    tags = element.get("tags", {})

                    # Create a dictionary with id, geometry, and all tags
                    relation_dict = {
                        "id": relation_id,
                        "geometry": geometry,
                        "type": "relation",
                        "relation_type": relation_type,
                    }
                    relation_dict.update(tags)  # Add all tags as columns
                    relation_data.append(relation_dict)

                except Exception as e:
                    # print(f"Warning: Could not create multipolygon for relation {relation_id}: {e}")
                    continue
            # Handle other relation types (routes, restrictions, etc.)
            else:
                # For non-multipolygon relations, we could:
                # 1. Skip them (current approach)
                # 2. Create a MultiLineString or GeometryCollection from members
                # 3. Create a custom representation

                # For demonstration, we'll just log and skip
                # print(f"Skipping non-multipolygon relation: {relation_id}, type: {relation_type}")

                # If you want to include non-multipolygon relations with their tags but without geometry:
                tags = element.get("tags", {})
                relation_dict = {
                    "id": relation_id,
                    "geometry": None,
                    "type": "relation",
                    "relation_type": relation_type,
                }
                relation_dict.update(tags)
                relation_data.append(relation_dict)

    # Create relations GeoDataFrame
    if relation_data:
        # Filter out entries with None geometries
        valid_relations = [r for r in relation_data if r["geometry"] is not None]
        if valid_relations:
            relations_gdf = gpd.GeoDataFrame(valid_relations, crs="EPSG:4326")
        else:
            relations_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            # print("No valid relation geometries found.")
    else:
        relations_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        # print("No relation features found.")

    return nodes_gdf, ways_gdf, relations_gdf


def get_gdf(json_data):
    # Process the OSM data
    nodes_gdf, ways_gdf, relations_gdf = process_osm_data(json_data)

    valid_gdfs = []
    if not nodes_gdf.empty:
        valid_gdfs.append(nodes_gdf)
    if not ways_gdf.empty:
        valid_gdfs.append(ways_gdf)
    if (
        not relations_gdf.empty
        and "geometry" in relations_gdf
        and not relations_gdf["geometry"].isna().all()
    ):
        # Remove any rows with None geometries
        valid_relations = relations_gdf[~relations_gdf["geometry"].isna()]
        if not valid_relations.empty:
            valid_gdfs.append(valid_relations)

    if valid_gdfs:
        combined_gdf = pd.concat(valid_gdfs, ignore_index=True)
        combined_gdf = combined_gdf.reindex(
            columns=[
                "type",
                "id",
                "geometry",
                "tourism",
                "waterway",
                "name",
                "building",
                "addr:street",
                "addr:housenumber",
                "addr:city",
                "addr:postcode",
                "addr:state",
                "addr:country",
                "building:use",
                "amenity",
                "smoking",
                "wheelchair",
                "historic",
            ]
        )
        # combined_gdf['amenity'] = combined_gdf['tourism']
        return combined_gdf
    else:
        return None


def _fetch_one(overpass_response):
    try:
        query = process_overpass(overpass_response)
        json_data = get_json(query)
        gdf = get_gdf(json_data)
        if gdf is not None and not gdf.empty:
            return gdf
    except Exception as e:
        print(f"[overpass error] {e}")
    return None


def get_response(overpassResponses):
    with ThreadPoolExecutor(max_workers=min(len(overpassResponses), 4)) as ex:
        results = list(ex.map(_fetch_one, overpassResponses))
    gdf_list = [g for g in results if g is not None and not g.empty]

    if not gdf_list:
        raise ValueError("No geospatial data returned from Overpass queries")

    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    combined_gdf["feature"] = (
        combined_gdf["tourism"].fillna("")
        + combined_gdf["amenity"].fillna("")
        + combined_gdf["historic"].fillna("")
        + combined_gdf["waterway"].fillna("")
    )
    combined_gdf = combined_gdf.sample(n=min(len(combined_gdf), 2_000))
    return combined_gdf
