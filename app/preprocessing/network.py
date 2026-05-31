import os
import sys
import hashlib
import json
import pandas as pd
import geopandas as gpd
import pandana as pdna
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
from shapely import wkt
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging
import gc # Importado globalmente para garantir disponibilidade

from app.preprocessing.utils import infer_column
from app.config import settings
import time # Importando time para o sleep
from unidecode import unidecode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _normalize_city(value):
    if pd.isnull(value):
        return ""
    return unidecode(str(value).strip().lower())

def _filter_by_city(gdf, city_column, city_name):
    city_norm = _normalize_city(city_name)
    return gdf[gdf[city_column].apply(_normalize_city) == city_norm]

def get_cache_key(city_name, polygon):
    """
    Generate a unique cache key based on the city name and the area's bounding box.
    """
    logger.debug("Generating cache key for city='%s' with polygon bounds=%s", city_name, polygon.bounds)
    key_str = city_name.lower() + str(polygon.bounds)
    return hashlib.sha1(key_str.encode()).hexdigest()

def load_network_from_cache(cache_key):
    """
    Attempt to load the network graph from cache.
    Converts geometry attributes from WKT to Shapely objects.
    Returns the graph if found, otherwise returns None.
    """
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    logger.info("Attempting to load network from cache: %s", cache_file)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            # Convert geometry attributes from WKT to Shapely objects
            for node in data.get("nodes", []):
                if "geometry" in node and isinstance(node["geometry"], str):
                    node["geometry"] = wkt.loads(node["geometry"])
            for edge in data.get("links", []):
                if "geometry" in edge and isinstance(edge["geometry"], str):
                    edge["geometry"] = wkt.loads(edge["geometry"])
            graph = nx.node_link_graph(data)
            logger.info("Network successfully loaded from cache.")
            return graph
        except Exception as e:
            logger.info("Cache file not found: %s", cache_file)
    return None

def save_network_to_cache(graph, cache_key):
    """
    Convert geometry attributes to WKT and save the network graph to cache.
    """
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    logger.info("Saving network to cache: %s", cache_file)
    try:
        data = nx.node_link_data(graph)
        # Convert geometry to WKT for nodes
        for node in data.get("nodes", []):
            if "geometry" in node and hasattr(node["geometry"], "wkt"):
                node["geometry"] = node["geometry"].wkt
        # Convert geometry to WKT for edges (links)
        for edge in data.get("links", []):
            if "geometry" in edge and hasattr(edge["geometry"], "wkt"):
                edge["geometry"] = edge["geometry"].wkt
        with open(cache_file, "w") as f:
            json.dump(data, f)
        logger.info("Network saved to cache under key=%s", cache_key)
    except Exception as e:
        logger.error(f"Error saving network to cache: {e}")

def compute_distance_matrix(demands_gdf, ubs_gdf, city_name=None, max_distance=50000, num_threads=1):
    """
    1. Attempts to download the network using the defined endpoints.
    2. If the network is already cached for the given city, it will be loaded from cache.
    3. If the download fails, the buffer is increased and retried until 'max_attempts' is reached.
    4. Timeout is set to 2500s (~41 minutes).
    """
    logger.info("Starting compute_distance_matrix with city_name='%s', max_distance=%d, num_threads=%d",
                city_name, max_distance, num_threads)
    
    # Set timeout (applies to all attempts)
    ox.settings.timeout = 2500

    # List of endpoints to use.
    endpoints = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter"
    ]
    
    logger.debug("Reprojecting demands_gdf and ubs_gdf to EPSG:4326")
    demands_gdf = demands_gdf.reset_index(drop=True)
    ubs_gdf = ubs_gdf.reset_index(drop=True)

    demands_gdf = demands_gdf.to_crs(epsg=4326)
    ubs_gdf = ubs_gdf.to_crs(epsg=4326)

    if city_name:
        logger.info("Filtering demands and opportunities by city_name='%s' (case-insensitive).", city_name)
        demand_city_col = infer_column(demands_gdf, settings.CITY_POSSIBLE_COLUMNS)
        ubs_city_col = infer_column(ubs_gdf, settings.CITY_POSSIBLE_COLUMNS)
        if demand_city_col:
            demands_gdf = _filter_by_city(demands_gdf, demand_city_col, city_name)
        else:
            logger.warning("Could not infer city column for demands; skipping demand city filter.")
        if ubs_city_col:
            ubs_gdf = _filter_by_city(ubs_gdf, ubs_city_col, city_name)
        else:
            logger.warning("Could not infer city column for opportunities; skipping opportunity city filter.")

    # Initial buffer (in degrees) to expand if points fall outside the network.
    # Buffer reduzido para 0.02 (aprox 2.2km) para otimizar memória em áreas densas
    buffer_size = 0.02
    combined_geom = demands_gdf.unary_union.union(ubs_gdf.unary_union)

    max_attempts = 3
    attempt = 0

    graph = None

    # Attempt to load the network from cache if city_name is provided.
    if city_name:
        cache_key = get_cache_key(city_name, combined_geom)
        graph = load_network_from_cache(cache_key)

    # If not found in cache, attempt to download the network.
    if graph is None:
        logger.info("Graph not found in cache. Attempting download from Overpass.")
        while attempt < max_attempts:
            area_of_interest = combined_geom.buffer(buffer_size)

            # Try each endpoint until the network is successfully downloaded.
            graph = None
            for endpoint in endpoints:
                try:
                    ox.settings.overpass_endpoint = endpoint
                    logger.info("Attempt %d - Overpass endpoint: %s (buffer=%.3f)", attempt+1, endpoint, buffer_size)
                    graph = ox.graph_from_polygon(
                        area_of_interest,
                        network_type='drive',
                        simplify=True
                    )
                    break

                except Exception as e:
                    logger.warning("Failed to download network from %s: %s", endpoint, e)

            # Se falhar em todos os endpoints...
            if graph is None:
                # MODIFICAÇÃO: Não aumente o buffer se o erro for de conexão/download em cidades grandes
                # Isso evita o loop da morte onde a área fica maior e mais pesada a cada retry
                logger.error("Failed on all endpoints for attempt %d.", attempt+1)
                
                # Apenas aumente o buffer se tiver certeza que é um problema geométrico, 
                # mas neste caso de timeout de rede, é melhor manter ou aumentar minimamente.
                # buffer_size = min(buffer_size * 1.5, 1.0) # <--- REMOVIDO AUMENTO AGRESSIVO
                
                # Opção: Esperar um pouco antes de tentar de novo para não ser bloqueado pela API
                logger.info("Waiting 10 seconds before next attempt...")
                time.sleep(10)
                
                attempt += 1
                continue

            # Network downloaded successfully; break out of loop.
            logger.info("Network downloaded successfully.")
            break

        # If city_name is provided and the network was downloaded (not from cache), save it to cache.
        if city_name and graph is not None:
            logger.info("Saving newly downloaded graph to cache for city='%s'", city_name)
            save_network_to_cache(graph, cache_key)

    # If after max_attempts the network could not be obtained, raise an error.
    if attempt == max_attempts and (graph is None):
        logger.error("Unable to obtain the network or map the points after %d attempts.", max_attempts)
        raise RuntimeError("Unable to obtain the network or map the points after several attempts.")

    
    # Convert the graph into GeoDataFrames.
    logger.info("Converting graph to GeoDataFrames.")
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes = nodes.reset_index()
    edges = edges.reset_index()

    # Adjust type for cross-platform compatibility.
    if sys.platform.startswith("win"):
        node_dtype = np.int32
    else:
        node_dtype = np.intp

    from_nodes = edges['u'].map(dict(zip(nodes['osmid'], nodes.index))).astype(node_dtype)
    to_nodes = edges['v'].map(dict(zip(nodes['osmid'], nodes.index))).astype(node_dtype)
    edge_weights = pd.DataFrame(edges['length'].astype(np.float64))

    # Create the Pandana network.
    logger.info("Initializing Pandana Network engine...")
    network = pdna.Network(
        nodes['x'].values,
        nodes['y'].values,
        from_nodes,
        to_nodes,
        edge_weights
    )

    # --- OTIMIZAÇÃO DE MEMÓRIA ---
    logger.info("Cleaning up raw graph data from memory to free RAM...")
    del graph
    del nodes
    del edges
    del from_nodes
    del to_nodes
    del edge_weights
    gc.collect() # Força o Garbage Collector a rodar AGORA
    # -----------------------------
    
    # Só agora faça o processamento pesado
    logger.debug("Locating nearest nodes...")

    # Extract coordinates.
    logger.debug("Locating nearest nodes for demand and opportunity points.")
    demand_coords = np.array(list(zip(demands_gdf['geometry'].x, demands_gdf['geometry'].y)))
    ubs_coords = np.array(list(zip(ubs_gdf['geometry'].x, ubs_gdf['geometry'].y)))

    # Locate the nearest nodes in the network.
    demand_nodes_array = network.get_node_ids(demand_coords[:, 0], demand_coords[:, 1])
    ubs_nodes_array = network.get_node_ids(ubs_coords[:, 0], ubs_coords[:, 1])

    invalid_demand_nodes = np.where(demand_nodes_array == -1)[0]
    invalid_ubs_nodes = np.where(ubs_nodes_array == -1)[0]

    # If points fall outside the network...
    if len(invalid_demand_nodes) > 0 or len(invalid_ubs_nodes) > 0:
        logger.warning(
            "%d demand points and %d opportunity points are outside the network.",
            len(invalid_demand_nodes), len(invalid_ubs_nodes)
        )
        # Nota: Como o grafo já foi baixado e deletado, não podemos retentar aumentando buffer neste fluxo sem refazer tudo.
        # A lógica original tentava aumentar o buffer AQUI, o que não funciona mais com a limpeza de memória acima.
        logger.warning("Continuing with valid nodes only.")
    else:
        logger.info("All points have been successfully mapped onto the network.")

    # Remove invalid records, if any.
    if len(invalid_demand_nodes) > 0:
        logger.info("Removing %d invalid demand records.", len(invalid_demand_nodes))
        demands_gdf = demands_gdf.drop(demands_gdf.index[invalid_demand_nodes]).reset_index(drop=True)
        demand_coords = np.delete(demand_coords, invalid_demand_nodes, axis=0)
        demand_nodes_array = np.delete(demand_nodes_array, invalid_demand_nodes)

    if len(invalid_ubs_nodes) > 0:
        logger.info("Removing %d invalid opportunity records.", len(invalid_ubs_nodes))
        ubs_gdf = ubs_gdf.drop(ubs_gdf.index[invalid_ubs_nodes]).reset_index(drop=True)
        ubs_coords = np.delete(ubs_coords, invalid_ubs_nodes, axis=0)
        ubs_nodes_array = np.delete(ubs_nodes_array, invalid_ubs_nodes)

    # Convert arrays to Series with the same index as the GeoDataFrames.
    demand_nodes = pd.Series(demand_nodes_array, index=demands_gdf.index)
    ubs_nodes = pd.Series(ubs_nodes_array, index=ubs_gdf.index)

    # Pre-calculate distances up to max_distance.
    logger.info("Precomputing distances up to %d meters in Pandana Network.", max_distance)
    network.precompute(max_distance)

    num_demands = len(demand_nodes)
    num_ubs = len(ubs_nodes)
    logger.info("Calculating shortest paths for %d demands vs %d opportunities...", num_demands, num_ubs)
    distances = np.empty((num_demands, num_ubs), dtype=np.float32)

    def compute_row(i):
        orig_node = demand_nodes.iloc[i]
        orig_nodes = np.full(num_ubs, orig_node, dtype=int)
        distances_row = network.shortest_path_lengths(orig_nodes, ubs_nodes.values)
        return i, distances_row

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_row, i) for i in range(num_demands)]
        for future in futures:
            i, distances_row = future.result()
            distances[i, :] = distances_row

    # Replace infinities with NaN.
    distances[np.isinf(distances)] = np.nan
    distance_df = pd.DataFrame(distances, index=demands_gdf.index, columns=ubs_gdf.index)
    logger.info("Distance matrix computed with shape: %s", distance_df.shape)

    # Infer the ID columns.
    logger.info("Inferring ID columns for demands and opportunities.")
    col_demand_id = infer_column(demands_gdf, settings.DEMAND_ID_POSSIBLE_COLUMNS)
    col_name = infer_column(ubs_gdf, settings.NAME_POSSIBLE_COLUMNS)

    demands_ids = demands_gdf[col_demand_id].values
    ubs_names = ubs_gdf[col_name].values

    distance_df.index = demands_ids
    distance_df.columns = ubs_names

    logger.info("compute_distance_matrix completed successfully.")
    
    # CORREÇÃO CRÍTICA: Retornar None para os objetos deletados da memória
    return distance_df, network, None, None, None, demand_nodes, ubs_nodes
