import logging
import geopandas as gpd
from unidecode import unidecode
import pandas as pd

from app.preprocessing.geoprocessing import process_geometries
from app.preprocessing.utils import infer_column
from app.config import settings

logger = logging.getLogger(__name__)

def _prepare_data_error(message):
    return {"error": message}, None, None, None, None, None, None, None

def _normalize_text(value):
    if pd.isnull(value):
        return ""
    return unidecode(str(value).strip().lower())

def prepare_data(opportunities_file, demands_file, state=None, city=None):
    logger.info("Reading GeoDataFrames from uploaded files.")
    opportunities_gdf = gpd.read_file(opportunities_file.file)
    demands_gdf = gpd.read_file(demands_file.file)

    logger.info("Calling process_geometries on both GDFs.")
    opportunities_gdf = process_geometries(opportunities_gdf)
    demands_gdf = process_geometries(demands_gdf)

    logger.info("Inferring column names for demands and opportunities.")
    col_demand_id = infer_column(demands_gdf, settings.DEMAND_ID_POSSIBLE_COLUMNS)
    col_name = infer_column(opportunities_gdf, settings.NAME_POSSIBLE_COLUMNS)
    col_city = infer_column(opportunities_gdf, settings.CITY_POSSIBLE_COLUMNS)
    col_city_demand = infer_column(demands_gdf, settings.CITY_POSSIBLE_COLUMNS)
    col_state_opportunities = infer_column(opportunities_gdf, settings.STATE_POSSIBLE_COLUMNS)
    col_state_demand = infer_column(demands_gdf, settings.STATE_POSSIBLE_COLUMNS)

    if not col_demand_id or not col_name or not col_city or not col_state_opportunities or not col_state_demand:
        logger.error("Could not infer all necessary columns. Check the input data.")
        return _prepare_data_error("Could not infer all necessary columns. Please check the input data.")

    if city and not col_city_demand:
        logger.error("Could not infer city column in demands data.")
        return _prepare_data_error("Could not infer the city column in demands data. Please check the input data.")

    if state:
        logger.info("Filtering by state='%s'.", state)
        opportunities_gdf = opportunities_gdf[opportunities_gdf[col_state_opportunities] == state]
        demands_gdf = demands_gdf[demands_gdf[col_state_demand] == state]
    else:
        logger.info("No state provided. Using entire dataset for allocation.")

    if city:
        logger.info("Filtering by city='%s'.", city)
        city_norm = _normalize_text(city)

        opportunities_gdf = opportunities_gdf[opportunities_gdf[col_city].apply(_normalize_text) == city_norm]
        demands_gdf = demands_gdf[demands_gdf[col_city_demand].apply(_normalize_text) == city_norm]

    logger.info("prepare_data completed successfully.")
    return None, demands_gdf, opportunities_gdf, col_demand_id, col_name, col_city, col_state_opportunities, col_state_demand
