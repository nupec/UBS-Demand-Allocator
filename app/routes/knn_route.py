import json
import uuid
import os
import pandas as pd
import io
import logging
import zipfile
import math

from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from enum import Enum
from app.preprocessing.common import prepare_data
from app.methods.knn_model import allocate_demands_knn
from app.preprocessing.utils import infer_column
from app.config import settings
from app.lib.convert_numpy import convert_numpy_types
from unidecode import unidecode

from app.analysis.reporting import (
    analyze_allocation,
    create_allocation_charts,
    create_coverage_stats,
    create_distance_hist,
    generate_allocation_pdf,
    create_distance_boxplot,
    save_summary_table_image,
    create_summary_table
)

logger = logging.getLogger(__name__)
router = APIRouter()

def _normalize_text(value) -> str:
    if pd.isnull(value):
        return ""
    return unidecode(str(value).strip().lower())

def _filter_by_normalized_column(df: pd.DataFrame, column: str, expected_value: str) -> pd.DataFrame:
    expected_norm = _normalize_text(expected_value)
    return df[df[column].apply(_normalize_text) == expected_norm]

def _json_safe(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return convert_numpy_types(value)

def dataframe_to_feature_collection(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        lat = row.get("Origin_Lat")
        lon = row.get("Origin_Lon")
        geometry = None
        if pd.notna(lat) and pd.notna(lon):
            geometry = {
                "type": "Point",
                "coordinates": [float(lon), float(lat)]
            }

        properties = {
            column: _json_safe(value)
            for column, value in row.items()
        }
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": properties
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

class MethodEnum(str, Enum):
    pandana_real_distance = "pandana_real_distance"
    pysal = "pysal"
    valhalla = "valhalla"  

class OutputFormatEnum(str, Enum):
    csv = "csv"
    geojson = "geojson"
    json = "json"

@router.post("/allocate_demands_knn/")
def allocate_demands_knn_api(
    opportunities_file: UploadFile,
    demands_file: UploadFile,
    num_threads: int = Query(0, description="Número de threads (0 = todos os núcleos)"),
    state: str = Query("", description="State (optional)"),
    city: str = Query("", description="City (optional)"),
    cities: str = Query("", description="Optional JSON array of cities for multi-city allocation"),
    k: int = Query(1, description="Number of neighbors for KNN"),
    method: MethodEnum = Query(MethodEnum.pysal, description="Choose the allocation method"),
    output_format: OutputFormatEnum = Query(OutputFormatEnum.csv, description="Output format: 'csv', 'geojson', or 'json'"),
    eda: bool = Query(False, description="If true, also perform EDA and return a ZIP with allocation and analysis results")
):
    threads = num_threads if num_threads > 0 else os.cpu_count()
    
    logger.info("Received request to allocate demands using KNN.")
    logger.info("Parameters: state=%s, city=%s, cities=%s, k=%d, method=%s, output_format=%s, eda=%s",
                state, city, cities, k, method, output_format, eda)

    opp_bytes = opportunities_file.file.read()
    dem_bytes = demands_file.file.read()

    cities_list = []
    if cities:
        try:
            cities_list = json.loads(cities)
            if not isinstance(cities_list, list):
                raise ValueError("Parameter 'cities' must be a JSON array.")
        except Exception as e:
            logger.exception("Error parsing 'cities' parameter.")
            raise HTTPException(status_code=400, detail="Invalid 'cities' parameter. Must be a JSON array string.")

    # Função auxiliar interna para preparar dados a partir dos bytes lidos
    def prepare_data_from_bytes(state: str = "", city_filter: str = ""):
        from app.preprocessing.common import prepare_data
        from io import BytesIO
        
        class FakeUploadFile:
            def __init__(self, content):
                self.file = content
        
        fake_opp = FakeUploadFile(io.BytesIO(opp_bytes))
        fake_dem = FakeUploadFile(io.BytesIO(dem_bytes))
        return prepare_data(fake_opp, fake_dem, state=state, city=city_filter)

    results_df_list = []

    if cities_list:
        logger.info("Allocating demands for multiple cities: %s", cities_list)
        for city_name in cities_list:
            error, demands_gdf, opportunities_gdf, col_demand_id, col_name, col_city, col_state_opp, col_state_dem = prepare_data_from_bytes(
                state=state, city_filter=None
            )
            if error:
                logger.error("Error in prepare_data: %s", error)
                raise HTTPException(status_code=400, detail=str(error))
            col_city_dem = infer_column(demands_gdf, settings.CITY_POSSIBLE_COLUMNS)
            if not col_city_dem:
                raise HTTPException(status_code=400, detail="Could not infer the city column in demands data.")

            demands_city = _filter_by_normalized_column(demands_gdf, col_city_dem, city_name)

            if col_city:
                opp_city = _filter_by_normalized_column(opportunities_gdf, col_city, city_name)
            else:
                opp_city = opportunities_gdf

            if demands_city.empty or opp_city.empty:
                logger.warning("No records found for city '%s'. Skipping.", city_name)
                continue
            
            logger.info("Allocating demands for city: %s", city_name)
            partial_df = allocate_demands_knn(
                demands_city,
                opp_city,
                col_demand_id,
                col_name,
                col_city,
                col_state_opp,
                k=k,
                method=method, 
                city_name=city_name,
                num_threads=threads
            )
            partial_df["city_allocated"] = city_name
            results_df_list.append(partial_df)
        
        if not results_df_list:
            raise HTTPException(status_code=404, detail="No records found for any provided city.")
        result_df = pd.concat(results_df_list, ignore_index=True)
    else:
        logger.info("Allocating demands for single city='%s' or entire region if blank.", city)
        error, demands_gdf, opportunities_gdf, col_demand_id, col_name, col_city, col_state_opp, col_state_dem = prepare_data_from_bytes(
            state=state, city_filter=city
        )
        if error:
            logger.error("Error in prepare_data: %s", error)
            raise HTTPException(status_code=400, detail=str(error))
        
        result_df = allocate_demands_knn(
            demands_gdf,
            opportunities_gdf,
            col_demand_id,
            col_name,
            col_city,
            col_state_opp,
            k=k,
            method=method,
            city_name=city,
            num_threads=threads
        )

    logger.info("Allocation completed successfully. Number of rows in result: %d", len(result_df))

    if eda:
        logger.info("EDA option enabled. Generating analysis results.")
        import geopandas as gpd
        demanda_gdf = gpd.read_file(io.BytesIO(dem_bytes))
        
        merged_df, summary = analyze_allocation(result_df, demanda_gdf)
        logger.info("Analysis completed. Merged allocation shape: %s, Summary shape: %s", merged_df.shape, summary.shape)
        
        chart1_buf, chart2_buf = create_allocation_charts(summary)
        coverage_stats = create_coverage_stats(merged_df)
        distance_hist_buf = create_distance_hist(merged_df)
        resumo = create_summary_table(summary)
        table_image = save_summary_table_image(resumo)
        box_plot = create_distance_boxplot(merged_df)
        pdf_buf = generate_allocation_pdf(summary, merged_df)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            allocation_csv = result_df.to_csv(index=False)
            zipf.writestr("allocation_result.csv", allocation_csv)
            
            merged_json = merged_df.to_json(orient="records", force_ascii=False)
            zipf.writestr("merged_allocation.json", merged_json)
            
            summary_csv = summary.to_csv(index=False)
            zipf.writestr("summary.csv", summary_csv)
            
            zipf.writestr("chart_population.png", chart1_buf.getvalue())
            zipf.writestr("chart_racial.png", chart2_buf.getvalue())
            zipf.writestr("Table_resumo.png", table_image.getvalue())
            zipf.writestr("report.pdf", pdf_buf.getvalue())
            
            if not coverage_stats.empty:
                zipf.writestr("coverage_stats.csv", coverage_stats.to_csv(index=False))
            if distance_hist_buf:
                zipf.writestr("distance_hist.png", distance_hist_buf.getvalue())
            if box_plot:
                zipf.writestr("Box_plot.png", box_plot.getvalue())    
        
        zip_buffer.seek(0)
        logger.info("EDA analysis generated successfully. Returning ZIP file.")
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=allocation_eda_result.zip"}
        )
    else:
        file_id = str(uuid.uuid4())
        OUTPUT_DIR = "/tmp/api_output/"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Saving allocation output to directory: %s", OUTPUT_DIR)

        if output_format == "csv":
            output_file = os.path.join(OUTPUT_DIR, f"allocation_result_{file_id}.csv")
            result_df.to_csv(output_file, index=False)
            logger.info("Returning CSV file: %s", output_file)
            return FileResponse(output_file, media_type="text/csv", filename=os.path.basename(output_file))
        elif output_format == "geojson":
            output_file = os.path.join(OUTPUT_DIR, f"allocation_result_{file_id}.geojson")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataframe_to_feature_collection(result_df), f, ensure_ascii=False)
            logger.info("Returning GeoJSON file: %s", output_file)
            return FileResponse(output_file, media_type="application/geo+json", filename=os.path.basename(output_file))
        elif output_format == "json":
            logger.info("Returning JSON response directly.")
            records = [
                {column: _json_safe(value) for column, value in row.items()}
                for row in result_df.to_dict(orient="records")
            ]
            return JSONResponse(content=records)
        else:
            logger.error("Invalid output format requested: %s", output_format)
            raise HTTPException(status_code=400, detail="Invalid output format. Use 'csv', 'geojson', or 'json'.")
