from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from app.preprocessing.common import prepare_data
from app.routes.knn_route import dataframe_to_feature_collection


class UploadWrapper:
    def __init__(self, file_obj):
        self.file = file_obj


def _write_geojson(path: Path, rows: dict, points: list[Point]) -> None:
    gdf = gpd.GeoDataFrame(rows, geometry=points, crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")


def test_prepare_data_error_contract_returns_eight_items(tmp_path):
    opp_path = tmp_path / "opportunities.geojson"
    dem_path = tmp_path / "demands.geojson"
    _write_geojson(opp_path, {"foo": [1]}, [Point(-46.6, -23.5)])
    _write_geojson(dem_path, {"bar": [1]}, [Point(-46.7, -23.6)])

    with open(opp_path, "rb") as opp_f, open(dem_path, "rb") as dem_f:
        result = prepare_data(UploadWrapper(opp_f), UploadWrapper(dem_f))

    assert len(result) == 8
    assert result[0]["error"]


def test_prepare_data_filters_demands_by_inferred_city_column(tmp_path):
    opp_path = tmp_path / "opportunities.geojson"
    dem_path = tmp_path / "demands.geojson"

    _write_geojson(
        opp_path,
        {
            "NOME": ["UBS Centro", "UBS Norte"],
            "MUNICIPIO": ["Macapa", "Santana"],
            "UF": ["AP", "AP"],
        },
        [Point(-51.05, 0.03), Point(-51.18, -0.04)],
    )
    _write_geojson(
        dem_path,
        {
            "CD_SETOR": ["1", "2"],
            "MUNICIPIO": ["Macapa", "Santana"],
            "UF": ["AP", "AP"],
        },
        [Point(-51.06, 0.04), Point(-51.17, -0.03)],
    )

    with open(opp_path, "rb") as opp_f, open(dem_path, "rb") as dem_f:
        err, demands_gdf, opps_gdf, *_ = prepare_data(
            UploadWrapper(opp_f),
            UploadWrapper(dem_f),
            state="AP",
            city="Macapa",
        )

    assert err is None
    assert list(demands_gdf["CD_SETOR"]) == ["1"]
    assert list(opps_gdf["NOME"]) == ["UBS Centro"]


def test_dataframe_to_feature_collection_uses_origin_geometry():
    df = pd.DataFrame(
        [
            {
                "demand_id": "A",
                "Origin_Lat": -3.1,
                "Origin_Lon": -60.0,
                "distance_km": 1.5,
            }
        ]
    )

    geojson = dataframe_to_feature_collection(df)

    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"][0]["geometry"] == {
        "type": "Point",
        "coordinates": [-60.0, -3.1],
    }
    assert geojson["features"][0]["properties"]["demand_id"] == "A"
