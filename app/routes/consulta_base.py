from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
import math  
from datetime import datetime
from io import BytesIO
from typing import Dict, Tuple, Optional
from unidecode import unidecode 

import geopandas as gpd
import pandas as pd
import requests
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.analysis.socioeconomic_analys import analyze_knn_allocation
from app.config import settings
from app.lib.convert_numpy import convert_numpy_types
from app.methods.knn_model import allocate_demands_knn
from app.preprocessing.common import prepare_data
from app.preprocessing.utils import get_polygon_path 
from app.analysis.reporting import (
    analyze_allocation,  
    create_allocation_charts,
    create_coverage_stats,
    create_distance_boxplot,
    create_distance_hist,
    create_summary_table,
    generate_allocation_pdf,
    save_summary_table_image,
    gerar_perguntas_respostas,
)

# ------------------------------------------------------------------ #
#  Configuração básica                                               #
# ------------------------------------------------------------------ #
num_threads = os.cpu_count()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/consulta_base")

BACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEMANDS_BASE_DIR   = os.path.join(BACK_ROOT, "data", "geojson_por_estado_cidade")
OPPORTUNITIES_PATH = os.path.join(BACK_ROOT, "data", "opportunities.geojson")


# --- ATUALIZAÇÃO PARA DOCKER ---
SHARED_DIR = "/shared"
os.makedirs(SHARED_DIR, exist_ok=True)
FRONT_DATA_DIR   = SHARED_DIR
FRONT_CONFIG_DIR = SHARED_DIR
FRONTEND_UPLOAD_URL = os.getenv("FRONTEND_UPLOAD_URL")
MAP_UPLOAD_TOKEN = os.getenv("MAP_UPLOAD_TOKEN")

# ------------------------------------------------------------------ #
#  Cache de ZIPs (TTL simples)                                       #
# ------------------------------------------------------------------ #
ZIP_CACHE: Dict[str, Tuple[datetime, bytes]] = {}
ZIP_TTL_MIN = 30


def _clean_zip_cache() -> None:
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(minutes=ZIP_TTL_MIN)
    for k, (ts, _) in list(ZIP_CACHE.items()):
        if ts < cutoff:
            del ZIP_CACHE[k]


# ------------------------------------------------------------------ #
#  Funções de apoio                                                  #
# ------------------------------------------------------------------ #
def _uid() -> str:
    return uuid.uuid4().hex[:7]


def _resolve_demands_file_path(uf: str, municipio: str) -> str:
    city_base = municipio.upper().replace(" ", "_")
    candidates = [f"{city_base}.geojson", f"{unidecode(city_base)}.geojson"]

    seen = set()
    for filename in candidates:
        if filename in seen:
            continue
        seen.add(filename)

        path = os.path.join(DEMANDS_BASE_DIR, uf.upper(), filename)
        if os.path.exists(path):
            return path

    raise HTTPException(
        status_code=404,
        detail=f"Arquivo de dados para o município '{municipio}' não encontrado.",
    )

# Função para calcular Zoom e Centro baseado na Bounding Box
def calculate_optimal_view(gdf: gpd.GeoDataFrame):
    """
    Calcula o centro (lat, lon) e o nível de zoom ideal para enquadrar
    toda a geometria do GeoDataFrame na tela do Kepler.gl.
    """
    if gdf.empty or not hasattr(gdf, 'total_bounds'):
        # Fallback padrão se algo der errado
        return -14.2350, -51.9253, 4

    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Centro
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    
    # Cálculo do Zoom
    # A lógica baseia-se na largura em graus. 
    # 360 graus = Zoom 0. Cada nível de zoom divide a área por 2.
    # Adicionamos um padding (margem) subtraindo do zoom final.
    
    dif_lon = maxx - minx
    dif_lat = maxy - miny
    
    # Pega a maior dimensão para garantir que caiba
    max_dif = max(dif_lon, dif_lat)
    
    if max_dif == 0:
        return center_lat, center_lon, 12 
        
    # Fórmula heurística para Web Mercator
    # zoom = log2(360 / max_dif) + offset
    # O offset ajusta o tamanho da janela. 
    zoom = math.log2(360 / max_dif) + 1.2
    
    # Trava limites razoáveis
    zoom = max(min(zoom, 16), 4)
    
    return center_lat, center_lon, zoom


def _format_access_class(distance_km) -> str:
    try:
        distance = float(distance_km)
    except (TypeError, ValueError):
        return "Sem distância"

    if math.isnan(distance):
        return "Sem distância"
    if distance > 4:
        return "Crítico (> 4 km)"
    if distance > 2:
        return "Atenção (2-4 km)"
    return "Adequado (≤ 2 km)"


def _format_priority_from_distance(distance_km) -> str:
    try:
        distance = float(distance_km)
    except (TypeError, ValueError):
        return "Sem distância"

    if math.isnan(distance):
        return "Sem distância"
    if distance > 4:
        return "Crítica"
    if distance > 2:
        return "Atenção"
    return "Adequada"


def _short_label(value, max_len: int = 28) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len - 1]}…"


def _build_map_allocation_df(merged_df: pd.DataFrame, fallback_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara um CSV enxuto e legível para o Kepler, com campos úteis no tooltip."""
    source_df = merged_df if not merged_df.empty else fallback_df
    wanted_columns = [
        "demand_id",
        "CD_SETOR",
        "SITUACAO",
        "REGIÃO",
        "Destination_State",
        "Destination_City",
        "opportunity_name",
        "distance_km",
        "distance_mean",
        "distance_variance",
        "DEMANDA",
        "RAÇA NEGRA TOTAL",
        "RAÇA PARDA TOTAL",
        "RAÇA INDÍGENA TOTAL",
        "RAÇA AMARELA TOTAL",
        "Origin_Lat",
        "Origin_Lon",
        "Destination_Lat",
        "Destination_Lon",
        "fallback_used",
    ]
    present_columns = [col for col in wanted_columns if col in source_df.columns]
    map_df = source_df[present_columns].copy()

    def text_series(column: str, default: str = "") -> pd.Series:
        if column in map_df.columns:
            return map_df[column].fillna(default).astype(str)
        return pd.Series([default] * len(map_df), index=map_df.index, dtype="object")

    for col in ["distance_km", "DEMANDA", "Origin_Lat", "Origin_Lon", "Destination_Lat", "Destination_Lon"]:
        if col in map_df.columns:
            map_df[col] = pd.to_numeric(map_df[col], errors="coerce")
    if "DEMANDA" not in map_df.columns:
        map_df["DEMANDA"] = 1

    distances = pd.to_numeric(map_df.get("distance_km", pd.Series(dtype=float)), errors="coerce")
    map_df["classe_acesso"] = distances.apply(_format_access_class)
    map_df["setor_acima_4km"] = distances.apply(lambda value: "Sim" if pd.notna(value) and value > 4 else "Não")
    map_df["prioridade_acesso"] = distances.apply(_format_priority_from_distance)

    population = pd.to_numeric(map_df.get("DEMANDA", pd.Series([0] * len(map_df))), errors="coerce").fillna(0)
    ppi_columns = ["RAÇA NEGRA TOTAL", "RAÇA PARDA TOTAL", "RAÇA INDÍGENA TOTAL"]
    ppi_total = pd.Series([0] * len(map_df), index=map_df.index, dtype="float64")
    for col in ppi_columns:
        if col in map_df.columns:
            ppi_total = ppi_total + pd.to_numeric(map_df[col], errors="coerce").fillna(0)

    map_df["Setor censitário"] = text_series("CD_SETOR")
    if (map_df["Setor censitário"] == "").all():
        map_df["Setor censitário"] = text_series("demand_id")
    map_df["UBS alocada"] = text_series("opportunity_name")
    map_df["População total"] = population.round(0)
    map_df["População PPI"] = ppi_total.round(0)
    map_df["Percentual PPI"] = (ppi_total / population.where(population > 0)).fillna(0).mul(100).round(1)
    map_df["Distância até UBS (km)"] = distances.round(2)
    map_df["Classe de acesso"] = map_df["classe_acesso"]
    map_df["Setor crítico"] = map_df["setor_acima_4km"]
    map_df["Prioridade de acesso"] = map_df["prioridade_acesso"]
    if "fallback_used" in map_df.columns:
        fallback_values = map_df["fallback_used"].astype(str).str.lower().isin(["true", "1", "sim", "yes"])
        map_df["Método fallback"] = fallback_values.map({True: "Sim", False: "Não"})
    else:
        map_df["Método fallback"] = "Não"

    # Camada crítica dedicada: linhas abaixo de 4 km ficam sem coordenadas e não são renderizadas.
    if {"Origin_Lat", "Origin_Lon"}.issubset(map_df.columns):
        critical_mask = distances > 4
        map_df["Critical_Lat"] = map_df["Origin_Lat"].where(critical_mask)
        map_df["Critical_Lon"] = map_df["Origin_Lon"].where(critical_mask)
    else:
        map_df["Critical_Lat"] = None
        map_df["Critical_Lon"] = None

    return map_df


def _build_ubs_map_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega UBS uma única vez para evitar centenas de pontos repetidos no Kepler."""
    required = {"opportunity_name", "Destination_Lat", "Destination_Lon", "distance_km"}
    if merged_df.empty or not required.issubset(merged_df.columns):
        return pd.DataFrame()

    data = merged_df.copy()
    for col in ["Destination_Lat", "Destination_Lon", "distance_km", "DEMANDA"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Destination_Lat", "Destination_Lon"])
    if data.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["opportunity_name", "Destination_Lat", "Destination_Lon"]
    for (name, lat, lon), group in data.groupby(group_cols, dropna=False):
        distances = pd.to_numeric(group["distance_km"], errors="coerce")
        if "DEMANDA" in group.columns:
            population = pd.to_numeric(group["DEMANDA"], errors="coerce").fillna(0)
        else:
            population = pd.Series([1] * len(group), index=group.index)
        above_4_mask = distances > 4
        row = {
            "opportunity_name": name,
            "Destination_Lat": lat,
            "Destination_Lon": lon,
            "setores_alocados": int(len(group)),
            "populacao_alocada": float(population.sum()),
            "distancia_media_km": float(distances.mean()) if distances.notna().any() else None,
            "distancia_maxima_km": float(distances.max()) if distances.notna().any() else None,
            "setores_acima_4km": int(above_4_mask.sum()),
            "populacao_acima_4km": float(population[above_4_mask.fillna(False)].sum()),
        }
        if "Destination_City" in group.columns:
            row["Destination_City"] = group["Destination_City"].dropna().iloc[0] if group["Destination_City"].notna().any() else None
        if "Destination_State" in group.columns:
            row["Destination_State"] = group["Destination_State"].dropna().iloc[0] if group["Destination_State"].notna().any() else None
        if "fallback_used" in group.columns:
            row["fallback_used"] = "Sim" if group["fallback_used"].astype(str).str.lower().isin(["true", "1", "sim", "yes"]).any() else "Não"
        rows.append(row)

    ubs_df = pd.DataFrame(rows)
    if not ubs_df.empty:
        ubs_df["classe_ubs"] = pd.to_numeric(ubs_df["distancia_media_km"], errors="coerce").apply(_format_access_class)
        ubs_df["prioridade_ubs"] = pd.to_numeric(ubs_df["distancia_media_km"], errors="coerce").apply(_format_priority_from_distance)
        critical_mask = (
            (pd.to_numeric(ubs_df["distancia_media_km"], errors="coerce") > 4)
            | (pd.to_numeric(ubs_df["setores_acima_4km"], errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(ubs_df["populacao_acima_4km"], errors="coerce").fillna(0) > 0)
        )
        ubs_df["ubs_critica"] = critical_mask.map({True: "Sim", False: "Não"})
        ubs_df["Critical_UBS_Lat"] = ubs_df["Destination_Lat"].where(critical_mask)
        ubs_df["Critical_UBS_Lon"] = ubs_df["Destination_Lon"].where(critical_mask)
        ubs_df["UBS"] = ubs_df["opportunity_name"].astype(str)
        ubs_df["UBS curta"] = ubs_df["UBS"].apply(_short_label)
        ubs_df["População alocada"] = pd.to_numeric(ubs_df["populacao_alocada"], errors="coerce").fillna(0).round(0)
        ubs_df["Setores alocados"] = pd.to_numeric(ubs_df["setores_alocados"], errors="coerce").fillna(0).round(0)
        ubs_df["Distância média (km)"] = pd.to_numeric(ubs_df["distancia_media_km"], errors="coerce").round(2)
        ubs_df["Distância máxima (km)"] = pd.to_numeric(ubs_df["distancia_maxima_km"], errors="coerce").round(2)
        ubs_df["Setores críticos"] = pd.to_numeric(ubs_df["setores_acima_4km"], errors="coerce").fillna(0).round(0)
        ubs_df["População crítica"] = pd.to_numeric(ubs_df["populacao_acima_4km"], errors="coerce").fillna(0).round(0)
        ubs_df["Classe da UBS"] = ubs_df["classe_ubs"]
        ubs_df["Prioridade UBS"] = ubs_df["prioridade_ubs"]
    return ubs_df


def build_kepler_config(
    csv_filename: str,
    center_lat: float,
    center_lon: float,
    poly_filename: str = None, 
    ubs_filename: str = None,
    zoom: float = 10, 
    lat_o: str = "Origin_Lat",
    lon_o: str = "Origin_Lon",
    lat_d: str = "Destination_Lat",
    lon_d: str = "Destination_Lon",
) -> dict:
    """Cria dinamicamente a configuração do KeplerGL."""
    origin_id, heatmap_id, critical_id, dest_id, critical_ubs_id, arc_id, line_id = (_uid() for _ in range(7))

    ACCESS_RANGE = {
        "name": "Acessibilidade IPSUM",
        "type": "sequential",
        "category": "Custom",
        "colors": [
            "#1F9BB4",
            "#2DD4BF",
            "#A3E635",
            "#FACC15",
            "#FB923C",
            "#EF4444",
            "#7F1D1D",
        ],
        "reversed": False,
    }

    FLOW_RANGE = {
        "name": "Fluxos por distância",
        "type": "sequential",
        "category": "Custom",
        "colors": ["#38BDF8", "#22D3EE", "#A3E635", "#FACC15", "#FB923C", "#EF4444", "#7F1D1D"],
        "reversed": False,
    }

    def field(name: str, field_type: str = "real") -> dict:
        return {"name": name, "type": field_type}

    def text_label(name: str, size: int = 12, offset=None) -> list[dict]:
        return [
            {
                "field": field(name, "string"),
                "color": [255, 255, 255],
                "size": size,
                "offset": offset or [0, -8],
                "anchor": "start",
                "alignment": "center",
            }
        ]

    def point_layer(
        layer_id,
        label,
        data_id,
        color,
        lat,
        lon,
        radius,
        opacity,
        color_field=None,
        size_field=None,
        color_range=None,
        radius_range=None,
        fixed_radius=False,
        outline=False,
        stroke_color=None,
        is_visible=True,
        text_labels=None,
    ):
        return {
            "id": layer_id,
            "type": "point",
            "config": {
                "dataId": data_id,
                "label": label,
                "color": color,
                "highlightColor": [252, 242, 26, 255],
                "columns": {"lat": lat, "lng": lon, "altitude": None},
                "isVisible": is_visible,
                "visConfig": {
                    "radius": radius,
                    "fixedRadius": fixed_radius,
                    "opacity": opacity,
                    "outline": outline,
                    "thickness": 2,
                    "strokeColor": stroke_color,
                    "colorRange": color_range or ACCESS_RANGE,
                    "radiusRange": radius_range or [4, 26],
                    "filled": True,
                },
                "hidden": False,
                "textLabel": text_labels or [],
            },
            "visualChannels": {
                "colorField": color_field,
                "colorScale": "quantile",
                "strokeColorField": None,
                "strokeColorScale": "quantile",
                "sizeField": size_field,
                "sizeScale": "sqrt",
            },
        }

    origin_layer = point_layer(
        origin_id,
        "Setores de demanda por distância",
        csv_filename,
        [255, 77, 77],
        lat_o,
        lon_o,
        5,
        0.28,
        color_field=None,
        size_field=field("DEMANDA", "integer"),
        radius_range=[2, 10],
    )

    heatmap_layer = {
        "id": heatmap_id,
        "type": "heatmap",
        "config": {
            "dataId": csv_filename,
            "label": "Calor de demanda territorial",
            "color": [251, 146, 60],
            "highlightColor": [252, 242, 26, 255],
            "columns": {"lat": lat_o, "lng": lon_o},
            "isVisible": False,
            "visConfig": {
                "opacity": 0.62,
                "colorRange": ACCESS_RANGE,
                "radius": 42,
            },
            "hidden": False,
            "textLabel": [],
        },
        "visualChannels": {
            "weightField": field("DEMANDA", "integer"),
            "weightScale": "linear",
        },
    }

    critical_layer = point_layer(
        critical_id,
        "Setores críticos (> 4 km)",
        csv_filename,
        [255, 77, 77],
        "Critical_Lat",
        "Critical_Lon",
        14,
        0.9,
        color_field=None,
        size_field=field("DEMANDA", "integer"),
        radius_range=[8, 24],
        fixed_radius=False,
        outline=True,
        stroke_color=[255, 209, 102],
    )

    if ubs_filename:
        destination_layer = point_layer(
            dest_id,
            "UBS por carga alocada",
            ubs_filename,
            [31, 186, 214],
            lat_d,
            lon_d,
            11,
            0.94,
            color_field=None,
            size_field=field("populacao_alocada", "integer"),
            radius_range=[8, 28],
            outline=True,
            stroke_color=[255, 255, 255],
            text_labels=text_label("UBS curta", size=11, offset=[0, -10]),
        )
        critical_ubs_layer = point_layer(
            critical_ubs_id,
            "UBS críticas",
            ubs_filename,
            [31, 186, 214],
            "Critical_UBS_Lat",
            "Critical_UBS_Lon",
            15,
            0.95,
            color_field=None,
            size_field=field("populacao_acima_4km", "integer"),
            radius_range=[12, 32],
            outline=True,
            stroke_color=[255, 209, 102],
            text_labels=text_label("UBS curta", size=12, offset=[0, -12]),
        )
    else:
        destination_layer = point_layer(
            dest_id,
            "UBS alocadas",
            csv_filename,
            [31, 186, 214],
            lat_d,
            lon_d,
            10,
            0.88,
            fixed_radius=True,
            outline=True,
            stroke_color=[255, 255, 255],
        )
        critical_ubs_layer = None

    arc_layer = {
        "id": arc_id,
        "type": "arc",
        "config": {
            "dataId": csv_filename,
            "label": "Fluxos de alocação",
            "color": [56, 189, 248],
            "highlightColor": [255, 255, 255],
            "columns": {"lat0": lat_o, "lng0": lon_o, "lat1": lat_d, "lng1": lon_d},
            "isVisible": True,
            "visConfig": {
                "opacity": 0.18,
                "thickness": 0.5,
                "sizeRange": [0, 6],
                "colorRange": FLOW_RANGE,
            },
            "hidden": False,
            "textLabel": [],
        },
        "visualChannels": {
            "colorField": field("distance_km"),
            "colorScale": "quantile",
            "sizeField": field("DEMANDA", "integer"),
            "sizeScale": "sqrt",
        },
    }

    line_layer = {
        "id": line_id,
        "type": "line",
        "config": {
            "dataId": csv_filename,
            "label": "Linhas origem-destino",
            "color": [148, 163, 184],
            "highlightColor": [252, 242, 26, 255],
            "columns": {
                "lat0": lat_o, "lng0": lon_o, "alt0": None,
                "lat1": lat_d, "lng1": lon_d, "alt1": None
            },
            "isVisible": False,
            "visConfig": {"opacity": 0.32, "thickness": 0.8},
            "hidden": False,
            "textLabel": [],
        },
    }

    layers = [heatmap_layer, origin_layer, critical_layer, destination_layer]
    if critical_ubs_layer:
        layers.append(critical_ubs_layer)
    layers.extend([arc_layer, line_layer])

    # --- Adicionar Camada de Polígono ---
    if poly_filename:
        poly_layer_id = _uid()
        polygon_layer = {
            "id": poly_layer_id,
            "type": "geojson",
            "config": {
                "dataId": poly_filename, 
                "label": "Limite Municipal",
                "color": [18, 147, 154],
                "columns": {
                    "geojson": "geometria"
                },
                "isVisible": True,
                "visConfig": {
                    "opacity": 0.08,
                    "strokeOpacity": 0.55,
                    "thickness": 1.2,
                    "strokeColor": [250, 204, 21],
                    "radius": 10,
                    "sizeRange": [0, 10],
                    "radiusRange": [0, 50],
                    "heightRange": [0, 0],
                    "elevationScale": 5,
                    "stroked": True,
                    "filled": True,
                    "enable3d": False,
                    "wireframe": False
                },
                "hidden": False,
                "textLabel": []
            },
            "visualChannels": {
                "colorField": None,
                "colorScale": "quantile",
                "strokeColorField": None,
                "strokeColorScale": "quantile",
                "sizeField": None,
                "sizeScale": "linear"
            }
        }
        layers.insert(0, polygon_layer)

    fields_to_show = {
        csv_filename: [
            {"name": "Setor censitário"},
            {"name": "População total"},
            {"name": "População PPI"},
            {"name": "Percentual PPI"},
            {"name": "UBS alocada"},
            {"name": "Distância até UBS (km)"},
            {"name": "Classe de acesso"},
            {"name": "Setor crítico"},
            {"name": "Método fallback"},
        ]
    }
    if ubs_filename:
        fields_to_show[ubs_filename] = [
            {"name": "UBS"},
            {"name": "População alocada"},
            {"name": "Setores alocados"},
            {"name": "Distância média (km)"},
            {"name": "Distância máxima (km)"},
            {"name": "Setores críticos"},
            {"name": "População crítica"},
            {"name": "Classe da UBS"},
            {"name": "Prioridade UBS"},
            {"name": "ubs_critica"},
        ]
    if poly_filename:
        fields_to_show[poly_filename] = [
            {"name": "nome_municipio"},
            {"name": "sigla_uf"},
            {"name": "id_municipio"},
        ]

    filters = [
        {
            "dataId": [csv_filename],
            "id": _uid(),
            "name": ["Classe de acesso"],
            "type": "multiSelect",
            "value": ["Adequado (≤ 2 km)", "Atenção (2-4 km)", "Crítico (> 4 km)", "Sem distância"],
            "enlarged": True,
            "plotType": "histogram",
            "yAxis": None,
        },
        {
            "dataId": [csv_filename],
            "id": _uid(),
            "name": ["Setor crítico"],
            "type": "multiSelect",
            "value": ["Sim", "Não"],
            "enlarged": False,
            "plotType": "histogram",
            "yAxis": None,
        },
    ]
    if ubs_filename:
        filters.extend([
            {
                "dataId": [ubs_filename],
                "id": _uid(),
                "name": ["Classe da UBS"],
                "type": "multiSelect",
                "value": ["Adequado (≤ 2 km)", "Atenção (2-4 km)", "Crítico (> 4 km)", "Sem distância"],
                "enlarged": False,
                "plotType": "histogram",
                "yAxis": None,
            },
            {
                "dataId": [ubs_filename],
                "id": _uid(),
                "name": ["ubs_critica"],
                "type": "multiSelect",
                "value": ["Sim", "Não"],
                "enlarged": False,
                "plotType": "histogram",
                "yAxis": None,
            },
        ])

    return {
        "version": "v1",
        "config": {
            "visState": {
                "filters": filters,
                "layers": layers,
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": fields_to_show,
                        "enabled": True,
                    }
                },
                "layerBlending": "normal",
                "splitMaps": [],
            },
            # [CONFIGURAÇÃO DE VISUALIZAÇÃO AQUI]
            "mapState": {
                "bearing": 0,       
                "dragRotate": True,
                "latitude": round(center_lat, 6),
                "longitude": round(center_lon, 6),
                "pitch": 45,       
                "zoom": zoom,       
                "isSplit": False,
            },
            "mapStyle": {"styleType": "dark"},
        },
    }


def _upload_to_frontend(
    map_id: str,
    csv_path: str,
    cfg_path: str,
    poly_path: str = None,
    ubs_path: str = None,
) -> str | None:
    """Envia CSV, JSON e camadas auxiliares opcionais ao endpoint /api/upload_map."""
    if not FRONTEND_UPLOAD_URL:
        logger.info("FRONTEND_UPLOAD_URL não definido – pulando upload.")
        return None
    logger.info("Enviando mapa para front‑end: %s", FRONTEND_UPLOAD_URL)
    
    open_files = []
    
    try:
        data_payload = {
            "map_id": map_id
        }
        
        f_csv = open(csv_path, "rb")
        open_files.append(f_csv)
        
        f_cfg = open(cfg_path, "rb")
        open_files.append(f_cfg)
        
        files_payload = {
            "csv_file": ("map.csv", f_csv, "text/csv"),
            "cfg_file": ("map.json", f_cfg, "application/json"),
        }
        
        if poly_path and os.path.exists(poly_path):
            f_poly = open(poly_path, "rb")
            open_files.append(f_poly)
            files_payload["poly_file"] = (os.path.basename(poly_path), f_poly, "text/csv")
            logger.info(f"Incluindo arquivo de polígono no upload: {poly_path}")

        if ubs_path and os.path.exists(ubs_path):
            f_ubs = open(ubs_path, "rb")
            open_files.append(f_ubs)
            files_payload["ubs_file"] = (os.path.basename(ubs_path), f_ubs, "text/csv")
            logger.info(f"Incluindo arquivo agregado de UBS no upload: {ubs_path}")

        headers = {}
        if MAP_UPLOAD_TOKEN:
            headers["X-Upload-Token"] = MAP_UPLOAD_TOKEN

        resp = requests.post(FRONTEND_UPLOAD_URL, data=data_payload, files=files_payload, headers=headers, timeout=30)
        resp.raise_for_status()
        
        link = resp.json().get("link")
        logger.info("Upload concluído – link recebido: %s", link)
        return link
        
    except Exception as e:
        logger.error("Falha no upload: %s", e, exc_info=True)
        return None
    finally:
        for f in open_files:
            try:
                f.close()
            except Exception:
                pass

@router.get("/ufs")
def get_ufs():
    """Obtém a lista de UFs dinamicamente listando os diretórios."""
    try:
        if not os.path.exists(DEMANDS_BASE_DIR):
             raise FileNotFoundError("Diretório base de demandas não encontrado.")
        ufs = [d for d in os.listdir(DEMANDS_BASE_DIR) if os.path.isdir(os.path.join(DEMANDS_BASE_DIR, d)) and len(d) == 2]
        return sorted(ufs)
    except Exception as e:
        logger.error("Erro ao listar UFs do diretório: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Não foi possível carregar a lista de estados: {e}"})


@router.get("/municipios")
def get_municipios(uf: str = Query("")):
    """Obtém a lista de municípios dinamicamente listando os arquivos."""
    if not uf:
        return []
    try:
        uf_path = os.path.join(DEMANDS_BASE_DIR, uf.upper())
        if not os.path.isdir(uf_path):
            return [] 
        
        files = [f for f in os.listdir(uf_path) if f.lower().endswith('.geojson')]
        cities = [os.path.splitext(f)[0].replace('_', ' ') for f in files]
        return sorted(cities)
    except Exception as e:
        logger.error("Erro ao listar municípios para a UF '%s': %s", uf, e)
        return JSONResponse(status_code=500, content={"error": f"Não foi possível carregar os municípios para {uf}: {e}"})

# ------------------------------------------------------------------ #
#  Rota principal                                                    #
# ------------------------------------------------------------------ #
@router.get("/resultado_completo")
def consulta_completa(uf: str, municipio: str, tipo: str = Query("pysal")):
    try:
        logger.info("Consulta UF=%s | Mun=%s | Tipo=%s", uf, municipio, tipo)

        # ---------- 1. prepara dados ----------
        demands_file_path = _resolve_demands_file_path(uf, municipio)
        logger.info("Arquivo de demanda selecionado: %s", demands_file_path)

        class _Buf:
            def __init__(self, f): self.file = f

        with open(demands_file_path, "rb") as dem_f, open(OPPORTUNITIES_PATH, "rb") as opp_f:
            err, demands_gdf, opps_gdf, col_did, col_name, col_city, col_state_opp, _ = prepare_data(
                _Buf(opp_f), _Buf(dem_f), state=uf, city=municipio
            )
        
        if err:
            return JSONResponse(status_code=500, content={"erro": err})
        if demands_gdf.empty or opps_gdf.empty:
            return JSONResponse(status_code=404, content={"erro": "Sem dados suficientes para a localidade."})

        # ---------- 2. KNN + resumo ----------
        df_knn = allocate_demands_knn(
            demands_gdf, opps_gdf,
            col_did, col_name, col_city, col_state_opp,
            k=1, method=tipo, city_name=municipio, num_threads=num_threads
        )
        allocation_dict, city_summary = analyze_knn_allocation(df_knn, demands_gdf, opps_gdf, settings=settings)

        # ---------- 3. artefatos EDA ----------
        merged_df, summary_ubs = analyze_allocation(df_knn, demands_gdf)
        perguntas_guiadas       = gerar_perguntas_respostas(summary_ubs, merged_df)

        chart_pop_buf, chart_racial_buf = create_allocation_charts(summary_ubs)
        coverage_stats_df = create_coverage_stats(merged_df)
        hist_buf    = create_distance_hist(merged_df)
        boxplot_buf = create_distance_boxplot(merged_df)
        resumo_df   = create_summary_table(summary_ubs)
        resumo_img_buf = save_summary_table_image(resumo_df)
        pdf_buf = generate_allocation_pdf(summary_ubs, merged_df)

        # ---------- 4. ZIP em cache ----------
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("allocation_result.csv", df_knn.to_csv(index=False))
            z.writestr("allocation_merged.json", merged_df.to_json(orient="records", force_ascii=False))
            z.writestr("city_summary.json", json.dumps(convert_numpy_types(city_summary)))
            z.writestr("ubs_summary.csv", summary_ubs.to_csv(index=False))
            z.writestr("chart_population.png", chart_pop_buf.getvalue())
            z.writestr("chart_racial.png", chart_racial_buf.getvalue())
            z.writestr("table_indicadores.png", resumo_img_buf.getvalue())
            z.writestr("report.pdf", pdf_buf.getvalue())
            if not coverage_stats_df.empty:
                z.writestr("coverage_stats.csv", coverage_stats_df.to_csv(index=False))
            if hist_buf:
                z.writestr("distance_hist.png", hist_buf.getvalue())
            if boxplot_buf:
                z.writestr("distance_boxplot.png", boxplot_buf.getvalue())
        zip_buffer.seek(0)

        cache_key = f"{uf}_{municipio}_{tipo}"
        _clean_zip_cache()
        ZIP_CACHE[cache_key] = (datetime.utcnow(), zip_buffer.getvalue())

        # ---------- 5. gera CSV + JSON para o mapa ----------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_id    = f"knn_{timestamp}"
        csv_file  = f"{map_id}.csv"
        cfg_file  = f"{map_id}.json"
        ubs_file  = f"ubs_{map_id}.csv"
        
        poly_file = f"poly_{map_id}.csv"

        tmpdir   = tempfile.mkdtemp(prefix="vis_upload_")
        csv_path = os.path.join(tmpdir, csv_file)
        cfg_path = os.path.join(tmpdir, cfg_file)
        ubs_path = os.path.join(tmpdir, ubs_file)

        map_df = _build_map_allocation_df(merged_df, df_knn)
        ubs_map_df = _build_ubs_map_df(merged_df)
        map_df.to_csv(csv_path, index=False)
        has_ubs_layer = not ubs_map_df.empty
        if has_ubs_layer:
            ubs_map_df.to_csv(ubs_path, index=False)
            logger.info("Camada agregada de UBS criada com %d registros.", len(ubs_map_df))

        # Lógica do Polígono
        data_root = os.path.join(BACK_ROOT, "data")
        source_poly_path = get_polygon_path(data_root, uf, municipio)
        
        has_poly = False
        dest_poly_path = None
        if source_poly_path:
            dest_poly_path = os.path.join(tmpdir, poly_file)
            shutil.copy(source_poly_path, dest_poly_path)
            has_poly = True
            logger.info("Polígono copiado com sucesso para diretório temporário.")
        try:
            center_lat, center_lon, dynamic_zoom = calculate_optimal_view(demands_gdf)
        except Exception as e:
            logger.warning(f"Erro ao calcular zoom dinâmico: {e}. Usando padrão.")
            center_lat = df_knn["Origin_Lat"].mean()
            center_lon = df_knn["Origin_Lon"].mean()
            dynamic_zoom = 10

        kepler_cfg = build_kepler_config(
            csv_filename=csv_file, 
            center_lat=center_lat, 
            center_lon=center_lon,
            poly_filename=poly_file if has_poly else None,
            ubs_filename=ubs_file if has_ubs_layer else None,
            zoom=dynamic_zoom 
        )
        
        kepler_cfg["label"] = f"Alocação – {municipio}/{uf}"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(kepler_cfg, f, ensure_ascii=False, indent=2)

        # ---------- 6. Upload ----------
        map_link = _upload_to_frontend(
            map_id,
            csv_path,
            cfg_path,
            poly_path=dest_poly_path if has_poly else None,
            ubs_path=ubs_path if has_ubs_layer else None,
        )

        if map_link:
            try:
                shutil.rmtree(tmpdir)
            except OSError as e:
                logger.warning(f"Não foi possível remover o diretório temporário {tmpdir}: {e}")
        else:
            logger.info(f"Modo legado: movendo arquivos para o diretório compartilhado: {SHARED_DIR}")
            shutil.move(csv_path, os.path.join(FRONT_DATA_DIR, csv_file))
            shutil.move(cfg_path, os.path.join(FRONT_CONFIG_DIR, cfg_file))
            if has_ubs_layer:
                shutil.move(ubs_path, os.path.join(FRONT_DATA_DIR, ubs_file))
                logger.info(f"Arquivo agregado de UBS movido para {FRONT_DATA_DIR}/{ubs_file}")
            
            if has_poly and dest_poly_path:
                shutil.move(dest_poly_path, os.path.join(FRONT_DATA_DIR, poly_file))
                logger.info(f"Arquivo de polígono movido para {FRONT_DATA_DIR}/{poly_file}")

            try:
                os.rmdir(tmpdir)
            except OSError as e:
                logger.warning(f"Não foi possível remover o diretório temporário vazio {tmpdir}: {e}")
            
            map_link = f"/map/{map_id}"

        # ---------- 7. resposta ----------
        return {
            "alocacao":  convert_numpy_types(allocation_dict),
            "eda":       convert_numpy_types(city_summary),
            "info":      {"UF": uf, "Município": municipio, "Distância": tipo},
            "map_url":   map_link,
            "perguntas": perguntas_guiadas,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Erro inesperado em /consulta_base/resultado_completo")
        return JSONResponse(status_code=500, content={"erro": str(e)})


# ------------------------------------------------------------------ #
#  Download do ZIP                                                   #
# ------------------------------------------------------------------ #
@router.get("/download_zip")
def download_zip(uf: str, municipio: str, tipo: str = "pysal"):
    cache_key = f"{uf}_{municipio}_{tipo}"
    cached = ZIP_CACHE.get(cache_key)
    if not cached:
        return JSONResponse(status_code=404, content={"erro": "ZIP não disponível. Execute a consulta primeiro."})

    _clean_zip_cache()
    _, zip_bytes = cached
    headers = {"Content-Disposition": f"attachment; filename=alocacao_{cache_key}.zip"}
    return StreamingResponse(BytesIO(zip_bytes), media_type="application/zip", headers=headers)


@router.get("/download_pdf")
def download_zip(uf: str, municipio: str, tipo: str = "pysal"):
    cache_key = f"{uf}_{municipio}_{tipo}"
    cached = ZIP_CACHE.get(cache_key)
    
    if not cached:
        return JSONResponse(status_code=404, content={"erro": "Relatório PDF não disponível. Execute a consulta primeiro."})

    _, zip_bytes = cached
    
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
            if "report.pdf" not in z.namelist():
                return JSONResponse(status_code=404, content={"erro": "O arquivo report.pdf não foi encontrado dentro do pacote."})
            pdf_data = z.read("report.pdf")

        headers = {"Content-Disposition": f"attachment; filename=relatorio_{cache_key}.pdf"}
        return StreamingResponse(BytesIO(pdf_data), media_type="application/pdf", headers=headers)
        
    except Exception as e:
        logger.error(f"Erro ao extrair PDF do cache: {e}")
        return JSONResponse(status_code=500, content={"erro": "Erro ao processar o arquivo PDF."})
