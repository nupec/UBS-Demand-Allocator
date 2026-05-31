from io import BytesIO
import textwrap
from datetime import datetime

# Importações de Análise de Dados e Geo
import logging
import pandas as pd
import geopandas as gpd
import numpy as np

# Importações de Plotagem (Gráficos)
import matplotlib.pyplot as plt
from pandas.plotting import table
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator

# Importações de Geração de PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from app.config import settings
from app.preprocessing.utils import infer_column

logger = logging.getLogger(__name__)


BRAND = colors.HexColor("#1F4E5F")
MUTED = colors.HexColor("#667085")
LIGHT_BG = colors.HexColor("#F3F7F7")
GRID = colors.HexColor("#D7E0E0")
TEXT = colors.HexColor("#1F2933")
VULNERABILITY_COLUMNS = ["pct_negros", "pct_pardos", "pct_indigenas"]
MAX_LISTED_NAMES = 8
MAX_MAP_LABELS = 12
MAX_MAP_ARCS = 350


def _safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_int(value) -> str:
    return f"{int(round(_safe_float(value))):,}".replace(",", ".")


def _fmt_km(value) -> str:
    return f"{_safe_float(value):.2f} km".replace(".", ",")


def _fmt_pct(value) -> str:
    return f"{_safe_float(value):.1f}%".replace(".", ",")


def _make_pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverKicker",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#D67B45"),
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=BRAND,
            alignment=TA_LEFT,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverBody",
            parent=styles["BodyText"],
            fontSize=11,
            leading=16,
            textColor=TEXT,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TocItemTitle",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=TEXT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TocItemText",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=BRAND,
            alignment=TA_LEFT,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["BodyText"],
            fontSize=11,
            leading=16,
            textColor=MUTED,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=BRAND,
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            textColor=TEXT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontSize=8.8,
            leading=12,
            textColor=TEXT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Warning",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=12,
            textColor=colors.HexColor("#92400E"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricValue",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            textColor=BRAND,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontSize=7.6,
            leading=9,
            textColor=TEXT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["TableCell"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )
    return styles


def _draw_page_footer(canvas_obj, doc):
    if doc.page == 1:
        return

    canvas_obj.saveState()
    page_width, _ = letter
    canvas_obj.setStrokeColor(GRID)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(doc.leftMargin, 0.48 * inch, page_width - doc.rightMargin, 0.48 * inch)
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.setFillColor(MUTED)
    canvas_obj.drawString(doc.leftMargin, 0.32 * inch, "Relatório de alocação UBS")
    canvas_obj.drawRightString(page_width - doc.rightMargin, 0.32 * inch, f"Página {doc.page}")
    canvas_obj.restoreState()


def _image_flowable(buffer: BytesIO, width: float, height: float) -> Image:
    buffer.seek(0)
    image = Image(buffer, width=width, height=height)
    image.hAlign = "CENTER"
    return image


def _fallback_used(merged_df: pd.DataFrame) -> bool:
    if merged_df.empty or "fallback_used" not in merged_df.columns:
        return False

    flags = merged_df["fallback_used"].astype(str).str.lower()
    return flags.isin(["true", "1", "sim", "yes"]).any()


def _format_name_list(values, limit: int = MAX_LISTED_NAMES) -> str:
    names = []
    for value in values:
        name = str(value).strip()
        if not name or name.lower() in {"nan", "none"} or name == "-":
            continue
        names.append(name)

    if not names:
        return "sem UBS identificadas"

    visible = names[:limit]
    text = ", ".join(visible)
    hidden_count = len(names) - len(visible)
    if hidden_count > 0:
        text += f" e mais {hidden_count}"
    return text


def _callout_table(text: str, styles, warning: bool = False) -> Table:
    style_name = "Warning" if warning else "Callout"
    background = colors.HexColor("#FFF7ED") if warning else LIGHT_BG
    border = colors.HexColor("#FDBA74") if warning else GRID
    table = Table([[Paragraph(text, styles[style_name])]], colWidths=[7.05 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    table.hAlign = "LEFT"
    return table


def _build_cover_page(summary: pd.DataFrame, merged_df: pd.DataFrame, city_name: str, generated_at: str, styles) -> list:
    fallback = _fallback_used(merged_df)
    title = "Relatório de Alocação UBS"
    intro = (
        "O projeto IPSUM apoia o planejamento territorial da Atenção Primária à Saúde, "
        "integrando setores de demanda, estabelecimentos de saúde e métricas de acesso "
        "para transformar dados geográficos em sinais operacionais."
    )
    scope = (
        "Este documento resume a distribuição da população entre UBS, identifica cargas "
        "assistenciais, distâncias de acesso e recortes socioeconômicos úteis para priorização "
        "de ações locais."
    )

    flowables = [
        Spacer(1, 0.35 * inch),
        Paragraph("PROJETO IPSUM", styles["CoverKicker"]),
        Paragraph(title, styles["CoverTitle"]),
        Paragraph(f"<b>Município analisado:</b> {city_name}", styles["CoverBody"]),
        Spacer(1, 0.12 * inch),
        _build_metrics_table(summary, merged_df, styles),
        Spacer(1, 0.28 * inch),
        Paragraph(intro, styles["CoverBody"]),
        Paragraph(scope, styles["CoverBody"]),
    ]

    if fallback:
        flowables.extend(
            [
                Spacer(1, 0.08 * inch),
                _callout_table(
                    (
                        "Atenção: esta execução usou fallback geodésico em vez de distâncias "
                        "de rede do Valhalla. Os resultados continuam úteis como aproximação, "
                        "mas não devem ser interpretados como deslocamento viário real."
                    ),
                    styles,
                    warning=True,
                ),
            ]
        )

    flowables.extend(
        [
            Spacer(1, 0.4 * inch),
            _callout_table(
                (
                    f"<b>Gerado em:</b> {generated_at}<br/>"
                    "Escopo: alocação KNN de setores de demanda para UBS, com análise de "
                    "cobertura, perfil populacional e gráficos sintéticos para tomada de decisão."
                ),
                styles,
            ),
        ]
    )
    return flowables


def _build_summary_page(summary: pd.DataFrame, merged_df: pd.DataFrame, styles) -> list:
    total_population = summary["total_population"].sum() if "total_population" in summary else 0
    total_ubs = summary["opportunity_name"].nunique() if "opportunity_name" in summary else 0
    dist = _distance_stats(merged_df)

    items = [
        (
            "1. Visão geral",
            "Indicadores principais, escopo da análise e sinais de atenção para o município.",
        ),
        (
            "2. Leitura executiva",
            "Principais achados sobre demanda, distâncias e perfil socioeconômico.",
        ),
        (
            "3. Resumo por UBS",
            "Tabela comparativa com população alocada, setores, distância média, distância máxima, PPI e analfabetismo.",
        ),
        (
            "4. Mapa da alocação",
            "Visão estática dos vínculos origem-destino, com destaque limitado para preservar legibilidade.",
        ),
        (
            "5. Gráficos de distribuição",
            "Histograma e boxplot das distâncias até a UBS alocada.",
        ),
        (
            "6. Comparativos por UBS",
            "Ranking de maior demanda e composição racial das UBS com maior população alocada.",
        ),
        (
            "7. Perguntas-guia",
            "Respostas objetivas para apoiar investigação local e priorização operacional.",
        ),
    ]

    rows = []
    for title, description in items:
        rows.append([Paragraph(title, styles["TocItemTitle"]), Paragraph(description, styles["TocItemText"])])

    toc = Table(rows, colWidths=[1.75 * inch, 5.25 * inch])
    toc.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FBFB")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    toc.hAlign = "LEFT"

    flowables = [
        Paragraph("Sumário", styles["ReportTitle"]),
        Paragraph(
            (
                "Este sumário organiza a leitura do relatório. Em municípios grandes, os quadros "
                "priorizam síntese visual e rankings para evitar que gráficos fiquem poluídos."
            ),
            styles["ReportSubtitle"],
        ),
        toc,
        Spacer(1, 0.24 * inch),
        _callout_table(
            (
                f"<b>Resumo rápido:</b> {_fmt_int(total_population)} pessoas, "
                f"{_fmt_int(total_ubs)} UBS, média de {_fmt_km(dist['mean'])} até a UBS alocada "
                f"e {_fmt_int(dist['above_4_population'])} pessoas acima de 4 km."
            ),
            styles,
        ),
    ]

    if _fallback_used(merged_df):
        flowables.extend(
            [
                Spacer(1, 0.12 * inch),
                _callout_table(
                    (
                        "Nota metodológica: esta versão do relatório foi produzida com fallback "
                        "geodésico. Após o ajuste do limite de matriz do Valhalla, gere novamente "
                        "o relatório para obter distâncias de rede viária."
                    ),
                    styles,
                    warning=True,
                ),
            ]
        )

    return flowables


def _distance_axis_limit(distances: pd.Series, quantile: float = 0.98, minimum: float = 4.5) -> float:
    if distances.empty:
        return minimum
    percentile = _safe_float(distances.quantile(quantile), minimum)
    maximum = _safe_float(distances.max(), minimum)
    return min(max(max(percentile * 1.2, minimum), 1.0), max(maximum, minimum))


def _distance_stats(merged_df: pd.DataFrame) -> dict:
    if merged_df.empty or "distance_km" not in merged_df.columns:
        return {"mean": 0, "median": 0, "max": 0, "above_4_count": 0, "above_4_population": 0}

    distances = pd.to_numeric(merged_df["distance_km"], errors="coerce").dropna()
    above_4 = merged_df[pd.to_numeric(merged_df["distance_km"], errors="coerce") > 4]
    above_4_population = above_4["DEMANDA"].sum() if "DEMANDA" in above_4.columns else len(above_4)

    return {
        "mean": distances.mean() if not distances.empty else 0,
        "median": distances.median() if not distances.empty else 0,
        "max": distances.max() if not distances.empty else 0,
        "above_4_count": len(above_4),
        "above_4_population": above_4_population,
    }


def _build_executive_findings(summary: pd.DataFrame, merged_df: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["Não há dados suficientes para gerar a leitura executiva."]

    total_population = summary["total_population"].sum() if "total_population" in summary else 0
    busiest = summary.loc[summary["total_population"].idxmax()]
    farthest = summary.loc[summary["avg_distance"].idxmax()]
    dist = _distance_stats(merged_df)
    vulnerability = summary[VULNERABILITY_COLUMNS].sum(axis=1) if all(col in summary for col in VULNERABILITY_COLUMNS) else pd.Series([0] * len(summary))
    most_vulnerable_idx = vulnerability.idxmax() if len(vulnerability) else None
    most_vulnerable = summary.loc[most_vulnerable_idx] if most_vulnerable_idx is not None else None

    busiest_share = (_safe_float(busiest["total_population"]) / total_population * 100) if total_population else 0
    findings = [
        (
            f"A maior concentração de demanda está em <b>{busiest['opportunity_name']}</b>, "
            f"com {_fmt_int(busiest['total_population'])} pessoas alocadas "
            f"({_fmt_pct(busiest_share)} do total analisado)."
        ),
        (
            f"A UBS com maior distância média é <b>{farthest['opportunity_name']}</b>, "
            f"com {_fmt_km(farthest['avg_distance'])}. A média geral ficou em {_fmt_km(dist['mean'])} "
            f"e a mediana em {_fmt_km(dist['median'])}."
        ),
        (
            f"Foram identificados {_fmt_int(dist['above_4_count'])} setores acima do raio de 4 km, "
            f"representando aproximadamente {_fmt_int(dist['above_4_population'])} pessoas."
        ),
    ]

    if most_vulnerable is not None:
        vuln_pct = vulnerability.loc[most_vulnerable_idx]
        findings.append(
            f"A maior proporção combinada de população preta, parda e indígena aparece em "
            f"<b>{most_vulnerable['opportunity_name']}</b>: {_fmt_pct(vuln_pct)}."
        )

    return findings


def _build_metrics_table(summary: pd.DataFrame, merged_df: pd.DataFrame, styles) -> Table:
    total_population = summary["total_population"].sum() if "total_population" in summary else 0
    total_ubs = summary["opportunity_name"].nunique() if "opportunity_name" in summary else 0
    total_sectors = len(merged_df)
    dist = _distance_stats(merged_df)

    cards = [
        ("População alocada", _fmt_int(total_population)),
        ("UBS analisadas", _fmt_int(total_ubs)),
        ("Setores analisados", _fmt_int(total_sectors)),
        ("Distância média", _fmt_km(dist["mean"])),
        ("Pessoas acima de 4 km", _fmt_int(dist["above_4_population"])),
    ]
    row = [
        [
            Paragraph(value, styles["MetricValue"]),
            Paragraph(label, styles["MetricLabel"]),
        ]
        for label, value in cards
    ]
    table = Table([row], colWidths=[1.38 * inch] * len(cards))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _build_ubs_table(summary: pd.DataFrame, merged_df: pd.DataFrame, styles) -> Table:
    coverage = create_coverage_stats(merged_df)
    max_distance_by_ubs = {}
    if not coverage.empty and "max_distance" in coverage:
        max_distance_by_ubs = coverage.set_index("opportunity_name")["max_distance"].to_dict()

    ordered = summary.sort_values("total_population", ascending=False) if "total_population" in summary else summary
    rows = [
        [
            Paragraph("UBS", styles["TableHeader"]),
            Paragraph("Pop.", styles["TableHeader"]),
            Paragraph("Set.", styles["TableHeader"]),
            Paragraph("Dist. média", styles["TableHeader"]),
            Paragraph("Dist. máx.", styles["TableHeader"]),
            Paragraph("PPI", styles["TableHeader"]),
            Paragraph("Analf.", styles["TableHeader"]),
        ]
    ]

    for _, row in ordered.iterrows():
        name = row.get("opportunity_name", "N/A")
        vulnerable_pct = sum(_safe_float(row.get(col, 0)) for col in VULNERABILITY_COLUMNS)
        rows.append(
            [
                Paragraph(str(name), styles["TableCell"]),
                _fmt_int(row.get("total_population", 0)),
                _fmt_int(row.get("total_demands", 0)),
                _fmt_km(row.get("avg_distance", 0)),
                _fmt_km(max_distance_by_ubs.get(name, 0)),
                _fmt_pct(vulnerable_pct),
                _fmt_int(row.get("pessoas_analfabetas", 0)),
            ]
        )

    table = Table(rows, colWidths=[2.65 * inch, 0.62 * inch, 0.42 * inch, 0.72 * inch, 0.72 * inch, 0.58 * inch, 0.58 * inch], repeatRows=1)
    table.hAlign = "LEFT"
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFB")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def analyze_allocation(allocation_df: pd.DataFrame, demanda_gdf: gpd.GeoDataFrame):
    logger.info("Iniciando analyze_allocation: convertendo chaves para string.")

    try:
        allocation_df["demand_id"] = allocation_df["demand_id"].astype(str)
        demanda_gdf["CD_SETOR"] = demanda_gdf["CD_SETOR"].astype(str)
    except KeyError as e:
        logger.error(f"Erro de chave em analyze_allocation: {e}. Verifique se 'demand_id' e 'CD_SETOR' existem.")
        return pd.DataFrame(), pd.DataFrame()

    logger.info("Fazendo merge do allocation_df com demanda_gdf (chaves: demand_id <-> CD_SETOR).")
    merged_df = allocation_df.merge(
        demanda_gdf, left_on="demand_id", right_on="CD_SETOR", how="left"
    )
    
    logger.info("Merge concluído. merged_df shape: %s", merged_df.shape)

    # Evitar problemas de serialização com geometria
    if "geometry" in merged_df.columns:
        logger.info("Convertendo geometria para WKT para evitar recursion issues.")
        merged_df["geometry"] = merged_df["geometry"].apply(lambda geom: geom.wkt if geom is not None else None)

    logger.info("Agrupando dados por 'opportunity_name' e calculando estatísticas.")

    try:
        merged_df["total_alfabetizados"] = merged_df[
        ["15 A 19 ANOS, ALFABETIZADAS",
         "20 A 24 ANOS, ALFABETIZADAS",
         "25 A 29 ANOS, ALFABETIZADAS",
         "30 A 34 ANOS, ALFABETIZADAS",
         "35 A 39 ANOS, ALFABETIZADAS",
         "40 A 44 ANOS, ALFABETIZADAS",
         "45 A 49 ANOS, ALFABETIZADAS",
         "50 A 54 ANOS, ALFABETIZADAS",
         "55 A 59 ANOS, ALFABETIZADAS",
         "60 A 64 ANOS, ALFABETIZADAS",
         "65 A 69 ANOS, ALFABETIZADAS",
         "70 A 79 ANOS, ALFABETIZADAS",
         "80 ANOS OU MAIS, ALFABETIZADAS"]].sum(axis=1)
    except KeyError:
        logger.warning("Não foi possível calcular 'total_alfabetizados'. Colunas não encontradas. Definindo como 0.")
        merged_df["total_alfabetizados"] = 0

    group = merged_df.groupby("opportunity_name")
    
    # Lista de agregações dinâmicas para colunas que podem ou não existir
    agg_dict = {
        "total_demands": ("demand_id", "count"),
        "avg_distance": ("distance_km", "mean"),
        "total_population": ("DEMANDA", "sum"),
        "total_negros": ("RAÇA NEGRA TOTAL", "sum"),
        "total_pardos": ("RAÇA PARDA TOTAL", "sum"),
        "total_indigenas": ("RAÇA INDÍGENA TOTAL", "sum"),
        "total_amarela": ("RAÇA AMARELA TOTAL", "sum"),
        "total_15_19":("15-19 ANOS", "sum"),
        "total_20_24":("20-24 ANOS", "sum"),
        "total_25_29":("25-29 ANOS", "sum"),
        "total_30_34":("30-34 ANOS", "sum"),
        "total_35_39":("35-39 ANOS", "sum"),
        "total_40_44":("40-44 ANOS", "sum"),
        "total_45_49":("45-49 ANOS", "sum"),
        "total_50_54":("50-54 ANOS", "sum"),
        "total_55_59":("55-59 ANOS", "sum"),
        "total_60_64":("60-64 ANOS", "sum"),
        "total_65_69":("65-69 ANOS", "sum"),
        "total_70_79":("70-79 ANOS", "sum"),
        "total_80_mais":("80 ANOS OU MAIS", "sum"),
        "total_alfabetizados":("total_alfabetizados", "sum")
    }

    # Filtra o dicionário de agregação para incluir apenas colunas que existem no merged_df
    valid_agg_dict = {}
    for key, (col, agg_func) in agg_dict.items():
        if col in merged_df.columns:
            valid_agg_dict[key] = (col, agg_func)
        else:
            logger.warning(f"Coluna de agregação '{col}' (para '{key}') não encontrada. Será ignorada.")

    if not valid_agg_dict:
        logger.error("Nenhuma coluna de agregação válida encontrada. Encerrando a análise.")
        return pd.DataFrame(), pd.DataFrame()
        
    summary = group.agg(**valid_agg_dict).reset_index()

    logger.debug("Resumo de estatísticas (primeiras linhas):\n%s", summary.head())

    # Calcula percentuais de cada grupo, se houver população registrada
    logger.info("Calculando percentuais de grupos raciais.")
    summary["pct_negros"] = (summary["total_negros"] / summary["total_population"] * 100) if "total_negros" in summary and "total_population" in summary else 0
    summary["pct_pardos"] = (summary["total_pardos"] / summary["total_population"] * 100) if "total_pardos" in summary and "total_population" in summary else 0
    summary["pct_indigenas"] = (summary["total_indigenas"] / summary["total_population"] * 100) if "total_indigenas" in summary and "total_population" in summary else 0
    summary["pct_amarela"] = (summary["total_amarela"] / summary["total_population"] * 100) if "total_amarela" in summary and "total_population" in summary else 0
    summary["pessoas_analfabetas"] = (summary.get('total_population', 0) - summary.get("total_alfabetizados", 0))

    logger.info("Criando colunas agregadas de faixa etária.")
    summary["total_15_29_anos"] = (summary.get("total_15_19", 0) + summary.get("total_20_24", 0) + summary.get("total_25_29", 0))
    summary["total_30_49_anos"] = (summary.get("total_30_34", 0) + summary.get("total_35_39", 0) + summary.get("total_40_44", 0) + summary.get("total_45_49", 0))
    summary["total_50_64_anos"] = (summary.get("total_50_54", 0) + summary.get("total_55_59", 0) + summary.get("total_60_64", 0))
    summary["total_65_mais_anos"] = (summary.get("total_65_69", 0) + summary.get("total_70_79", 0) + summary.get("total_80_mais", 0))
    
    # Tenta extrair o nome da cidade a partir dos aliases configurados.
    city_column = infer_column(merged_df, settings.CITY_POSSIBLE_COLUMNS)
    if city_column:
        summary['city_name'] = merged_df[city_column].iloc[0] if not merged_df.empty else "N/A"
    else:
        logger.warning("Coluna de cidade não encontrada no merged_df. 'city_name' será 'N/A'.")
        summary['city_name'] = "N/A"

    logger.info("analyze_allocation concluído.")
    return merged_df, summary

def create_allocation_charts(summary: pd.DataFrame):
    logger.info("Iniciando criação dos gráficos de alocação.")
    
    if summary.empty:
        logger.warning("DataFrame de sumário vazio. Não é possível criar gráficos.")
        return BytesIO(), BytesIO()

    plt.style.use("default")

    # Gráfico 1: Top UBS por população atendida
    fig1, ax1 = plt.subplots(figsize=(11, 6.2))
    top_opp = summary.sort_values(by="total_population", ascending=True).tail(10).copy()
    labels = [textwrap.fill(str(name), width=30) for name in top_opp["opportunity_name"]]
    bars1 = ax1.barh(
        labels,
        top_opp["total_population"],
        color="#2F6F73",
        edgecolor="#1F4E5F",
        linewidth=0.8,
    )

    ax1.set_title("UBS por população alocada", fontsize=15, fontweight="bold", color="#1F2933", loc="left")
    ax1.set_xlabel("Pessoas alocadas", fontsize=10)
    ax1.set_ylabel("")
    ax1.tick_params(axis="both", labelsize=9)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.set_axisbelow(True)

    for bar in bars1:
        width = bar.get_width()
        ax1.annotate(
            _fmt_int(width),
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="#1F2933",
        )

    buf1 = BytesIO()
    fig1.tight_layout()
    fig1.savefig(buf1, format="png", dpi=180, bbox_inches="tight")
    buf1.seek(0)
    plt.close(fig1)
    logger.info("Gráfico 1 (População) criado com sucesso.")

    # Gráfico 2: composição racial por UBS em barras empilhadas.
    fig2, ax2 = plt.subplots(figsize=(11, 6.2))
    racial = summary.sort_values(by="total_population", ascending=True).tail(10).copy()
    labels = [textwrap.fill(str(name), width=28) for name in racial["opportunity_name"]]
    components = [
        ("Preta", "pct_negros", "#335C67"),
        ("Parda", "pct_pardos", "#D98E4A"),
        ("Indígena", "pct_indigenas", "#4F772D"),
        ("Amarela", "pct_amarela", "#E0B84F"),
    ]
    left = np.zeros(len(racial))
    for label, column, color in components:
        values = pd.to_numeric(racial.get(column, 0), errors="coerce").fillna(0).to_numpy()
        ax2.barh(labels, values, left=left, label=label, color=color)
        left += values

    others = np.clip(100 - left, 0, 100)
    ax2.barh(labels, others, left=left, label="Demais grupos", color="#D9DEE5")
    ax2.set_title("Composição racial da população alocada por UBS", fontsize=15, fontweight="bold", color="#1F2933", loc="left")
    ax2.set_xlabel("Percentual da população alocada", fontsize=10)
    ax2.set_ylabel("")
    ax2.set_xlim(0, 100)
    ax2.tick_params(axis="both", labelsize=8.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.25)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8)

    buf2 = BytesIO()
    fig2.tight_layout()
    fig2.savefig(buf2, format="png", dpi=180, bbox_inches="tight")
    buf2.seek(0)
    plt.close(fig2)
    logger.info("Gráfico 2 (Composição Racial) criado com sucesso.")

    logger.info("Criação de gráficos concluída.")
    return buf1, buf2

def create_coverage_stats(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Exemplo de estatísticas adicionais sobre a cobertura, se desejar.
    Agrupa por 'opportunity_name' e gera min, max, etc. da 'distance_km'.
    """
    logger.info("Gerando estatísticas de cobertura (coverage_stats).")
    if "distance_km" not in merged_df.columns:
        logger.warning("Não há coluna 'distance_km' no merged_df; coverage_stats será vazio.")
        return pd.DataFrame()

    group_opp = merged_df.groupby("opportunity_name")

    coverage_stats = group_opp.agg(
        total_demands=("demand_id", "count"),
        avg_distance=("distance_km", "mean"),
        max_distance=("distance_km", "max"),
        min_distance=("distance_km", "min")
    ).reset_index()

    logger.debug("coverage_stats:\n%s", coverage_stats.head())
    return coverage_stats

def create_distance_boxplot(merged_df: pd.DataFrame):
    """
    Cria um boxplot de 'distance_km' para visualizar a dispersão e possíveis outliers nas distâncias.
    """
    logger.info("Gerando boxplot de 'distance_km'.")
    if "distance_km" not in merged_df.columns:
        logger.warning("Não há coluna 'distance_km' no merged_df; não será gerado boxplot.")
        return None

    distances = pd.to_numeric(merged_df["distance_km"], errors="coerce").dropna()
    if distances.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 3.4))
    axis_limit = _distance_axis_limit(distances, quantile=0.98)
    max_distance = distances.max()
    ax.boxplot(
        distances,
        vert=False,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "#DDEEEF", "color": "#1F4E5F"},
        medianprops={"color": "#D67B45", "linewidth": 2},
        whiskerprops={"color": "#1F4E5F"},
        capprops={"color": "#1F4E5F"},
        flierprops={"marker": "o", "markerfacecolor": "#D67B45", "markeredgecolor": "#D67B45", "markersize": 4},
    )
    ax.axvline(4, color="#B42318", linestyle="--", linewidth=1.2, label="Raio de referência: 4 km")
    ax.set_title("Dispersão das distâncias até a UBS alocada", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel("Distância (km)")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    if max_distance > axis_limit:
        ax.set_xlim(left=0, right=axis_limit)
        ax.text(
            0.99,
            0.15,
            f"Eixo limitado ao P98. Máximo observado: {_fmt_km(max_distance)}.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#667085",
        )
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def create_distance_hist(merged_df: pd.DataFrame) -> BytesIO:
    """
    Cria um histograma da coluna 'distance_km' utilizando apenas Matplotlib.
    """
    logger.info("Gerando histograma de 'distance_km' com Matplotlib.")

    if "distance_km" not in merged_df.columns:
        logger.warning("Não há coluna 'distance_km' no merged_df; histograma não será gerado.")
        return None

    distances = pd.to_numeric(merged_df["distance_km"], errors="coerce").dropna()
    if distances.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    axis_limit = _distance_axis_limit(distances, quantile=0.98)
    max_distance = distances.max()
    plot_distances = distances.clip(upper=axis_limit) if max_distance > axis_limit else distances

    bins = min(24, max(6, int(np.sqrt(len(plot_distances)))))
    ax.hist(
        plot_distances,
        bins=bins,
        color="#6DA6A8",
        edgecolor="white",
        alpha=0.95,
    )

    mean = distances.mean()
    median = distances.median()
    ax.axvline(mean, color="#1F4E5F", linewidth=2, label=f"Média: {_fmt_km(mean)}")
    ax.axvline(median, color="#D67B45", linewidth=2, linestyle=":", label=f"Mediana: {_fmt_km(median)}")
    ax.axvline(4, color="#B42318", linewidth=1.4, linestyle="--", label="Referência: 4 km")

    ax.set_title("Distribuição das distâncias até UBS", fontsize=14, weight="bold", loc="left")
    ax.set_xlabel("Distância até UBS (km)", fontsize=10)
    ax.set_ylabel("Setores", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(left=0, right=axis_limit)
    locator_step = 1 if axis_limit <= 10 else 5
    ax.xaxis.set_major_locator(MultipleLocator(locator_step))
    if max_distance > axis_limit:
        ax.text(
            0.99,
            0.76,
            f"Valores acima de {_fmt_km(axis_limit)} foram agrupados no último intervalo. Máximo: {_fmt_km(max_distance)}.",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#667085",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D7E0E0", "alpha": 0.9},
        )

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf


def create_allocation_map(merged_df: pd.DataFrame) -> BytesIO | None:
    """Gera uma visão estática dos arcos origem-destino usados no mapa Kepler."""
    required = {"Origin_Lat", "Origin_Lon", "Destination_Lat", "Destination_Lon", "opportunity_name"}
    if merged_df.empty or not required.issubset(set(merged_df.columns)):
        logger.warning("Colunas de coordenadas insuficientes para gerar mapa de alocação.")
        return None

    data = merged_df.copy()
    for col in ["Origin_Lat", "Origin_Lon", "Destination_Lat", "Destination_Lon", "distance_km"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Origin_Lat", "Origin_Lon", "Destination_Lat", "Destination_Lon"])
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    distances = data["distance_km"] if "distance_km" in data else pd.Series([1] * len(data))
    norm = plt.Normalize(vmin=distances.min(), vmax=distances.max() if distances.max() > distances.min() else distances.min() + 1)
    cmap = plt.cm.YlOrRd

    # Limita a quantidade de arcos para preservar legibilidade em municípios grandes.
    plot_data = data.sort_values("distance_km", ascending=False).head(MAX_MAP_ARCS) if "distance_km" in data else data.head(MAX_MAP_ARCS)
    for _, row in plot_data.iterrows():
        color = cmap(norm(_safe_float(row.get("distance_km", 0))))
        arc = FancyArrowPatch(
            (row["Origin_Lon"], row["Origin_Lat"]),
            (row["Destination_Lon"], row["Destination_Lat"]),
            connectionstyle="arc3,rad=0.18",
            arrowstyle="-",
            linewidth=0.75,
            alpha=0.45,
            color=color,
        )
        ax.add_patch(arc)

    size_source = data["DEMANDA"] if "DEMANDA" in data else pd.Series([80] * len(data))
    sizes = 28 + 90 * (size_source / size_source.max()) if size_source.max() else 45
    scatter = ax.scatter(
        data["Origin_Lon"],
        data["Origin_Lat"],
        c=distances,
        cmap=cmap,
        norm=norm,
        s=sizes,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
        label="Setores de demanda",
        zorder=3,
    )

    agg_spec = {"total_demands": ("opportunity_name", "size")}
    if "DEMANDA" in data:
        agg_spec["total_population"] = ("DEMANDA", "sum")
    if "distance_km" in data:
        agg_spec["avg_distance"] = ("distance_km", "mean")

    destinations = (
        data.groupby(["Destination_Lat", "Destination_Lon", "opportunity_name"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
    if "total_population" not in destinations:
        destinations["total_population"] = destinations["total_demands"]
    ax.scatter(
        destinations["Destination_Lon"],
        destinations["Destination_Lat"],
        marker="s",
        s=55,
        color="#1F4E5F",
        edgecolor="white",
        linewidth=0.8,
        label="UBS",
        zorder=4,
    )

    highlighted = destinations.sort_values("total_population", ascending=False).head(MAX_MAP_LABELS)
    legend_lines = []
    for idx, (_, row) in enumerate(highlighted.iterrows(), start=1):
        ax.text(
            row["Destination_Lon"],
            row["Destination_Lat"],
            str(idx),
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        legend_lines.append(
            f"{idx}. {textwrap.shorten(str(row['opportunity_name']), width=34, placeholder='...')}"
        )

    if legend_lines:
        hidden_count = max(len(destinations) - len(highlighted), 0)
        legend_text = "\n".join(legend_lines)
        if hidden_count:
            legend_text += f"\n+ {hidden_count} UBS sem rótulo no mapa"
        ax.text(
            0.01,
            0.02,
            legend_text,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            color="#1F2933",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D7E0E0", "alpha": 0.94},
        )

    ax.set_title("Mapa estático da alocação: setores conectados às UBS", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8)
    ax.text(
        0.99,
        0.02,
        f"Arcos exibidos: {len(plot_data)} maiores distâncias. UBS destacadas: {len(highlighted)} de {len(destinations)}.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#667085",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D7E0E0", "alpha": 0.9},
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Distância até UBS (km)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    lon_values = pd.concat([data["Origin_Lon"], data["Destination_Lon"]])
    lat_values = pd.concat([data["Origin_Lat"], data["Destination_Lat"]])
    lon_pad = max((lon_values.max() - lon_values.min()) * 0.12, 0.01)
    lat_pad = max((lat_values.max() - lat_values.min()) * 0.12, 0.01)
    ax.set_xlim(lon_values.min() - lon_pad, lon_values.max() + lon_pad)
    ax.set_ylim(lat_values.min() - lat_pad, lat_values.max() + lat_pad)

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=190, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def gerar_perguntas_respostas(summary: pd.DataFrame, merged_df: pd.DataFrame):
    perguntas_respostas = [
        (
            "Este relatório é guiado por perguntas de capacidade, acessibilidade e perfil "
            "socioeconômico. As respostas abaixo resumem os principais sinais observados "
            "nos dados processados."
        ),
        "",
    ]
    
    if summary.empty or merged_df.empty:
        logger.warning("DataFrames de sumário ou merge vazios. Perguntas-guia não podem ser geradas.")
        perguntas_respostas.append("DADOS INSUFICIENTES PARA GERAR RESPOSTAS.")
        return perguntas_respostas

    total_population = summary["total_population"].sum() if "total_population" in summary else 0
    dist = _distance_stats(merged_df)

    perguntas_respostas.append("1. Capacidade Instalada vs. Demanda Alocada\n")

    top_ubs = summary.sort_values(by="total_population", ascending=False).head(3)
    perguntas_respostas.append("Pergunta: Quais UBS concentram a maior demanda?")
    top_ubs_list = _format_name_list(top_ubs["opportunity_name"], limit=3)
    perguntas_respostas.append(f"Resposta: As maiores cargas estão em {top_ubs_list}.")

    media_atendimento = summary["total_population"].mean()
    criticas = summary[summary["total_population"] > media_atendimento * 1.5]
    perguntas_respostas.append("Pergunta: Há UBS muito acima da média de atendimento?")
    if not criticas.empty:
        criticas_list = _format_name_list(
            criticas.sort_values("total_population", ascending=False)["opportunity_name"],
            limit=MAX_LISTED_NAMES,
        )
        perguntas_respostas.append(
            f"Resposta: Sim. Foram identificadas {_fmt_int(len(criticas))} UBS; principais: {criticas_list}. "
            "Elas atendem mais de 50% acima da média observada."
        )
    else:
        perguntas_respostas.append("Resposta: Não há UBS acima de 50% da média de atendimento.")

    subutilizadas = summary[summary["total_population"] < media_atendimento * 0.5]
    perguntas_respostas.append("Pergunta: Há UBS com baixa carga relativa?")
    if not subutilizadas.empty:
        subutilizadas_list = _format_name_list(
            subutilizadas.sort_values("total_population", ascending=True)["opportunity_name"],
            limit=MAX_LISTED_NAMES,
        )
        perguntas_respostas.append(
            f"Resposta: Sim. Foram identificadas {_fmt_int(len(subutilizadas))} UBS; principais: {subutilizadas_list}. "
            "Elas estão abaixo de 50% da média populacional alocada."
        )
    else:
        perguntas_respostas.append("Resposta: Todas as UBS estão acima de 50% da média populacional alocada.")

    perguntas_respostas.append("\n")

    perguntas_respostas.append("2. Perfil Socioeconômico da População\n")

    vulneraveis = summary[
        (summary.get("pct_negros", 0) + summary.get("pct_pardos", 0) + summary.get("pct_indigenas", 0)) > 80
    ]
    perguntas_respostas.append("Pergunta: Há UBS atendendo áreas com concentração muito alta de população preta, parda e indígena?")
    if not vulneraveis.empty:
        vulneraveis_list = _format_name_list(vulneraveis["opportunity_name"], limit=MAX_LISTED_NAMES)
        perguntas_respostas.append(
            f"Resposta: Sim. Foram identificadas {_fmt_int(len(vulneraveis))} UBS; principais: {vulneraveis_list}."
        )
    else:
        perguntas_respostas.append("Resposta: Nenhuma UBS ultrapassou 80% nesse indicador combinado.")

    perguntas_respostas.append("Pergunta: Onde consultar a composição por UBS?")
    perguntas_respostas.append("Resposta: A tabela inicial e o gráfico de composição racial mostram a comparação entre UBS.")

    perguntas_respostas.append("Pergunta: Os dados indicam concentração social que merece atenção?")
    if not vulneraveis.empty:
        perguntas_respostas.append(
            "Resposta: Sim. A concentração elevada em algumas UBS sugere necessidade de olhar territorial mais fino."
        )
    else:
        maior_ppi = (summary.get("pct_negros", 0) + summary.get("pct_pardos", 0) + summary.get("pct_indigenas", 0)).max()
        perguntas_respostas.append(f"Resposta: Não há concentração acima de 80%, mas o maior valor observado foi {_fmt_pct(maior_ppi)}.")

    perguntas_respostas.append("\n")

    perguntas_respostas.append("3. Acessibilidade e Proximidade\n")

    perguntas_respostas.append("Pergunta: Qual é a distância média entre setor de demanda e UBS alocada?")
    perguntas_respostas.append(f"Resposta: A distância média geral é {_fmt_km(dist['mean'])}, com mediana de {_fmt_km(dist['median'])}.")

    perguntas_respostas.append("Pergunta: Há setores acima do raio de referência de 4 km?")
    if dist["above_4_count"]:
        perc = 100 * dist["above_4_count"] / len(merged_df)
        perguntas_respostas.append(
            f"Resposta: Sim. {_fmt_int(dist['above_4_count'])} setores ({_fmt_pct(perc)}) estão acima de 4 km, somando aproximadamente {_fmt_int(dist['above_4_population'])} pessoas."
        )
    else:
        perguntas_respostas.append("Resposta: Não há setores acima de 4 km.")

    perguntas_respostas.append("Pergunta: Qual é a dispersão das distâncias?")
    std_distance = pd.to_numeric(merged_df["distance_km"], errors="coerce").std() if "distance_km" in merged_df else 0
    perguntas_respostas.append(
        f"Resposta: O desvio-padrão entre setores é {_fmt_km(std_distance)}, e o maior deslocamento observado é {_fmt_km(dist['max'])}."
    )

    return perguntas_respostas


def generate_allocation_pdf(summary: pd.DataFrame, merged_df: pd.DataFrame):
    logger.info("Iniciando geração do PDF de relatório.")
    pdf_buffer = BytesIO()
    styles = _make_pdf_styles()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=0.48 * inch,
        leftMargin=0.48 * inch,
        topMargin=0.52 * inch,
        bottomMargin=0.62 * inch,
        title="Relatório de Alocação UBS",
        author="IPSUM",
    )

    story = []
    city_name = summary["city_name"].iloc[0] if not summary.empty and "city_name" in summary else "N/A"
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    total_population = summary["total_population"].sum() if "total_population" in summary else 0
    total_ubs = summary["opportunity_name"].nunique() if "opportunity_name" in summary else 0

    story.extend(_build_cover_page(summary, merged_df, city_name, generated_at, styles))
    story.append(PageBreak())
    story.extend(_build_summary_page(summary, merged_df, styles))
    story.append(PageBreak())

    story.append(Paragraph("Relatório de Alocação UBS", styles["ReportTitle"]))
    story.append(
        Paragraph(
            f"<b>Município:</b> {city_name} &nbsp;&nbsp; <b>Gerado em:</b> {generated_at}",
            styles["ReportSubtitle"],
        )
    )
    story.append(_build_metrics_table(summary, merged_df, styles))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            (
                "Este relatório consolida a alocação dos setores de demanda às UBS, "
                "com foco em carga assistencial, distância de acesso e perfil socioeconômico "
                "da população atendida."
            ),
            styles["Body"],
        )
    )
    if _fallback_used(merged_df):
        story.append(Spacer(1, 0.08 * inch))
        story.append(
            _callout_table(
                (
                    "Esta execução usou fallback geodésico porque o Valhalla não retornou uma matriz válida. "
                    "Recalcule após reiniciar o serviço Valhalla com o novo limite de matriz para obter distâncias viárias."
                ),
                styles,
                warning=True,
            )
        )

    story.append(Paragraph("Leitura executiva", styles["SectionTitle"]))
    for finding in _build_executive_findings(summary, merged_df):
        story.append(Paragraph(f"• {finding}", styles["Body"]))

    story.append(Paragraph("Resumo por UBS", styles["SectionTitle"]))
    story.append(_build_ubs_table(summary, merged_df, styles))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "PPI soma os percentuais de população preta, parda e indígena alocados a cada UBS.",
            styles["Small"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Mapa da alocação", styles["SectionTitle"]))
    story.append(
        Paragraph(
            (
                "A imagem abaixo reproduz, em versão estática para o PDF, os arcos origem-destino "
                "usados na visualização do Kepler. As linhas conectam setores de demanda às UBS "
                "alocadas; a cor acompanha a distância do deslocamento."
            ),
            styles["Body"],
        )
    )
    map_buffer = create_allocation_map(merged_df)
    if map_buffer:
        story.append(_image_flowable(map_buffer, width=7.0 * inch, height=4.35 * inch))
    else:
        story.append(Paragraph("Mapa indisponível: coordenadas de origem ou destino ausentes.", styles["Body"]))

    story.append(PageBreak())
    story.append(Paragraph("Gráficos de distribuição", styles["SectionTitle"]))
    hist_buffer = create_distance_hist(merged_df)
    box_plot = create_distance_boxplot(merged_df)
    chart_pop, chart_racial = create_allocation_charts(summary)

    if hist_buffer:
        story.append(KeepTogether([Paragraph("Distribuição das distâncias", styles["Body"]), _image_flowable(hist_buffer, 6.8 * inch, 3.35 * inch)]))
        story.append(Spacer(1, 0.14 * inch))
    if box_plot:
        story.append(KeepTogether([Paragraph("Dispersão das distâncias", styles["Body"]), _image_flowable(box_plot, 6.8 * inch, 2.1 * inch)]))

    story.append(PageBreak())
    story.append(Paragraph("Comparativos por UBS", styles["SectionTitle"]))
    if chart_pop:
        story.append(KeepTogether([Paragraph("População alocada por UBS", styles["Body"]), _image_flowable(chart_pop, 6.85 * inch, 3.75 * inch)]))
        story.append(Spacer(1, 0.16 * inch))
    if chart_racial:
        story.append(KeepTogether([Paragraph("Composição racial por UBS", styles["Body"]), _image_flowable(chart_racial, 6.85 * inch, 3.65 * inch)]))

    story.append(PageBreak())
    story.append(Paragraph("Perguntas-guia", styles["SectionTitle"]))
    story.append(
        Paragraph(
            (
                "As respostas abaixo sintetizam os principais riscos operacionais e sociais observados "
                "nos dados. Elas servem como leitura inicial para priorização e investigação local."
            ),
            styles["Body"],
        )
    )
    for line in gerar_perguntas_respostas(summary, merged_df):
        clean_line = str(line).strip()
        if not clean_line:
            story.append(Spacer(1, 0.04 * inch))
            continue
        if clean_line[0].isdigit() and "." in clean_line[:3]:
            story.append(Paragraph(f"<b>{clean_line}</b>", styles["Body"]))
        else:
            story.append(Paragraph(clean_line.replace("   - ", "• "), styles["Body"]))

    story.append(Spacer(1, 0.16 * inch))
    story.append(
        Paragraph(
            (
                f"Base analisada: {_fmt_int(total_population)} pessoas alocadas em "
                f"{_fmt_int(total_ubs)} UBS. As distâncias estão em quilômetros e refletem "
                "a alocação calculada no momento da geração do relatório."
            ),
            styles["Small"],
        )
    )

    doc.build(story, onFirstPage=_draw_page_footer, onLaterPages=_draw_page_footer)
    pdf_buffer.seek(0)
    logger.info("Relatório PDF gerado com sucesso.")
    return pdf_buffer


def create_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando criação da Tabela Resumo de Indicadores.")

    if summary.empty or "total_population" not in summary.columns:
        logger.warning("Sumário vazio ou sem 'total_population'. Retornando tabela vazia.")
        return pd.DataFrame(columns=['Indicador', 'Valor'])

    mais_sobrecarregada = summary.loc[summary['total_population'].idxmax()]['opportunity_name']
    ocupacao_max = summary['total_population'].max()

    mais_subutilizada = summary.loc[summary['total_population'].idxmin()]['opportunity_name']
    ocupacao_min = summary['total_population'].min()

    media_ocupacao = summary['total_population'].mean()
    mediana_ocupacao = summary['total_population'].median()
    desvio_padrao_ocupacao = summary['total_population'].std()

    resumo = pd.DataFrame({
        'Indicador': [
            'UBS mais sobrecarregada',
            'UBS mais subutilizada',
            'Média de ocupação',
            'Mediana de ocupação',
            'Desvio-padrão de ocupação'
        ],
        'Valor': [
            f"{mais_sobrecarregada} ({ocupacao_max:.0f} pessoas)",
            f"{mais_subutilizada} ({ocupacao_min:.0f} pessoas)",
            f"{media_ocupacao:.2f}",
            f"{mediana_ocupacao:.2f}",
            f"{desvio_padrao_ocupacao:.2f}"
        ]
    })

    logger.info("Tabela Resumo criada com sucesso.")
    return resumo

def save_summary_table_image(resumo: pd.DataFrame) -> BytesIO:
    logger.info("Iniciando criação da imagem da Tabela Resumo.")
    
    if resumo.empty:
        logger.warning("DataFrame de resumo vazio. Não é possível gerar imagem da tabela.")
        return BytesIO() # Retorna buffer vazio

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    tabela = table(ax, resumo, loc='center', cellLoc='center', colWidths=[0.4, 0.6])
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)

    # Estilização
    for key, cell in tabela.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2)
        if key[0] == 0:
            cell.set_facecolor('#C4DFDF')
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)

    logger.info("Imagem da Tabela Resumo criada com sucesso.")
    return buf
