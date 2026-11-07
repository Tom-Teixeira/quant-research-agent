from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class QuantState(TypedDict, total=False):
    raw_data: object
    data_path: str

    cleaned_texts: list
    data_quality_report: dict

    factors: object
    factor_meta: dict

    signal_df: object
    quality: dict

    backtest: dict
    report: str

    retry_count: int
    diagnostics: list

    db_status: dict
    db_stats: dict
    start_date: str
    end_date: str
    index_code: str
    ts_code: str
    query: str
    evidence: list
    rag_mode: str
    rag_route: str


Messages = Annotated[list, add_messages]
