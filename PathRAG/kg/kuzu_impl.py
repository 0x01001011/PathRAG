from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass

import kuzu

from ..storage import NetworkXStorage


@dataclass
class KuzuGraphStorage(NetworkXStorage):
    """Graph storage backed by Kùzu DB with in-memory NetworkX operations."""

    def __post_init__(self):
        super().__post_init__()
        self._db_dir = os.path.join(
            self.global_config["working_dir"], f"kuzu_{self.namespace}"
        )
        os.makedirs(self._db_dir, exist_ok=True)
        self._db = kuzu.Database(self._db_dir)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()
        if self._graph.number_of_nodes() == 0:
            self._load_from_kuzu()

    def _init_schema(self):
        try:
            self._conn.execute(
                "CREATE NODE TABLE Entity(id STRING, entity_type STRING, description STRING, source_id STRING, PRIMARY KEY(id))"
            )
        except RuntimeError as e:
            if "already exists" not in str(e):
                raise

        try:
            self._conn.execute(
                "CREATE REL TABLE Relation(FROM Entity TO Entity, weight DOUBLE, description STRING, keywords STRING, source_id STRING)"
            )
        except RuntimeError as e:
            if "already exists" not in str(e):
                raise

    def _load_from_kuzu(self):
        nodes_res = self._conn.execute(
            "MATCH (e:Entity) RETURN e.id, e.entity_type, e.description, e.source_id"
        )
        while nodes_res.has_next():
            node_id, typ, desc, src = nodes_res.get_next()
            self._graph.add_node(
                node_id,
                entity_type=typ,
                description=desc,
                source_id=src,
            )
        nodes_res.close()
        edges_res = self._conn.execute(
            "MATCH (a:Entity)-[r:Relation]->(b:Entity) RETURN a.id, b.id, r.weight, r.description, r.keywords, r.source_id"
        )
        while edges_res.has_next():
            s, t, w, desc, kw, src = edges_res.get_next()
            self._graph.add_edge(
                s,
                t,
                weight=w,
                description=desc,
                keywords=kw,
                source_id=src,
            )
        edges_res.close()

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        await super().upsert_node(node_id, node_data)
        await asyncio.to_thread(
            self._conn.execute,
            "MERGE (n:Entity {id:$id}) SET n.entity_type=$t, n.description=$d, n.source_id=$s",
            {
                "id": node_id,
                "t": node_data.get("entity_type", ""),
                "d": node_data.get("description", ""),
                "s": node_data.get("source_id", ""),
            },
        )

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        await super().upsert_edge(source_node_id, target_node_id, edge_data)
        query = (
            "MATCH (a:Entity {id:$src}), (b:Entity {id:$tgt}) "
            "MERGE (a)-[r:Relation]->(b) "
            "SET r.weight=$w, r.description=$d, r.keywords=$k, r.source_id=$s"
        )
        await asyncio.to_thread(
            self._conn.execute,
            query,
            {
                "src": source_node_id,
                "tgt": target_node_id,
                "w": edge_data.get("weight", 1.0),
                "d": edge_data.get("description", ""),
                "k": edge_data.get("keywords", ""),
                "s": edge_data.get("source_id", ""),
            },
        )

    async def delete_node(self, node_id: str):
        await super().delete_node(node_id)
        await asyncio.to_thread(
            self._conn.execute,
            "MATCH (n:Entity {id:$id}) DETACH DELETE n",
            {"id": node_id},
        )

    async def index_done_callback(self):
        await super().index_done_callback()
