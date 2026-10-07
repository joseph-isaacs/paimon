# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import Any, Iterator, List, Optional, Set, Tuple

import pyarrow as pa
from pyarrow import RecordBatch

from pypaimon.common.file_io import FileIO
from pypaimon.common.predicate import Predicate
from pypaimon.read.reader.iface.record_batch_reader import RecordBatchReader
from pypaimon.schema.data_types import DataField, PyarrowFieldParser
from pypaimon.table.special_fields import SpecialFields


_VIEW_TO_PLAIN = {pa.string_view(): pa.utf8(), pa.binary_view(): pa.binary()}


class FormatVortexReader(RecordBatchReader):
    """
    A Format Reader that reads record batch from a Vortex file,
    and filters it based on the provided predicate and projection.
    """

    # row_indices: from IndexedSplit (ANN vector search), discrete local row offsets within the file.
    # shard_range: from SlicedSplit (parallel shard scan), a contiguous [start, end) row range within the file.
    def __init__(self, file_io: FileIO, file_path: str, read_fields: List[DataField],
                 push_down_predicate: Any, batch_size: int = 1024,
                 row_indices: Optional[List[int]] = None,
                 shard_range: Optional[Tuple[int, int]] = None,
                 predicate_fields: Optional[Set[str]] = None,
                 paimon_predicate: Optional[Predicate] = None):
        import vortex

        from pypaimon.read.reader.vortex_utils import (
            open_vortex_file, paimon_predicate_to_vortex)
        vortex_file = open_vortex_file(file_io, file_path)

        self.read_fields = read_fields
        self._read_field_names = [f.name for f in read_fields]

        # Identify which fields exist in the file and which are missing
        arrow_schema = vortex_file.dtype.to_arrow_schema()
        file_schema_names = set(arrow_schema.names)
        self.existing_fields = [f.name for f in read_fields if f.name in file_schema_names]
        self.missing_fields = [f.name for f in read_fields if f.name not in file_schema_names]

        columns_for_vortex = self.existing_fields if self.existing_fields else None

        # A non-None ``push_down_predicate`` means the caller relies on this reader to filter
        # exactly, so whatever cannot be evaluated natively is applied to the output in Arrow.
        if paimon_predicate is not None:
            vortex_expr, exact = paimon_predicate_to_vortex(paimon_predicate, arrow_schema)
        elif push_down_predicate is not None:
            try:
                from vortex.arrow.expression import arrow_to_vortex
                vortex_expr, exact = arrow_to_vortex(push_down_predicate, arrow_schema), True
            except Exception:
                vortex_expr, exact = None, False
        else:
            vortex_expr, exact = None, True
        self._post_filter = push_down_predicate if not exact else None

        # Scan with Vortex's natural (layout-aligned) splits rather than forcing
        # ``batch_size``-row splits, which fragments the scan into many tiny
        # tasks. Batches are re-sliced to ``batch_size`` on the Arrow side.
        if row_indices is not None:
            array_iter = vortex_file.scan(
                columns_for_vortex, expr=vortex_expr, indices=vortex.array(row_indices))
        elif shard_range is not None:
            array_iter = vortex_file.to_repeated_scan(
                columns_for_vortex, expr=vortex_expr).execute(row_range=shard_range)
        else:
            array_iter = vortex_file.scan(columns_for_vortex, expr=vortex_expr)

        projected = [arrow_schema.field(name) for name in self.existing_fields] \
            if self.existing_fields else list(arrow_schema)
        target_schema = pa.schema([f.with_type(_VIEW_TO_PLAIN.get(f.type, f.type)) for f in projected])
        try:
            # Converting straight to string/binary avoids copying every view-typed batch.
            arrow_reader = array_iter.to_arrow(schema=target_schema)
        except TypeError:
            # vortex-data releases without the ``schema`` argument.
            arrow_reader = array_iter.to_arrow()
        self.record_batch_reader = self._sliced_batches(arrow_reader, batch_size)

        self._output_schema = (
            PyarrowFieldParser.from_paimon_schema(read_fields) if read_fields else None
        )

    @staticmethod
    def _sliced_batches(reader: pa.RecordBatchReader, batch_size: int) -> Iterator[RecordBatch]:
        for batch in reader:
            batch = FormatVortexReader._cast_view_types(batch)
            if batch_size <= 0 or batch.num_rows <= batch_size:
                yield batch
                continue
            for offset in range(0, batch.num_rows, batch_size):
                yield batch.slice(offset, batch_size)

    @staticmethod
    def _cast_view_types(batch: RecordBatch) -> RecordBatch:
        """Cast all string_view/binary_view columns to string/binary."""
        columns = []
        fields = []
        changed = False
        for i in range(batch.num_columns):
            col = batch.column(i)
            field = batch.schema.field(i)
            if col.type == pa.string_view():
                col = col.cast(pa.utf8())
                field = field.with_type(pa.utf8())
                changed = True
            elif col.type == pa.binary_view():
                col = col.cast(pa.binary())
                field = field.with_type(pa.binary())
                changed = True
            columns.append(col)
            fields.append(field)
        if changed:
            return pa.RecordBatch.from_arrays(columns, schema=pa.schema(fields))
        return batch

    def read_arrow_batch(self) -> Optional[RecordBatch]:
        while True:
            batch = self._read_unfiltered_batch()
            if batch is None or self._post_filter is None:
                return batch
            table = pa.Table.from_batches([batch]).filter(self._post_filter)
            if table.num_rows > 0:
                return table.combine_chunks().to_batches()[0]

    def _read_unfiltered_batch(self) -> Optional[RecordBatch]:
        try:
            batch = next(self.record_batch_reader)

            if not self.missing_fields:
                return batch

            def _type_for_missing(name: str) -> pa.DataType:
                if self._output_schema is not None:
                    idx = self._output_schema.get_field_index(name)
                    if idx >= 0:
                        return self._output_schema.field(idx).type
                return pa.null()

            missing_columns = [
                pa.nulls(batch.num_rows, type=_type_for_missing(name))
                for name in self.missing_fields
            ]

            # Reconstruct the batch with all fields in the correct order
            all_columns = []
            out_fields = []
            for field_name in self._read_field_names:
                if field_name in self.existing_fields:
                    column_idx = self.existing_fields.index(field_name)
                    all_columns.append(batch.column(column_idx))
                    out_fields.append(batch.schema.field(column_idx))
                else:
                    column_idx = self.missing_fields.index(field_name)
                    col_type = _type_for_missing(field_name)
                    all_columns.append(missing_columns[column_idx])
                    nullable = not SpecialFields.is_system_field(field_name)
                    out_fields.append(pa.field(field_name, col_type, nullable=nullable))
            return pa.RecordBatch.from_arrays(all_columns, schema=pa.schema(out_fields))

        except StopIteration:
            return None

    def close(self):
        self.record_batch_reader = None
