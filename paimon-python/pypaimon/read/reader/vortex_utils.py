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

import datetime
import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import pyarrow as pa

from pypaimon.common.file_io import FileIO
from pypaimon.common.options.config import OssOptions
from pypaimon.common.predicate import Like, Predicate


def to_vortex_specified(file_io: FileIO, file_path: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """Convert path and extract storage options for Vortex store.from_url().

    Returns (url, store_kwargs) where store_kwargs can be passed as
    keyword arguments to ``vortex.store.from_url(url, **store_kwargs)``.
    For local paths store_kwargs is None.
    """
    if hasattr(file_io, 'file_io'):
        file_io = file_io.file_io()

    if hasattr(file_io, 'get_merged_properties'):
        properties = file_io.get_merged_properties()
    else:
        properties = file_io.properties if hasattr(file_io, 'properties') and file_io.properties else None

    scheme, _, _ = file_io.parse_location(file_path)
    file_path_for_vortex = file_io.to_filesystem_path(file_path)

    store_kwargs = None

    if scheme in {'file', None} or not scheme:
        if not os.path.isabs(file_path_for_vortex):
            file_path_for_vortex = os.path.abspath(file_path_for_vortex)
        return file_path_for_vortex, None

    # For remote schemes, keep the original URI so vortex can parse it
    file_path_for_vortex = file_path

    if scheme == 'oss' and properties:
        parsed = urlparse(file_path)
        bucket = parsed.netloc

        store_kwargs = {
            'endpoint': f"https://{bucket}.{properties.get(OssOptions.OSS_ENDPOINT)}",
            'access_key_id': properties.get(OssOptions.OSS_ACCESS_KEY_ID),
            'secret_access_key': properties.get(OssOptions.OSS_ACCESS_KEY_SECRET),
            'virtual_hosted_style_request': 'true',
        }
        if properties.contains(OssOptions.OSS_SECURITY_TOKEN):
            store_kwargs['session_token'] = properties.get(OssOptions.OSS_SECURITY_TOKEN)

        file_path_for_vortex = file_path_for_vortex.replace('oss://', 's3://')

    return file_path_for_vortex, store_kwargs


def _resolve_vortex_location(file_io: FileIO, file_path: str):
    """Return ``(path, store)``: a local path or URL with no store, or an object key in ``store``."""
    url, store_kwargs = to_vortex_specified(file_io, file_path)
    if not store_kwargs:
        return url, None
    from vortex import store

    parsed = urlparse(url)
    bucket_store = store.from_url(f"{parsed.scheme}://{parsed.netloc}", **store_kwargs)
    return parsed.path.lstrip('/'), bucket_store


def open_vortex_file(file_io: FileIO, file_path: str):
    import vortex

    path, store = _resolve_vortex_location(file_io, file_path)
    return vortex.open(path, store=store)


def write_vortex_file(file_io: FileIO, file_path: str, data: pa.Table):
    import vortex

    path, store = _resolve_vortex_location(file_io, file_path)
    # Streams the table's batches into the file without converting it to a Vortex array first.
    vortex.io.write(data, path, store=store)


_COMPARISONS = {
    'equal': lambda c, v: c == v,
    'notEqual': lambda c, v: c != v,
    'lessThan': lambda c, v: c < v,
    'lessOrEqual': lambda c, v: c <= v,
    'greaterThan': lambda c, v: c > v,
    'greaterOrEqual': lambda c, v: c >= v,
}

_TEMPORAL_LITERAL_TYPES = (datetime.date, datetime.datetime)


class _Unsupported(Exception):
    pass


def _is_utf8(arrow_type: pa.DataType) -> bool:
    return (pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)
            or arrow_type == pa.string_view())


def _escape_like(value: str) -> str:
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def _literal(arrow_type: pa.DataType, value: Any):
    """Build a Vortex literal typed exactly like the column; Vortex does not coerce types."""
    import vortex

    if value is None:
        raise _Unsupported()
    if pa.types.is_temporal(arrow_type) and not isinstance(value, _TEMPORAL_LITERAL_TYPES):
        # Integer temporal literals have format-dependent units; leave them to Arrow.
        raise _Unsupported()
    if arrow_type == pa.string_view():
        arrow_type = pa.string()
    elif arrow_type == pa.binary_view():
        arrow_type = pa.binary()
    try:
        return vortex.array(pa.array([value], type=arrow_type)).scalar_at(0)
    except (pa.ArrowException, TypeError, ValueError, OverflowError) as e:
        raise _Unsupported() from e


def _leaf_to_vortex(predicate: Predicate, schema: pa.Schema):
    import vortex.expr as ve

    if predicate.field is None or predicate.field not in schema.names:
        raise _Unsupported()
    arrow_type = schema.field(predicate.field).type
    column = ve.column(predicate.field)
    method = predicate.method
    literals = predicate.literals or []

    if method == 'isNull':
        return ve.is_null(column)
    if method == 'isNotNull':
        return ve.is_not_null(column)
    if method in _COMPARISONS:
        return _COMPARISONS[method](column, _literal(arrow_type, literals[0]))
    if method in ('between', 'notBetween'):
        between = ve.between(column, _literal(arrow_type, literals[0]),
                             _literal(arrow_type, literals[1]))
        return between if method == 'between' else ve.not_(between)
    if method == 'in':
        # SQL IN never matches a null literal.
        values = [v for v in literals if v is not None]
        if not values:
            raise _Unsupported()
        return ve.or_collect(column == _literal(arrow_type, v) for v in values)
    if method == 'notIn':
        # Any null literal makes SQL NOT IN unknown for every row; leave that to Arrow.
        if not literals or any(v is None for v in literals):
            raise _Unsupported()
        return ve.and_collect(column != _literal(arrow_type, v) for v in literals)
    if method in ('startsWith', 'endsWith', 'contains', 'like'):
        if not _is_utf8(arrow_type) or not isinstance(literals[0], str):
            raise _Unsupported()
        value = literals[0]
        if method == 'like':
            # Rejects the escape sequences Paimon's LIKE rejects.
            try:
                Like._sql_like_to_regex(value)
            except ValueError as e:
                raise _Unsupported() from e
            pattern = value
        elif method == 'startsWith':
            pattern = _escape_like(value) + '%'
        elif method == 'endsWith':
            pattern = '%' + _escape_like(value)
        else:
            pattern = '%' + _escape_like(value) + '%'
        return ve.like(column, pattern)
    raise _Unsupported()


def _to_vortex(predicate: Predicate, schema: pa.Schema) -> Tuple[Optional[Any], bool]:
    import vortex.expr as ve

    if predicate.method == 'and':
        children = [_to_vortex(p, schema) for p in predicate.literals]
        converted = [expr for expr, _ in children if expr is not None]
        # Dropping a conjunct keeps every matching row, so a partial AND is a safe pre-filter.
        return (ve.and_collect(converted) if converted else None,
                all(exact for _, exact in children))
    if predicate.method == 'or':
        children = [_to_vortex(p, schema) for p in predicate.literals]
        if any(expr is None or not exact for expr, exact in children):
            return None, False
        return ve.or_collect(expr for expr, _ in children), True
    try:
        return _leaf_to_vortex(predicate, schema), True
    except _Unsupported:
        return None, False


def paimon_predicate_to_vortex(
        predicate: Optional[Predicate], schema: pa.Schema) -> Tuple[Optional[Any], bool]:
    """Convert a Paimon predicate to a Vortex filter expression.

    Returns ``(expr, exact)``. ``expr`` keeps every row the predicate matches and may be None.
    When ``exact`` is False the caller must still apply the full predicate to the scan output.
    """
    if predicate is None:
        return None, True
    return _to_vortex(predicate, schema)
