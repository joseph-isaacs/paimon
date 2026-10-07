################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import datetime
import decimal
import random
import shutil
import sys
import tempfile
import unittest

import pyarrow as pa

from pypaimon import CatalogFactory, Schema


@unittest.skipIf(sys.version_info < (3, 11), "vortex-data requires Python >= 3.11")
class VortexPredicateTest(unittest.TestCase):
    """Filtered Vortex reads must match filtered Parquet reads of the same data."""

    SEED = 0x5EED
    NUM_ROWS = 300
    STRINGS = ['abc', 'a%c', 'a_c', 'a\\c', 'xyz', 'ab', '', 'bca']

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        cls.catalog = CatalogFactory.create({'warehouse': cls.tempdir})
        cls.catalog.create_database('default', False)

        rnd = random.Random(cls.SEED)

        def maybe_null(value):
            return None if rnd.random() < 0.1 else value

        rows = [{
            'id': i,
            'i32': maybe_null(rnd.randint(-5, 5)),
            'f64': maybe_null(rnd.choice([-1.5, 0.0, 2.25, 3.0])),
            's': maybe_null(rnd.choice(cls.STRINGS)),
            'bin': maybe_null(rnd.choice([b'ab', b'cd', b'\x00\x01'])),
            'd': maybe_null(datetime.date(2024, 1, 1) + datetime.timedelta(days=rnd.randint(0, 3))),
            'dec': maybe_null(decimal.Decimal(rnd.choice(['1.25', '2.50', '-3.75']))),
            'flag': maybe_null(rnd.choice([True, False])),
        } for i in range(cls.NUM_ROWS)]
        cls.pa_schema = pa.schema([
            pa.field('id', pa.int64(), nullable=False),
            ('i32', pa.int32()),
            ('f64', pa.float64()),
            ('s', pa.string()),
            ('bin', pa.binary()),
            ('d', pa.date32()),
            ('dec', pa.decimal128(10, 2)),
            ('flag', pa.bool_()),
        ])
        data = pa.Table.from_pylist(rows, schema=cls.pa_schema)
        cls.tables = {}
        for fmt in ('parquet', 'vortex'):
            schema = Schema.from_pyarrow_schema(cls.pa_schema, options={'file.format': fmt})
            cls.catalog.create_table(f'default.t_{fmt}', schema, False)
            table = cls.catalog.get_table(f'default.t_{fmt}')
            write_builder = table.new_batch_write_builder()
            table_write = write_builder.new_write()
            table_commit = write_builder.new_commit()
            table_write.write_arrow(data)
            table_commit.commit(table_write.prepare_commit())
            table_write.close()
            table_commit.close()
            cls.tables[fmt] = table

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    def _read_ids(self, fmt, make_predicate):
        table = self.tables[fmt]
        predicate = make_predicate(table.new_read_builder().new_predicate_builder())
        read_builder = table.new_read_builder().with_filter(predicate)
        result = read_builder.new_read().to_arrow(read_builder.new_scan().plan().splits())
        return sorted(result.column('id').to_pylist())

    def _leaves(self):
        d0 = datetime.date(2024, 1, 2)
        dec = decimal.Decimal('1.25')
        return [
            lambda pb: pb.equal('i32', 3),
            lambda pb: pb.not_equal('i32', 3),
            lambda pb: pb.less_than('i32', 0),
            lambda pb: pb.less_or_equal('f64', 0.0),
            lambda pb: pb.greater_than('f64', 1.0),
            lambda pb: pb.greater_or_equal('d', d0),
            lambda pb: pb.between('i32', -2, 2),
            lambda pb: pb.not_between('i32', -2, 2),
            lambda pb: pb.is_in('i32', [1, 2, 3]),
            lambda pb: pb.is_in('s', ['abc', 'xyz', None]),
            lambda pb: pb.is_not_in('i32', [1, 2]),
            lambda pb: pb.is_not_in('i32', [1, None]),
            lambda pb: pb.is_null('s'),
            lambda pb: pb.is_not_null('d'),
            lambda pb: pb.equal('s', 'a%c'),
            lambda pb: pb.equal('bin', b'cd'),
            lambda pb: pb.equal('dec', dec),
            lambda pb: pb.equal('flag', True),
            lambda pb: pb.startswith('s', 'a'),
            lambda pb: pb.startswith('s', 'a%'),
            lambda pb: pb.endswith('s', 'c'),
            lambda pb: pb.contains('s', '_'),
            lambda pb: pb.contains('s', '\\'),
            lambda pb: pb.like('s', 'a_c'),
            lambda pb: pb.like('s', '%b%'),
        ]

    def _assert_same(self, make_predicate, description):
        self.assertEqual(
            self._read_ids('vortex', make_predicate),
            self._read_ids('parquet', make_predicate),
            description)

    def test_leaf_predicates_match_parquet(self):
        for i, leaf in enumerate(self._leaves()):
            self._assert_same(leaf, f'leaf {i}')

    def test_compound_predicates_match_parquet(self):
        rnd = random.Random(self.SEED)
        leaves = self._leaves()
        for trial in range(40):
            picked = rnd.sample(leaves, 3)

            def make_predicate(pb, picked=picked, shape=trial % 4):
                a, b, c = (leaf(pb) for leaf in picked)
                if shape == 0:
                    return pb.and_predicates([a, b])
                if shape == 1:
                    return pb.or_predicates([a, b])
                if shape == 2:
                    return pb.and_predicates([pb.or_predicates([a, b]), c])
                return pb.or_predicates([pb.and_predicates([a, b]), c])

            self._assert_same(make_predicate, f'trial {trial}')


if __name__ == '__main__':
    unittest.main()
