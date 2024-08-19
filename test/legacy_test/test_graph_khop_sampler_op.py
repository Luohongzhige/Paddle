#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from op_test import OpTest


class TestGraphKhopSamplerOp(OpTest):
    def setUp(self):
        num_nodes = 20
        edges = np.random.randint(num_nodes, size=(100, 2))
        edges = np.unique(edges, axis=0)
        edges_id = np.arange(0, len(edges))
        sorted_edges = edges[np.argsort(edges[:, 1])]
        sorted_eid = edges_id[np.argsort(edges[:, 1])]

        # Calculate dst index cumsum counts.
        dst_count = np.zeros(num_nodes)
        dst_src_dict = {}
        for dst in range(0, num_nodes):
            true_index = sorted_edges[:, 1] == dst
            dst_count[dst] = np.sum(true_index)
            dst_src_dict[dst] = sorted_edges[:, 0][true_index]
        dst_count = dst_count.astype("int64")
        colptr = np.cumsum(dst_count)
        colptr = np.insert(colptr, 0, 0)

        self.row = sorted_edges[:, 0].astype("int64")
        self.colptr = colptr.astype("int64")
        self.sorted_eid = sorted_eid.astype("int64")
        self.nodes = np.unique(np.random.randint(num_nodes, size=5)).astype(
            "int64"
        )
        self.sample_sizes = [5, 5]
        self.dst_src_dict = dst_src_dict

    def test_check_output(self):
        self.check_output(check_pir=True)
