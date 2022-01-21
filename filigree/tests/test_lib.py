# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os

import pytest

from filigree import _lib


@pytest.fixture
def basn6a16():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, "pngsuite", "basn6a16.png")
    return _lib.Image(path)


def test_nonexistent_image():
    # An attempt to load an image that does not exist throws
    with pytest.raises(RuntimeError, match="Magick: unable to open image"):
        _lib.Image("nonexistent.png")


def test_init_image(basn6a16):
    # Normally loading an image should result in a valid image
    assert basn6a16.valid


def test_image_get_data(basn6a16):
    print(basn6a16.get_data())
