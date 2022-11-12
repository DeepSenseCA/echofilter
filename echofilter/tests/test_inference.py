"""
Tests for inference.py.
"""

# This file is part of Echofilter.
#
# Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import pathlib
import tempfile

import pytest

from .. import inference
from ..ui import inference_cli
from .base_test import BaseTestCase


class test_get_color_palette(BaseTestCase):
    """
    Tests for get_color_palette.
    """

    def test_get_color_palette_base(self):
        inference.get_color_palette(include_xkcd=False)

    def test_get_color_palette_xkcd(self):
        inference.get_color_palette(include_xkcd=True)


class test_hexcolor2rgb8(BaseTestCase):
    """
    Tests for hexcolor2rgb8.
    """

    def test_hexcolor2rgb8_pass(self):
        input = (0.5, 0.5, 0.5)
        out = inference.hexcolor2rgb8(input)
        self.assertEqual(out, input)

    def test_hexcolor2rgb8_white(self):
        input = "#ffffff"
        out = inference.hexcolor2rgb8(input)
        self.assertEqual(out, (255, 255, 255))

    def test_hexcolor2rgb8_black(self):
        input = "#000000"
        out = inference.hexcolor2rgb8(input)
        self.assertEqual(out, (0, 0, 0))

    def test_hexcolor2rgb8_grey(self):
        input = "#888888"
        out = inference.hexcolor2rgb8(input)
        self.assertEqual(out, (136, 136, 136))


class test_run_inference(BaseTestCase):
    """
    Tests for run_inference.
    """

    def test_dryrun(self):
        inference.run_inference(
            self.resource_directory,
            dry_run=True,
        )

    def test_run_downfacing(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_downfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
            )
            basefile = os.path.splitext(self.testfile_downfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_run_upfacing(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_noclobber_bottom(self):
        with tempfile.TemporaryDirectory() as outdirname:
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            fname = os.path.join(outdirname, basefile + ".bottom.evl")
            pathlib.Path(fname).touch()
            with pytest.raises(EnvironmentError):
                inference.run_inference(
                    self.testfile_upfacing,
                    source_dir=self.resource_directory,
                    output_dir=outdirname,
                )

    def test_noclobber_surface(self):
        with tempfile.TemporaryDirectory() as outdirname:
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            fname = os.path.join(outdirname, basefile + ".surface.evl")
            pathlib.Path(fname).touch()
            with pytest.raises(EnvironmentError):
                inference.run_inference(
                    self.testfile_upfacing,
                    source_dir=self.resource_directory,
                    output_dir=outdirname,
                )

    def test_noclobber_turbulence(self):
        with tempfile.TemporaryDirectory() as outdirname:
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            fname = os.path.join(outdirname, basefile + ".turbulence.evl")
            pathlib.Path(fname).touch()
            with pytest.raises(EnvironmentError):
                inference.run_inference(
                    self.testfile_upfacing,
                    source_dir=self.resource_directory,
                    output_dir=outdirname,
                )

    def test_rerun_skip(self):
        with tempfile.TemporaryDirectory() as outdirname:
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            pathlib.Path(os.path.join(outdirname, basefile + ".bottom.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".surface.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".turbulence.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".regions.evr")).touch()
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                skip_existing=True,
            )

    def test_rerun_overwrite(self):
        with tempfile.TemporaryDirectory() as outdirname:
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            pathlib.Path(os.path.join(outdirname, basefile + ".bottom.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".surface.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".turbulence.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".regions.evr")).touch()
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                overwrite_existing=True,
            )

    def test_no_bottom(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                generate_bottom_line=False,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_absent(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_no_surface(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                generate_surface_line=False,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_absent(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_no_turbulence(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                generate_turbulence_line=False,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_absent(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_with_patches(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                minimum_patch_area=25,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_with_logitsmoothing(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                logit_smoothing_sigma=2,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_run_verbose(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                verbose=10,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_run_quiet(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.testfile_upfacing,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                verbose=0,
            )
            basefile = os.path.splitext(self.testfile_upfacing)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_run_directory(self):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                self.resource_directory,
                output_dir=outdirname,
            )


class test_cli(BaseTestCase):
    """
    Tests for command line interface.
    """

    def test_help(self):
        with pytest.raises(SystemExit):
            inference_cli.cli(["--help"])

    def test_help_short(self):
        with pytest.raises(SystemExit):
            inference_cli.cli(["-h"])

    def test_version(self):
        with pytest.raises(SystemExit):
            inference_cli.cli(["--version"])

    def test_version_short(self):
        with pytest.raises(SystemExit):
            inference_cli.cli(["-V"])

    def test_show_checkpoints(self):
        with pytest.raises(SystemExit):
            inference_cli.cli(["--list-checkpoints"])

    def test_show_colors(self):
        with pytest.raises(SystemExit):
            inference_cli.cli(["--list-colors"])

    def test_dryrun(self):
        inference_cli.cli([self.resource_directory, "--dry-run"])

    def test_dryrun_short(self):
        inference_cli.cli([self.resource_directory, "-n"])
