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
from parametrize import parametrize

from .. import inference
from ..raw.loader import evl_loader
from ..ui import inference_cli
from .base_test import BaseTestCase

EXPECTED_STATS = {
    "GP_20200125T160020_first240_Sv_raw.csv": {
        "timestamps": 242,
        "surface_depths": [31, 32],
        "turbulence_depths": [35, 41],
        "bottom_depths": [49, 50],
    },
    "Survey17_GR4_N5W_E_first240_Sv_raw.csv": {
        "timestamps": 242,
        "surface_depths": [0, 1],
        "turbulence_depths": [0, 19],
        "bottom_depths": [49, 57],
    },
    "Survey17_GR4_N5W_E_first50-redact_Sv_raw.csv": {
        "timestamps": 52,
        "surface_depths": [0, 1],
        "turbulence_depths": [0, 19],
        "bottom_depths": [49, 57],
    },
    "dec2017_20180108T045216_first600_Sv_raw.csv": {
        "timestamps": 602,
        "surface_depths": [15, 17],
        "turbulence_depths": [22, 47],
        "bottom_depths": [49, 50],
    },
    "mar2018_20180513T015216_first120_Sv_raw.csv": {
        "timestamps": 122,
        "surface_depths": [6, 8],
        "turbulence_depths": [7, 16],
        "bottom_depths": [49, 50],
    },
    "mar2018_20180513T015216_first720_Sv_raw.csv": {
        "timestamps": 722,
        "surface_depths": [6, 8],
        "turbulence_depths": [7, 16],
        "bottom_depths": [49, 50],
    },
    "sep2018_20181027T022221_first720_Sv_raw.csv": {
        "timestamps": 722,
        "surface_depths": [11, 14],
        "turbulence_depths": [20, 48],
        "bottom_depths": [49, 50],
    },
}


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

    def check_lines(self, input_fname, output_dirname, lines=None):
        stats = EXPECTED_STATS[input_fname]
        if lines is None:
            lines = [
                k.replace("_depths", "") for k in stats.keys() if k != "timestamps"
            ]
        basefile = os.path.splitext(input_fname)[0]
        for line_name in lines:
            fname = os.path.join(output_dirname, f"{basefile}.{line_name}.evl")
            ts, depths = evl_loader(fname)
            self.assertEqual(len(ts), len(depths))
            self.assertEqual(len(ts), stats["timestamps"])
            self.assertGreaterEqual(min(depths), stats[f"{line_name}_depths"][0])
            self.assertLessEqual(max(depths), stats[f"{line_name}_depths"][1])

    def test_dryrun(self):
        inference.run_inference(
            self.resource_directory,
            dry_run=True,
        )

    @parametrize("test_fname", EXPECTED_STATS.keys())
    def test_run_files(self, test_fname):
        with tempfile.TemporaryDirectory() as outdirname:
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
            )
            basefile = os.path.splitext(test_fname)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))
            self.check_lines(test_fname, outdirname)

    def test_noclobber_bottom(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            basefile = os.path.splitext(test_fname)[0]
            fname = os.path.join(outdirname, basefile + ".bottom.evl")
            pathlib.Path(fname).touch()
            with pytest.raises(EnvironmentError):
                inference.run_inference(
                    test_fname,
                    source_dir=self.resource_directory,
                    output_dir=outdirname,
                )

    def test_noclobber_surface(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            basefile = os.path.splitext(test_fname)[0]
            fname = os.path.join(outdirname, basefile + ".surface.evl")
            pathlib.Path(fname).touch()
            with pytest.raises(EnvironmentError):
                inference.run_inference(
                    test_fname,
                    source_dir=self.resource_directory,
                    output_dir=outdirname,
                )

    def test_noclobber_turbulence(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            basefile = os.path.splitext(test_fname)[0]
            fname = os.path.join(outdirname, basefile + ".turbulence.evl")
            pathlib.Path(fname).touch()
            with pytest.raises(EnvironmentError):
                inference.run_inference(
                    test_fname,
                    source_dir=self.resource_directory,
                    output_dir=outdirname,
                )

    def test_rerun_skip(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            basefile = os.path.splitext(test_fname)[0]
            pathlib.Path(os.path.join(outdirname, basefile + ".bottom.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".surface.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".turbulence.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".regions.evr")).touch()
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                skip_existing=True,
            )

    def test_rerun_overwrite(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            basefile = os.path.splitext(test_fname)[0]
            pathlib.Path(os.path.join(outdirname, basefile + ".bottom.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".surface.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".turbulence.evl")).touch()
            pathlib.Path(os.path.join(outdirname, basefile + ".regions.evr")).touch()
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                overwrite_existing=True,
            )
            self.check_lines(test_fname, outdirname)

    def test_no_bottom(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                generate_bottom_line=False,
            )
            basefile = os.path.splitext(test_fname)[0]
            self.assert_file_absent(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_no_surface(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                generate_surface_line=False,
            )
            basefile = os.path.splitext(test_fname)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_absent(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_no_turbulence(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                generate_turbulence_line=False,
            )
            basefile = os.path.splitext(test_fname)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_absent(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_with_patches(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                minimum_patch_area=25,
            )
            basefile = os.path.splitext(test_fname)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_with_logitsmoothing(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                logit_smoothing_sigma=2,
            )
            basefile = os.path.splitext(test_fname)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_run_verbose(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                verbose=10,
            )
            basefile = os.path.splitext(test_fname)[0]
            self.assert_file_exists(os.path.join(outdirname, basefile + ".bottom.evl"))
            self.assert_file_exists(os.path.join(outdirname, basefile + ".surface.evl"))
            self.assert_file_exists(
                os.path.join(outdirname, basefile + ".turbulence.evl")
            )
            self.assert_file_exists(os.path.join(outdirname, basefile + ".regions.evr"))

    def test_run_quiet(self):
        with tempfile.TemporaryDirectory() as outdirname:
            test_fname = self.testfile_upfacing
            inference.run_inference(
                test_fname,
                source_dir=self.resource_directory,
                output_dir=outdirname,
                verbose=0,
            )
            basefile = os.path.splitext(test_fname)[0]
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
