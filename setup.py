#!/usr/bin/env python

import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command
from setuptools.command.test import test as TestCommand


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


def process_requirements(requirements):
    """Parse requirements, handling git repositories"""
    EGG_MARK = "#egg="
    required = []
    dependency_links = []
    for i_req, req in enumerate(requirements):
        if (
            req.startswith("-e git:")
            or req.startswith("-e git+")
            or req.startswith("git:")
            or req.startswith("git+")
        ):
            if not EGG_MARK in req:
                raise ValueError(
                    "Dependency to a git repository should have the format:\n"
                    "git+ssh://git@github.com/org/repo@ref#egg=package\n"
                    "But line {} contained:\n{}".format(i_req, req)
                )
            package_name = req[req.find(EGG_MARK) + len(EGG_MARK) :]
            required.append(package_name)
            dependency_links.append(req)
        else:
            required.append(req)
    return required, dependency_links


def process_requirements_file(fname):
    """Parse requirements from a given filename."""
    return process_requirements(read(fname).splitlines())


install_requires, dependency_links = process_requirements_file("requirements.txt")

extras_require = {}
for tag in ("dev", "docs", "test", "train", "plots"):
    extras_require[tag], extra_dep_links = process_requirements_file(
        "requirements-{}.txt".format(tag)
    )
    dependency_links += extra_dep_links


# If there are any extras, add a catch-all case that includes everything.
# This assumes that entries in extras_require are lists (not single strings).
if extras_require:
    extras_require["all"] = sorted({x for v in extras_require.values() for x in v})


# Import meta data from __meta__.py
#
# We use exec for this because __meta__.py runs its __init__.py first,
# __init__.py may assume the requirements are already present, but this code
# is being run during the `python setup.py install` step, before requirements
# are installed.
# https://packaging.python.org/guides/single-sourcing-package-version/
meta = {}
exec(read("echofilter/__meta__.py"), meta)


# Import the README and use it as the long-description.
# If your readme path is different, add it here.
possible_readme_names = ["README.rst", "README.md", "README.txt", "README"]

# Handle turning a README file into long_description
long_description = meta["description"]
readme_fname = ""
for fname in possible_readme_names:
    try:
        long_description = read(fname)
    except IOError:
        # doesn't exist
        continue
    else:
        # exists
        readme_fname = fname
        break

# Infer the content type of the README file from its extension.
# If the contents of your README do not match its extension, manually assign
# long_description_content_type to the appropriate value.
readme_ext = os.path.splitext(readme_fname)[1]
if readme_ext.lower() == ".rst":
    long_description_content_type = "text/x-rst"
elif readme_ext.lower() == ".md":
    long_description_content_type = "text/markdown"
else:
    long_description_content_type = "text/plain"


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds...")
            here = os.path.abspath(os.path.dirname(__file__))
            rmtree(here, "dist")
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution...")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine...")
        os.system("twine upload dist/*")

        self.status("Pushing git tags...")
        os.system("git tag v{0}".format(meta["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    # Essential details on the package and its dependencies
    name=meta["name"],
    version=meta["version"],
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_dir={meta["name"]: os.path.join(".", meta["path"])},
    # If any package contains *.txt or *.rst files, include them:
    package_data={"echofilter": ["checkpoints.yaml"]},
    install_requires=install_requires,
    extras_require=extras_require,
    dependency_links=dependency_links,
    # Metadata to display on PyPI
    author=meta["author"],
    author_email=meta["author_email"],
    description=meta["description"],
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    license=meta["license"],
    url=meta["url"],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    # Could also include keywords, download_url, project_urls, etc.
    entry_points={
        "console_scripts": [
            "echofilter=echofilter.__main__:main",
            "echofilter-train=echofilter.ui.train_cli:main",
            "echofilter-generate-shards=echofilter.generate_shards:main",
            "ev2csv=echofilter.ev2csv:main",
        ],
    },
    # Custom commands
    cmdclass={"test": PyTest, "upload": UploadCommand},
)
