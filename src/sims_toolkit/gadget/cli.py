import pathlib

import attr
import click
from colored import attr as c_attr, fg, stylize
from sims_toolkit.gadget.snapshot import File, FileFormat
from tabulate import tabulate

FG_CYAN = fg("cyan_2")
FG_RED = fg("red")
FG_ORANGE = fg("orange_1")
BOLD_TXT = c_attr("bold")


def info_label(text: str):
    """Stylize an information label."""
    return stylize(text, BOLD_TXT)


def describe_file(file: File):
    """Show the basic info of a GADGET-2 snapshot.

    :param file: The File instance that represents the snapshot.
    :return: The information as a string.
    """
    if file.format is FileFormat.ALT:
        file_format = "Enhanced (equivalent to SnapFormat=2)"
    else:
        file_format = "Default"
    header = file.header
    file_size = file.size / 1024 ** 2
    num_par_spec_dict = attr.asdict(header.num_par_spec)
    mass_spec_dict = attr.asdict(header.mass_spec)
    num_par_total_dict = attr.asdict(header.num_par_total)
    par_types = [par_type.capitalize() for par_type in num_par_spec_dict]
    par_nums = [int(num_par) for num_par in num_par_spec_dict.values()]
    par_masses = [mass for mass in mass_spec_dict.values()]
    total_par_nums = [int(num_par) for num_par in num_par_total_dict.values()]
    table_headers = ["Type", "Number", "Mass", "Total Number"]
    table_value_formats = ["s", "d", ".3G", "d"]
    table_values = list(zip(par_types, par_nums, par_masses, total_par_nums))
    par_spec_table = tabulate(table_values,
                              headers=table_headers,
                              floatfmt=table_value_formats)
    blocks = [bloc.id or "NOT IDENTIFIED" for bloc in file.inspect()]
    snap_blocks = ", ".join(blocks)

    snap_info_tpl = f"""
{stylize("SNAPSHOT INFORMATION", FG_RED + BOLD_TXT)}
{stylize("====================", FG_RED + BOLD_TXT)}

{stylize("File Information", FG_CYAN + BOLD_TXT)}
----------------

Path:       {file.name}
Format:     {file_format}
Size:       {file_size:.2F}MB

{stylize("Simulation Information", FG_CYAN + BOLD_TXT)}
----------------------

{info_label("Time")}                {header.time:.5G}
{info_label("Redshift")}            {header.redshift:.5G}
{info_label("Flag Sfr")}            {header.flag_sfr:.5G}
{info_label("Flag Feedback")}       {header.flag_feedback:.5G}
{info_label("Flag Cooling")}        {header.flag_cooling:.5G}
{info_label("Number of Files")}     {header.num_files_snap}
{info_label("Box Size")}            {header.box_size:.5E}
{info_label("Omega0")}              {header.omega_zero:.5G}
{info_label("OmegaLambda")}         {header.omega_lambda:.5G}
{info_label("Hubble Param")}        {header.hubble_param:.5G}

{par_spec_table}

{info_label("Stored Snapshot Blocks")}: {snap_blocks}
    """
    return snap_info_tpl


@click.group()
def main():
    """Sims-Toolkit Command Line Interface."""
    pass


@main.group()
def gadget_snap():
    """Command group for GADGET-2 snapshot operations."""
    pass


@gadget_snap.command()
@click.argument("path", type=click.Path(exists=True))
def describe(path: str):
    """Describe the GADGET-2 snapshot contents.

    :param path:
    """
    snap = File(pathlib.Path(path))
    description = describe_file(snap)
    click.echo_via_pager(description)


if __name__ == '__main__':
    main()