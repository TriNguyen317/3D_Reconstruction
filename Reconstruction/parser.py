import argparse

from .config import Config


def parse_cli() -> dict:
    """Parse command line arguments and return a dictionary with the input arguments. If --gui is specified, run the GUI interface and return the arguments from the GUI."""

    parser = argparse.ArgumentParser(
        description="Matching with hand-crafted and deep-learning based local features and image retrieval."
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Project directoryt, containing a folder 'images', in which all the images are present and where the results will be saved.",
        default=None,
    )
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        help="Folder containing images to process. If not specified, an 'images' folder inside the project folder is assumed.",
        default=None,
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        help="Define the pipeline (combination of local feature extractor and matcher) to use for the matching.",
        choices=Config.get_pipelines(),
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="Path of a YAML configuration file that contains user-defined options. If not specified, the default configuration for the selected matching configuration is used.",
        default=None,
    )
    (
        parser.add_argument(
            "-q",
            "--quality",
            type=str,
            choices=["lowest", "low", "medium", "high", "highest"],
            default="high",
            help="Set the image resolution for the matching. High means full resolution images, medium is half res, low is 1/4 res, highest is x2 upsampling. Default is high.",
        ),
    )
    parser.add_argument(
        "-t",
        "--tiling",
        type=str,
        choices=["none", "preselection", "grid", "exhaustive"],
        default="none",
        help="Set the tiling strategy for the matching. Default is none.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=[
            "matching_lowres",
            "bruteforce",
            "sequential",
            "retrieval",
            "custom_pairs",
            "covisibility",
        ],
        default="matching_lowres",
        help="Matching strategy",
    )
    parser.add_argument(
        "--pair_file", type=str, default=None, help="Specify pairs for matching"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Image overlap, if using sequential overlap strategy",
        default=1,
    )
    parser.add_argument(
        "--global_feature",
        choices=Config.get_retrieval_names(),
        default="netvlad",
        help="Specify image retrieval method",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Path to the COLMAP database to be use for covisibility pair selection.",
    )
    parser.add_argument(
        "--upright",
        action="store_true",
        help="Enable the estimation of the best image rotation for the matching (useful in case of aerial datasets).",
        default=False,
    )
    parser.add_argument(
        "--skip_reconstruction",
        action="store_true",
        help="Skip reconstruction step carried out with pycolmap. This step is necessary to export the solution in Bundler format for Agisoft Metashape.",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of output folder",
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--camera_options",
        help="Path to camera options yaml file, e.g. config/cameras.yaml",
        default="./config/cameras.yaml",
    )
    args = parser.parse_args()



    return vars(args)
