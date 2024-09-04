from importlib import import_module
import argparse
import pycolmap
import enlighten

import yaml
from Reconstruction import logger, timer
from Reconstruction.config import Config
from Reconstruction.image_matching import ImageMatching
from Reconstruction.io.h5_to_db import export_to_colmap
from Reconstruction.parser import parse_cli


#python main.py --dir D:\Sinh_vien\Project\Reference\3D-Reconstruction\Data\scan4 --pipeline superpoint+superglue  -f  
if __name__=="__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()

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
        default="sequential",
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
    parser.add_argument(
        "--saveply",
        action="store_true",
        default=False,
        help="Save the result to output folder",
    )
    
    args = parser.parse_args()
    imgs_dir = args["image_dir"]
    output_dir = args["output_dir"]
    database_path = output_dir / "database.db"
    
    if args['pipeline'] == "superpoint+superglue":
        # Build configuration
        config = Config(args)
        imgs_dir = config.general["image_dir"]
        output_dir = config.general["output_dir"]
        # Initialize ImageMatching class
        img_matching = ImageMatching(
            imgs_dir=imgs_dir,
            output_dir=output_dir,
            matching_strategy=config.general["matching_strategy"],
            local_features=config.extractor["name"],
            matching_method=config.matcher["name"],
            pair_file=config.general["pair_file"],
            retrieval_option=config.general["retrieval"],
            overlap=config.general["overlap"],
            existing_colmap_model=config.general["db_path"],
            custom_config=config.as_dict(),
        )

        # Generate pairs to be matched
        pair_path = img_matching.generate_pairs()
        timer.update("generate_pairs")

        # Extract features
        feature_path = img_matching.extract_features()
        timer.update("extract_features")

        # Matching
        match_path = img_matching.match_pairs(feature_path)
        timer.update("matching")

        # If features have been extracted on "upright" images, this function bring features back to their original image orientation
        if config.general["upright"]:
            img_matching.rotate_back_features(feature_path)
            timer.update("rotate_back_features")

        # Export in colmap format
        with open(config.general["camera_options"], "r") as file:
            camera_options = yaml.safe_load(file)
        
        export_to_colmap(
            img_dir=imgs_dir,
            feature_path=feature_path,
            match_path=match_path,
            database_path=database_path,
            camera_options=camera_options,
        )
        timer.update("export_to_colmap")

        num_images = pycolmap.Database(database_path).num_images

        output = output_dir+"/mvs"
        
        with enlighten.Manager() as manager:
            with manager.counter(total=num_images, desc="Images registered:") as pbar:
                pbar.update(0, force=True)
                recs = pycolmap.incremental_mapping(
                    database_path,
                    imgs_dir,
                    output,
                    initial_image_pair_callback=lambda: pbar.update(2),
                    next_image_callback=lambda: pbar.update(1),
                )

            
    elif args['pipeline'] == "SIFT":
        output = output_dir+"/mvs"
    
        pycolmap.extract_features(database_path, imgs_dir)
        pycolmap.match_exhaustive(database_path)
        num_images = pycolmap.Database(database_path).num_images


        with enlighten.Manager() as manager:
            with manager.counter(total=num_images, desc="Images registered:") as pbar:
                pbar.update(0, force=True)
                recs = pycolmap.incremental_mapping(
                    database_path,
                    imgs_dir,
                    output,
                    initial_image_pair_callback=lambda: pbar.update(2),
                    next_image_callback=lambda: pbar.update(1),
        )

    if args["saveply"]:
        reconstruction = pycolmap.Reconstruction(output_dir)
        reconstruction.write_text(output_dir)  # text format
        print("Result is saving in {}".format(output_dir+"rec1.ply"))
        reconstruction.export_PLY("rec1.ply")  # PLY format