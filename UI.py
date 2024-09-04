import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox, Progressbar, Style

import os
import pycolmap
import enlighten
import yaml
import numpy as np
from Reconstruction import logger, timer
from Reconstruction.config import Config
from Reconstruction.image_matching import ImageMatching
from Reconstruction.io.h5_to_db import export_to_colmap
from Reconstruction.parser import parse_cli

import pathlib
import sys
import threading
import open3d as o3d
from pycolmap import logging

class RedirectedOutput:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass 

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Giao diện")

        self.img_folder = ""
        self.output_folder = ""
        self.pipeline = "superpoint+superglue"
        self.run_without_matching = tk.BooleanVar(value=False)
        self.db_file = ""
        
        # Left frame for buttons and options
        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Right frame for output display
        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Button to select image folder
        self.btn_select_img_folder = tk.Button(left_frame, text="Chọn folder chứa ảnh", command=self.select_img_folder)
        self.btn_select_img_folder.pack(pady=10)

        # Label to display selected image folder
        self.lbl_img_folder = tk.Label(left_frame, text="Folder chứa ảnh: Chưa chọn")
        self.lbl_img_folder.pack(pady=10)

        # Button to select output folder
        self.btn_select_output_folder = tk.Button(left_frame, text="Chọn folder output", command=self.select_output_folder)
        self.btn_select_output_folder.pack(pady=10)

        # Label to display selected output folder
        self.lbl_output_folder = tk.Label(left_frame, text="Folder output: Chưa chọn")
        self.lbl_output_folder.pack(pady=10)

        # Combobox for pipeline selection
        self.pipeline_var = tk.StringVar()
        self.combobox = Combobox(left_frame, textvariable=self.pipeline_var, state="readonly")
        self.combobox['values'] = ("superpoint+superglue", "sift+kornia_matcher")
        self.combobox.current(0)  # set default value
        self.combobox.pack(pady=10)

        # Checkbox for runwithoutmatching
        self.chk_run_without_matching = tk.Checkbutton(left_frame, text="Run without matching", variable=self.run_without_matching, command=self.toggle_db_file)
        self.chk_run_without_matching.pack(pady=10)

        # Button to select database file
        self.btn_select_db_file = tk.Button(left_frame, text="Chọn file database", command=self.select_db_file, state=tk.DISABLED)
        self.btn_select_db_file.pack(pady=10)

        # Button to run
        self.btn_run = tk.Button(left_frame, text="Run", command=self.run)
        self.btn_run.pack(pady=10)

        # Button to run matching only
        self.btn_matching = tk.Button(left_frame, text="Matching Only", command=self.run_matching)
        self.btn_matching.pack(pady=10)

        # Progress bar
        self.progress = Progressbar(left_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(pady=10)

        # Customize combobox style
        style = Style()
        style.theme_use('default')
        style.configure("TCombobox", arrowcolor='black', arrowsize=12)  # Adjust arrowsize if necessary

        # Text widget to display output
        self.text_output = tk.Text(right_frame, wrap=tk.WORD)
        self.text_output.pack(fill=tk.BOTH, expand=True)

        # Redirect stdout to the text widget
        sys.stdout = RedirectedOutput(self.text_output)
        sys.stderr = RedirectedOutput(self.text_output)

    
    def select_img_folder(self):
        self.img_folder = filedialog.askdirectory()
        if self.img_folder:
            self.lbl_img_folder.config(text=f"Folder chứa ảnh: {self.img_folder}")

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        if self.output_folder:
            self.lbl_output_folder.config(text=f"Folder output: {self.output_folder}")

    def toggle_db_file(self):
        if self.run_without_matching.get():
            self.btn_select_db_file.config(state=tk.NORMAL)
        else:
            self.btn_select_db_file.config(state=tk.DISABLED)
            self.db_file = ""  # Clear db_file if checkbox is unchecked

    def select_db_file(self):
        self.db_file = filedialog.askopenfilename(filetypes=[("Database files", "*.db")])
        if self.db_file:
            print(f"File database được chọn: {self.db_file}")

    def run(self):
        if not self.img_folder or not self.output_folder:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn cả folder chứa ảnh và folder output")
            return
        
        if self.run_without_matching.get() and not self.db_file:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn file database")
            return
        
        self.pipeline = self.pipeline_var.get()
        self.progress["value"] = 0
        self.root.update_idletasks()
        threading.Thread(target=self.process_images).start()

    def run_matching(self):
        if not self.img_folder or not self.output_folder:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn cả folder chứa ảnh và folder output")
            return

        self.pipeline = self.pipeline_var.get()
        self.progress["value"] = 0
        self.root.update_idletasks()
        threading.Thread(target=self.match_only).start()

    def process_images(self):
        print("...............Running..............")
        # Define arguments manually
        args = {
            "dir": self.img_folder,
            "images": self.img_folder,
            "pipeline": self.pipeline,
            "outs": pathlib.Path(os.path.join(self.output_folder, "output")),
            "config_file": None,
            "quality": "medium",
            "tiling": "none",
            "strategy": "sequential",
            "pair_file": None,
            "overlap": 1,
            "global_feature": "netvlad",
            "db_path": self.db_file if self.run_without_matching.get() else None,
            "upright": False,
            "skip_reconstruction": False,
            "force": True,
            "verbose": False,
            "camera_options": "./Reconstruction/config/cameras.yaml",
            "saveply": True,
        }

        imgs_dir = args["images"]
        output_dir = self.output_folder
        database_path = os.path.join(output_dir, "database.db")
        
        if not self.run_without_matching.get():
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

            # Matching if not running without matching
            match_path = img_matching.match_pairs(feature_path)
            timer.update("matching")

            # # If features have been extracted on "upright" images, this function brings features back to their original image orientation
            # if config.general["upright"]:
            #     img_matching.rotate_back_features(feature_path)
            #     timer.update("rotate_back_features")

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

        sparse = os.path.join(output_dir, "sparse")
        dense = os.path.join(output_dir, "dense")
        with enlighten.Manager() as manager:
            with manager.counter(total=num_images, desc="Images registered:") as pbar:
                pbar.update(0, force=True)
                recs = pycolmap.incremental_mapping(
                    database_path,
                    imgs_dir,
                    sparse,
                    initial_image_pair_callback=lambda: pbar.update(2),
                    next_image_callback=lambda: pbar.update(1),
                )
                #recs[0].write(sparse)
        
        for idx, rec in recs.items():
            logging.info(f"#{idx} {rec.summary()}")

        if args["saveply"]:
            reconstruction = pycolmap.Reconstruction(os.path.join(sparse, "0"))
            reconstruction.write_text(os.path.join(sparse, "0"))  # text format
            reconstruction.export_PLY(os.path.join(output_dir, "rec.ply"))  # PLY format

        self.progress["value"] = 100
        self.root.update_idletasks()
        
        point_cloud = o3d.io.read_point_cloud(os.path.join(output_dir, "rec.ply"))
        o3d.visualization.draw_geometries([point_cloud])
        #point_cloud.scale(0.01, center=point_cloud.get_center())
        #point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Sử dụng thuật toán Poisson
        point_cloud.estimate_normals()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=11)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.035)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        #mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
        
        messagebox.showinfo("Hoàn thành", "Quá trình đã hoàn thành")

    def match_only(self):
        # Define arguments manually
        args = {
            "dir": self.img_folder,
            "images": self.img_folder,
            "pipeline": self.pipeline,
            "config_file": None,
            "quality": "high",
            "tiling": "none",
            "strategy": "sequential",
            "pair_file": None,
            "overlap": 1,
            "global_feature": "netvlad",
            "db_path": None,
            "upright": False,
            "skip_reconstruction": False,
            "force": True,
            "verbose": False,
            "camera_options": "./config/cameras.yaml",
            "saveply": False,
        }

        imgs_dir = args["images"]
        output_dir = self.output_folder
        database_path = os.path.join(output_dir, "database.db")
        
        if args['pipeline'] == "superpoint+superglue":
            # Build configuration
            config = Config(args)

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

        elif args['pipeline'] == "SIFT":
            output = os.path.join(output_dir, "mvs")
        
            pycolmap.extract_features(database_path, imgs_dir)
            pycolmap.match_exhaustive(database_path)

        self.progress["value"] = 100
        self.root.update_idletasks()
        messagebox.showinfo("Hoàn thành", "Matching đã hoàn thành")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
