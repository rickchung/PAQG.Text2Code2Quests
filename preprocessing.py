"""
Clean and preprocess some datasets for the study.
This script is not about producing data for an nn.Module.
You can find more detail for that case in `nn_data_processor.py`.
"""

from pathlib import Path

path_thinkjava2 = Path("Datasets", "ThinkJava2-master")
path_chapters = {
    'variable': Path(path_thinkjava2, 'ch02.tex'),
    'io': Path(path_thinkjava2, 'ch03.tex'),
    'methods_testing': Path(path_thinkjava2, 'ch04.tex'),
    'conditionals': Path(path_thinkjava2, 'ch05.tex'),
    'loops_strings': Path(path_thinkjava2, 'ch06.tex'),
    'arrays_references': Path(path_thinkjava2, 'ch07.tex'),
}
