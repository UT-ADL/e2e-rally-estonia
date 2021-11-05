import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from video_creator import create_prediction_video

ROOT_PATH = Path("/media/romet/data2/datasets/rally-estonia/dataset")

DATASETS = [
    "2021-05-20-12-36-10_e2e_sulaoja_20_30",
    "2021-05-20-12-43-17_e2e_sulaoja_20_30",
    "2021-05-20-12-51-29_e2e_sulaoja_20_30",
    "2021-05-20-13-44-06_e2e_sulaoja_10_10",
    "2021-05-20-13-51-21_e2e_sulaoja_10_10",
    "2021-05-20-13-59-00_e2e_sulaoja_10_10",
    "2021-05-28-15-07-56_e2e_sulaoja_20_30",
    "2021-05-28-15-17-19_e2e_sulaoja_20_30",
    "2021-06-07-14-06-31_e2e_rec_ss6",
    "2021-06-07-14-09-18_e2e_rec_ss6",
    "2021-06-07-14-36-16_e2e_rec_ss6",
    "2021-06-09-13-14-51_e2e_rec_ss2",
    "2021-06-09-13-55-03_e2e_rec_ss2_backwards",
    "2021-06-09-14-58-11_e2e_rec_ss3",
    "2021-06-09-15-42-05_e2e_rec_ss3_backwards",
    "2021-06-09-16-24-59_e2e_rec_ss13",
    "2021-06-09-16-50-22_e2e_rec_ss13_backwards",
    "2021-06-10-12-59-59_e2e_ss4",
    "2021-06-10-13-19-22_e2e_ss4_backwards",
    "2021-06-10-13-51-34_e2e_ss12",
    "2021-06-10-14-02-24_e2e_ss12_backwards",
    "2021-06-10-14-44-24_e2e_ss3_backwards",
    "2021-06-10-15-03-16_e2e_ss3_backwards",
    "2021-06-14-11-08-19_e2e_rec_ss14",
    "2021-06-14-11-22-05_e2e_rec_ss14",
    "2021-06-14-11-43-48_e2e_rec_ss14_backwards",
    "2021-09-24-11-19-25_e2e_rec_ss10",
    "2021-09-24-11-40-24_e2e_rec_ss10_2",
    "2021-09-24-12-02-32_e2e_rec_ss10_3",
    "2021-09-24-12-21-20_e2e_rec_ss10_backwards",
    "2021-09-24-13-39-38_e2e_rec_ss11",
    "2021-09-30-13-57-00_e2e_rec_ss14",
    "2021-09-30-15-03-37_e2e_ss14_from_half_way",
    "2021-09-30-15-20-14_e2e_ss14_backwards",
    "2021-10-07-11-44-52_e2e_rec_ss3_backwards",
    "2021-10-07-13-22-35_e2e_rec_ss4_backwards",
    "2021-10-11-17-10-23_e2e_rec_last_part",
    "2021-10-11-17-14-40_e2e_rec_backwards",
    "2021-10-11-17-20-12_e2e_rec_backwards",
    "2021-10-20-14-55-47_e2e_rec_vastse_ss13_17",
    "2021-10-20-15-11-29_e2e_rec_vastse_ss13_17_back",
    "2021-10-20-13-57-51_e2e_rec_neeruti_ss19_22",
    "2021-10-20-14-15-07_e2e_rec_neeruti_ss19_22_back",
    "2021-10-26-10-49-06_e2e_rec_ss20_elva",
    "2021-10-26-11-08-59_e2e_rec_ss20_elva_back",
    "2021-10-25-17-31-48_e2e_rec_ss2_arula",
    "2021-10-25-17-06-34_e2e_rec_ss2_arula_back",
    "2021-10-11-14-50-59_e2e_rec_vahi",
    "2021-10-14-13-08-51_e2e_rec_vahi_backwards",

    "2021-05-28-15-19-48_e2e_sulaoja_20_30",
    "2021-06-07-14-20-07_e2e_rec_ss6",
    "2021-06-14-11-22-05_e2e_rec_ss14",
    "2021-10-07-11-05-13_e2e_rec_ss3",
    "2021-09-24-14-03-45_e2e_rec_ss11_backwards",

    "2021-10-11-16-06-44_e2e_rec_ss2",
    "2021-09-30-15-56-59_e2e_ss14_attempt_2"
]


def create_video(dataset_folder):
    model_path = 'models/20211102005651_autumn-v2/best.pt'
    create_prediction_video(ROOT_PATH / dataset_folder, model_path=model_path)


def main():
    pool = ProcessPoolExecutor(max_workers=16)
    list(pool.map(create_video, DATASETS))

if __name__ == '__main__':
    main()
