import argparse
import os
from glob import iglob
from os.path import join
import pandas as pd

DATASETS = ["TVD", "SFU", "OIV6", "HIEVE"]


def _read_df_rec(path, fn_regex=r"summary.csv"):
    return pd.concat(
        (pd.read_csv(f) for f in iglob(join(path, "**", fn_regex), recursive=True)),
        ignore_index=True,
    )


def pd_append(df1, df2):
    out = pd.concat([df1, df2], ignore_index=True)
    out.reset_index()
    return out


def classwise_computation(result_df, classes: dict):
    classwise = pd.DataFrame(columns=result_df.columns)
    classwise.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for tag, item in classes.items():
        output = compute_class_wise_results(result_df, tag, item)
        classwise = pd_append(classwise, output)

    return classwise


def compute_class_wise_results(result_df, name, sequences):
    samples = None
    num_points = prev_num_points = -1
    output = pd.DataFrame(columns=result_df.columns)
    output.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    for seq in sequences:
        d = result_df.loc[(result_df["Dataset"] == seq)]

        if samples is None:
            samples = d
        else:
            samples = pd_append(samples, d)

        if prev_num_points == -1:
            num_points = prev_num_points = d.shape[0]
        else:
            assert prev_num_points == d.shape[0]

    samples["length"] = samples["num_of_coded_frame"] / samples["fps"]

    for i in range(num_points):
        # print(f"Set - {i}")
        points = samples.iloc[range(i, samples.shape[0], num_points)]
        total_length = points["length"].sum()

        # print(points)

        new_row = {
            output.columns[0]: [
                name,
            ],
            output.columns[1]: [
                i,
            ],
        }
        for column in output.columns[2:]:
            if column == "end_accuracy":
                new_row[column] = -1
                continue

            weighted = points[column] * points["length"]
            new_row[column] = [
                (1 / total_length) * weighted.sum(),
            ]

        output = pd_append(output, pd.DataFrame(new_row))

    return output


def add_columns(result_df, is_remote_inference: bool = False):
    result_df.insert(
        loc=4, column="y_psnr", value=["" for i in range(result_df.shape[0])]
    )
    if not is_remote_inference:
        result_df.insert(
            loc=6, column="feat_cov_plus_nn_part_1", value=result_df["nn_part_1"].tolist()
        )
        result_df.insert(
            loc=8,
            column="encode_total",
            value=(result_df["feat_cov_plus_nn_part_1"] + result_df["encode"]).tolist(),
        )
        result_df.insert(
            loc=10, column="inv_conv_plus_nn_part_2", value=result_df["nn_part_2"].tolist()
        )
        result_df.insert(
            loc=12,
            column="dec_total",
            value=(result_df["inv_conv_plus_nn_part_2"] + result_df["decode"]).tolist(),
        )
    return result_df


def generate_sfu_csv(result_path, dataset_name, is_remote_inference: bool = False):
    seq_list = [
        "Traffic_2560x1600_30",
        "Kimono_1920x1080_24",
        "ParkScene_1920x1080_24",
        "Cactus_1920x1080_50",
        "BasketballDrive_1920x1080_50",
        "BQTerrace_1920x1080_60",
        "BasketballDrill_832x480_50",
        "BQMall_832x480_60",
        "PartyScene_832x480_50",
        "RaceHorsesC_832x480_30",
        "BasketballPass_416x240_50",
        "BQSquare_416x240_60",
        "BlowingBubbles_416x240_50",
        "RaceHorses_416x240_30",
    ]

    classes = {
        "CLASS_AB": [
            "Traffic_2560x1600_30",
            "Kimono_1920x1080_24",
            "ParkScene_1920x1080_24",
            "Cactus_1920x1080_50",
            "BasketballDrive_1920x1080_50",
            "BQTerrace_1920x1080_60",
        ],
        "CLASS_C": [
            "BasketballDrill_832x480_50",
            "BQMall_832x480_60",
            "PartyScene_832x480_50",
            "RaceHorsesC_832x480_30",
        ],
        "CLASS_D": [
            "BasketballPass_416x240_50",
            "BQSquare_416x240_60",
            "BlowingBubbles_416x240_50",
            "RaceHorses_416x240_30",
        ],
    }

    result_df = _read_df_rec(result_path)

    # sort
    sorterIndex = dict(zip(seq_list, range(len(seq_list))))
    result_df["ds_rank"] = result_df["Dataset"].map(sorterIndex)
    result_df.sort_values(["ds_rank", "qp"], ascending=[True, True], inplace=True)
    result_df.drop(columns=["ds_rank"], inplace=True)

    classwise_result_df = classwise_computation(result_df, classes)

    # drop columns
    result_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    # concatenate classwise results
    result_df = pd_append(result_df, classwise_result_df)

    # add columns
    result_df = add_columns(result_df, is_remote_inference)

    # save
    final_csv_path = os.path.join(result_path, f"final_{dataset_name}.csv")
    result_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(result_df)
    print(f"Final CSV Saved at: {final_csv_path}")


def generate_csv_by_class(result_df, sequences_by_class: dict, is_remote_inference: bool = False):
    # sort
    result_df = result_df.sort_values(by=["Dataset", "qp"], ascending=[True, True])

    classwise_result_df = classwise_computation(result_df, sequences_by_class)

    # drop columns
    result_df.drop(columns=["fps", "num_of_coded_frame"], inplace=True)

    # concatenate class-wise results
    result_df = pd_append(result_df, classwise_result_df)

    # add columns
    result_df = add_columns(result_df, is_remote_inference)

    return result_df


def generate_tvd_csv(result_path, dataset_name):
    seqs_by_class = {"Overall": ["TVD-01", "TVD-02", "TVD-03"]}

    result_df = generate_csv_by_class(_read_df_rec(result_path), seqs_by_class)

    # save
    final_csv_path = os.path.join(result_path, f"final_{dataset_name}.csv")
    result_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(result_df)
    print(f"Final CSV Saved at: {final_csv_path}")


def generate_hieve_csv(result_path, dataset_name):
    hieve_1080p = {"HIEVE-1080P": ["13", "16"]}
    hieve_720p = {"HIEVE-720P": ["2", "17", "18"]}

    all_results_df = _read_df_rec(result_path)
    output_df = []
    for seqs_by_class in [hieve_1080p, hieve_720p]:
        result_df = generate_csv_by_class(all_results_df, seqs_by_class)
        if len(output_df) == 0:
            output_df = result_df
        else:
            output_df = pd_append(output_df, result_df)

    # save
    final_csv_path = os.path.join(result_path, f"final_{dataset_name}.csv")
    output_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(output_df)
    print(f"Final CSV Saved at: {final_csv_path}")


def generate_csv(result_path, dataset_name, is_remote_inference: bool = False):
    result_df = _read_df_rec(result_path)

    # sort
    result_df = result_df.sort_values(by=["Dataset", "qp"], ascending=[True, True])

    # add columns
    result_df = add_columns(result_df, is_remote_inference)

    # save
    final_csv_path = os.path.join(result_path, f"final_{dataset_name}.csv")
    result_df.to_csv(final_csv_path, sep=",", encoding="utf-8")
    print(result_df)
    print(f"Final CSV Saved at: {final_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--result_path", required=True)
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=DATASETS,
        required=True,
    )
    parser.add_argument(
        "--remote_inference",
        action="store_true",
        default=False,
        help="Collect results from a remote inference pipeline.",
    )

    args = parser.parse_args()

    if (args.dataset) == "SFU":
        generate_sfu_csv(args.result_path, args.dataset, args.remote_inference)
    elif (args.dataset) == "OIV6":
        generate_csv(args.result_path, args.dataset, args.remote_inference)
    elif (args.dataset) == "TVD":
        generate_tvd_csv(args.result_path, args.dataset, args.remote_inference)
    elif (args.dataset) == "HIEVE":
        generate_hieve_csv(args.result_path, args.dataset, args.remote_inference)
    else:
        print("Not Implemented!")
