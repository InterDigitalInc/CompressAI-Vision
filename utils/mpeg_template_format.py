from glob import iglob
from os.path import join
import pandas as pd
import os
import argparse

def _read_df_rec(path, fn_regex=r'summary.csv'):
    return pd.concat((pd.read_csv(f) for f in iglob(join(path, '**', fn_regex), recursive=True)), ignore_index=True)

def _add_columns(result_df):
    result_df.insert(loc=4, column='y_psnr', value=['' for i in range(result_df.shape[0])])
    result_df.insert(loc=6, column='feat_cov_plus_nn_part_1', value=result_df['nn_part_1'].tolist())
    result_df.insert(loc=8, column='encode_total', value=(result_df['feat_cov_plus_nn_part_1'] + result_df['encode']).tolist())
    result_df.insert(loc=10, column='inv_conv_plus_nn_part_2', value=result_df['nn_part_2'].tolist())
    result_df.insert(loc=12, column='dec_total', value=(result_df['inv_conv_plus_nn_part_2'] + result_df['decode']).tolist())
    return result_df

def _generate_sfu_csv(result_path, dataset_name):

    result_df = _read_df_rec(result_path)
    final_csv_path = os.path.join(result_path, f'final_{dataset_name}.csv')

    result_df[['name', 'w_h', 'fps']] = result_df['Dataset'].str.split(
        pat='_', 
        expand=True,
    )
    result_df[['w', 'h']] = result_df['w_h'].str.split(
        pat='x', 
        expand=True,
    )

    # sort
    result_df['total_pixel'] = result_df['w'].astype(int) * result_df['h'].astype(int)
    result_df = result_df.sort_values(by=['total_pixel', 'Dataset', 'qp'], ascending=[False, False, True])
    result_df = result_df.drop(columns=['w', 'h','total_pixel', 'name', 'w_h', 'fps'])

    # add columns
    result_df = _add_columns(result_df)

    # save
    result_df.to_csv(final_csv_path, sep=',', encoding='utf-8') 
    print(result_df)
    print(f"Final CSV Saved at: {final_csv_path}")

def _generate_csv(result_path, dataset_name):

    result_df = _read_df_rec(result_path)
    final_csv_path = os.path.join(result_path, f'final_{dataset_name}.csv')

    # sort
    result_df = result_df.sort_values(by=['Dataset', 'qp'], ascending=[True, True])

    # add columns
    result_df = _add_columns(result_df)

    # save
    result_df.to_csv(final_csv_path, sep=',', encoding='utf-8') 
    print(result_df)
    print(f"Final CSV Saved at: {final_csv_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', required=True)
    parser.add_argument('--dataset', required=True)

    args = parser.parse_args()

    if (args.dataset) == 'SFU':
        _generate_sfu_csv(args.result_path, args.dataset)
    elif (args.dataset) == 'TVD' or (args.dataset) == 'OIV6' or (args.dataset) == 'HIEVE':
        _generate_csv(args.result_path, args.dataset)
    else:
        print("Not Implemented!")

