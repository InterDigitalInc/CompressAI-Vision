output_dir="/mnt/wekamount/scratch_fcvcm/eimran/runs/"

grid submit --gridfile run_oiv6.yaml 
grid submit --gridfile run_sfu.yaml 
grid submit --gridfile run_tvd.yaml 
grid submit --gridfile run_hieve.yaml

# tar the package
sbatch --dependency=singleton --job-name=generate_csv tar_bitstreams.sh ${output_dir}

#TODO: @eimran plot RD-curves
