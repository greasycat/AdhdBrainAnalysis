demo: demo_data demo_analysis

data:
	uv run python download_derivatives.py --ids control_subjects_ids.txt adhd_subjects_ids.txt --type fmriprep --tasks rest scap stopsignal taskswitch --output-dir derivatives/

demo_data:
	uv run python download_derivatives.py --ids demo_ids.txt --type task --tasks scap --output-dir derivatives/

demo_analysis:
	uv run python main.py --demo

full:
	uv run python main.py
