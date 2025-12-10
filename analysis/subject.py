from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import os


# Create a class to handle subject information
class SubjectInfo:
    def __init__(
        self,
        subject_id: str,
        raw_data_dir: Path | None = None,
        derivatives_dir: Path | None = None,
        working_dir: Path | None = None,
        results_dir: Path | None = None,
    ):
        if raw_data_dir is None:
            raw_data_dir = Path("data")
        if derivatives_dir is None:
            derivatives_dir = Path("derivatives")
        if working_dir is None:
            working_dir = Path("working")
        if results_dir is None:
            results_dir = Path("results")

        working_dir.mkdir(parents=True, exist_ok=True)

        self.subject_id: str = subject_id
        self.raw_data_dir: Path = raw_data_dir
        self.derivatives_dir: Path = derivatives_dir
        self.working_dir: Path = working_dir
        self.results_dir: Path = results_dir

    def get_confounds_file(self, task_name: str):
        confounds_file = (
            self.derivatives_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-{task_name}_bold_confounds.tsv"
        )
        if not confounds_file.exists():
            raise FileNotFoundError(f"Confounds file not found: {confounds_file}")
        return confounds_file

    def get_event_file(self, task_name: str):
        event_file = (
            self.raw_data_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-{task_name}_events.tsv"
        )
        if not event_file.exists():
            raise FileNotFoundError(f"Event file not found: {event_file}")
        return event_file

    def get_subject_id(self):
        return self.subject_id
    
    def filter_confounds(self, task_name: str, confounds_columns: list[str]):
        confounds_file = self.get_confounds_file(task_name)
        confounds_df = pd.read_csv(confounds_file, sep="\t", na_values="n/a")
        selected_columns = confounds_df[confounds_columns]

        # map na to 0
        selected_columns = selected_columns.fillna(0).astype(float)

        selected_columns = selected_columns.round(3)

        # get the file name without the extension and add filtered to the end before the extension
        confounds_folder = self.working_dir / self.subject_id
        confounds_folder.mkdir(parents=True, exist_ok=True)

        confounds_file_name = (
            confounds_folder
            / f"{self.subject_id}_task-{task_name}_confounds_filtered.tsv"
        )
        selected_columns.to_csv(confounds_file_name, sep=" ", header=False, index=False)
        return confounds_file_name
    
    def get_anatomical_file(self):
        anat_file = (
            self.derivatives_dir
            / self.subject_id
            / "anat"
            / f"{self.subject_id}_T1w_space-MNI152NLin2009cAsym_preproc.nii.gz"
        )
        if not anat_file.exists():
            raise FileNotFoundError(f"Anatomical file not found: {anat_file}")
        return anat_file

    def get_standard_brain_mask(self, task_name: str):
        mask_file = (
            self.derivatives_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-{task_name}_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"
        )
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
        return mask_file

    def get_transformed_brain_mask(self, task_name: str):
        mask_file = (
            self.derivatives_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-{task_name}_bold_space-MNI152NLin2009cAsym_brainmask_processed.nii.gz"
        )
        return mask_file

    def get_processed_bold_file(self, task_name: str):
        bold_file = (
            self.derivatives_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-{task_name}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"
        )
        if not bold_file.exists():
            raise FileNotFoundError(f"Bold file not found: {bold_file}")
        return bold_file

    def get_smoothed_bold_file(self, task_name: str):
        bold_file = (
            self.derivatives_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-{task_name}_bold_space-MNI152NLin2009cAsym_preproc_smoothed.nii.gz"
        )
        return bold_file

    def get_events(self, task_name: str):
        events_file = self.get_event_file(task_name)
        if not events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")

        events_df = pd.read_csv(events_file, sep="\t", na_values="n/a")
        return events_df
    
    def get_working_task_dir(self, task_name: str):
        task_dir = self.working_dir / self.subject_id / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir
    
    def get_output_dir(self):
        task_dir = self.results_dir / self.subject_id 
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def create_fsl_explanatory_variable_file(
        self,
        task_name: str,
        events_df: pd.DataFrame,
        ev_name: str,
        reference_duration_column: str | None = None,
        reference_amplitude_column: str | None = None,
        fixed_duration: float = 1.0,
        fixed_amplitude: float = 1.0,
    ):
        events_df = events_df.dropna(subset=["onset"])
        if events_df.empty:
            warnings.warn(f"No events found in the events file for {ev_name}")
            


        onsets = events_df["onset"].to_list()
        if reference_duration_column is not None: 
            # if we have a reference duration column
            durations = events_df[reference_duration_column].to_list()
        else: 
            # if we don't have a reference duration column, use a fixed duration
            durations = [fixed_duration] * len(onsets)

        if reference_amplitude_column is not None:
            # center the amplitude around 0
            amplitudes = events_df[reference_amplitude_column]
            amplitudes = (
                amplitudes.apply(lambda x: x - np.mean(amplitudes)).round(3).to_list()
            )

        else:
            amplitudes = [fixed_amplitude] * len(onsets)

        fsl_explanatory_variable_df = pd.DataFrame(
            {"0": onsets, "1": durations, "2": amplitudes}
        )

        ev_file_folder = self.working_dir / self.subject_id / task_name 
        ev_file_folder.mkdir(parents=True, exist_ok=True)
        ev_file_path = ev_file_folder / f"{ev_name}.txt"

        fsl_explanatory_variable_df.to_csv(
            ev_file_path,
            sep="\t",
            header=False,
            index=False,
        )

        return ev_file_path


    def create_fsl_evs_scap(self):
        events_df = self.get_events("scap")

        # separate correct and incorrect events
        correct_events = events_df[events_df["ResponseAccuracy"] == "CORRECT"]
        incorrect_events = events_df[events_df["ResponseAccuracy"] == "INCORRECT"]

        evs = []
        n_regressors = 25 # number of regressors as stated in the original paper

        # initialize orthogonalization dictionary to remove reaction time correlation PS: a weird FSL format
        orthogonalization = {x: {y:0 for y in range(1,n_regressors+1)} for x in range(1,n_regressors+1)}

        # load and delay levels as stated in the original paper
        load_levels = [1, 3, 5, 7]
        delay_levels = [1.5, 3.0, 4.5]

        for load_level in load_levels:
            for delay_level in delay_levels:
                condition_mask = (correct_events["Load"] == load_level) & (
                    correct_events["Delay"] == delay_level
                )
                condition_events = correct_events[condition_mask]

                if condition_events.empty:
                    warnings.warn(
                        f"No events found for load {load_level} and delay {delay_level}"
                    )
                
                # condition_events["onset"] = condition_events["onset"] + delay_level
                condition_events.loc[:, "onset"] = condition_events["onset"] + delay_level

                fixed_ev = self.create_fsl_explanatory_variable_file(
                    "scap",
                    condition_events,
                    f"LOAD{load_level}_DELAY{delay_level}",
                    fixed_duration=3.0,
                    fixed_amplitude=1.0,
                )

                reaction_time_ev = self.create_fsl_explanatory_variable_file(
                    "scap",
                    condition_events,
                    f"LOAD{load_level}_DELAY{delay_level}_rt",
                    reference_duration_column="ReactionTime",
                    fixed_amplitude=1.0,
                )

                evs.append(fixed_ev)
                fix_index = len(evs) 

                evs.append(reaction_time_ev)
                reaction_time_index = len(evs)
                orthogonalization[reaction_time_index][fix_index] = 1
                orthogonalization[reaction_time_index][0] = 1


        
        if not incorrect_events.empty:
            error_ev = self.create_fsl_explanatory_variable_file(
                "scap",
                incorrect_events,
                "Error",
                fixed_duration=3.0,
                fixed_amplitude=1.0,
            )
            evs.append(error_ev)
        

        valid_ev_files = []
        for ev_file in evs:
            if os.path.exists(ev_file) and os.path.getsize(ev_file) > 0:
                valid_ev_files.append(ev_file.absolute().as_posix())
            else:
                print(f"Warning: Empty or missing EV file: {ev_file}")
        
        # Validation
        expected_regressors = len(load_levels) * len(delay_levels) * 2 + 1  # 25
        if len(valid_ev_files) != expected_regressors:
            print(f"Warning: Expected {expected_regressors} regressors, got {len(valid_ev_files)}")

        
        return {
            'event_files': valid_ev_files,
            'orthogonalization': orthogonalization
        }

    def get_task_stats(self, task_name: str, stat_name: str):
        stat_file = (
            self.derivatives_dir
            / "task"
            / self.subject_id
            / f"{task_name}.feat"
            / "stats"
            / f"{stat_name}.nii.gz"
        )
        if not stat_file.exists():
            raise FileNotFoundError(f"Cope file not found: {stat_file}")
        return stat_file
    
    def get_rest_image(self):
        rest_image = (
            self.derivatives_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"
        )
        return rest_image
    
    def get_rest_confounds_file(self):
        confounds_file = (
            self.derivatives_dir
            / self.subject_id
            / "func"
            / f"{self.subject_id}_task-rest_bold_confounds.tsv"
        )
        return confounds_file