from twinweaver import DataManager, ConverterPretrain, Config
import pandas as pd


class ConvertToText:
    def __init__(self):
        # Set basics
        self.config = Config()  # Override values here to customize pipeline

        # Manually set from constant
        self.config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        self.config.constant_birthdate_column = "birthyear"

        # Load data
        df_events = pd.read_csv("./examples/example_data/events.csv")
        df_constant = pd.read_csv("./examples/example_data/constant.csv")
        df_constant_description = pd.read_csv("./examples/example_data/constant_description.csv")

        # Init data managers
        self.dm = DataManager(config=self.config)
        self.dm.load_indication_data(
            df_events=df_events, df_constant=df_constant, df_constant_description=df_constant_description
        )
        self.dm.process_indication_data()
        self.dm.setup_unique_mapping_of_events()
        self.dm.setup_dataset_splits()

        #: set up converter
        self.converter = ConverterPretrain(config=self.config, dm=self.dm)

        # Due to internal data issue, we skip this one when doing reverse checks
        self.reverse_patient_skip_list = ["electrophoresis m spike"]

    def convert_full_to_string_for_one_patient(self, patientid):
        patient_data = self.dm.get_patient_data(patientid)

        #: convert patient data using ConverterPretrain
        p_converted = self.converter.forward_conversion(patient_data["events"], patient_data["constant"])

        #: convert extras into JSON using df.to_json
        internal_meta = p_converted["meta"].copy()
        internal_meta["split"] = self.dm.get_patient_split(patientid=patientid)
        p_converted["meta"] = {
            "patientid": patientid,
            "split": self.dm.get_patient_split(patientid=patientid),
            "constant": p_converted["meta"]["processed_constant"].to_json(orient="split"),
            "events": p_converted["meta"]["events"].to_json(orient="split"),
        }

        return [(p_converted, internal_meta)]

    def assess_reverse_conversion(self, all_patient_data) -> None:
        """
        Assesses the reverse conversion for a single patient to ensure data integrity.

        Parameters
        ----------
        all_patient_data : list of tuples
            A list of tuples, each containing converted patient data and internal metadata.
        """

        # Split up
        converted_data, internal_meta = all_patient_data[0]

        # Log that testing patient
        print("Assessing reverse conversion for patient" + str(converted_data["meta"]["patientid"]))

        #: do reverse conversion using ConverterPretrain
        p_reverse_converted = self.converter.reverse_conversion(
            converted_data["text"], internal_meta, self.dm.unique_events
        )

        #: check differences appropriately
        diff = self.converter.get_difference_in_event_dataframes(
            internal_meta["events"],
            p_reverse_converted["events"],
            skip_genetic=True,
            skip_vals_list=self.reverse_patient_skip_list,
        )

        #: assert that no differences are found, and print patientid if issues are found
        assert diff.shape[0] == 0, f"Patient {internal_meta['patientid']} has differences in reverse conversion: {diff}"


################################### Actual running #######################################
# NOTE: run this from the root folder of twinweaver

converter = ConvertToText()

all_patientids = converter.dm.all_patientids.copy()
all_patientids = all_patientids[:10]


for idx, patientid in enumerate(all_patientids):
    print(idx)

    #: go through all patients and convert them
    patient_data = converter.convert_full_to_string_for_one_patient(patientid)

    # Add check whether it is correct
    converter.assess_reverse_conversion(patient_data)


print("Finished")
