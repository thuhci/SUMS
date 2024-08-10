import numpy as np
import pandas as pd
import cv2
import glob
import os
from scipy.interpolate import interp1d
from dataset.data_loader.BaseLoader import BaseLoader

class SUMSLoader(BaseLoader):
    def __init__(self, name, data_path, config_data):
        """Initializes an THUSPO2 dataloader."""
        self.info = config_data.INFO  
        print(data_path)
        print("fsahfsjadjfk")
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories in the specified path (suitable for the THUSPO2 dataset)."""
        "Get all 060200, 060201... files; data_path needs to be changed"
        print(data_path)
        data_dirs = glob.glob(data_path + os.sep + '0602*')
        if not data_dirs:
            raise ValueError(self.dataset_name + ' Data path is empty!')
        dirs = list()
        # data_dirs absolute path
        for data_dir in data_dirs:
            
            subject = int(os.path.split(data_dir)[-1]) # File name directly 060200
            d_dirs = os.listdir(data_dir)
            print(d_dirs)
            # v01 v02 v03 v04
            for dir in d_dirs:    
                items_dirs = os.listdir(data_dir + os.sep + dir) # avi csv 
                for item in items_dirs:
                    if "avi" in item: # If returning all together here
                        dirs.append({'index': dir[1:],
                                    'path': data_dir + os.sep + dir + os.sep +item,
                            'subject': subject,
                            'type': item.split('_')[-1].split('.')[0]
                        })
                
        print(dirs)    
        return dirs
        

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        # Split according to tags v01, v02, v03, v04
        
        data_info = dict()
        for data in data_dirs:
            # index = data['index']
            # data_dir = data['path']
            subject = data['subject']
            # type = data['type'] # face or finger
            # Create a data directory dictionary indexed by subject number
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)
        
        subj_list = list(data_info.keys())  # Get all subject numbers
        subj_list = sorted(subj_list)  # Sort subject numbers
        
        num_subjs = len(subj_list)  # Total number of subjects      
        
        # Get data set split (according to start/end ratio)
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print('Subjects ID used for split:', [subj_list[i] for i in subj_range])

        # Add file paths that meet the split range to the new list
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]
        
        print(data_dirs_new)
        return data_dirs_new        
        
        


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i,  file_list_dict):
    
        
        
        # Read video frames
        video_file = data_dirs[i]['path']
        frames = self.read_video(video_file)

        # Get the directory of the current video
        video_dir = os.path.dirname(video_file)

        # Extract subject ID and experiment ID from the directory path
        subject_id = video_dir.split(os.sep)[-2]
        experiment_id = video_dir.split(os.sep)[-1]  # Assuming experiment ID follows subject ID
        print(f"subject_id: {subject_id}, experiment_id: {experiment_id}")
        # Get BVP, frame timestamps, and SpO2 files
        bvp_file = os.path.join(video_dir, "BVP.csv")
        timestamp_file = os.path.join(video_dir, "frames_timestamp.csv")
        spo2_file = os.path.join(video_dir, "SpO2.csv")

        # Read frame timestamps
        frame_timestamps = self.read_frame_timestamps(timestamp_file)

        # Read BVP data and timestamps
        bvp_timestamps, bvp_values = self.read_bvp(bvp_file)

        # Read SpO2 data and timestamps
        spo2_timestamps, spo2_values = self.read_spo2(spo2_file)

        # Resample BVP data to match video frames
        resampled_bvp = self.synchronize_and_resample(bvp_timestamps, bvp_values, frame_timestamps)
        

        # Resample SpO2 data to match video frames
        resampled_spo2 = self.synchronize_and_resample(spo2_timestamps, spo2_values, frame_timestamps)
        

        # Process frames, BVP signals, and SpO2 signals according to the configuration
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = resampled_bvp
            spo2 = resampled_spo2
        # Label once here
        if "face" in video_file:
            frames_clips, bvps_clips, spo2_clips = self.preprocess(frames, bvps, spo2, config_preprocess, "face")
            # print(f"face Frames clips shape: {frames_clips.shape}")
            # print(f"BVP clips shape: {bvps_clips.shape}")
            # print(f"SpO2 clips shape: {spo2_clips.shape}")
            filename = f"{subject_id}_{experiment_id}"
            input_name_list, label_name_list, spo2_name_list = self.save_multi_process(frames_clips, bvps_clips, spo2_clips, filename)
            file_list_dict[i] = input_name_list
        else:
            frames_clips, _, _ = self.preprocess(frames, None, None, config_preprocess, "finger")
            # print(f"finger Frames clips shape: {frames_clips.shape}"
            filename = f"{subject_id}_{experiment_id}_finger"
            input_name_list = self.save_multi_process_no_labels(frames_clips, filename)
            file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """Load preprocessed data listed in the file list."""
        type_info = self.info.TYPE
        state = self.info.STATE
        print(f"type_info: {type_info}, state: {state}")
        
        file_list_path = self.file_list_path   # Get file list path
        file_list_df = pd.read_csv(file_list_path)  # Read file list
        inputs_temp = file_list_df['input_files'].tolist()  # Get input file list
        inputs_face = [] 
        inputs_finger = [] 
        # v01 v02 v03 v04 face finger configuration information
        for each_input in inputs_temp:
            # print(f"each_input: {each_input}")
            info = each_input.split(os.sep)[-1].split('_')

            state = int(info[1][-1])

            if info[2] == "face":   # face finger
                type = 1
            else:
                type = 2
            # print(f"info:{info}, state: {state}, type: {type}")
            # Filter data according to configuration information

            if (state in self.info.STATE) and (type in self.info.TYPE) and type == 1:
                inputs_face.append(each_input)
            # finger 2
            if (state in self.info.STATE) and (type in self.info.TYPE) \
                and type == 2:
                inputs_finger.append(each_input)
            # print(f"inputs_face_len: {len(inputs_face)}, inputs_finger_len: {len(inputs_finger)}")
        if not inputs_face and not inputs_finger:
            raise ValueError(self.dataset_name + ' Dataset loading error!')
        # single face finger both   
        if len(type_info) == 1:
            if type_info[0] == 1: # face
                inputs_face = sorted(inputs_face)
                labels_bvp = [input_file.replace("face_input", "hr") for input_file in inputs_face]  
                labels_spo2 = [input_file.replace("face_input", "spo2") for input_file in inputs_face]  
                self.inputs = inputs_face
                self.labels_bvp = labels_bvp
                self.labels_spo2 = labels_spo2
                self.preprocessed_data_len = len(inputs_face)   
            else: # finger    
                inputs_finger = sorted(inputs_finger)
                labels_bvp = [input_file.replace("finger_input", "hr") for input_file in inputs_finger]  
                labels_spo2 = [input_file.replace("finger_input", "spo2") for input_file in inputs_finger]  
                self.inputs_finger = inputs_finger
                self.labels_bvp = labels_bvp
                self.labels_spo2 = labels_spo2
                self.preprocessed_data_len = len(inputs_finger)   
        else:
            inputs_face = sorted(inputs_face)
            inputs_finger = sorted(inputs_finger)
            labels_bvp = [input_file.replace("face_input", "hr") for input_file in inputs_face]  
            labels_spo2 = [input_file.replace("face_input", "spo2") for input_file in inputs_face]  
            self.inputs = inputs_face
            self.inputs_finger = inputs_finger
            self.labels_bvp = labels_bvp
            self.labels_spo2 = labels_spo2
            # Mixed training also only requires one of the lengths
            self.preprocessed_data_len = len(inputs_face)   
            print(f"inputs_face: {inputs_face[20]}")
            print(f"inputs_finger: {inputs_finger[20]}")

            

    @staticmethod
    def read_spo2(spo2_file):
        """Reads a SpO2 signal file with timestamps."""
        data = pd.read_csv(spo2_file)
        timestamps = data['timestamp'].values
        spo2_values = data['spo2'].values
        return timestamps, spo2_values

    @staticmethod
    def read_bvp(bvp_file):
        """Reads a BVP signal file with timestamps."""
        data = pd.read_csv(bvp_file)
        timestamps = data['timestamp'].values
        bvp_values = data['bvp'].values
        return timestamps, bvp_values

    @staticmethod
    def read_frame_timestamps(timestamp_file):
        """Reads timestamps for each video frame."""
        data = pd.read_csv(timestamp_file)
        return data['timestamp'].values

    @staticmethod
    def synchronize_and_resample(timestamps_data, data_values, timestamps_frames):
        """Synchronize and resample data to match video frame timestamps."""
        interpolator = interp1d(timestamps_data, data_values, bounds_error=False, fill_value="extrapolate")
        resampled_data = interpolator(timestamps_frames)
        return resampled_data

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames."""
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.array(frames)

    def preprocess(self, frames, bvps, spo2, config_preprocess, video_type):
        """Preprocesses a pair of data."""
        if video_type == "face":
            DO_CROP_FACE = True
        else:
            DO_CROP_FACE = False
        frames = self.crop_face_resize(
            frames,
            DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)
        
        
        

        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)

        if bvps is not None and spo2 is not None:
            if config_preprocess.LABEL_TYPE == "Raw":
                pass
            elif config_preprocess.LABEL_TYPE == "DiffNormalized":
                bvps = BaseLoader.diff_normalize_label(bvps)
            elif config_preprocess.LABEL_TYPE == "Standardized":
                bvps = BaseLoader.standardized_label(bvps)
            else:
                raise ValueError("Unsupported label type!")

            if config_preprocess.DO_CHUNK:
                frames_clips, bvps_clips, spo2_clips = self.chunk(data, bvps, spo2, config_preprocess.CHUNK_LENGTH, video_type)
            else:
                frames_clips = np.array([data])
                bvps_clips = np.array([bvps])
                spo2_clips = np.array([spo2])

            return frames_clips, bvps_clips, spo2_clips
        else:
            if config_preprocess.DO_CHUNK:
                frames_clips, _, _ = self.chunk(data, None, None, config_preprocess.CHUNK_LENGTH, video_type)
            else:
                frames_clips = np.array([data])

            return frames_clips, None, None

    def save_multi_process_no_labels(self, frames_clips, filename):
        """Save all the chunked data with multi-thread processing (no labels for finger)."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)

        count = 0
        input_path_name_list = []

        for i in range(len(frames_clips)):
            input_path_name = os.path.join(self.cached_path, f"{filename}_input_{count}.npy")
            input_path_name_list.append(input_path_name)
            np.save(input_path_name, frames_clips[i])
            count += 1

        return input_path_name_list

