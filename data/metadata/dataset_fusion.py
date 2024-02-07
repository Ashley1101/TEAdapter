import json
import cv2
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch
# from basicsr.utils import img2tensor
import audioldm2.utilities.audio as Audio
from audioldm2.utilities.tools import load_json
from audioldm2.utils import random_uniform, normalize_wav


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

class AudioDataset(Dataset):
    def __init__(
        self,
        config=None,
        split="train",
        extra_cond_opt=["chord", "melody"],
        waveform_only=False,
        add_ons=[],
        dataset_json_path=None,  
        use_gt=True,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.config = config
        self.split = split
        self.extra_cond_opt = extra_cond_opt
        self.pad_wav_start_sample = 0  # If none, random choose
        self.trim_wav = False
        self.waveform_only = waveform_only
        self.add_ons = [eval(x) for x in add_ons]
        self.beat_start_end = {}
        self.use_gt = use_gt
        
        print("Add-ons:", self.add_ons)

        self.build_setting_parameters()

        # For an external dataset
        if dataset_json_path is not None:
            assert type(dataset_json_path) == str
            print("Load metadata from %s" % dataset_json_path)
            self.data = load_json(dataset_json_path)["data"]
            self.extra_cond_data = {}
            self.id2label, self.index_dict, self.num2label = {}, {}, {}
        else:
            self.metadata_root = load_json(self.config["metadata_root"])
            self.dataset_name = self.config["data"][self.split]
            assert split in self.config["data"].keys(), (
                "The dataset split %s you specified is not present in the config. You can choose from %s"
                % (split, self.config["data"].keys())
            )
            
        for extra_cond_name in self.extra_cond_opt:
            assert extra_cond_name in self.config["data"]["extra_cond"], (
                "The extra condition %s you specified is not present in the config. You can choose from %s"
                % (split, self.config["data"]["extra_cond"])
            )
        self.build_dataset()
        self.build_id_to_label()

        self.build_dsp()
        self.label_num = len(self.index_dict)
        print("Dataset initialize finished")

    def __getitem__(self, index):
        (
            fname,
            waveform,
            stft,
            log_mel_spec,
            ta_kaldi_fbank,
            label_vector,  # the one-hot representation of the audio class
            # the metadata of the sampled audio file and the mixup audio file (if exist)
            (datum, mix_datum),
            random_start,
            extra_fname,
            extra_cond_log_mel_spec,
            extra_cond_waveform,
        ) = self.feature_extraction(index)
        text = self.get_sample_text_caption(datum, mix_datum, label_vector)

        if self.use_gt:
            data = {
                "text": text,  # list
                "fname": self.text_to_filename(text)
                if (len(fname) == 0)
                else fname,  # list
                # tensor, [batchsize, class_num]
                "label_vector": "" if (label_vector is None) else label_vector.float(),
                # tensor, [batchsize, 1, samples_num]
                "waveform": "" if (waveform is None) else waveform.float(),
                # tensor, [batchsize, t-steps, f-bins]
                "stft": "" if (stft is None) else stft.float(),
                # tensor, [batchsize, t-steps, mel-bins]
                "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),
                "ta_kaldi_fbank": "" if (ta_kaldi_fbank is None) else ta_kaldi_fbank.float(),
                "duration": self.duration,
                "sampling_rate": self.sampling_rate,
                "random_start_sample_in_original_audio_file": random_start,
                "extra_cond_log_mel_spec": extra_cond_log_mel_spec,
                "extra_cond_waveform": extra_cond_waveform,
            }
        else:
            waveform_zero = torch.zeros_like(waveform)
            stft_zero = torch.zeros_like(stft)
            log_mel_spec_zero = torch.zeros_like(log_mel_spec)
            ta_kaldi_fbank_zero = torch.zeros_like(ta_kaldi_fbank)
            data = {
                "text": text,  # list
                "fname": self.text_to_filename(text)
                if (len(fname) == 0)
                else fname,  # list
                # tensor, [batchsize, class_num]
                "label_vector": "" if (label_vector is None) else label_vector.float(),
                # tensor, [batchsize, 1, samples_num]
                "waveform": "" if (waveform is None) else waveform_zero.float(),
                # tensor, [batchsize, t-steps, f-bins]
                "stft": "" if (stft is None) else stft_zero.float(),
                # tensor, [batchsize, t-steps, mel-bins]
                "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec_zero.float(),
                "ta_kaldi_fbank": "" if (ta_kaldi_fbank is None) else ta_kaldi_fbank_zero.float(),
                "duration": self.duration,
                "sampling_rate": self.sampling_rate,
                "random_start_sample_in_original_audio_file": random_start,
                "extra_cond_log_mel_spec": extra_cond_log_mel_spec,
                "extra_cond_waveform": extra_cond_waveform,
            }

        for add_on in self.add_ons:
            data.update(add_on(self.config, data, self.data[index]))

        if data["text"] is None:
            print("Warning: The model return None on key text", fname)
            data["text"] = ""

        return data

    def text_to_filename(self, text):
        return text.replace(" ", "_").replace("'", "_").replace('"', "_")

    def get_dataset_root_path(self, dataset):
        assert dataset in self.metadata_root.keys()
        return self.metadata_root[dataset]

    def get_dataset_metadata_path(self, dataset, key):
        # key: train, test, val, class_label_indices
        try:
            if dataset in self.metadata_root["metadata"]["path"].keys():
                return self.metadata_root["metadata"]["path"][dataset][key]
        except:
            raise ValueError(
                'Dataset %s does not metadata "%s" specified' % (dataset, key)
            )
            # return None

    def __len__(self):
        return len(self.data)

    def feature_extraction(self, index):
        if index > len(self.data) - 1:
            print(
                "The index of the dataloader is out of range: %s/%s"
                % (index, len(self.data))
            )
            index = random.randint(0, len(self.data) - 1)

        # Read wave file and extract feature
        while True:
            try:
                label_indices = np.zeros(self.label_num, dtype=np.float32)
                datum = self.data[index]
                (
                    log_mel_spec,
                    stft,
                    mix_lambda,
                    waveform,
                    random_start,
                    p_start,
                ) = self.read_audio_file(datum["wav"], type='audio')

                waveform = torch.FloatTensor(waveform)
                ta_kaldi_fbank = self.extract_kaldi_fbank_feature(waveform=waveform, sampling_rate=self.sampling_rate, log_mel_spec=log_mel_spec)
                
                extra_cond_log_mel_spec = {}
                extra_cond_waveform = {}
                for extra_cond_name in self.extra_cond_opt:
                    if extra_cond_name == 'chord':
                        extra_datum = self.extra_cond_data[extra_cond_name][index]
                        (
                          log_mel_spec_,
                          stft_,
                          mix_lambda_,
                          waveform_,
                          random_start_,
                          p_start_,
                        ) = self.read_audio_file(extra_datum["wav"], type=extra_cond_name, random_start=random_start, p_start=p_start)
                    else:
                        extra_datum = self.extra_cond_data[extra_cond_name][index]
                        (
                          log_mel_spec_,
                          stft_,
                          mix_lambda_,
                          waveform_,
                          random_start_,
                          p_start_,
                        ) = self.read_audio_file(extra_datum["wav"], type='audio')
                    extra_cond_log_mel_spec[extra_cond_name]=log_mel_spec_
                    extra_cond_waveform[extra_cond_name]=waveform_

                mix_datum = None
                if self.label_num > 0 and "labels" in datum.keys():
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] = 1.0

                # If the key "label" is not in the metadata, return all zero vector
                label_indices = torch.FloatTensor(label_indices)
                break
            except Exception as e:
                index = (index + 1) % len(self.data)
                print(
                    "Error encounter during audio feature extraction: ", e, datum["wav"]
                )
                continue

        # The filename of the wav file
        fname = datum["wav"]
        extra_fname = {}
        for extra_cond_name in self.extra_cond_opt:
            extra_fname[extra_cond_name]=extra_datum["wav"]
        # t_step = log_mel_spec.size(0)
        # waveform = torch.FloatTensor(waveform[..., : int(self.hopsize * t_step)])


        return (
            fname,
            waveform,
            stft,
            log_mel_spec,
            ta_kaldi_fbank,
            label_indices,
            (datum, mix_datum),
            random_start,
            extra_fname,
            extra_cond_log_mel_spec,
            extra_cond_waveform,
        )

    # def augmentation(self, log_mel_spec):
    #     assert torch.min(log_mel_spec) < 0
    #     log_mel_spec = log_mel_spec.exp()

    #     log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
    #     # this is just to satisfy new torchaudio version.
    #     log_mel_spec = log_mel_spec.unsqueeze(0)
    #     if self.freqm != 0:
    #         log_mel_spec = self.frequency_masking(log_mel_spec, self.freqm)
    #     if self.timem != 0:
    #         log_mel_spec = self.time_masking(
    #             log_mel_spec, self.timem)  # self.timem=0

    #     log_mel_spec = (log_mel_spec + 1e-7).log()
    #     # squeeze back
    #     log_mel_spec = log_mel_spec.squeeze(0)
    #     log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
    #     return log_mel_spec

    def build_setting_parameters(self):
        # Read from the json config
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        # self.freqm = self.config["preprocessing"]["mel"]["freqm"]
        # self.timem = self.config["preprocessing"]["mel"]["timem"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.mixup = self.config["augmentation"]["mixup"]

        # Calculate parameter derivations
        # self.waveform_sample_length = int(self.target_length * self.hopsize)

        # if (self.config["balance_sampling_weight"]):
        #     self.samples_weight = np.loadtxt(
        #         self.config["balance_sampling_weight"], delimiter=","
        #     )

        if "train" not in self.split:
            self.mixup = 0.0
            # self.freqm = 0
            # self.timem = 0

    def _relative_path_to_absolute_path(self, metadata, dataset_name):
        root_path = self.get_dataset_root_path(dataset_name)
        for i in range(len(metadata["data"])):
            assert "wav" in metadata["data"][i].keys(), metadata["data"][i]
            assert metadata["data"][i]["wav"][0] != "/", (
                "The dataset metadata should only contain relative path to the audio file: "
                + str(metadata["data"][i]["wav"])
            )
            metadata["data"][i]["wav"] = os.path.join(
                root_path, metadata["data"][i]["wav"]
            )
        return metadata

    def build_dataset(self):
        self.data = []
        self.extra_cond_data = {}
        print("Build dataset split %s from %s" % (self.split, self.dataset_name))
        if type(self.dataset_name) is str:
            data_json = load_json(
                self.get_dataset_metadata_path(self.dataset_name, key=self.split)
            )
            data_json = self._relative_path_to_absolute_path(
                data_json, self.dataset_name
            )
            self.data = data_json["data"]

            # load meta data of chord audio (start time, end time)
            beat_start_end_json = load_json(
                self.get_dataset_metadata_path(self.dataset_name, key="beat_start_end")
            )["data"]
            for row in beat_start_end_json:
                track_id = row['trackid']
                self.beat_start_end[track_id] = {"start_time": row['start_time'], "end_time": row['end_time']}

            for extra_cond_name in self.extra_cond_opt:
                print("Add extra condition named %s" % extra_cond_name)
                extra_data_json = load_json(
                    self.get_dataset_metadata_path(self.dataset_name, key=self.split+"_"+extra_cond_name)
                )
                extra_data_json = self._relative_path_to_absolute_path(
                    extra_data_json, self.dataset_name
                )
                self.extra_cond_data[extra_cond_name] = extra_data_json["data"]
          
            
        elif type(self.dataset_name) is list:
            beat_start_end_json = []
            for dataset_name in self.dataset_name:
                data_json = load_json(
                    self.get_dataset_metadata_path(dataset_name, key=self.split)
                )
                data_json = self._relative_path_to_absolute_path(
                    data_json, dataset_name
                )
                self.data += data_json["data"]

                # load meta data of chord audio (start time, end time)
                tmp_json = load_json(
                    self.get_dataset_metadata_path(dataset_name, key="beat_start_end")
                )
            beat_start_end_json += tmp_json["data"]
            for row in beat_start_end_json:
                track_id = row['trackid']
                self.beat_start_end[track_id] = {"start_time": row['start_time'], "end_time": row['end_time']}

            for extra_cond_name in self.extra_cond_opt:
                print("Add extra condition named %s" % extra_cond_name)
                extra_data_json = load_json(
                    self.get_dataset_metadata_path(dataset_name, key=self.split+"_"+ extra_cond_name)
                )
                extra_data_json = self._relative_path_to_absolute_path(
                    extra_data_json, dataset_name
                )
                self.extra_cond_data[extra_cond_name] += extra_data_json["data"]

        else:
            raise Exception("Invalid data format")
        

        print("Data size: {}".format(len(self.data)))

    def build_dsp(self):
        self.STFT = Audio.stft.TacotronSTFT(
            self.config["preprocessing"]["stft"]["filter_length"],
            self.config["preprocessing"]["stft"]["hop_length"],
            self.config["preprocessing"]["stft"]["win_length"],
            self.config["preprocessing"]["mel"]["n_mel_channels"],
            self.config["preprocessing"]["audio"]["sampling_rate"],
            self.config["preprocessing"]["mel"]["mel_fmin"],
            self.config["preprocessing"]["mel"]["mel_fmax"],
        )
        # self.stft_transform = torchaudio.transforms.Spectrogram(
        #     n_fft=1024, hop_length=160
        # )
        # self.melscale_transform = torchaudio.transforms.MelScale(
        #     sample_rate=16000, n_stft=1024 // 2 + 1, n_mels=64
        # )

    def build_id_to_label(self):
        id2label = {}
        id2num = {}
        num2label = {}
        class_label_indices_path = self.get_dataset_metadata_path(
            dataset=self.config["data"]["class_label_indices"],
            key="class_label_indices",
        )
        if class_label_indices_path is not None:
            df = pd.read_csv(class_label_indices_path)
            for _, row in df.iterrows():
                index, mid, display_name = row["index"], row["mid"], row["display_name"]
                id2label[mid] = display_name
                id2num[mid] = index
                num2label[index] = display_name
            self.id2label, self.index_dict, self.num2label = id2label, id2num, num2label
        else:
            self.id2label, self.index_dict, self.num2label = {}, {}, {}

    def resample(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        # waveform = librosa.resample(waveform, sr, self.sampling_rate)
        return waveform

        # if sr == 16000:
        #     return waveform
        # if sr == 32000 and self.sampling_rate == 16000:
        #     waveform = waveform[::2]
        #     return waveform
        # if sr == 48000 and self.sampling_rate == 16000:
        #     waveform = waveform[::3]
        #     return waveform
        # else:
        #     raise ValueError(
        #         "We currently only support 16k audio generation. You need to resample you audio file to 16k, 32k, or 48k: %s, %s"
        #         % (sr, self.sampling_rate)
        #     )

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

    def random_segment_wav(self, waveform, target_length, type='audio', random_start=None, filename=None, sr=None):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0
        
        chord_start = (int)(self.beat_start_end[filename]['start_time'] * sr)
        chord_end = (int)(self.beat_start_end[filename]['end_time'] * sr)
        if type == 'audio':
            # Original Audio
            if chord_end - chord_start < target_length:
                random_start = int(random_uniform(max(0, chord_end-target_length), min(chord_start, waveform_length - target_length)))
            else:
                random_start = int(random_uniform(chord_start, chord_end-target_length))
        else:
            # chord audio
            random_start = (int)(random_start * sr / self.sampling_rate)
            random_start -= chord_start

        return waveform[:, random_start : random_start + target_length], random_start


    def pad_wav(self, waveform, target_length, type='audio', t_start=None, p_start=None, filename=None, sr = 48000):
        waveform_length = waveform.shape[-1]
        # assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform, 0

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if type == 'audio':
            if self.pad_wav_start_sample is None:
                rand_start = int(random_uniform(0, target_length - waveform_length))
            else:
                rand_start = 0
        else:
            chord_start = (int)(self.beat_start_end[filename]["start_time"] * self.sampling_rate) 
            chord_end = (int)(self.beat_start_end[filename]["end_time"] * self.sampling_rate) 
            if t_start > 0:
                rand_start = chord_start - t_start
            else:
                rand_start = p_start + chord_start

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav, rand_start

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, 
                      filename, 
                      type='audio', 
                      random_start=None, 
                      p_start=None):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)
        trackid = os.path.splitext(os.path.basename(filename))[0]
        if type == 'audio':
            waveform, random_start = self.random_segment_wav(
                    waveform, target_length=int(sr * self.duration), type=type, filename=trackid, sr=sr)
        else:
            waveform, _ = self.random_segment_wav(
                    waveform, int(sr * self.duration), type=type, random_start=random_start, filename=trackid, sr=sr)

        waveform = self.resample(waveform, sr)
        if type == 'audio':
            random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = normalize_wav(waveform)

        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        if type == 'audio':
            waveform, p_start = self.pad_wav(
                waveform, int(self.sampling_rate * self.duration), type=type, filename=trackid
            )
        else:
            waveform, p_start = self.pad_wav(
                waveform, int(self.sampling_rate * self.duration), type=type, t_start=random_start, p_start=p_start, filename=trackid, sr=sr
            )
        return waveform, random_start, p_start

    def mix_two_waveforms(self, waveform1, waveform2):
        mix_lambda = np.random.beta(5, 5)
        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
        return normalize_wav(mix_waveform), mix_lambda

    def read_audio_file(self, filename, type='audio', random_start=None, p_start=None):
        if os.path.exists(filename):
            waveform, random_start, p_start = self.read_wav_file(filename, type, random_start, p_start)
        else:
            print(
                'Warning [dataset.py]: The wav path "',
                filename,
                '" is not find in the metadata. Use empty waveform instead.',
            )
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0

        mix_lambda = 0.0
        # log_mel_spec, stft = self.wav_feature_extraction_torchaudio(waveform) # this line is faster, but this implementation is not aligned with HiFi-GAN
        if not self.waveform_only:
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            # Load waveform data only
            # Use zero array to keep the format unified
            log_mel_spec, stft = None, None

        return log_mel_spec, stft, mix_lambda, waveform, random_start, p_start

    def get_sample_text_caption(self, datum, mix_datum, label_indices):
        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += " " + self.label_indices_to_text(mix_datum, label_indices)
        return text

    # This one is significantly slower than "wav_feature_extraction_torchaudio" if num_worker > 1
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft

    # @profile
    # def wav_feature_extraction_torchaudio(self, waveform):
    #     waveform = waveform[0, ...]
    #     waveform = torch.FloatTensor(waveform)

    #     stft = self.stft_transform(waveform)
    #     mel_spec = self.melscale_transform(stft)
    #     log_mel_spec = torch.log(mel_spec + 1e-7)

    #     log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    #     stft = torch.FloatTensor(stft.T)

    #     log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
    #     return log_mel_spec, stft

    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

    def _read_datum_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        random_index = torch.randint(0, len(caption_keys), (1,))[0].item()
        return datum[caption_keys[random_index]]

    def _is_contain_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        return len(caption_keys) > 0

    def label_indices_to_text(self, datum, label_indices):
        if self._is_contain_caption(datum):
            return self._read_datum_caption(datum)
        elif "label" in datum.keys():
            name_indices = torch.where(label_indices > 0.1)[0]
            # description_header = "This audio contains the sound of "
            description_header = ""
            labels = ""
            for id, each in enumerate(name_indices):
                if id == len(name_indices) - 1:
                    labels += "%s." % self.num2label[int(each)]
                else:
                    labels += "%s, " % self.num2label[int(each)]
            return description_header + labels
        else:
            return ""  # TODO, if both label and caption are not provided, return empty string

    def frequency_masking(self, log_mel_spec, freqm):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(random_uniform(freqm // 8, freqm))
        mask_start = int(random_uniform(start=0, end=freq - mask_len))
        log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
        return log_mel_spec

    def time_masking(self, log_mel_spec, timem):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(random_uniform(timem // 8, timem))
        mask_start = int(random_uniform(start=0, end=tsteps - mask_len))
        log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
        return log_mel_spec


    def extract_kaldi_fbank_feature(self, waveform, sampling_rate, log_mel_spec):
        norm_mean = -4.2677393
        norm_std = 4.5689974

        if sampling_rate != 16000:
            waveform_16k = torchaudio.functional.resample(
                waveform, orig_freq=sampling_rate, new_freq=16000
            )
        else:
            waveform_16k = waveform

        waveform_16k = waveform_16k - waveform_16k.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform_16k,
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )

        TARGET_LEN = log_mel_spec.size(0)

        # cut and pad
        n_frames = fbank.shape[0]
        p = TARGET_LEN - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[:TARGET_LEN, :]

        fbank = (fbank - norm_mean) / (norm_std * 2)

        return fbank  # [500 (duration), 128]


