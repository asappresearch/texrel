import copy


class WeightsRecorder(object):
    def __init__(self, num_timesteps_back):
        self.num_timesteps_back = num_timesteps_back
        self.records = []

    def record(self, state_dict):
        self.records.append(copy.deepcopy(state_dict))
        if len(self.records) > self.num_timesteps_back:
            self.records = self.records[-self.num_timesteps_back:]

    def get_record(self, timestep_back_idx):
        return self.records[-timestep_back_idx]
