import os
from io import open
import torch
import numpy

class TimeseriesTorch(object):
    def __init__(self, path, num_features):
        self.train = self.load_series(os.path.join(path, 'train.dat'), num_features)
        self.valid = self.load_series(os.path.join(path, 'dev.dat'), num_features)
        self.test = self.load_series(os.path.join(path, 'test.dat'), num_features)

    def load_series(self, path, num_features):
        """Load health data to Tensor objects."""
        assert os.path.exists(path)
        
        # Get sequence length
        with open(path, 'r', encoding="utf8") as f:
            timesteps = 0
            for line in f:
                timesteps += 1

        # Load multidimensional data with targets
        with open(path, 'r', encoding="utf8") as f:
            steps = torch.Tensor(timesteps, num_features)
            targets = torch.Tensor(timesteps)
            pos = 0
            for line in f:
                healthdata = line.split()
                values = []
                # remove encid,sepsis,severity,timestep
                # from multidimensional data
                for value in healthdata[4:4+num_features]:
                    values.append(float(value))
                # keep severity as target
                severity = healthdata[2]
                    
                steps[pos] = torch.from_numpy(numpy.array(values))
                targets[pos] = float(severity)
                pos += 1
        return [steps,targets]

class TimeseriesNumPy(object):
    def __init__(self, path, num_features):
        self.train = self.load_series(os.path.join(path, 'train.dat'), num_features)
        self.valid = self.load_series(os.path.join(path, 'dev.dat'), num_features)
        self.test = self.load_series(os.path.join(path, 'test.dat'), num_features)

    def load_series(self, path, num_features):
        """Load health data to NumPy arrays."""
        assert os.path.exists(path)
        
        # Get sequence length
        with open(path, 'r', encoding="utf8") as f:
            timesteps = 0
            for line in f:
                timesteps += 1

        # Load multidimensional data with targets
        with open(path, 'r', encoding="utf8") as f:
            stepsl = []
            targetsl = []
            for line in f:
                healthdata = line.split()
                values = []
                # remove encid,sepsis,severity,timestep
                # from multidimensional data
                for value in healthdata[4:4+num_features]:
                    values.append(float(value))
                # keep severity as target
                severity = float(healthdata[2])
                    
                stepsl.append(values)
                targetsl.append(severity)

        steps = numpy.array(stepsl,dtype=float)
        targets = numpy.array(targetsl,dtype=float)
        
        return [steps,targets]

