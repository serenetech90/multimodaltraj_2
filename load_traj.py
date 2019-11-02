import numpy as np
import pickle
import os
import random
import glob
import torch
import math

class DataLoader():

    def __init__(self, args, datasets=[0, 1, 2, 3, 4,5,6], sel=None ,start=0, processFrame=False , infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        parent_dir = '/home/siri0005/Documents/multimodaltraj/data'
        # '/fakepath/Documents/self-growing-spatial-graph/self-growing-gru-offline_avgPool/data'
        # '/Data/stanford_campus_dataset/annotations/'
        # List of data directories where world-coordinates data is stored

        self.data_dirs = [
            parent_dir + '/eth/hotel/',
            parent_dir + '/eth/univ/',
            parent_dir + '/ucy/zara/zara01/',
            parent_dir + '/ucy/zara/zara02/',
            parent_dir + '/ucy/univ/',
            parent_dir + '/annotation_tc.txt'
            ]

        # parent_dir + '/stanford/bookstore/',
        # parent_dir + '/stanford/hyang/',
        # parent_dir + '/stanford/coupa/',
        # parent_dir + '/stanford/deathCircle/',
        # parent_dir + '/stanford/gates/',
        # parent_dir + '/stanford/nexus/',
        # parent_dir + '/crowds/'
        # parent_dir + '/sdd/pedestrians/quad/',
        # parent_dir + '/sdd/pedestrians/hyang/',
        # parent_dir + '/sdd/pedestrians/coupa/',
        # parent_dir + '/sdd/gates/',
        # parent_dir + '/sdd/little/',
        # parent_dir + '/sdd/deathCircle/'
        # parent_dir + '/eth/',
        # parent_dir + '/hotel/',
        # parent_dir + '/zara/',
        # parent_dir + '/crowds/',

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir = parent_dir

        # Store the arguments
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.pred_len = args.pred_len
        self.obs_len = args.obs_len
        self.diff = self.obs_len

        # Validation arguments
        # self.val_fraction = 0.2
        # Define the path in which the process data would be stored
        self.current_dir = self.used_data_dirs[start]
        # self.frame_pointer = self.seed

        files = self.used_data_dirs[start] + "*.csv"  # '.txt'
        data_files = sorted(glob.glob(files))


        if sel is None:
            if len(data_files) > 1:
                print([x for x in range(len(data_files))])
                print(data_files)
                sel = input('select which file you want for loading:')

        self.dataset_pointer = sel #str(data_files[int(sel)])[-5]
        self.load_dataset(data_files[int(sel)])
        self.sel_file = self.current_dir + '/trajectories_{0}.cpkl'.format(int(self.dataset_pointer))

        # If the file doesn't exist or forcePreProcess is true
        processFrame = os.path.exists(self.current_dir + '/trajectories_{0}.cpkl'.format(int(self.dataset_pointer)))
        if not processFrame:
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.current_dir + '/trajectories_{0}.cpkl'.format(int(self.dataset_pointer)),
                                  seed=self.seed)

        # Load the processed data from the pickle file
        self.load_trajectories(self.current_dir + 'trajectories_{0}.cpkl'.format(int(self.dataset_pointer)))
        self.num_batches = int((len(self.frameList)/self.seq_length)/self.batch_size)

    def load_trajectories(self, data_file):
        ''' Load set of pre-processed trajectories from pickled file '''

        f = open(data_file, 'rb')
        self.trajectories = pickle.load(file=f)

        return self.trajectories

    def load_dataset(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file

        f = open(data_file, 'rb')
        self.raw_data = np.genfromtxt(fname=data_file, delimiter=',') # remove
        f.close()
        # Get all the data from the pickle file
        # self.data = self.raw_data[:,2:4]
        self.frameList = self.raw_data[0,:]
        self.pedsPerFrameList = self.raw_data[0:4,:]
        self.vislet = self.raw_data[4:6,:]
        self.seed = self.frameList[0]
        self.frame_pointer = self.seed

    def next_step(self, targets={}):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = {}
        # Iteration index
        b = -1
        pc = 1
        max_idx = max(self.frameList)
        # unique_frames = np.unique(self.frameList)

        max_log = math.log(max_idx, self.diff)
        idx = self.frame_pointer
        while b < self.batch_size:
            b += 1
            # Get the frame pointer for the current dataset
            # While there is still seq_length number of frames left in the current dataset
            # treat frames list indices as arithmetic series
            c = (max_idx - (idx + 1))
            if c <= 0:
                break
            else:
                c = math.log(abs(c), self.diff)
            if c <= max_log:
                # All the data in this sequence
                # try:
                rang = range(int(self.frame_pointer) , int(self.frame_pointer+(self.batch_size*self.obs_len)), self.diff)
                try:
                    traj_batch = [{idx:self.trajectories[idx]} for idx in rang]
                except KeyError:
                    break

                iter_traj = iter(traj_batch)
                for idx in traj_batch:#range(int(self.frame_pointer), int(self.frame_pointer+self.obs_len+1)):
                    (idx , _), = idx.items()
                    source_frame = self.trajectories[idx]
                    # Number of unique peds in this sequence of frames
                    if len(source_frame):
                        x_batch[idx] = source_frame
                        if pc % self.obs_len == 0:
                            try:
                                (idx_c,_), = next(iter_traj).items()
                            except StopIteration:
                                break
                            for i in range(int(self.pred_len)):
                                # idx_c = next(idx_c)
                                for j in range(len(self.trajectories[idx_c])):
                                    # if self.trajectories[idx_c] not in targets:
                                    (id, pos), = self.trajectories[idx_c][j].items()
                                    if len(targets) == 0:
                                        targets = {int(id): [pos]}
                                    elif int(id) not in targets:
                                        targets.update({int(id): [pos]})
                                    else:
                                        targets[int(id)].append(pos)
                    pc += 1
                    try:
                        next(iter_traj)
                    except StopIteration:
                        break
                    # iter_traj = iter(self.trajectories[tmp])
                self.frame_pointer += self.diff
            else:
                self.tick_frame_pointer(valid=False)

        return x_batch, targets, self.frame_pointer #, d

    # if idx_c + self.pred_len <= max_idx:
    #     idx_c += 1 # self.pred_len
    # else:
    #     self.tick_frame_pointer(valid=False, incr=self.diff)
    #     continue
    # except KeyError:
    #     self.tick_frame_pointer(valid=False, incr=self.diff)
    #     continue

    def frame_preprocess(self,  data_file, seed=0):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        frame_data = {i:{} for i in self.frameList}

        self.pedsPerFrameList = np.transpose(self.pedsPerFrameList)
        # self.frameList[1] - seed
        while self.frame_pointer <= max(self.frameList):
            x = [{ped: [pos_x, pos_y]} for (ind, ped, pos_x, pos_y) in self.pedsPerFrameList
                 if ind == self.frame_pointer]
            frame_data[self.frame_pointer] = x

            self.tick_frame_pointer(incr=self.diff)

        f = open(data_file, "wb")
        pickle.dump(frame_data, f, protocol=2)
        f.close()

    def tick_frame_pointer(self, valid=False, incr = 8):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            self.frame_pointer += incr
        else:
            # Go to the next dataset
            self.dataset_pointer += 1
            self.valid_frame_pointer = 0
            if self.dataset_pointer >= len(self.valid_data):
                self.dataset_pointer = 0

    def reset_data_pointer(self, valid=False, dataset_pointer=0, frame_pointer=0):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            # self.dataset_pointer = 0
            self.frame_pointer = self.seed
        else:
            self.dataset_pointer = dataset_pointer
            self.frame_pointer = frame_pointer
