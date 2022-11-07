# TODO: Switch to PyTorch
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from sklearn.cluster import KMeans
import heapq
# from data_helper import *
# from baselines import *
from scipy.optimize import linprog
from collections import defaultdict

import copy

import os
import random
import pickle


# helper functions
def get_key(date_string, secs):
    date = [int(d) for d in date_string[:-4].split('-')]
    units = [3000, 12, 30, 24, 60, 60]
    carry_over = 0
    date[-1] += secs
    for i in reversed(range(6)):
        tmp = date[i]
        date[i] = (date[i] + carry_over) % units[i]
        carry_over = (tmp + carry_over) // units[i]
    return "{}-{}-{}-{}-{}-{}".format(date[0], date[1], date[2], date[3], date[4], date[5])


class KnobPlanner:
    def __init__(self,
                categories,
                knob_cost,
                hours_plan_ahead,
                time_interval,
                knob_order,
                corr,
                config_belong,
                runtimes,
                input_hours=96,
                linear_programming=True,
                verbose=False):
        """
        KnobPlanner: Forecast workload and assign knobs to categories
        - categories (np([[float]])): K means cluster centers
        - knob_cost ([[float]]): cost of each placement (row) for each knob(col)
            per second
        - hours_plan_ahead (float): number of hours to plan ahead
        - time_interval (float): how many seconds is each data interval long.
            Needed for determining how many data points 1 input/output day
            consists of
        - knob_order (dict): knob name (as in processed files) and position in
            categories and knob_cost
        - corr: correlation matrix
        - config_belong: to which config does a runtime-cost pair belong (if
            several placements / config, then not 0,1,2,3, ...)
        - runtimes
        - input_hours (float): hours of history that forecast NN gets as input
        - linear_programming (Bool): Use linear programming or 0-1 knapsack to
          assign knob settings to categories
        """

        self.categories = categories
        self.knob_cost = knob_cost
        self.hours_plan_ahead = plan_ahead
        self.num_cluster = categories.shape[0]
        self.num_knobs = categories.shape[1]
        self.knob_place = knob_cost.shape[0]
        self.time_interval = time_interval
        self.knob_order = knob_order
        self.verbose = verbose
        self.runtimes = runtimes
        # print("rt constr", runtimes)
        self.corr = corr
        self.config_belong = config_belong
        # print("belong", config_belong)
        assert config_belong.shape == runtimes.shape == knob_cost.shape
        assert corr.shape == (self.num_cluster,self.num_cluster)

        # self.forecast = WorkloadForecast()
        self.forecast = Sequential()
        x_shape = (input_hours*2*int(3600/time_interval),)
        self.forecast.add(Dense(100, input_shape=x_shape, activation='relu'))
        self.forecast.add(Dense(30, activation='relu'))
        self.forecast.add(Dense(self.num_cluster, activation='relu'))
        self.forecast.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

        # hyperparams
        self.linear_programming = linear_programming
        self.input_hours = input_hours


    def assign_knobs_lin_prog(self, mixture, budget):
        # YOLO calls per second
        A = np.zeros((3*self.num_cluster+1, self.knob_place*self.num_cluster), dtype=float)
        b = np.zeros((3*self.num_cluster+1,), dtype=float)
        c = np.zeros((self.knob_place*self.num_cluster,), dtype=float)

        # enforce that distributions add up to 1
        for i in range(self.num_cluster):
            for j in range(i*self.knob_place, (i+1)*self.knob_place):
                A[2*i][j] = 1
                A[2*i+1][j] = -1
            b[2*i] = 1
            b[2*i+1] = -1

        # enforce runtime / buffering contraint
        # runtimes of different knob placements
        for i in range(self.num_cluster):
            for j in range(self.num_cluster):
                for k in range(self.knob_place):
                    A[2*self.num_cluster+i][j*self.knob_place+k] = self.runtimes[k] * self.corr[i][j] # mixture[j]
            b[2*self.num_cluster+i] = 0.0
        # print("rt", self.runtimes)

        # enforce below budget
        for i, m in enumerate(mixture):
            for j in range(self.knob_place):
                A[3*self.num_cluster][i*self.knob_place+j] \
                    = m*self.knob_cost[j] *self.hours_plan_ahead*3600
        b[3*self.num_cluster] = budget #/(self.hours_plan_ahead*3600)

        # score TODO maybe just put the mutilplying factors only where we estimate
        # the score. could be better for num stab?
        for i, center in enumerate(self.categories):
            for j in range(self.knob_place):
                c[i*self.knob_place+j] = -mixture[i] * center[self.config_belong[j]] \
                    * self.hours_plan_ahead*3600/self.time_interval


        # # debug
        # for i in range(3*self.num_cluster+1):
        #     for j in range(self.knob_place*self.num_cluster):
        #         if A[i][j] >= 0.0:
        #             print(" {}".format(round(A[i][j], 2)), end=" ")
        #         else:
        #             print(round(A[i][j], 2), end=" ")
        #
        #     print(" ||  {}".format(round(b[i], 2)))
        #     # print()
        # print("c", c)


        # Solve linear program
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0,1)) #, method='lstsq', options = {"presolve":True})

        print("\nKnob planner (lin solve):", res.message)
        print('Expected score:', round(-res.fun, ndigits=4))
        if self.verbose:
            sol = list(res.x)
            print("Solution linear program:")
            for i in range(self.num_cluster):
                for j in range(self.knob_place):
                    print(round(sol[i*self.knob_place+j], ndigits=4), end=" ")
                print()

        res_copy = copy.copy(res.x)

        return res_copy, -res.fun


    def assign_knobs_knap_sack(self, mixture, budget):
        raise NotImplementedError

        # TODO
        # populate heap
        heap = []
        for p, (c_i, c) in zip(mixture, enumerate(self.categories)):
            for i, k in enumerate(self.category_cost):
                # TODO: Is this the right way to calculate in the cost
                cost = p*24*3600*30/k - p*24*3600*30/150
                if c[i] - c[-1] > 0 and cost > 0:
                    heap.append([cost/(c[i] - c[-1]), cost, i, c_i])

        budget -= 24*3600/5

        # elimate all with cost 0
        # TODO: If pred. mixture is 0 you should still include in case that pred is wrong
        heap = [h for h in heap if h[0] > 0]

        heapq.heapify(heap)

        knob_map = [5]*self.num_cluster
        while len(heap) > 0:
            (_, cost, knob, center) = heapq.heappop(heap)

            if cost < budget:
                knob_map[center] = knob
                budget -= cost

                # elimate all cheaper knobs of that center
                heap = [h for h in heap if h[3] != center or self.categories[center][h[2]] > self.categories[center][knob]]

                # adjust costs of same center knobs
                for i, h in enumerate(heap):
                    if h[3] == center:
                        heap[i][1] = h[1] - cost
                        # denom = cluster_centers[centers][h[2]] - cluster_centers[centers][knob]
                        # if denom > 0:
                        heap[i][0] = heap[i][1]/(self.categories[center][h[2]] - self.categories[center][knob])

                heapq.heapify(heap)

        print("remaining budget:", budget)
        print(knob_map)


    def sample_input_output(self, score, start_day, end_day, num_samples,
                            keys_list, kmeans):
        points_per_hour = int(3600/self.time_interval)

        # history input_hours on from rand. start and histogram for output_days
        X = np.empty((num_samples, self.input_hours*points_per_hour*2))
        y = np.empty((num_samples, self.num_cluster))

        for i in range(num_samples):
            day = random.randrange(start_day, end_day-self.hours_plan_ahead/24)
            hour = random.randrange(24)
            start = "2021-11-{:02d}-{:02d}-00-00".format(day, hour)

            # find starting key
            l = 0
            while keys_list[l] < start:
                l += 1

            # get history (input vector)
            inp_v = []
            for j in range(l, l+points_per_hour*self.input_hours):
                sc = -1
                while sc == -1:
                    knob = random.randrange(6)
                    sc = score[keys_list[j]][knob]
                inp_v.append(knob)
                inp_v.append(sc)
            X[i,:] = inp_v

            # get histogram (output vector)
            histo_vecs = []
            for j in range(l+points_per_hour*self.input_hours, l+points_per_hour*(self.input_hours+self.hours_plan_ahead)):
                v = score[keys_list[j]]
                # assert all([x != -1 for x in v])
                if all([x != -1 for x in v]):
                    histo_vecs.append(v)
            labels = kmeans.predict(histo_vecs)
            histo = np.bincount(labels)
            y[i, :histo.shape[0]] = histo


        # for i in range(self.input_hours*points_per_hour*2):
        #     print("{},".format(X[0,i]), end="")


        # Cast to np array and normalize
        # TODO: When normalizing X you need to make sure that knob idx and score more or less same size
        X /= np.linalg.norm(X)
        y /= y.sum(axis=1)[:,None]
        return X, y


    def get_traindata(self, foldername, num_samples, num_test_samples=0,
        test_start=""):
        """
        Read in data from all workload processing files. Get execution histories
            (x) and target histograms (y).
        - foldername (string): path to workload processing files
        - test_start (string): where to start test split (e.g.
            "2021-11-15-00-00-00"). "" means no test split
        """

        # workload processing files must have this structure:
        # file (date-time.mp4),    knob name, second offset, runtime,           score
        # 2021-11-10-09-47-18.mp4, 75,        0,             6.203967332839966, 70

        # get all knob setting names
        # knob_names = set()
        # for file in os.listdir(foldername):
        #     with open(os.path.join(foldername, file), "r") as f:
        #         lines = f.readlines()
        #     for l in lines:
        #         if len(l.split(",")) < 5 or l.split(",")[3] == "NA" or l[:7] == "file_id":
        #             continue
        #         knob_name = l.split(",")[1]
        #         knob_names.add(knob_name)
        # knobs = dict()
        # for i, knob_name in enumerate(knob_names):
        #     knobs[knob_name] = i

        # build dict which maps time -> knob score vector
        score = defaultdict(lambda: [-1]*6)
        keys_list = []
        for file in os.listdir(foldername):
            with open(os.path.join(foldername, file), "r") as f:
                lines = f.readlines()

            for l in lines:
                if len(l.split(",")) < 5 or l.split(",")[3] == "NA" or l[:7] == "file_id":
                    continue

                key = get_key(l.split(",")[0], int(l.split(",")[2]))
                keys_list.append(key)
                knob = self.knob_order[l.split(",")[1]]
                sc = int(l.split(",")[4])
                score[key][knob] = sc

        # sample data points
        keys_list.sort() # for sampling
        # TODO: Just get k means from previous offline step
        kmeans = KMeans(n_clusters=self.num_cluster).fit(self.categories) # for building histograms
        kmeans.cluster_centers_ = self.categories
        start_day = 1 # minmum 1, day of Nov where data starts
        end_day = 23 # max 29, day of Nov where data ends
        X_train, y_train = self.sample_input_output(score, start_day, end_day,
            num_samples, keys_list, kmeans)
        return X_train, y_train, [], []


    def fit(self, data_folder="", weights_file=""):
        # load weights
        if weights_file != "":
            self.forecast.load_weights(weights_file)

        # train
        if train_data != "":
            # test_start = "2021-11-15-00-00-00"
            X_train, y_train, X_test, y_test = self.get_traindata(data_folder,
                num_samples=100)

            # fit
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath="100-150-80-30-6.ckpt",
                save_weights_only=True,
                monitor='val_mean_absolute_error',
                mode='max',
                save_best_only=True)

            self.forecast.fit(X_train, y_train, epochs=20, batch_size=128,
                            # validation_data=(X_test, y_test),
                            callbacks=[model_checkpoint_callback])


    def plan(self, input, budget, use_gt_histo=False, histogram=None):
        # predict input
        if use_gt_histo:
            # TODO: I think you need data_helper.py for this
            # histogram = gt_mixture("2021-11-05-09-46-46-cont.csv", "recursive_kmeans_centers")
            # # covid
            # histogram = np.array([0.12535677, 0.21901565, 0.07405493, 0.20308848, 0.37848417])
            #
            # histogram = np.array([0.12183039, 0.22210391, 0.07476686, 0.01886752, 0.56243131])
            #
            # # only boxes
            # histogram = np.array([0.50521163, 0.15577302, 0.02518716, 0.26182526, 0.05200293])
            #
            # # covid score
            # histogram = np.array([0.64522465, 0.04769037, 0.07938028, 0.02672185, 0.20098286])

            histogram = histogram

        else:
            histogram = np.array(self.forecast(input))[0]
            histogram /= histogram.sum()

        if self.verbose:
            print(histogram)

        # assign knobs to categories
        if self.linear_programming:
            return self.assign_knobs_lin_prog(histogram, budget)
        else:
            return self.assign_knobs_knap_sack(histogram, budget)



# ===========================================
# prelim, debug
# ===========================================

if __name__ == "__main__":

    debug = True
    if debug:
        # debug /prelim
        categories = np.array([
                                [0.19876441515714305,0.2665842943432608,0.07968149368514332,0.11781987918702441,0.03421197144420063,],
                                [771.4942196531788,945.3110549132957,320.49205202312123,364.0317919075147,82.75722543352592,],
                                [374.02707228579453,362.7608149595306,171.3402176946693,159.41027072285797,57.38403572425341,],
                              ])
        histogram = np.array([0.8515349185195576,0.06466998667321316,0.0837950948072292,])
        std_devs = np.array([
                               [5.7754953429820395,7.423810394314598,2.391213001954317,3.835365952601099,1.6950723215497423,],
                               [118.8861924338638,232.6228742641138,80.57378423391121,95.43783435006958,82.5754669646941,],
                               [132.1872597756308,123.76269274784866,69.21308364285483,67.08149835107194,60.351812208855684,],
                             ])
        corr = np.array([
                         [0.97561846187639,0.0020729798742483733,0.02230855824936163,],
                         [0.0019898697539797393,0.8010130246020261,0.19699710564399422,],
                         [0.24581473214285715,0.1328125,0.6213727678571429,],
                        ])

        knob_costs = [21900, 84900, 4765, 15095, 3635]

    #           VETL          no cloud       no buffer
    allowed = ["everything", "buffer_only", "cloud_only", "none"]
    allowed = allowed[0]

    plan_dict = {
                  0: (np.array([0]*len(knob_costs)), np.array(knob_costs), np.array([i for i in range(len(knob_costs))]))
                }

    with open("covid_bw.place", "rb") as f:
        hw_dict = pickle.load(f)

    budgets = np.linspace(24*3600*1000, 3628800000.0/2, 10) #np.linspace(5590080, 3628800000.0/2, 10)
    cloud_budgets = budgets
    cores_list = [2, 4, 8, 16, 24, 32, 48]

    plan_ahead = 24 # plan 48 hours ahead
    knob_order = { "10": 0, "20": 1, "30": 2, "50": 3, "75": 4, "200": 5 }

    buffering_allowed = True
    cloud_allowed = False

    if not buffering_allowed and not cloud_allowed:
        cloud_budgets = np.array([24*3600*1000])

    final_scores = [[] for _ in range(len(cores_list))]
    final_costs = [[] for _ in range(len(cores_list))]

    realtime = 5000
    num_secs = 5 # cost is over how many secs

    # get quality sorted. if a knob config doesnt fit into buffer, we can use
    # next worse one that fits
    quality_sort = []
    for i_qual in range(categories.shape[0]):
        quality_sort.append(np.argsort(-categories[i_qual]))

    # get runtimes for planning
    runtimes, knob_cost, config_belong = plan_dict[0]
    runtimes = np.array(runtimes) #- realtime
    knob_cost = np.array(knob_cost) / num_secs
    config_belong = np.array(config_belong)

    for budget in cloud_budgets:

        for i, num_cores in enumerate(cores_list):

            knob_cost = np.zeros(categories.shape[1])
            # cur = 0
            for r, c, b in zip(hw_dict[num_cores][0], hw_dict[num_cores][1], hw_dict[num_cores][2]):
                if c == 0:
                    knob_cost[b] = r

            runtimes = np.zeros(len(knob_cost))
            config_belong = np.array([i for i in range(len(knob_cost))])
            knob_cost = np.array(knob_cost)

            # plan
            kp = KnobPlanner(categories, knob_cost, plan_ahead, time_interval=2, knob_order=knob_order, verbose = True, corr=corr, runtimes=runtimes, config_belong= config_belong)
            plan, score = kp.plan(input=None, budget=budget, use_gt_histo=True, histogram=histogram)

            # get runtimes for hardware
            runtimes, knob_cost, config_belong = hw_dict[num_cores]
            runtimes = (np.array(runtimes) - realtime)
            knob_cost = np.array(knob_cost) / num_secs
            config_belong = np.array(config_belong)

            # get config priorities
            config_prio = []
            # runtime, cost, config = hw_dict[num_cores]
            for qual_sort in quality_sort:
                cluster_prio = []
                for c in qual_sort:
                    config_cluster_prio = []
                    # get all placements
                    for r, co, con in zip(runtimes, knob_cost, config_belong):
                        if con == c:

                            if not buffering_allowed and r > 0:
                                continue
                            if not cloud_allowed and co > 0:
                                continue
                            config_cluster_prio.append((r, co, con))
                    cluster_prio += sorted(config_cluster_prio, reverse=True)
                config_prio.append(cluster_prio)

            # knob switcher --> test on preproc input
            buffer_size = 180*1000
            buffer = 0

            input = "covid_proc_60.csv"
            runtimes, knob_cost, config_belong = plan_dict[0]
            target_place = plan.reshape((categories.shape[0], knob_cost.shape[0]))

            place_counts = np.ones(target_place.shape)

            file = open(input, 'r')
            file.readline()

            score_sum = 0
            cost_sum = 0
            # policy = 0
            use_knob = 2

            if not buffering_allowed and not cloud_allowed:
                none_dict = { 2: 2, 4: 2, 8: 3, 16: 0, 24: 0, 32: 0, 48: 1 }
                use_knob = none_dict[num_cores]

            consec_zeros = 0
            last_skip = 100

            while True:

                # print(buffer)
                line = file.readline()
                if not line:
                    break

                cur_score = int(line.split(",")[1+use_knob])

                score_sum += cur_score

                # nearest cluster (dynamics)
                dynamics = np.argmin(np.abs(categories[:, use_knob] - cur_score))

                if int(line.split(",")[1]) == 0:
                    consec_zeros += 1
                else:
                    consec_zeros = 0

                if not buffering_allowed and not cloud_allowed:
                    continue

                # all placements that overflow the buffer impossible
                buffer_space = buffer_size - buffer
                ratio_error = target_place[dynamics] - (place_counts[dynamics]/np.sum(place_counts[dynamics]))

                knob_place = np.argmax(ratio_error)

                # search start idx of config prio
                idx = 0
                while config_prio[dynamics][idx][2] != knob_place:
                    idx += 1

                # search for next best placement
                while config_prio[dynamics][idx][0] > buffer_space:
                    idx += 1

                # add cost etc.
                buffer = max(0, buffer + config_prio[dynamics][idx][0])
                cost_sum += config_prio[dynamics][idx][1]
                use_knob = config_prio[dynamics][idx][2]
                place_counts[dynamics][use_knob] += 1

            final_scores[i].append(score_sum)
            final_costs[i].append(cost_sum)

    # print
    print("cost,qual,cores")
    for j, budget in enumerate(cloud_budgets):
        for i, num_cores in enumerate(cores_list):
            # cost = num_cores*3600*1000*24+1.5*budget
            print("{},{},{}".format(final_costs[i][j], final_scores[i][j], num_cores))
