import copy

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.optimizers import RMSprop

dim_actions = 2
dim_states = 4


def make_model(hidden_size=64, dim_states=4):
    model = Sequential()
    model.add(Dense(hidden_size,
                    input_shape=(dim_states, ),
                    activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(2))
    model.compile(loss='mse', optimizer='sgd')
    return model


class Agent():
    def __init__(self,
                 explore=0.1,
                 discount=0.9,
                 hidden_size=64,
                 memory_limit=5000):

        self.Q = make_model()
        self.Q_target = make_model()
        self.batch_size = 32
        self.step_count = 0
        self.target_switch = False
        self.target_update = 100

        # experience replay:
        # remember states to "reflect" on later
        self.memory = deque([], maxlen=memory_limit)

        self.explore = explore
        self.discount = discount

    def act(self, state):
        if np.random.rand() <= self.explore:
            return np.random.randint(0, 2)
        s = np.asarray(state).reshape(1, 4)
        q = self.Q.predict(s)
        choice = np.argmax(q[0])
        return choice

    def remember(self, state, action, next_state, reward):
        # the deque object will automatically keep a fixed length
        self.memory.append((state, action, next_state, reward))

    def _prep_batch(self, batch_size):
        self.step_count += 1
        if batch_size > self.memory.maxlen:
            Warning(
                'batch size should not be larger than max memory size. Setting batch size to memory size')
            batch_size = self.memory.maxlen

        batch_size = min(batch_size, len(self.memory))

        inputs = []
        targets = []

        # prep the batch
        # inputs are states, outputs are values over actions
        batch = random.sample(list(self.memory), batch_size)
        random.shuffle(batch)
        for state, action, next_state, reward in batch:
            inputs.append(state)
            s = np.asarray(state).reshape(1, 4)
            if self.target_switch:
                target = self.Q_target.predict(s)[0]
            else:
                target = self.Q.predict(s)[0]
            # debug, "this should never happen"
            assert not np.array_equal(state, next_state)

            # non-zero reward indicates terminal state
            if reward == 0:
                target[action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                ns = np.asarray(next_state).reshape(1, 4)
                if self.target_switch:
                    Q_sa = np.max(self.Q_target.predict(ns)[0])
                else:
                    Q_sa = np.max(self.Q.predict(ns)[0])
                target[action] = reward + self.discount * Q_sa
            targets.append(target)

        # to numpy matrices
        return np.vstack(inputs), np.vstack(targets)

    def flashback(self):
        inputs, targets = self._prep_batch(self.batch_size)
        loss = self.Q.train_on_batch(inputs, targets)
        if self.step_count % self.target_update == 0:
            self.update_target_network()
        pass

    def update_target_network(self):
        self.target_switch = True
        weights = self.Q.get_weights()
        self.Q_target.set_weights(weights)
        pass

    def save(self, fname):
        self.Q.save_weights(fname)

    def load(self, fname):
        self.Q.load_weights(fname)
        print(self.Q.get_weights())
