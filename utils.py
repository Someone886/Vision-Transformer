# Utils.py
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
import flax
from flax.training import train_state, checkpoints
import os
import json
import tqdm
import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

import optax

from tqdm import  tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

class Trainer:
    
    H = W = 32
    C = 3
    metadata_filename = "metadata.txt"
    
    def __init__(self, ckpt_dir:str, model_class: nn.Module, model_h_params: dict, batch_size: int, optimizer_name: str,
                 optimizer_args: dict, preloaded=False):
        """
            :param ckpt_dir : Path to directory where checkpoints are saved/should be saved
            :param model_class : Class of flax module that represents the model
            :param model_h_params: *args that model_class takes
            :param batch_size
            :param optimizer_name: Choose from 'sgd', 'adam', 'adamw'
            :param optimizer_args: *args that the corresponding optax optimizer takes
            :param preloaded: If true all args except model_class are ignored and loaded from metadata
        """
        
        self.ckpt_dir = ckpt_dir
        
        self.model_class = model_class
        self.model_h_params = model_h_params
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.dropout_key = random.PRNGKey(0)
        self.dropout_step = 0
        
        assert os.path.isdir(ckpt_dir), "Path is not directory or does not exist"
        
        if not preloaded:
            self.initialise()
            self.opt_state = self.tx.init(self.params)
        else:
            self.__load_model()
        self.create_functions()
    
    def initialise(self):
        try:
            if self.optimizer_name == "sgd":
                self.tx = optax.sgd(**self.optimizer_args)
            elif self.optimizer_name == "adam":
                self.tx = optax.adam(**self.optimizer_args)
            elif self.optimizer_name == "adamw":
                self.tx = optax.adamw(**self.optimizer_args)
            else:
                raise ValueError("Invalid optimizer name")
        except:
            raise ValueError("Invalid optimizer args")

        key1, key2 = random.split(random.PRNGKey(0), 2)
        x = random.uniform(key1, (self.H, self.W, self.C))
        x.shape

        try:
            self.model = self.model_class(**self.model_h_params)
        except:
            raise ValueError("Invalid model args")

        model_init_param = self.model.init(key2, x, train=False)
        self.params = model_init_param
    
   
    def create_functions(self):
    
        def cross_entropy(params, x_batched, y_batched):
            # Define the squared loss for a single pair (x,y)
            def loss(x, y):
                if 'batch_stats' in params:
                    pred, new_params = self.model.apply({'params': params['params'], 'batch_stats': params['batch_stats']}, x, mutable=['batch_stats'])
                else:
                    dropout_rng = jax.random.fold_in(self.dropout_key, self.dropout_step)
                    pred = self.model.apply(params, x, train=True, rngs={'dropout': dropout_rng})
                labels_onehot = jax.nn.one_hot(y, num_classes=10)
                return optax.softmax_cross_entropy(logits=pred, labels=labels_onehot).mean()

            # Vectorize the previous to compute the average of the loss on all samples.
            return jnp.mean(jax.vmap(loss)(x_batched,y_batched), axis=0)

        def num_correct(params, x_batched, y_batched):
            def correct(x, y):
                if 'batch_stats' in params:
                    pred, new_parmas = self.model.apply({'params': params['params'], 'batch_stats': params['batch_stats']}, x, mutable=['batch_stats'])
                else:
                    pred = self.model.apply(params, x, train=False)
                pred_label = jnp.argmax(pred)
                return pred_label == y
            return jnp.sum(jax.vmap(correct)(x_batched, y_batched), axis=0)

        self.cross_entropy = jax.jit(cross_entropy)
        self.num_correct = jax.jit(num_correct)
            
    def get_accuracy(self, params, X, y):
        generator = yield_batches(X, y, B, False, False)
        acc = 0
        for X_batch, y_batch in generator:
            acc += self.num_correct(params, X_batch, y_batch)
        return acc/X.shape[0]
    
    def batched_apply(self, x_batched):
         return jax.vmap(self.model.apply,(None, 0))(self.params, x_batched)

    def visualize_test(self, num_img=4):
        # Visualize some images and results
        params = self.params
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        def visualise(x, y):
          if 'batch_stats' in params:
            logits, _ = self.model.apply({'params': params['params'], 'batch_stats': params['batch_stats']}, x, mutable=['batch_stats'])
          else:
            logits = self.model.apply(params, x, train=False)

          plt.imshow(x, interpolation='nearest')
          print(f"True label = {labels[int(y)]}", f"\tModel label = {labels[int(logits.argmax(axis=-1))]}")
          plt.show()

        i = 0
        data_loader = yield_batches(valid_X, valid_y, self.batch_size, False, False)
        for batch in data_loader:
          X, Y = batch
          X = X * jnp.array([0.2023, 0.1994, 0.2010]) + jnp.array([0.4914, 0.4822, 0.4465])
          plt.figure()
          train = False
          for i, (x, y) in enumerate(zip(X, Y)):
              visualise(x,y)
              if i >= num_img:
                break
          break
        return
  
    
    def __save_model(self, step):
        assert os.path.isdir(self.ckpt_dir), "Path is not directory or does not exist"
        save_metadata = {
            "batch_size": self.batch_size,
            "optimizer_name": self.optimizer_name,
            "optimizer_args": self.optimizer_args,
            "model_h_params": self.model_h_params
        }
        with open(os.path.join(self.ckpt_dir, self.metadata_filename), 'w') as f:
            f.write(json.dumps(save_metadata))
            
        state = train_state.TrainState.create(apply_fn=self.model.apply,
                                              params=self.params,
                                              tx=self.tx)
        try:
            checkpoints.save_checkpoint(ckpt_dir=self.ckpt_dir, target=state, step=step)
        except:
            print(f"\t\tSkipping because already saved better ckpt")
    
    def __load_model(self):
        assert os.path.isdir(self.ckpt_dir), "Path is not directory or does not exist"
        assert os.path.isfile(os.path.join(self.ckpt_dir, self.metadata_filename)), "Save metadata does not exist"
        
        ckpt = checkpoints.latest_checkpoint(self.ckpt_dir)
        assert not (ckpt is None), "Checkpt file not found"
        
        
        with open(os.path.join(self.ckpt_dir, self.metadata_filename)) as f:
            save_metadata = json.loads(f.read())
        self.batch_size = save_metadata["batch_size"]
        self.optimizer_name = save_metadata["optimizer_name"]
        self.optimizer_args = save_metadata["optimizer_args"]
        self.model_h_params = save_metadata["model_h_params"]
        self.initialise()

        state = checkpoints.restore_checkpoint(ckpt, target=None)
        self.params = state["params"]
        self.opt_state = state["opt_state"]
        
    def train_model(self, num_epochs, train_X, train_y, valid_X=None, valid_y=None, data_augmentation=True):
        loss_grad_fn = jax.value_and_grad(self.cross_entropy)
        num_iters = int(jnp.ceil(len(train_X)/self.batch_size))
        
        train_accs = []
        val_accs = []
        print(num_epochs)
        for epoch in range(num_epochs):
            train_batch_generator = yield_batches(train_X, train_y, self.batch_size, False, data_augmentation)
            print("\nEpoch =", epoch)
            for i, (batched_x, batched_y) in tqdm(enumerate(train_batch_generator), total=num_iters):
                self.dropout_step += 1
                loss, grads = loss_grad_fn(self.params, batched_x, batched_y)
                updates, opt_state = self.tx.update(grads, self.opt_state)
                params = optax.apply_updates(self.params, updates)
                self.opt_state, self.params = opt_state, params

            
            step=epoch
            train_acc = float(self.get_accuracy(self.params, train_X, train_y))
            train_accs.append(train_acc)
            print(f'Training acc :{train_acc}')
            if not valid_X is None and not valid_y is None:
                valacc = self.get_accuracy(self.params, valid_X, valid_y)
                val_accs.append(valacc)
                print(f'Validation acc {float(valacc)}')
                step=valacc
            
            try:    
                print(f'\t\tSaving checkpoint...')
                self.__save_model(step)
            except Exception as e:
                print(f"\t\tSkipping because already saved better ckpt")
        
        return train_accs, val_accs


def downloading(path, source='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if sys.version_info[0] == 2:
        from urllib import urlopen
    else:
        from urllib.request import urlopen
    
    import tarfile
    if not os.path.exists(path):
        os.makedirs(path)
    u = urlopen(source)
    with tarfile.open(fileobj=u, mode='r|gz') as f:
        f.extractall(path=path)
    u.close()


def loading(path):
    downloading(path)

    train_data = [pickle.load(open(os.path.join(path, 'cifar-10-batches-py', 'data_batch_%d' % (i + 1)), 'rb'),
                              encoding='latin1') for i in range(5)]

    X_train = np.vstack([d['data'] for d in train_data])
    y_train = np.hstack([np.asarray(d['labels'], np.int8) for d in train_data])

    test_data = pickle.load(open(os.path.join(path, 'cifar-10-batches-py', 'test_batch'), 'rb'), encoding='latin1')
    X_test = test_data['data']
    y_test = np.asarray(test_data['labels'], np.int8)

    X_train = X_train.reshape(-1, 3, 32, 32)
    X_train = X_train.transpose(0,2,3,1)
    X_test = X_test.reshape(-1, 3, 32, 32)
    X_test = X_test.transpose(0,2,3,1)

    return X_train, y_train, X_test, y_test



def identity(x):
    return x


def rotation(x, prob):
    return jax.lax.cond(prob, jnp.rot90, identity, x)


def horizontal_flip(x, prob):
    return jax.lax.cond(prob, jnp.fliplr, identity, x)
    
    
def vertical_flip(x, prob):
    return jax.lax.cond(prob, jnp.flipud, identity, x)


def pighead(x):
    r_start = random.randint(key=random.PRNGKey(1126), shape=(1,), minval=0, maxval=7)[0]
    c_start = random.randint(key=random.PRNGKey(1126), shape=(1,), minval=0, maxval=7)[0]
    x = jnp.pad(x, pad_width = np.array([[4, 4], [4, 4], [0, 0]]),\
                mode = "constant", constant_values=np.array([[0.4914], [0.4822], [0.4465]]))
    return jax.lax.dynamic_slice(x, (r_start, c_start, 0), (32, 32, 3))


def jigsaw(x, prob):
    return jax.lax.cond(prob, pighead, identity, x)


rotation_jit = jax.jit(jax.vmap(rotation, in_axes=(0, 0)))
horizontal_flip_jit = jax.jit(jax.vmap(horizontal_flip, in_axes=(0, 0)))
vertocal_flip_jit = jax.jit(jax.vmap(vertical_flip, in_axes=(0, 0)))
jigsaw_jit = jax.jit(jax.vmap(jigsaw, in_axes=(0, 0)))


def plot_samples(X, y, num_rows, num_columns, figsize=(12, 8), title=None):
    # Need to un-standardize X to clip pixel values to [0.0, 1.0]
    X = X * jnp.array([0.2023, 0.1994, 0.2010]) + jnp.array([0.4914, 0.4822, 0.4465])

    _, ax = plt.subplots(num_rows, num_columns, figsize=figsize)
    for i in range(num_rows * num_columns):
        img = X[i]
        label = str(y[i])
        ax[i // num_columns, i % num_columns].imshow(img.reshape(32, 32, 3))
        ax[i // num_columns, i % num_columns].set_title(label)
        ax[i // num_columns, i % num_columns].axis("off")

    if title:
        plt.suptitle(str(title))
    plt.show()


# If shuffle == True, then the original arrays would be shuffled as well.
# So if you want to keep the original ones, please make the copies before
# passing them to this function
def yield_batches(X, y, batch_size, shuffle=False,
                  contamination = True, key = 1126):
    assert len(X) == len(y)
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    for i in range(0, len(X) - batch_size + 1, batch_size):
        batch_indices = slice(i, i + batch_size)

        if contamination:
            yield contamination_factory(X[batch_indices], key = random.PRNGKey(key)), y[batch_indices]
        else:
            yield X[batch_indices], y[batch_indices]