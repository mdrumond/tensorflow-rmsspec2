import argparse
import time
import sys
import os
import shutil
from subprocess import call

parser = argparse.ArgumentParser(description='Run a training experiment')

parser.add_argument('-e', '--experiment', action='append',
                    choices=['sgd', 'rmsprop', 'rmsprop-no-mom',
                             'rmsspectral',
                             'rmsspectral-no-mom'])
parser.add_argument('--stage-dir',
                    default='/data/cifar10/cifar10_train')
parser.add_argument('-f', '--experiment-folder',
                    default='/data/cifar10/results')
parser.add_argument('-n', '--number-images', type=int,
                    default=128000)
parser.add_argument('-g', '--gpu-number', type=int,
                    default=0)
parser.add_argument('-t', action='store_true',
                    help='Test run, do not run QEMU, just prints command')
parser.add_argument('--assync', action='store_true',
                    help='Run assynchronous models')
parser.add_argument('--approx', action='store_true',
                    help='Run use approximated sharp for RMSSpectral')
parser.add_argument('-F', '--force', action='store_true',
                    help='Force overwriting of results folders')
parser.add_argument('-s', '--training-scripts',
                    default='cifar10_train.py')

args = parser.parse_args()


def ask_yes_no(question):
    done = False
    ans = False

    yes = set(['yes', 'y', 'ye', ''])
    no = set(['no', 'n'])

    while(not done):
        print("%s [y/n]" % question)
        choice = input().lower()
        if choice in yes:
            ans = True
            done = True
        elif choice in no:
            ans = False
            done = True
        else:
            print("Please respond with 'yes' or 'no'")

    return ans


class Experiment(object):
    def __init__(self, name, method, batch_size, lr, lr_decay,
                 momentum=None, epsilon=None, rms_decay=None,
                 use_approx_sharp=False,
                 use_locking=False,
                 gpu_number=0,
                 stage_dir=args.stage_dir):
        self.name = name
        self._method = method
        self._batch_size = batch_size
        self._lr = lr
        self._lr_decay = lr_decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._rms_decay = rms_decay
        self._use_approx_sharp = use_approx_sharp
        self._use_locking = use_locking
        self._stage_dir = os.path.join(stage_dir,name)
        self._gpu_number = gpu_number
        
    def print_flags(self, n_images):

        momentum = ["--momentum=%f" % self._momentum] if self._momentum else []
        epsilon = ["--epsilon=%f" % self._epsilon] if self._epsilon else []
        rms_decay = (["--rms-decay=%f" % self._rms_decay] if self._rms_decay
                     else [])
        locking = ["--use-locking"] if self._use_locking else []
        approx = ["--use-approx-sharp"] if self._use_approx_sharp else []

        max_steps = n_images / self._batch_size
        return (["--training-method=%s" % self._method,
                 "--train-dir=%s" % self._stage_dir,
                 "--max-steps=%d" % max_steps,
                 "--batch-size=%d" % self._batch_size,
                 "--learning-rate=%f" % self._lr,
                 "--learning-rate-decay=%f" % self._lr_decay,
                 "--gpu-number=%d" % self._gpu_number] +
                momentum + epsilon + rms_decay + locking + approx)

    def __str__(self):
        str_momentum = ("momentum: %f\n" % self._momentum if self._momentum
                        else "")
        str_epsilon = ("epsilon: %f\n" % self._epsilon if self._epsilon
                       else "")
        str_rms_decay = ("rms_decay: %f\n" % self._rms_decay if self._rms_decay
                         else "")
        str_locking = ("locking\n" if self._use_locking else
                       "not locking\n")
        return (("name:%s/%s\n batch-size:%d\n"
                "lr:%f lr decay:%f\n" %
                 (self.name, self._method,
                  self._batch_size,
                  self._lr, self._lr_decay)) +
                str_locking + str_momentum +
                str_rms_decay + str_epsilon)


exp_dic = {
    "sgd": Experiment(name="sgd", method="sgd", batch_size=128, lr=.05,
                      lr_decay=.5,
                      use_locking=not args.assync,
                      gpu_number=args.gpu_number),

    "rmsprop": Experiment(name="rmsprop", method="rmsprop", batch_size=128,
                          lr=0.02, lr_decay=.5,
                          momentum=.9, epsilon=0.1, rms_decay=.9,
                          use_locking=not args.assync,
                          gpu_number=args.gpu_number),

    "rmsprop-no-mom": Experiment(name="rmsprop-no-mom", method="rmsprop",
                                 batch_size=128,
                                 lr=0.02, lr_decay=.5,
                                 momentum=0, epsilon=0.1, rms_decay=.9,
                                 use_locking=not args.assync),

    "rmsspectral":  Experiment(name="rmsspectral", method="rmsspectral",
                               batch_size=1024, lr=.01,
                               lr_decay=.5, momentum=.9, epsilon=0.05,
                               rms_decay=.98,
                               use_locking=not args.assync,
                               gpu_number=args.gpu_number,
                               use_approx_sharp=args.approx),

    "rmsspectral-no-mom":  Experiment(name="rmsspectral-no-mom",
                                      method="rmsspectral",
                                      batch_size=128,
                                      lr=.0001, lr_decay=.9, momentum=0,
                                      epsilon=0.000001, rms_decay=.9,
                                      use_locking=not args.assync,
                                      gpu_number=args.gpu_number,
                                      use_approx_sharp=args.approx)
}


def run_experiment(exp, n_images, experiment_folder):

    print(">>> Going to run %s <<<" % exp.name)

    exp_dir = os.path.join(experiment_folder, exp.name)
    if os.path.isdir(exp_dir):
        if not args.force:
            ans = ask_yes_no("Output folder exists! overwrite it?")
            if ans:
                shutil.rmtree(exp_dir)
            else:
                print("Refusing to overwrite the experiment folder! Giving up")
                sys.exit(1)
        else:
            shutil.rmtree(exp_dir)

    os.makedirs(exp_dir)

    exp_info_path = os.path.join(exp_dir, "experiment.txt")

    with open(exp_info_path, "w+") as f:
        f.write("%s\n" % time.strftime("%c"))
        f.write("number of images processed: %s\n" % n_images)
        f.write("params: \n%s\n" % str(exp))

    train_cmd = (["python3", args.training_scripts] +
                 exp.print_flags(n_images))
    if args.t:
        print(train_cmd)
    else:
        checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
        call(train_cmd)
        shutil.copytree(args.stage_dir, checkpoints_dir)

print(args)
if args.experiment:
    for exp in args.experiment:
        run_experiment(exp_dic[exp], args.number_images,
                       args.experiment_folder)
else:
    for _, exp in exp_dic.items():
        run_experiment(exp, args.number_images, args.experiment_folder)
