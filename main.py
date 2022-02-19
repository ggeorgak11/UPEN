
from train_options import TrainOptions
from trainer import Trainer
from tester import NavTester
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError


def nav_testing(options, scene_id):
    tester = NavTester(options, scene_id)
    tester.test_navigation()


if __name__ == '__main__':

    mp.set_start_method('forkserver', force=True)
    options = TrainOptions().parse_args()

    if options.is_train:

        trainer = Trainer(options)
        trainer.train()

    else:
        scene_ids = options.scenes_list

        # Create iterables for map function
        n = len(scene_ids)
        options_list = [options] * n
        args = [*zip(options_list, scene_ids)]

        # isolate OpenGL context in each simulator instance
        with Pool(processes=options.gpu_capacity) as pool:
            pool.starmap(nav_testing, args)

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
