import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_load.replay_buffer import ReplayBuffer
from data_load.batchifier import Batchifier
from models.ppo_model import PPONetwork, PPOAgent, tf
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics
from sklearn.utils.extmath import softmax




DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=32
BPTT=10
asset_list=ASSET_LIST
IDX=0
NUM_EPOCHS = 100
INIT_PV=1000
NUM_HID=20
ASSETS = ASSET_LIST
LR = 3e-4
LOG_MINI_BATCH = 5000
randomize_train=False
overlapping_train=True
RB_FACTOR=3
OPTIMIZE_AFTER=RB_FACTOR*BSZ + 2

GAMMA = 0.99
MIX_FACTOR = 0.001
CLIP_NORM = 5.0


if __name__ == '__main__':

    batch_gen = Batchifier(data_path=DATA_PATH, bsz=1, bptt=BPTT, idx=IDX,
                           asset_list=ASSETS, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)



    agent = PPOAgent(bsz=1, clip_norm=CLIP_NORM, num_hid=NUM_HID,
                     num_assets=len(ASSETS), bptt=BPTT, lr=LR)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1,NUM_EPOCHS + 1):

            losses = []
            last_state = None
            last_action = None
            last_value = None
            actor_rewards = 0.0

            for idx, (bTrainX, bTrainY) in enumerate(batch_gen.load_train()):
                if (idx + 1) % OPTIMIZE_AFTER != 0:
                    actor_action, critic_value = agent.act_and_fetch(sess, last_state, last_action, last_value, actor_rewards, bTrainX, idx)
                    last_state, last_action, last_value = bTrainX, actor_action, critic_value

                    actor_rewards = np.dot(actor_action.flatten(), bTrainY[:,-1,:].flatten())
                    # actor_rewards += alloc_reward
                    continue
                else:
                    states, actions, values, advantage = agent.get_training_data()
                    assert len(states) == len(actions) and len(values) == len(advantage), "Error in setting up"
                    shuffle_idxs = np.arange(0, OPTIMIZE_AFTER - 2)
                    np.random.shuffle(shuffle_idxs)
                    bloss = 0.0
                    for batch in range(0, OPTIMIZE_AFTER - 2, BSZ):
                        loss, ratio = agent.train(sess, state=states[batch:batch+BSZ],
                                    actions=actions[batch:batch + BSZ],
                                    values=values[batch:batch + BSZ],
                                    advantage=advantage[batch:batch + BSZ]
                                    )
                        bloss += loss
                    losses.append(bloss/RB_FACTOR)
                    # print(losses)
            print("Epoch {} Loss: {}, validating...".format(epoch, np.mean(losses)))
            losses = []
            allocation_wts = []
            price_change_vec = []
            for bEvalX, bEvalYFat in batch_gen.load_test():
                bEvalY = bEvalYFat[:,-1,:]
                actor_action_test = agent.get_allocations(sess, bEvalX)
                assert bEvalY.shape == actor_action_test.shape
                price_change_vec.append(bEvalY)
                allocation_wts.append(actor_action_test)

            true_change_vec = np.concatenate(price_change_vec)
            allocation_wts = np.concatenate(allocation_wts)
            #
            # random_alloc_wts = softmax(np.random.random(allocation_wts.shape))
            test_date = "_".join(batch_gen.dp.test_dates[IDX])
            m = get_metrics(dt_range=test_date)
            print("Our Policy:")
            m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV)
            agent._reset_trajectories()